"""
tests/test_memory_index.py
==========================
Tests for Phase 2.3 — Memory-Efficient Index Structures.

Covers:
  - SortedPostingsIndex  (compact sorted postings list)
  - VarInt encode / decode + delta coding
  - CompressedBM25       (delta+VarInt compressed index)
  - MmapBM25             (memory-mapped backend)
  - StreamingBM25Builder (chunked on-disk ingestion)
"""

from __future__ import annotations

import os
import struct
import tempfile
from pathlib import Path
from typing import List

import pytest

# ---------------------------------------------------------------------------
# The module under test lives in flashbm25/memory_index.py
# ---------------------------------------------------------------------------
from flashbm25.memory_index import (
    CompressedBM25,
    MmapBM25,
    StreamingBM25Builder,
    _PyCompressedIndex,
    _encode_postings,
    _decode_postings,
    _varint_encode,
    _varint_decode,
)


# ===========================================================================
# Helpers
# ===========================================================================

SAMPLE_CORPUS: List[List[str]] = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "lay", "on", "the", "rug"],
    ["cats", "and", "dogs", "are", "great", "pets"],
    ["information", "retrieval", "with", "bm25", "is", "fast"],
    ["fast", "bm25", "index", "compressed", "delta", "varint"],
]


def _simple_tokenize(text: str) -> List[str]:
    return text.lower().split()


# ===========================================================================
# 1. VarInt encoding / decoding
# ===========================================================================

class TestVarInt:
    """Tests for the pure-Python VarInt helpers."""

    @pytest.mark.parametrize("value", [0, 1, 127, 128, 255, 300, 16383, 16384,
                                        2**21 - 1, 2**28, 2**35, 2**56])
    def test_roundtrip(self, value: int) -> None:
        encoded = _varint_encode(value)
        decoded, pos = _varint_decode(encoded, 0)
        assert decoded == value
        assert pos == len(encoded)

    def test_small_values_are_single_byte(self) -> None:
        for v in range(128):
            assert len(_varint_encode(v)) == 1

    def test_128_is_two_bytes(self) -> None:
        assert len(_varint_encode(128)) == 2

    def test_sequential_decode(self) -> None:
        """Multiple VarInts packed into one buffer decode correctly."""
        values = [0, 1, 127, 128, 300, 16384, 2**21]
        buf = b"".join(_varint_encode(v) for v in values)
        pos = 0
        for expected in values:
            got, pos = _varint_decode(buf, pos)
            assert got == expected
        assert pos == len(buf)

    def test_overflow_raises(self) -> None:
        # Craft a 10-byte continuation sequence to trigger overflow
        bad = bytes([0x80] * 10 + [0x01])
        with pytest.raises(ValueError, match="overflow"):
            _varint_decode(bad, 0)

    def test_truncated_raises(self) -> None:
        with pytest.raises(ValueError):
            _varint_decode(bytes([0x80]), 0)  # continuation set but no next byte


# ===========================================================================
# 2. Delta-coded posting list encoding
# ===========================================================================

class TestDeltaEncoding:
    """Tests for encode_postings / decode_postings."""

    def test_roundtrip_single_entry(self) -> None:
        entries = [(42, 3.0)]
        blob = _encode_postings(entries)
        decoded, _ = _decode_postings(blob, 0)
        assert len(decoded) == 1
        assert decoded[0][0] == 42
        assert abs(decoded[0][1] - 3.0) < 1e-5

    def test_roundtrip_multiple_entries(self) -> None:
        entries = [(0, 1.0), (5, 2.0), (100, 1.0), (999, 4.0)]
        blob = _encode_postings(entries)
        decoded, _ = _decode_postings(blob, 0)
        assert [(d, round(t, 4)) for d, t in decoded] == [
            (0, round(1.0, 4)), (5, round(2.0, 4)),
            (100, round(1.0, 4)), (999, round(4.0, 4))
        ]

    def test_empty_list(self) -> None:
        blob = _encode_postings([])
        decoded, _ = _decode_postings(blob, 0)
        assert decoded == []

    def test_compression_ratio(self) -> None:
        """Delta+VarInt should be smaller than raw (uint64 + float32) pairs."""
        # 1000 consecutive doc IDs with TF=1
        entries = [(i, 1.0) for i in range(1000)]
        blob = _encode_postings(entries)
        raw_size = 1000 * (8 + 4)  # uint64 + float32
        assert len(blob) < raw_size, (
            f"Expected compressed ({len(blob)}) < raw ({raw_size})"
        )


# ===========================================================================
# 3. _PyCompressedIndex — build, lookup, save, load
# ===========================================================================

class TestPyCompressedIndex:
    """Tests for the pure-Python CompressedIndex fallback."""

    def test_build_and_lookup(self) -> None:
        idx = _PyCompressedIndex.build(SAMPLE_CORPUS)
        assert idx.num_docs == len(SAMPLE_CORPUS)
        assert idx.num_terms > 0
        entries = idx.lookup("bm25")
        assert len(entries) == 2  # appears in docs 3 and 4
        doc_ids = [e[0] for e in entries]
        assert 3 in doc_ids and 4 in doc_ids

    def test_lookup_missing_term(self) -> None:
        idx = _PyCompressedIndex.build(SAMPLE_CORPUS)
        assert idx.lookup("nonexistent_xyz") == []

    def test_avg_dl(self) -> None:
        idx = _PyCompressedIndex.build(SAMPLE_CORPUS)
        expected = sum(len(d) for d in SAMPLE_CORPUS) / len(SAMPLE_CORPUS)
        assert abs(idx.avg_dl - expected) < 1e-6

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        idx = _PyCompressedIndex.build(SAMPLE_CORPUS)
        path = tmp_path / "test.fbcidx"
        idx.save(path)
        assert path.exists()
        loaded = _PyCompressedIndex.load(path)
        assert loaded.num_docs  == idx.num_docs
        assert loaded.num_terms == idx.num_terms
        assert abs(loaded.avg_dl - idx.avg_dl) < 1e-6
        assert loaded.lookup("bm25") == idx.lookup("bm25")

    def test_invalid_magic_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.fbcidx"
        bad.write_bytes(b"BADMAGIC" + b"\x00" * 100)
        with pytest.raises(ValueError, match="magic"):
            _PyCompressedIndex.load(bad)

    def test_compressed_bytes_smaller_than_raw(self) -> None:
        # Large corpus to make the ratio meaningful
        corpus = [["word"] * 50 for _ in range(200)]
        idx = _PyCompressedIndex.build(corpus)
        raw = idx.num_docs * 50 * (8 + 4)  # worst-case raw
        assert idx.compressed_bytes() < raw


# ===========================================================================
# 4. CompressedBM25 — high-level API
# ===========================================================================

class TestCompressedBM25:
    """Tests for the CompressedBM25 public wrapper."""

    def test_repr(self) -> None:
        bm = CompressedBM25(SAMPLE_CORPUS)
        r = repr(bm)
        assert "CompressedBM25" in r
        assert "num_docs=5" in r

    def test_get_scores_length(self) -> None:
        bm = CompressedBM25(SAMPLE_CORPUS)
        scores = bm.get_scores(["bm25", "fast"])
        assert len(scores) == len(SAMPLE_CORPUS)

    def test_relevant_doc_scores_higher(self) -> None:
        bm = CompressedBM25(SAMPLE_CORPUS)
        scores = bm.get_scores(["bm25"])
        # docs 3 and 4 contain "bm25"; doc 0 does not
        assert scores[3] > scores[0] or scores[4] > scores[0]

    def test_get_top_n(self) -> None:
        bm = CompressedBM25(SAMPLE_CORPUS)
        top = bm.get_top_n(["bm25", "fast"], n=2)
        assert len(top) == 2
        # results must be (score, doc_id) pairs, sorted descending by score
        assert top[0][0] >= top[1][0]

    def test_zero_scores_for_absent_query(self) -> None:
        bm = CompressedBM25(SAMPLE_CORPUS)
        scores = bm.get_scores(["completelymissingterm_xyz"])
        assert all(s == 0.0 for s in scores)

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        bm = CompressedBM25(SAMPLE_CORPUS)
        path = tmp_path / "bm25.fbcidx"
        bm.save(path)
        bm2 = CompressedBM25.load(path)
        scores1 = bm.get_scores(["bm25", "fast"])
        scores2 = bm2.get_scores(["bm25", "fast"])
        for s1, s2 in zip(scores1, scores2):
            assert abs(s1 - s2) < 1e-5

    def test_k1_b_affect_scores(self) -> None:
        bm_default = CompressedBM25(SAMPLE_CORPUS, k1=1.5, b=0.75)
        bm_custom  = CompressedBM25(SAMPLE_CORPUS, k1=2.0, b=0.5)
        s1 = bm_default.get_scores(["bm25"])
        s2 = bm_custom.get_scores(["bm25"])
        # They should differ for at least one document
        assert any(abs(a - b) > 1e-6 for a, b in zip(s1, s2))

    def test_num_docs_and_terms(self) -> None:
        bm = CompressedBM25(SAMPLE_CORPUS)
        assert bm.num_docs == 5
        assert bm.num_terms > 10


# ===========================================================================
# 5. MmapBM25 — memory-mapped backend
# ===========================================================================

class TestMmapBM25:
    """Tests for the MmapBM25 memory-mapped index backend."""

    def test_loads_and_queries(self, tmp_path: Path) -> None:
        # Build and save with CompressedBM25 first
        bm = CompressedBM25(SAMPLE_CORPUS)
        path = tmp_path / "mmap_test.fbcidx"
        bm.save(path)

        mmap_bm = MmapBM25(path)
        assert mmap_bm.num_docs == bm.num_docs
        assert mmap_bm.num_terms == bm.num_terms

        scores_normal = bm.get_scores(["cat", "bm25"])
        scores_mmap   = mmap_bm.get_scores(["cat", "bm25"])
        for s1, s2 in zip(scores_normal, scores_mmap):
            assert abs(s1 - s2) < 1e-5

    def test_close_releases_handles(self, tmp_path: Path) -> None:
        bm   = CompressedBM25(SAMPLE_CORPUS)
        path = tmp_path / "close_test.fbcidx"
        bm.save(path)
        mmap_bm = MmapBM25(path)
        mmap_bm.close()
        # After close, _mmap_file should be None
        assert mmap_bm._mmap_file is None

    def test_repr(self, tmp_path: Path) -> None:
        bm   = CompressedBM25(SAMPLE_CORPUS)
        path = tmp_path / "repr_test.fbcidx"
        bm.save(path)
        mmap_bm = MmapBM25(path)
        r = repr(mmap_bm)
        assert "MmapBM25" in r
        assert "num_docs=5" in r


# ===========================================================================
# 6. StreamingBM25Builder — chunked on-disk ingestion
# ===========================================================================

class TestStreamingBM25Builder:
    """Tests for the StreamingBM25Builder chunked ingestion workflow."""

    def test_single_chunk_matches_in_memory(self, tmp_path: Path) -> None:
        """With chunk_size > corpus, result should match CompressedBM25."""
        path = tmp_path / "streaming_single.fbcidx"
        builder = StreamingBM25Builder(path, chunk_size=1000)
        for doc_id, tokens in enumerate(SAMPLE_CORPUS):
            builder.add_tokens(doc_id, tokens)
        bm_stream = builder.build()

        bm_mem = CompressedBM25(SAMPLE_CORPUS)
        scores_stream = bm_stream.get_scores(["bm25", "cat"])
        scores_mem    = bm_mem.get_scores(["bm25", "cat"])
        for s1, s2 in zip(scores_stream, scores_mem):
            assert abs(s1 - s2) < 1e-4

    def test_multi_chunk_produces_correct_index(self, tmp_path: Path) -> None:
        """chunk_size=2 forces 3 shard flushes for 5 docs."""
        path = tmp_path / "streaming_multi.fbcidx"
        builder = StreamingBM25Builder(path, chunk_size=2)
        for doc_id, tokens in enumerate(SAMPLE_CORPUS):
            builder.add_tokens(doc_id, tokens)
        bm = builder.build()
        assert bm.num_docs == len(SAMPLE_CORPUS)
        scores = bm.get_scores(["fast"])
        assert len(scores) == len(SAMPLE_CORPUS)
        # "fast" is in docs 3 and 4
        assert scores[4] > scores[0]

    def test_add_text_with_tokenizer(self, tmp_path: Path) -> None:
        path  = tmp_path / "streaming_text.fbcidx"
        texts = [" ".join(tokens) for tokens in SAMPLE_CORPUS]
        builder = StreamingBM25Builder(path, tokenizer=_simple_tokenize)
        builder.add_batch(texts, start_id=0)
        bm = builder.build()
        assert bm.num_docs == len(SAMPLE_CORPUS)

    def test_add_batch_mixed(self, tmp_path: Path) -> None:
        path = tmp_path / "streaming_batch.fbcidx"
        builder = StreamingBM25Builder(path, tokenizer=_simple_tokenize)
        items = [" ".join(t) for t in SAMPLE_CORPUS[:3]] + SAMPLE_CORPUS[3:]
        next_id = builder.add_batch(items, start_id=0)
        bm = builder.build()
        assert next_id == len(SAMPLE_CORPUS)
        assert bm.num_docs == len(SAMPLE_CORPUS)

    def test_tmp_files_cleaned_up(self, tmp_path: Path) -> None:
        path = tmp_path / "cleanup_test.fbcidx"
        builder = StreamingBM25Builder(path, chunk_size=2, tmp_dir=tmp_path)
        for doc_id, tokens in enumerate(SAMPLE_CORPUS):
            builder.add_tokens(doc_id, tokens)
        builder.build()
        tmp_files = list(tmp_path.glob("_flashbm25_shard_*.tmp"))
        assert tmp_files == [], f"Leftover shard files: {tmp_files}"

    def test_repr(self, tmp_path: Path) -> None:
        path = tmp_path / "repr_stream.fbcidx"
        builder = StreamingBM25Builder(path)
        r = repr(builder)
        assert "StreamingBM25Builder" in r

    def test_requires_tokenizer_for_add_text(self, tmp_path: Path) -> None:
        path = tmp_path / "no_tok.fbcidx"
        builder = StreamingBM25Builder(path)
        with pytest.raises(RuntimeError, match="tokenizer not set"):
            builder.add_text(0, "hello world")

    def test_num_shards_property(self, tmp_path: Path) -> None:
        path = tmp_path / "shards.fbcidx"
        builder = StreamingBM25Builder(path, chunk_size=2)
        for doc_id, tokens in enumerate(SAMPLE_CORPUS):
            builder.add_tokens(doc_id, tokens)
        # 5 docs with chunk_size=2: flushes at doc 2, 4 (auto); last flush in build()
        assert builder.num_shards >= 2
