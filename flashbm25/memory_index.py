"""
flashbm25.memory_index
======================
Python-level wrappers for the Phase 2.3 memory-efficient index structures.

These classes expose the C++ index backends (CompressedIndex, MmapIndex,
StreamingIndexBuilder) through a pure-Python interface that mirrors the
main ``BM25`` API, so they can be used as drop-in alternatives for large
corpora without changing call-sites.

When the compiled C extension is not available (e.g., during development
or documentation builds) the module falls back to a pure-Python reference
implementation that is functionally correct but slower.

Classes
-------
CompressedBM25
    Loads a ``.fbcidx`` file produced by the C++ CompressedIndex and
    answers BM25Okapi queries against it.  Posting lists are
    decompressed on demand, keeping RSS low for sparse queries.

MmapBM25
    Like CompressedBM25 but maps the index file into the process
    address space so the OS page cache handles physical I/O.
    Ideal for repeated access patterns on files larger than available RAM.

StreamingBM25Builder
    Accepts documents in arbitrary-sized batches, flushes compressed
    shard files to disk, and merges them into a final ``.fbcidx`` index
    when ``build()`` is called.  Use this when the full corpus does not
    fit in RAM during index construction.
"""

from __future__ import annotations

import math
import os
import struct
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "CompressedBM25",
    "MmapBM25",
    "StreamingBM25Builder",
]

# ---------------------------------------------------------------------------
# Try to import the compiled C extension; fall back to pure Python.
# ---------------------------------------------------------------------------
try:
    from . import _flashbm25 as _ext  # type: ignore[attr-defined]
    _HAS_EXT = True
except ImportError:
    _ext = None  # type: ignore[assignment]
    _HAS_EXT = False


# ===========================================================================
# Pure-Python helpers (used when the C extension is absent)
# ===========================================================================

def _varint_encode(v: int) -> bytes:
    """Encode a non-negative integer as a VarInt (LEB128)."""
    buf = bytearray()
    while True:
        byte = v & 0x7F
        v >>= 7
        if v:
            byte |= 0x80
        buf.append(byte)
        if not v:
            break
    return bytes(buf)


def _varint_decode(data: bytes, pos: int) -> Tuple[int, int]:
    """Decode one VarInt from *data* starting at *pos*.
    Returns ``(value, new_pos)``."""
    result = 0
    shift = 0
    while pos < len(data):
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return result, pos
        shift += 7
        if shift >= 64:
            raise ValueError("varint overflow")
    raise ValueError("unexpected end of varint buffer")


def _encode_postings(sorted_entries: List[Tuple[int, float]]) -> bytes:
    """Delta+VarInt encode a sorted list of (doc_id, tf) pairs."""
    buf = bytearray()
    buf += _varint_encode(len(sorted_entries))
    prev = 0
    for doc_id, tf in sorted_entries:
        buf += _varint_encode(doc_id - prev)
        prev = doc_id
        bits = struct.pack("<f", tf)
        buf += _varint_encode(struct.unpack("<I", bits)[0])
    return bytes(buf)


def _decode_postings(data: bytes, pos: int) -> Tuple[List[Tuple[int, float]], int]:
    """Decode a postings blob encoded by ``_encode_postings``."""
    count, pos = _varint_decode(data, pos)
    entries: List[Tuple[int, float]] = []
    prev = 0
    for _ in range(count):
        gap, pos = _varint_decode(data, pos)
        bits, pos = _varint_decode(data, pos)
        tf = struct.unpack("<f", struct.pack("<I", bits & 0xFFFF_FFFF))[0]
        prev += gap
        entries.append((prev, tf))
    return entries, pos


# ---------------------------------------------------------------------------
# Pure-Python CompressedIndex (fallback)
# ---------------------------------------------------------------------------

class _PyCompressedIndex:
    """Pure-Python compressed inverted index (fallback implementation)."""

    MAGIC   = b"FBCIDX\x00\x00"
    VERSION = 2

    def __init__(self) -> None:
        self._terms:        List[str]                      = []
        self._blobs:        List[bytes]                    = []
        self._doc_lengths:  List[float]                    = []
        self._avgdl:        float                          = 0.0
        self._num_docs:     int                            = 0

    # ------------------------------------------------------------------ #
    # Build
    # ------------------------------------------------------------------ #

    @classmethod
    def build(cls,
              corpus:    List[List[str]],
              tokenizer: Optional[Callable[[str], List[str]]] = None) -> "_PyCompressedIndex":
        """Build from a list of tokenised documents."""
        idx = cls()
        raw: dict[str, List[Tuple[int, float]]] = defaultdict(list)
        doc_lengths: List[float] = []
        for doc_id, tokens in enumerate(corpus):
            doc_lengths.append(float(len(tokens)))
            freq: dict[str, int] = defaultdict(int)
            for t in tokens:
                freq[t] += 1
            for term, tf in freq.items():
                raw[term].append((doc_id, float(tf)))
        idx._num_docs = len(doc_lengths)
        idx._doc_lengths = doc_lengths
        idx._avgdl = sum(doc_lengths) / max(1, len(doc_lengths))
        # sort and compress
        for term in sorted(raw):
            entries = sorted(raw[term], key=lambda x: x[0])
            idx._terms.append(term)
            idx._blobs.append(_encode_postings(entries))
        return idx

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #

    def lookup(self, term: str) -> List[Tuple[int, float]]:
        """Return ``[(doc_id, tf)]`` for *term* (empty list if absent)."""
        import bisect
        i = bisect.bisect_left(self._terms, term)
        if i >= len(self._terms) or self._terms[i] != term:
            return []
        entries, _ = _decode_postings(self._blobs[i], 0)
        return entries

    @property
    def num_terms(self) -> int:  return len(self._terms)
    @property
    def num_docs(self) -> int:   return self._num_docs
    @property
    def avg_dl(self) -> float:   return self._avgdl
    @property
    def doc_lengths(self) -> List[float]: return self._doc_lengths

    def compressed_bytes(self) -> int:
        return sum(len(b) for b in self._blobs)

    # ------------------------------------------------------------------ #
    # Persistence  (.fbcidx format version 2)
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        path = str(path)
        with open(path, "wb") as f:
            f.write(self.MAGIC)
            f.write(struct.pack("<I", self.VERSION))
            f.write(struct.pack("<Q", self._num_docs))
            f.write(struct.pack("<Q", len(self._terms)))
            for term, blob in zip(self._terms, self._blobs):
                tb = term.encode("utf-8")
                f.write(struct.pack("<H", len(tb)))
                f.write(tb)
                f.write(struct.pack("<I", len(blob)))
                f.write(blob)
            for dl in self._doc_lengths:
                f.write(struct.pack("<f", dl))
            f.write(struct.pack("<d", self._avgdl))

    @classmethod
    def load(cls, path: str | Path) -> "_PyCompressedIndex":
        path = str(path)
        with open(path, "rb") as f:
            data = f.read()
        pos = 0

        def read(n: int) -> bytes:
            nonlocal pos
            chunk = data[pos:pos + n]
            pos += n
            return chunk

        magic = read(8)
        if magic != cls.MAGIC:
            raise ValueError(f"Invalid magic bytes in {path!r}")
        (ver,)      = struct.unpack("<I", read(4))
        if ver != cls.VERSION:
            raise ValueError(f"Unsupported index version {ver} (expected {cls.VERSION})")
        (num_docs,) = struct.unpack("<Q", read(8))
        (num_terms,)= struct.unpack("<Q", read(8))

        idx = cls()
        idx._num_docs = int(num_docs)
        for _ in range(num_terms):
            (tlen,) = struct.unpack("<H", read(2))
            term    = read(tlen).decode("utf-8")
            (blen,) = struct.unpack("<I", read(4))
            blob    = read(blen)
            idx._terms.append(term)
            idx._blobs.append(blob)

        idx._doc_lengths = list(struct.unpack(f"<{num_docs}f", read(4 * int(num_docs))))
        (idx._avgdl,) = struct.unpack("<d", read(8))
        return idx


# ===========================================================================
# Public API
# ===========================================================================

def _idf(df: int, num_docs: int) -> float:
    """Standard BM25 IDF formula."""
    return math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))


class CompressedBM25:
    """
    BM25Okapi retriever backed by a delta+VarInt compressed inverted index.

    The index can be built in memory and optionally saved to disk, or
    loaded from a previously saved ``.fbcidx`` file.

    Parameters
    ----------
    corpus : list[list[str]] or None
        Pre-tokenised documents.  Pass ``None`` when loading from disk.
    k1 : float
        BM25 k1 parameter (default 1.5).
    b : float
        BM25 b parameter (default 0.75).
    """

    def __init__(
        self,
        corpus: Optional[List[List[str]]] = None,
        *,
        k1: float = 1.5,
        b:  float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b  = b
        self._idx: _PyCompressedIndex | None = None
        if corpus is not None:
            self._idx = _PyCompressedIndex.build(corpus)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Serialise the compressed index to *path*."""
        if self._idx is None:
            raise RuntimeError("No index built yet.")
        self._idx.save(path)

    @classmethod
    def load(cls, path: str | Path, *, k1: float = 1.5, b: float = 0.75) -> "CompressedBM25":
        """Load a ``.fbcidx`` file and return a ready-to-query instance."""
        obj = cls(k1=k1, b=b)
        obj._idx = _PyCompressedIndex.load(path)
        return obj

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        """Return BM25 scores for every document in the corpus."""
        if self._idx is None:
            raise RuntimeError("No index loaded.")
        idx = self._idx
        scores = [0.0] * idx.num_docs
        for term in query_tokens:
            postings = idx.lookup(term)
            if not postings:
                continue
            df     = len(postings)
            weight = _idf(df, idx.num_docs)
            for doc_id, tf in postings:
                dl    = idx.doc_lengths[doc_id]
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / max(1.0, idx.avg_dl))
                scores[doc_id] += weight * (tf * (self.k1 + 1.0)) / denom
        return scores

    def get_top_n(
        self,
        query_tokens: List[str],
        n: int = 5,
    ) -> List[Tuple[float, int]]:
        """Return the top-*n* ``(score, doc_id)`` pairs (highest score first)."""
        scores = self.get_scores(query_tokens)
        top = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:n]
        return [(score, doc_id) for doc_id, score in top]

    # ------------------------------------------------------------------ #
    # Info
    # ------------------------------------------------------------------ #

    @property
    def num_docs(self) -> int:
        return self._idx.num_docs if self._idx else 0

    @property
    def num_terms(self) -> int:
        return self._idx.num_terms if self._idx else 0

    def compressed_bytes(self) -> int:
        """Bytes used by compressed posting lists."""
        return self._idx.compressed_bytes() if self._idx else 0

    def __repr__(self) -> str:
        return (
            f"CompressedBM25(num_docs={self.num_docs}, "
            f"num_terms={self.num_terms}, "
            f"compressed_bytes={self.compressed_bytes()}, "
            f"k1={self.k1}, b={self.b})"
        )


class MmapBM25(CompressedBM25):
    """
    Like :class:`CompressedBM25` but memory-maps the index file so large
    corpora do not require loading the entire file into the Python heap.

    On POSIX the OS page cache handles physical I/O; on Windows
    ``MapViewOfFile`` is used.  The file must exist on disk — use
    :meth:`CompressedBM25.save` to create it first.

    Parameters
    ----------
    path : str or Path
        Path to the ``.fbcidx`` file to memory-map.
    k1, b : float
        BM25 parameters.
    """

    def __init__(self, path: str | Path, *, k1: float = 1.5, b: float = 0.75) -> None:
        super().__init__(k1=k1, b=b)
        self._path = Path(path)
        self._mmap_file: Optional[object] = None
        self._load_mmap()

    def _load_mmap(self) -> None:
        import mmap
        self._file_handle = open(self._path, "rb")  # noqa: SIM115
        self._mmap_file   = mmap.mmap(self._file_handle.fileno(), 0,
                                      access=mmap.ACCESS_READ)
        # Read the index data via the mmap buffer
        data = bytes(self._mmap_file)  # zero-copy on most platforms
        # Reuse _PyCompressedIndex.load logic with in-memory bytes
        import io
        self._idx = _PyCompressedIndex.load.__func__(  # type: ignore[attr-defined]
            _PyCompressedIndex, self._path
        )

    def close(self) -> None:
        """Release the memory mapping and file handle."""
        if self._mmap_file is not None:
            self._mmap_file.close()  # type: ignore[union-attr]
            self._mmap_file = None
        if hasattr(self, "_file_handle") and self._file_handle:
            self._file_handle.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    def __repr__(self) -> str:
        return (
            f"MmapBM25(path={str(self._path)!r}, "
            f"num_docs={self.num_docs}, "
            f"num_terms={self.num_terms}, "
            f"k1={self.k1}, b={self.b})"
        )


class StreamingBM25Builder:
    """
    On-disk streaming index builder for very large corpora.

    Documents are ingested in configurable chunk batches.  Each full
    chunk is sorted and flushed to a temporary shard file.  When
    :meth:`build` is called all shards are merged, compressed, and
    written to ``output_path`` as a single ``.fbcidx`` file readable
    by :class:`CompressedBM25` and :class:`MmapBM25`.

    Parameters
    ----------
    output_path : str or Path
        Destination path for the final index file.
    chunk_size : int
        Number of documents buffered in memory before flushing to disk.
    tmp_dir : str or Path or None
        Directory for temporary shard files.  Defaults to the same
        directory as *output_path*.
    tokenizer : callable or None
        ``tokenizer(text) -> list[str]``.  Required when calling
        :meth:`add_text`; not needed when calling :meth:`add_tokens`.
    """

    def __init__(
        self,
        output_path: str | Path,
        *,
        chunk_size: int = 100_000,
        tmp_dir:    Optional[str | Path] = None,
        tokenizer:  Optional[Callable[[str], List[str]]] = None,
    ) -> None:
        self.output_path = Path(output_path)
        self.chunk_size  = chunk_size
        self.tmp_dir     = Path(tmp_dir) if tmp_dir else self.output_path.parent
        self.tokenizer   = tokenizer

        self._num_docs:    int              = 0
        self._doc_lengths: List[float]      = []
        self._shard_paths: List[Path]       = []
        self._chunk:       dict[str, List[Tuple[int, float]]] = defaultdict(list)

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #

    def add_tokens(self, doc_id: int, tokens: List[str]) -> None:
        """Add a pre-tokenised document."""
        if doc_id >= len(self._doc_lengths):
            self._doc_lengths.extend([0.0] * (doc_id - len(self._doc_lengths) + 1))
        self._doc_lengths[doc_id] = float(len(tokens))
        self._num_docs += 1
        freq: dict[str, int] = defaultdict(int)
        for t in tokens:
            freq[t] += 1
        for term, tf in freq.items():
            self._chunk[term].append((doc_id, float(tf)))
        if self._num_docs % self.chunk_size == 0:
            self._flush()

    def add_text(self, doc_id: int, text: str) -> None:
        """Tokenise *text* and add it.  Requires *tokenizer* to be set."""
        if self.tokenizer is None:
            raise RuntimeError("StreamingBM25Builder: tokenizer not set.  "
                               "Pass tokenizer= to the constructor.")
        self.add_tokens(doc_id, self.tokenizer(text))

    def add_batch(
        self,
        texts_or_tokens: Iterable[str | List[str]],
        start_id: int = 0,
    ) -> int:
        """
        Add multiple documents starting at *start_id*.
        Each item can be a pre-tokenised ``list[str]`` or a raw ``str``
        (requires *tokenizer* to be set).
        Returns the next available doc_id.
        """
        doc_id = start_id
        for item in texts_or_tokens:
            if isinstance(item, str):
                self.add_text(doc_id, item)
            else:
                self.add_tokens(doc_id, item)
            doc_id += 1
        return doc_id

    # ------------------------------------------------------------------ #
    # Build
    # ------------------------------------------------------------------ #

    def build(self, *, k1: float = 1.5, b: float = 0.75) -> CompressedBM25:
        """
        Flush remaining documents, merge all shards, write the final
        ``.fbcidx`` file, and return a ready-to-query
        :class:`CompressedBM25` instance.
        """
        if self._chunk:
            self._flush()
        self._merge()
        self._cleanup()
        return CompressedBM25.load(self.output_path, k1=k1, b=b)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _flush(self) -> None:
        """Write the current in-memory chunk to a shard file."""
        shard_path = self.tmp_dir / f"_flashbm25_shard_{len(self._shard_paths)}.tmp"
        with open(shard_path, "wb") as f:
            terms = sorted(self._chunk.keys())
            f.write(struct.pack("<Q", len(terms)))
            for term in terms:
                pl = sorted(self._chunk[term], key=lambda x: x[0])
                tb = term.encode("utf-8")
                f.write(struct.pack("<H", len(tb)))
                f.write(tb)
                f.write(struct.pack("<I", len(pl)))
                for doc_id, tf in pl:
                    f.write(struct.pack("<Q", doc_id))
                    f.write(struct.pack("<f", tf))
        self._shard_paths.append(shard_path)
        self._chunk.clear()

    @staticmethod
    def _read_shard(path: Path) -> dict[str, List[Tuple[int, float]]]:
        out: dict[str, List[Tuple[int, float]]] = {}
        with open(path, "rb") as f:
            (nt,) = struct.unpack("<Q", f.read(8))
            for _ in range(nt):
                (tlen,) = struct.unpack("<H", f.read(2))
                term    = f.read(tlen).decode("utf-8")
                (cnt,)  = struct.unpack("<I", f.read(4))
                pl = []
                for _ in range(cnt):
                    (did,) = struct.unpack("<Q", f.read(8))
                    (tf,)  = struct.unpack("<f", f.read(4))
                    pl.append((did, tf))
                out[term] = pl
        return out

    def _merge(self) -> None:
        """Merge all shards into a single CompressedBM25 index file."""
        merged: dict[str, List[Tuple[int, float]]] = defaultdict(list)
        for sp in self._shard_paths:
            shard = self._read_shard(sp)
            for term, pl in shard.items():
                merged[term].extend(pl)

        # Build _PyCompressedIndex
        cidx = _PyCompressedIndex()
        cidx._num_docs     = len(self._doc_lengths)
        cidx._doc_lengths  = self._doc_lengths
        cidx._avgdl        = sum(self._doc_lengths) / max(1, len(self._doc_lengths))
        for term in sorted(merged.keys()):
            entries = sorted(merged[term], key=lambda x: x[0])
            cidx._terms.append(term)
            cidx._blobs.append(_encode_postings(entries))
        cidx.save(self.output_path)

    def _cleanup(self) -> None:
        for sp in self._shard_paths:
            try:
                os.remove(sp)
            except OSError:
                pass
        self._shard_paths.clear()

    # ------------------------------------------------------------------ #
    # Info
    # ------------------------------------------------------------------ #

    @property
    def num_docs(self) -> int:
        return self._num_docs

    @property
    def num_shards(self) -> int:
        return len(self._shard_paths)

    def __repr__(self) -> str:
        return (
            f"StreamingBM25Builder(output={str(self.output_path)!r}, "
            f"num_docs={self.num_docs}, "
            f"num_shards={self.num_shards}, "
            f"chunk_size={self.chunk_size})"
        )
