"""
Tests for ROADMAP §2.2 — Parallel & Async Query.

Covers:
  - get_scores_batch  (sequential and threaded)
  - get_top_n_batch
  - aget_scores
  - aget_top_n
  - aget_scores_batch
  - n_jobs validation
  - type / value error guards
  - consistency: threaded results must match sequential results
  - all four BM25 variants inherit the mixin correctly
"""

from __future__ import annotations

import asyncio
from typing import List

import numpy as np
import pytest

from flashbm25 import BM25, BM25Adpt, BM25L, BM25Plus

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

CORPUS: List[str] = [
    "the quick brown fox jumps over the lazy dog",
    "never gonna give you up never gonna let you down",
    "to be or not to be that is the question",
    "all that glitters is not gold",
    "elementary my dear watson",
    "may the force be with you",
    "i am your father",
    "just keep swimming swimming swimming",
]

QUERIES: List[str] = [
    "quick fox",
    "never gonna",
    "be or not to be",
    "gold glitters",
]


@pytest.fixture(scope="module")
def bm25() -> BM25:
    return BM25(CORPUS)


# ---------------------------------------------------------------------------
# 1 + 2  get_scores_batch — shape, dtype, consistency
# ---------------------------------------------------------------------------


class TestGetScoresBatch:
    def test_shape_and_dtype(self, bm25: BM25) -> None:
        result = bm25.get_scores_batch(QUERIES)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (len(QUERIES), len(CORPUS))

    def test_sequential_matches_individual(self, bm25: BM25) -> None:
        batch = bm25.get_scores_batch(QUERIES, n_jobs=1)
        for i, q in enumerate(QUERIES):
            expected = np.array(bm25.get_scores(q), dtype=np.float32)
            np.testing.assert_allclose(batch[i], expected, rtol=1e-5)

    def test_threaded_matches_sequential(self, bm25: BM25) -> None:
        seq = bm25.get_scores_batch(QUERIES, n_jobs=1)
        threaded = bm25.get_scores_batch(QUERIES, n_jobs=2)
        np.testing.assert_allclose(threaded, seq, rtol=1e-5)

    def test_n_jobs_minus_one(self, bm25: BM25) -> None:
        result = bm25.get_scores_batch(QUERIES, n_jobs=-1)
        assert result.shape == (len(QUERIES), len(CORPUS))

    def test_single_query_list(self, bm25: BM25) -> None:
        result = bm25.get_scores_batch(["quick fox"])
        assert result.shape == (1, len(CORPUS))

    def test_raises_on_string_input(self, bm25: BM25) -> None:
        with pytest.raises(TypeError, match="sequence"):
            bm25.get_scores_batch("quick fox")  # type: ignore[arg-type]

    def test_raises_on_empty_list(self, bm25: BM25) -> None:
        with pytest.raises(ValueError, match="at least one"):
            bm25.get_scores_batch([])

    def test_raises_on_invalid_n_jobs(self, bm25: BM25) -> None:
        with pytest.raises(ValueError, match="n_jobs"):
            bm25.get_scores_batch(QUERIES, n_jobs=-2)


# ---------------------------------------------------------------------------
# 2  get_top_n_batch
# ---------------------------------------------------------------------------


class TestGetTopNBatch:
    def test_returns_list_of_lists(self, bm25: BM25) -> None:
        results = bm25.get_top_n_batch(QUERIES, n=3)
        assert isinstance(results, list)
        assert len(results) == len(QUERIES)
        for ranked in results:
            assert len(ranked) <= 3

    def test_threaded_matches_sequential(self, bm25: BM25) -> None:
        seq = bm25.get_top_n_batch(QUERIES, n=3, n_jobs=1)
        threaded = bm25.get_top_n_batch(QUERIES, n=3, n_jobs=2)
        assert seq == threaded

    def test_raises_on_string(self, bm25: BM25) -> None:
        with pytest.raises(TypeError):
            bm25.get_top_n_batch("query")  # type: ignore[arg-type]

    def test_raises_on_empty(self, bm25: BM25) -> None:
        with pytest.raises(ValueError):
            bm25.get_top_n_batch([])


# ---------------------------------------------------------------------------
# 3  Async interface
# ---------------------------------------------------------------------------


class TestAsyncInterface:
    def test_aget_scores_returns_ndarray(self, bm25: BM25) -> None:
        result = asyncio.run(bm25.aget_scores("quick fox"))
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (len(CORPUS),)

    def test_aget_scores_matches_sync(self, bm25: BM25) -> None:
        async def _run():
            return await bm25.aget_scores("quick fox")

        async_result = asyncio.run(_run())
        sync_result = np.array(bm25.get_scores("quick fox"), dtype=np.float32)
        np.testing.assert_allclose(async_result, sync_result, rtol=1e-5)

    def test_aget_top_n_returns_list(self, bm25: BM25) -> None:
        result = asyncio.run(bm25.aget_top_n("quick fox", n=3))
        assert isinstance(result, list)
        assert len(result) <= 3

    def test_aget_scores_batch_shape(self, bm25: BM25) -> None:
        async def _run():
            return await bm25.aget_scores_batch(QUERIES)

        result = asyncio.run(_run())
        assert isinstance(result, np.ndarray)
        assert result.shape == (len(QUERIES), len(CORPUS))

    def test_aget_scores_batch_matches_sync_batch(self, bm25: BM25) -> None:
        async def _run():
            return await bm25.aget_scores_batch(QUERIES, n_jobs=1)

        async_result = asyncio.run(_run())
        sync_result = bm25.get_scores_batch(QUERIES, n_jobs=1)
        np.testing.assert_allclose(async_result, sync_result, rtol=1e-5)

    def test_multiple_concurrent_aget_scores(self, bm25: BM25) -> None:
        async def _run():
            tasks = [bm25.aget_scores(q) for q in QUERIES]
            return await asyncio.gather(*tasks)

        results = asyncio.run(_run())
        assert len(results) == len(QUERIES)
        for r in results:
            assert isinstance(r, np.ndarray)
            assert r.shape == (len(CORPUS),)


# ---------------------------------------------------------------------------
# 4  GIL release — verified indirectly: threaded batch must not deadlock
#    and must return correct results under thread contention.
# ---------------------------------------------------------------------------


class TestGILRelease:
    def test_high_concurrency_batch(self, bm25: BM25) -> None:
        """Run a large batch with many workers; would deadlock if GIL not released."""
        big_queries = QUERIES * 20  # 80 queries
        result = bm25.get_scores_batch(big_queries, n_jobs=4)
        assert result.shape == (80, len(CORPUS))
        # Every block of 4 identical queries must produce the same scores.
        for i in range(0, 80, len(QUERIES)):
            np.testing.assert_allclose(result[i : i + len(QUERIES)], result[:len(QUERIES)], rtol=1e-5)

    def test_async_concurrent_does_not_block(self, bm25: BM25) -> None:
        async def _run():
            tasks = [bm25.aget_scores(q) for q in QUERIES * 5]
            return await asyncio.gather(*tasks)

        results = asyncio.run(_run())
        assert len(results) == len(QUERIES) * 5


# ---------------------------------------------------------------------------
# All variants inherit AsyncBatchMixin
# ---------------------------------------------------------------------------


class TestAllVariantsHaveBatchMethods:
    @pytest.mark.parametrize(
        "cls,kwargs",
        [
            (BM25, {}),
            (BM25L, {}),
            (BM25Plus, {}),
            (BM25Adpt, {}),
        ],
    )
    def test_get_scores_batch_available(self, cls, kwargs) -> None:
        model = cls(CORPUS, **kwargs)
        result = model.get_scores_batch(["quick fox", "lazy dog"])
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, len(CORPUS))

    @pytest.mark.parametrize(
        "cls,kwargs",
        [
            (BM25, {}),
            (BM25L, {}),
            (BM25Plus, {}),
            (BM25Adpt, {}),
        ],
    )
    def test_aget_scores_available(self, cls, kwargs) -> None:
        model = cls(CORPUS, **kwargs)
        result = asyncio.run(model.aget_scores("quick fox"))
        assert isinstance(result, np.ndarray)
        assert result.shape == (len(CORPUS),)
