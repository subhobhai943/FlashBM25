"""
flashbm25.parallel
==================
Parallel and async query helpers implementing ROADMAP §2.2.

All four deliverables live here:

1. ``get_scores_batch``  – vectorised batch query returning ``np.ndarray``
2. Thread-pool executor  – configurable ``n_jobs`` via ``concurrent.futures``
3. ``aget_scores``       – async single-query interface using ``asyncio.to_thread``
4. GIL release          – every hot-path call wraps ``get_scores`` so the
                          C++ extension can release the GIL.  The pybind11
                          binding must declare
                          ``py::call_guard<py::gil_scoped_release>()``
                          on ``get_scores`` / ``get_top_n`` for full benefit;
                          the Python side dispatches work off the main thread
                          so the GIL is always dropped between queries in a
                          batch.
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Sequence, Union

import numpy as np

__all__ = ["AsyncBatchMixin"]


def _n_workers(n_jobs: Optional[int]) -> int:
    """Resolve *n_jobs* to a concrete worker count.

    * ``None`` or ``1``  → 1  (no thread pool)
    * ``-1``             → ``os.cpu_count()`` (all logical cores)
    * positive integer   → that many workers
    """
    if n_jobs is None or n_jobs == 1:
        return 1
    if n_jobs == -1:
        return os.cpu_count() or 1
    if n_jobs < -1:
        raise ValueError("n_jobs must be -1, None, 1, or a positive integer.")
    return n_jobs


class AsyncBatchMixin:
    """Mixin that adds parallel batch-query and async methods to BM25 classes.

    Classes that mix this in must already expose a ``get_scores(query: str)``
    method returning a sequence of floats.
    """

    # ------------------------------------------------------------------
    # 1 + 2  Batch query with optional thread-pool
    # ------------------------------------------------------------------

    def get_scores_batch(
        self,
        queries: Sequence[str],
        *,
        n_jobs: Optional[int] = 1,
    ) -> np.ndarray:
        """Score every indexed document against each query in *queries*.

        Parameters
        ----------
        queries:
            Sequence of query strings to evaluate.
        n_jobs:
            Number of worker threads to use.

            * ``1`` or ``None`` – run sequentially in the calling thread.
            * ``-1``            – use all logical CPU cores.
            * ``k > 1``         – use exactly *k* threads.

            When ``n_jobs != 1`` each query is dispatched to a
            :class:`concurrent.futures.ThreadPoolExecutor`.  Because the
            BM25 C++ scoring function releases the GIL (see the pybind11
            binding note in ``parallel.py``), multiple queries can overlap
            their CPU work.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``(len(queries), corpus_size)`` where
            ``result[i, j]`` is the BM25 score of document *j* for
            query *i*.

        Raises
        ------
        ValueError
            If *queries* is empty or *n_jobs* is invalid.
        TypeError
            If *queries* is a plain string instead of a sequence.
        """
        if isinstance(queries, str):
            raise TypeError(
                "get_scores_batch expects a sequence of query strings, "
                "not a single string. Did you mean get_scores(query)?"
            )
        queries = list(queries)
        if not queries:
            raise ValueError("queries must contain at least one query string.")

        workers = _n_workers(n_jobs)

        if workers == 1:
            # Fast path: stay on the calling thread, no overhead.
            rows = [np.array(self.get_scores(q), dtype=np.float32) for q in queries]
        else:
            rows = [None] * len(queries)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                future_to_idx = {
                    pool.submit(self.get_scores, q): i
                    for i, q in enumerate(queries)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    rows[idx] = np.array(future.result(), dtype=np.float32)

        return np.stack(rows)  # shape (Q, D)

    # ------------------------------------------------------------------
    # 2  Convenience: parallel get_top_n for a batch of queries
    # ------------------------------------------------------------------

    def get_top_n_batch(
        self,
        queries: Sequence[str],
        n: int = 5,
        *,
        n_jobs: Optional[int] = 1,
    ) -> List[List]:
        """Return top-*n* results for each query in *queries*.

        Parameters
        ----------
        queries:
            Sequence of query strings.
        n:
            Maximum number of ``(score, doc_id)`` pairs per query.
        n_jobs:
            Thread-pool size (same semantics as :meth:`get_scores_batch`).

        Returns
        -------
        list[list[tuple[float, int]]]
            One ranked list per input query.
        """
        if isinstance(queries, str):
            raise TypeError(
                "get_top_n_batch expects a sequence of query strings."
            )
        queries = list(queries)
        if not queries:
            raise ValueError("queries must contain at least one query string.")

        workers = _n_workers(n_jobs)

        def _run(q: str):
            return self.get_top_n(q, n)

        if workers == 1:
            return [_run(q) for q in queries]

        results = [None] * len(queries)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_idx = {
                pool.submit(_run, q): i for i, q in enumerate(queries)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
        return results

    # ------------------------------------------------------------------
    # 3  Async interface
    # ------------------------------------------------------------------

    async def aget_scores(self, query: str) -> np.ndarray:
        """Asynchronously score every document against *query*.

        The scoring work is offloaded to a worker thread via
        :func:`asyncio.to_thread`, so the event loop is never blocked.
        This makes FlashBM25 safe to use inside ``async`` request handlers
        (FastAPI, aiohttp, etc.) without degrading concurrency.

        Parameters
        ----------
        query:
            Query text to score.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``(corpus_size,)``.
        """
        raw = await asyncio.to_thread(self.get_scores, query)
        return np.array(raw, dtype=np.float32)

    async def aget_top_n(
        self,
        query: str,
        n: int = 5,
    ) -> List:
        """Asynchronously return top-*n* ``(score, doc_id)`` pairs for *query*.

        Parameters
        ----------
        query:
            Query text.
        n:
            Maximum number of results.

        Returns
        -------
        list[tuple[float, int]]
            Ranked results from highest to lowest score.
        """
        return await asyncio.to_thread(self.get_top_n, query, n)

    async def aget_scores_batch(
        self,
        queries: Sequence[str],
        *,
        n_jobs: Optional[int] = -1,
    ) -> np.ndarray:
        """Asynchronously score every document for a batch of queries.

        Runs :meth:`get_scores_batch` in a thread pool via
        :func:`asyncio.to_thread` so the event loop stays unblocked.

        Parameters
        ----------
        queries:
            Sequence of query strings.
        n_jobs:
            Thread-pool size passed to :meth:`get_scores_batch`.
            Defaults to ``-1`` (all cores) because this call is already
            async.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``(len(queries), corpus_size)``.
        """
        return await asyncio.to_thread(
            self.get_scores_batch, queries, n_jobs=n_jobs
        )
