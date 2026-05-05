"""
flashbm25.parallel
==================
Parallel, async, and sparse query helpers implementing ROADMAP sections 2.2 and 2.4.

The batch-query deliverables live here:

1. ``get_scores_batch``  – vectorised batch query returning dense or sparse arrays
2. Thread-pool executor  – configurable ``n_jobs`` via ``concurrent.futures``
3. ``aget_scores``       – async single-query interface using ``asyncio.to_thread``
4. GIL release          – every hot-path binding releases the GIL around
                          C++ scoring; the Python side dispatches work off the
                          main thread so the GIL is dropped between queries in
                          a batch.
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Optional, Sequence, Union

import numpy as np

__all__ = ["AsyncBatchMixin"]

_DEFAULT_SPARSE_THRESHOLD = 10_000_000


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


def _rows_to_csr(rows: Sequence[np.ndarray]) -> Any:
    try:
        from scipy import sparse
    except ImportError as exc:
        raise ImportError(
            "SciPy is required for sparse batch score matrices. "
            "Install scipy or use the flashbm25[sparse] extra."
        ) from exc

    n_rows = len(rows)
    n_cols = int(rows[0].shape[0]) if rows else 0
    data_parts = []
    index_parts = []
    indptr = [0]

    for row in rows:
        row = np.asarray(row, dtype=np.float32)
        nonzero = np.flatnonzero(row)
        if nonzero.size:
            data_parts.append(row[nonzero])
            index_parts.append(nonzero.astype(np.int64, copy=False))
        indptr.append(indptr[-1] + int(nonzero.size))

    data = (
        np.concatenate(data_parts).astype(np.float32, copy=False)
        if data_parts
        else np.array([], dtype=np.float32)
    )
    indices = (
        np.concatenate(index_parts)
        if index_parts
        else np.array([], dtype=np.int64)
    )
    return sparse.csr_matrix(
        (data, indices, np.asarray(indptr, dtype=np.int64)),
        shape=(n_rows, n_cols),
        dtype=np.float32,
    )


class AsyncBatchMixin:
    """Mixin that adds parallel batch-query and async methods to BM25 classes.

    Classes that mix this in must already expose a ``get_scores(query: str)``
    method returning a float32-compatible score vector.
    """

    # ------------------------------------------------------------------
    # 1 + 2  Batch query with optional thread-pool
    # ------------------------------------------------------------------

    def get_scores_batch(
        self,
        queries: Sequence[str],
        *,
        n_jobs: Optional[int] = 1,
        sparse: Optional[bool] = None,
        sparse_threshold: int = _DEFAULT_SPARSE_THRESHOLD,
    ) -> Union[np.ndarray, Any]:
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
        sparse:
            Controls sparse output. ``False`` always returns a dense
            :class:`numpy.ndarray`; ``True`` returns a
            :class:`scipy.sparse.csr_matrix`; ``None`` returns CSR when
            ``len(queries) * corpus_size >= sparse_threshold``.
        sparse_threshold:
            Dense cell count where automatic sparse output begins when
            ``sparse`` is ``None``.

        Returns
        -------
        np.ndarray | scipy.sparse.csr_matrix
            Float32 score matrix of shape ``(len(queries), corpus_size)`` where
            ``result[i, j]`` is the BM25 score of document *j* for query *i*.

        Raises
        ------
        ValueError
            If *queries* is empty or *n_jobs* is invalid.
        TypeError
            If *queries* is a plain string instead of a sequence.
        ImportError
            If sparse output is requested but SciPy is unavailable.
        """
        if isinstance(queries, str):
            raise TypeError(
                "get_scores_batch expects a sequence of query strings, "
                "not a single string. Did you mean get_scores(query)?"
            )
        queries = list(queries)
        if not queries:
            raise ValueError("queries must contain at least one query string.")
        if sparse_threshold < 0:
            raise ValueError("sparse_threshold must be non-negative.")

        workers = _n_workers(n_jobs)

        if workers == 1:
            # Fast path: stay on the calling thread, no overhead.
            rows = [np.asarray(self.get_scores(q), dtype=np.float32) for q in queries]
        else:
            rows = [None] * len(queries)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                future_to_idx = {
                    pool.submit(self.get_scores, q): i
                    for i, q in enumerate(queries)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    rows[idx] = np.asarray(future.result(), dtype=np.float32)

        total_cells = len(rows) * int(rows[0].shape[0])
        use_sparse = sparse if sparse is not None else total_cells >= sparse_threshold
        if use_sparse:
            return _rows_to_csr(rows)

        return np.stack(rows).astype(np.float32, copy=False)  # shape (Q, D)

    # ------------------------------------------------------------------
    # 2  Convenience: parallel get_top_n for a batch of queries
    # ------------------------------------------------------------------

    def get_top_n_batch(
        self,
        queries: Sequence[str],
        n: int = 5,
        *,
        n_jobs: Optional[int] = 1,
    ) -> List[np.ndarray]:
        """Return top-*n* results for each query in *queries*.

        Parameters
        ----------
        queries:
            Sequence of query strings.
        n:
            Maximum number of structured ``(score, doc_id)`` records per query.
        n_jobs:
            Thread-pool size (same semantics as :meth:`get_scores_batch`).

        Returns
        -------
        list[np.ndarray]
            One structured ``(score, doc_id)`` record array per input query.
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
        return np.asarray(raw, dtype=np.float32)

    async def aget_top_n(
        self,
        query: str,
        n: int = 5,
    ) -> np.ndarray:
        """Asynchronously return top-*n* records for *query*.

        Parameters
        ----------
        query:
            Query text.
        n:
            Maximum number of results.

        Returns
        -------
        np.ndarray
            Structured ``(score, doc_id)`` record array from highest to lowest
            score.
        """
        return await asyncio.to_thread(self.get_top_n, query, n)

    async def aget_scores_batch(
        self,
        queries: Sequence[str],
        *,
        n_jobs: Optional[int] = -1,
        sparse: Optional[bool] = None,
        sparse_threshold: int = _DEFAULT_SPARSE_THRESHOLD,
    ) -> Union[np.ndarray, Any]:
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
        sparse:
            Sparse-output control passed to :meth:`get_scores_batch`.
        sparse_threshold:
            Automatic sparse-output threshold passed to
            :meth:`get_scores_batch`.

        Returns
        -------
        np.ndarray | scipy.sparse.csr_matrix
            Float32 score matrix of shape ``(len(queries), corpus_size)``.
        """
        return await asyncio.to_thread(
            self.get_scores_batch,
            queries,
            n_jobs=n_jobs,
            sparse=sparse,
            sparse_threshold=sparse_threshold,
        )
