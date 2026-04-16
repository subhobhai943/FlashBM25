"""
FlashBM25
=========
High-performance BM25 information-retrieval library.

The heavy lifting is done by a C++ extension (_flashbm25) compiled at install
time via pybind11 + scikit-build-core.  This module provides a thin, ergonomic
Python wrapper around the C++ BM25 class.

Quick start
-----------
>>> from flashbm25 import BM25
>>> corpus = [
...     "the cat sat on the mat",
...     "dogs are great pets",
... ]
>>> bm25 = BM25(corpus)
>>> bm25.get_scores("cat mat")
[1.23..., 0.0]
>>> bm25.get_top_n("cat mat", n=1)
[(1.23..., 0)]
"""

from __future__ import annotations

from typing import List, Tuple

try:
    from ._flashbm25 import BM25 as _BM25Core
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "FlashBM25 C++ extension (_flashbm25) could not be imported. "
        "Make sure you installed the package with: pip install flashbm25"
    ) from exc


__version__ = "0.1.0"
__all__ = ["BM25"]


class BM25:
    """
    Okapi BM25 ranking model backed by a C++ inverted index.

    Parameters
    ----------
    corpus : list[str]
        Documents to index.
    k1 : float
        Term saturation parameter (default 1.5).
    b : float
        Length normalisation parameter (default 0.75).
    epsilon : float
        IDF floor — prevents negative scores for very common terms (default 0.25).
    lowercase : bool
        Normalise text to lowercase before indexing (default True).
    """

    def __init__(
        self,
        corpus: List[str],
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
        lowercase: bool = True,
    ) -> None:
        if not corpus:
            raise ValueError("corpus must contain at least one document.")
        self._corpus = list(corpus)
        self._core   = _BM25Core(corpus, k1, b, epsilon, lowercase)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def k1(self) -> float:
        """Term saturation parameter."""
        return self._core.k1

    @property
    def b(self) -> float:
        """Length normalisation parameter."""
        return self._core.b

    @property
    def epsilon(self) -> float:
        """IDF floor value."""
        return self._core.epsilon

    @property
    def corpus_size(self) -> int:
        """Number of documents in the index."""
        return self._core.corpus_size

    @property
    def avg_doc_length(self) -> float:
        """Average document length (in tokens)."""
        return self._core.avg_doc_length

    # ── Core API ──────────────────────────────────────────────────────────────

    def get_scores(self, query: str) -> List[float]:
        """
        Return a BM25 score for every document in the corpus.

        Parameters
        ----------
        query : str
            Raw query string (tokenised the same way as the corpus).

        Returns
        -------
        list[float]
            Scores in the same order as the original corpus.
        """
        return self._core.get_scores(query)

    def get_top_n(self, query: str, n: int = 5) -> List[Tuple[float, int]]:
        """
        Return the top-*n* ``(score, doc_index)`` pairs, sorted descending.

        Parameters
        ----------
        query : str
        n : int
            Number of results (default 5).

        Returns
        -------
        list[tuple[float, int]]
        """
        return self._core.get_top_n(query, n)

    def get_top_n_docs(self, query: str, n: int = 5) -> List[str]:
        """
        Return the top-*n* documents from the corpus, sorted by score descending.

        Parameters
        ----------
        query : str
        n : int
            Number of results (default 5).

        Returns
        -------
        list[str]
        """
        return self._core.get_top_n_docs(self._corpus, query, n)

    def __repr__(self) -> str:
        return (
            f"BM25(corpus_size={self.corpus_size}, "
            f"avg_doc_length={self.avg_doc_length:.2f}, "
            f"k1={self.k1}, b={self.b})"
        )
