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

Variant selection
-----------------
>>> bm25l = BM25(corpus, variant="l")          # BM25L
>>> bm25p = BM25(corpus, variant="plus")       # BM25+
>>> bm25a = BM25(corpus, variant="adpt")       # BM25Adpt
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

try:
    from ._flashbm25 import (
        BM25 as _BM25Core,
        BM25Plus as _BM25PlusCore,
        BM25L as _BM25LCore,
        BM25Adpt as _BM25AdptCore,
        BM25F as _BM25FCore,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "FlashBM25 C++ extension (_flashbm25) could not be imported. "
        "Make sure you installed the package with: pip install flashbm25"
    ) from exc


__version__ = "0.2.0"
__all__ = ["BM25", "BM25L", "BM25Plus", "BM25Adpt", "BM25F"]

# ─── Valid variant names for the factory ──────────────────────────────────────
_VARIANT_MAP = {
    "okapi": "okapi",
    "bm25":  "okapi",
    "l":     "l",
    "bm25l": "l",
    "plus":  "plus",
    "bm25+": "plus",
    "bm25plus": "plus",
    "adpt":  "adpt",
    "adaptive": "adpt",
    "bm25adpt": "adpt",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  BM25 — Okapi BM25  (also the factory entry point via `variant=`)
# ═══════════════════════════════════════════════════════════════════════════════

class BM25:
    """
    Okapi BM25 ranking model backed by a C++ inverted index.

    When ``variant`` is specified, this acts as a factory and returns the
    corresponding variant class instead:

    * ``"okapi"`` (default) → :class:`BM25`
    * ``"l"`` → :class:`BM25L`
    * ``"plus"`` → :class:`BM25Plus`
    * ``"adpt"`` → :class:`BM25Adpt`

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
    variant : str or None
        BM25 variant to use.  One of ``"okapi"``, ``"l"``, ``"plus"``, ``"adpt"``.
        If ``None``, defaults to ``"okapi"`` (classic BM25).
    delta : float or None
        Lower-bound parameter for BM25L (default 0.5) and BM25+ (default 1.0).
        Only used when ``variant`` is ``"l"`` or ``"plus"``.
    """

    def __new__(
        cls,
        corpus: List[str],
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
        lowercase: bool = True,
        variant: Optional[str] = None,
        delta: Optional[float] = None,
    ):
        if variant is not None:
            key = _VARIANT_MAP.get(variant.lower())
            if key is None:
                raise ValueError(
                    f"Unknown variant {variant!r}. "
                    f"Valid variants: 'okapi', 'l', 'plus', 'adpt'."
                )
            if key == "l":
                return BM25L(corpus, k1=k1, b=b, delta=delta if delta is not None else 0.5,
                             epsilon=epsilon, lowercase=lowercase)
            elif key == "plus":
                return BM25Plus(corpus, k1=k1, b=b, delta=delta if delta is not None else 1.0,
                                epsilon=epsilon, lowercase=lowercase)
            elif key == "adpt":
                return BM25Adpt(corpus, k1=k1, b=b, epsilon=epsilon, lowercase=lowercase)
            # key == "okapi" → fall through to normal construction

        instance = super().__new__(cls)
        return instance

    def __init__(
        self,
        corpus: List[str],
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
        lowercase: bool = True,
        variant: Optional[str] = None,
        delta: Optional[float] = None,
    ) -> None:
        # If __new__ returned a different type, __init__ is still called but
        # we must not re-initialise the foreign object.
        if not isinstance(self, BM25):
            return
        if hasattr(self, "_core"):
            return  # already initialised
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


# ═══════════════════════════════════════════════════════════════════════════════
#  BM25L — Length-normalised lower-bound IDF
# ═══════════════════════════════════════════════════════════════════════════════

class BM25L:
    """
    BM25L (Lv & Zhai, 2011) — normalises TF before length normalisation.

    Reduces the over-penalisation of long documents that classic BM25 suffers
    from.  ``delta`` is typically in the range [0.0, 0.5].

    Parameters
    ----------
    corpus : list[str]
    k1 : float
    b : float
    delta : float
        Lower-bound addition (default 0.5).
    epsilon : float
    lowercase : bool
    """

    def __init__(
        self,
        corpus: List[str],
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 0.5,
        epsilon: float = 0.25,
        lowercase: bool = True,
    ) -> None:
        if not corpus:
            raise ValueError("corpus must contain at least one document.")
        self._corpus = list(corpus)
        self._core   = _BM25LCore(corpus, k1, b, delta, epsilon, lowercase)

    @property
    def k1(self) -> float:
        return self._core.k1

    @property
    def b(self) -> float:
        return self._core.b

    @property
    def delta(self) -> float:
        return self._core.delta

    @property
    def corpus_size(self) -> int:
        return self._core.corpus_size

    @property
    def avg_doc_length(self) -> float:
        return self._core.avg_doc_length

    @property
    def epsilon(self) -> float:
        return self._core.epsilon

    def get_scores(self, query: str) -> List[float]:
        """Return a BM25L score for every document in the corpus."""
        return self._core.get_scores(query)

    def get_top_n(self, query: str, n: int = 5) -> List[Tuple[float, int]]:
        """Return the top-*n* ``(score, doc_index)`` pairs, sorted descending."""
        return self._core.get_top_n(query, n)

    def get_top_n_docs(self, query: str, n: int = 5) -> List[str]:
        """Return the top-*n* documents, sorted by score descending."""
        scores = self.get_scores(query)
        top = self.get_top_n(query, n)
        return [self._corpus[idx] for _, idx in top if idx < len(self._corpus)]

    def __repr__(self) -> str:
        return (
            f"BM25L(corpus_size={self.corpus_size}, "
            f"avg_doc_length={self.avg_doc_length:.2f}, "
            f"k1={self.k1}, b={self.b}, delta={self.delta})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  BM25Plus — Lower-bound term frequency
# ═══════════════════════════════════════════════════════════════════════════════

class BM25Plus:
    """
    BM25+ (Lv & Zhai, 2011) — adds a lower-bound ``delta`` to the TF component.

    Ensures that a term appearing at least once in a document always contributes
    positively, fixing the "over-penalisation" problem of classic BM25.

    Parameters
    ----------
    corpus : list[str]
    k1 : float
    b : float
    delta : float
        Lower-bound addition (default 1.0).
    epsilon : float
    lowercase : bool
    """

    def __init__(
        self,
        corpus: List[str],
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 1.0,
        epsilon: float = 0.25,
        lowercase: bool = True,
    ) -> None:
        if not corpus:
            raise ValueError("corpus must contain at least one document.")
        self._corpus = list(corpus)
        self._core   = _BM25PlusCore(corpus, k1, b, delta, epsilon, lowercase)

    @property
    def k1(self) -> float:
        return self._core.k1

    @property
    def b(self) -> float:
        return self._core.b

    @property
    def delta(self) -> float:
        return self._core.delta

    @property
    def corpus_size(self) -> int:
        return self._core.corpus_size

    @property
    def avg_doc_length(self) -> float:
        return self._core.avg_doc_length

    @property
    def epsilon(self) -> float:
        return self._core.epsilon

    def get_scores(self, query: str) -> List[float]:
        """Return a BM25+ score for every document in the corpus."""
        return self._core.get_scores(query)

    def get_top_n(self, query: str, n: int = 5) -> List[Tuple[float, int]]:
        """Return the top-*n* ``(score, doc_index)`` pairs, sorted descending."""
        return self._core.get_top_n(query, n)

    def get_top_n_docs(self, query: str, n: int = 5) -> List[str]:
        """Return the top-*n* documents, sorted by score descending."""
        top = self.get_top_n(query, n)
        return [self._corpus[idx] for _, idx in top if idx < len(self._corpus)]

    def __repr__(self) -> str:
        return (
            f"BM25Plus(corpus_size={self.corpus_size}, "
            f"avg_doc_length={self.avg_doc_length:.2f}, "
            f"k1={self.k1}, b={self.b}, delta={self.delta})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  BM25Adpt — Adaptive k1 per term
# ═══════════════════════════════════════════════════════════════════════════════

class BM25Adpt:
    """
    BM25Adpt (Lv & Zhai, 2011) — adaptive k1 per query term.

    Instead of a single global ``k1``, each term gets its own ``k1`` based on
    its average term frequency across the corpus.  High-frequency terms get a
    higher k1 (slower saturation), rare terms get lower k1 (faster saturation).

    Parameters
    ----------
    corpus : list[str]
    k1 : float
        Base k1 used as the scaling anchor (default 1.5).
    b : float
    epsilon : float
    lowercase : bool
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
        self._core   = _BM25AdptCore(corpus, k1, b, epsilon, lowercase)

    @property
    def k1(self) -> float:
        """Base k1 parameter (before per-term adaptation)."""
        return self._core.k1

    @property
    def b(self) -> float:
        return self._core.b

    @property
    def corpus_size(self) -> int:
        return self._core.corpus_size

    @property
    def avg_doc_length(self) -> float:
        return self._core.avg_doc_length

    @property
    def epsilon(self) -> float:
        return self._core.epsilon

    def get_scores(self, query: str) -> List[float]:
        """Return a BM25Adpt score for every document in the corpus."""
        return self._core.get_scores(query)

    def get_top_n(self, query: str, n: int = 5) -> List[Tuple[float, int]]:
        """Return the top-*n* ``(score, doc_index)`` pairs, sorted descending."""
        return self._core.get_top_n(query, n)

    def get_top_n_docs(self, query: str, n: int = 5) -> List[str]:
        """Return the top-*n* documents, sorted by score descending."""
        top = self.get_top_n(query, n)
        return [self._corpus[idx] for _, idx in top if idx < len(self._corpus)]

    def __repr__(self) -> str:
        return (
            f"BM25Adpt(corpus_size={self.corpus_size}, "
            f"avg_doc_length={self.avg_doc_length:.2f}, "
            f"k1_base={self.k1}, b={self.b})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  BM25F — Field-weighted multi-field scoring (skeleton)
# ═══════════════════════════════════════════════════════════════════════════════

class BM25F:
    """
    BM25F (Robertson et al., 2004) — field-weighted multi-field BM25 scoring.

    Each document is a ``dict`` mapping field names to text.  Fields are
    independently tokenised and their term frequencies are combined using
    configurable per-field boost weights before the BM25 saturation function
    is applied.

    .. note::
        This is a **skeleton** in Phase 1.  Full per-field length normalisation
        and boost formula strategies land in Phase 3.

    Parameters
    ----------
    corpus : list[dict[str, str]]
        Each document is ``{"field_name": "text", ...}``.
    field_weights : dict[str, float]
        Mapping of field name → boost multiplier, e.g.
        ``{"title": 2.0, "body": 1.0}``.
    k1 : float
    epsilon : float
    lowercase : bool
    """

    def __init__(
        self,
        corpus: List[Dict[str, str]],
        field_weights: Dict[str, float],
        k1: float = 1.5,
        epsilon: float = 0.25,
        lowercase: bool = True,
    ) -> None:
        if not corpus:
            raise ValueError("corpus must contain at least one document.")
        self._corpus = list(corpus)
        self._core   = _BM25FCore(corpus, field_weights, k1, epsilon, lowercase)

    @property
    def k1(self) -> float:
        return self._core.k1

    @property
    def epsilon(self) -> float:
        return self._core.epsilon

    @property
    def corpus_size(self) -> int:
        return self._core.corpus_size

    def set_field_b(self, field: str, b: float) -> None:
        """Set the length-normalisation parameter for a specific field."""
        self._core.set_field_b(field, b)

    def get_scores(self, query: str) -> List[float]:
        """Return a BM25F score for every document in the corpus."""
        return self._core.get_scores(query)

    def get_top_n(self, query: str, n: int = 5) -> List[Tuple[float, int]]:
        """Return the top-*n* ``(score, doc_index)`` pairs, sorted descending."""
        return self._core.get_top_n(query, n)

    def __repr__(self) -> str:
        return f"BM25F(corpus_size={self.corpus_size}, k1={self.k1})"
