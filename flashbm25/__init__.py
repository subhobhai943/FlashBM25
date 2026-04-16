"""
FlashBM25
=========
High-performance BM25 information-retrieval library.

The heavy lifting is done by a C++ extension (_flashbm25) compiled at install
time via pybind11 + scikit-build-core. This module provides a thin, ergonomic
Python wrapper around the C++ BM25 classes and exposes a first-class tokenizer
layer for customization in Python.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

try:
    from ._flashbm25 import (
        BM25 as _BM25Core,
        BM25Adpt as _BM25AdptCore,
        BM25F as _BM25FCore,
        BM25L as _BM25LCore,
        BM25Plus as _BM25PlusCore,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "FlashBM25 C++ extension (_flashbm25) could not be imported. "
        "Make sure you installed the package with: pip install flashbm25"
    ) from exc

from .tokenizer import (
    ENGLISH_STOPWORDS,
    Tokenizer,
    _TokenEncoder,
    _build_tokenizer_callable,
)

__version__ = "0.2.0"
__all__ = [
    "BM25",
    "BM25L",
    "BM25Plus",
    "BM25Adpt",
    "BM25F",
    "Tokenizer",
    "ENGLISH_STOPWORDS",
]

_VARIANT_MAP = {
    "okapi": "okapi",
    "bm25": "okapi",
    "l": "l",
    "bm25l": "l",
    "plus": "plus",
    "bm25+": "plus",
    "bm25plus": "plus",
    "adpt": "adpt",
    "adaptive": "adpt",
    "bm25adpt": "adpt",
}


class _TokenizerSupportMixin:
    def _encode_query(self, query: str) -> str:
        tokenizer = getattr(self, "_query_tokenizer", None)
        encoder = getattr(self, "_token_encoder", None)
        if tokenizer is None or encoder is None:
            return query
        return encoder.encode_text(tokenizer(query))

    def _top_n_docs_from_corpus(self, query: str, n: int = 5) -> List[str]:
        top = self.get_top_n(query, n)
        return [self._corpus[idx] for _, idx in top if idx < len(self._corpus)]


def _prepare_text_corpus(
    corpus: Sequence[str],
    *,
    tokenizer,
    lowercase: bool,
    stopwords=None,
    extra_stopwords=None,
    stemmer: Optional[Callable[[str], str]] = None,
):
    query_tokenizer = _build_tokenizer_callable(
        tokenizer,
        lowercase=lowercase,
        stopwords=stopwords,
        extra_stopwords=extra_stopwords,
        stemmer=stemmer,
    )
    if query_tokenizer is None:
        return list(corpus), None, None

    tokenized_corpus = [query_tokenizer(doc) for doc in corpus]
    token_encoder = _TokenEncoder()
    token_encoder.fit_many(tokenized_corpus)
    encoded_corpus = [token_encoder.encode_text(tokens) for tokens in tokenized_corpus]
    return encoded_corpus, query_tokenizer, token_encoder


def _prepare_field_corpus(
    corpus: Sequence[Dict[str, str]],
    *,
    tokenizer,
    lowercase: bool,
    stopwords=None,
    extra_stopwords=None,
    stemmer: Optional[Callable[[str], str]] = None,
):
    query_tokenizer = _build_tokenizer_callable(
        tokenizer,
        lowercase=lowercase,
        stopwords=stopwords,
        extra_stopwords=extra_stopwords,
        stemmer=stemmer,
    )
    if query_tokenizer is None:
        return list(corpus), None, None

    tokenized_corpus = [
        {field: query_tokenizer(text) for field, text in doc.items()}
        for doc in corpus
    ]
    token_encoder = _TokenEncoder()
    token_encoder.fit_many(tokens for doc in tokenized_corpus for tokens in doc.values())
    encoded_corpus = [
        {field: token_encoder.encode_text(tokens) for field, tokens in doc.items()}
        for doc in tokenized_corpus
    ]
    return encoded_corpus, query_tokenizer, token_encoder


class BM25(_TokenizerSupportMixin):
    """
    Okapi BM25 ranking model backed by a C++ inverted index.

    When ``variant`` is specified, this acts as a factory and returns the
    corresponding variant class instead:

    * ``"okapi"`` (default) -> :class:`BM25`
    * ``"l"`` -> :class:`BM25L`
    * ``"plus"`` -> :class:`BM25Plus`
    * ``"adpt"`` -> :class:`BM25Adpt`
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
        tokenizer=None,
        stopwords=None,
        extra_stopwords=None,
        stemmer: Optional[Callable[[str], str]] = None,
    ):
        if variant is not None:
            key = _VARIANT_MAP.get(variant.lower())
            if key is None:
                raise ValueError(
                    f"Unknown variant {variant!r}. "
                    f"Valid variants: 'okapi', 'l', 'plus', 'adpt'."
                )
            if key == "l":
                return BM25L(
                    corpus,
                    k1=k1,
                    b=b,
                    delta=delta if delta is not None else 0.5,
                    epsilon=epsilon,
                    lowercase=lowercase,
                    tokenizer=tokenizer,
                    stopwords=stopwords,
                    extra_stopwords=extra_stopwords,
                    stemmer=stemmer,
                )
            if key == "plus":
                return BM25Plus(
                    corpus,
                    k1=k1,
                    b=b,
                    delta=delta if delta is not None else 1.0,
                    epsilon=epsilon,
                    lowercase=lowercase,
                    tokenizer=tokenizer,
                    stopwords=stopwords,
                    extra_stopwords=extra_stopwords,
                    stemmer=stemmer,
                )
            if key == "adpt":
                return BM25Adpt(
                    corpus,
                    k1=k1,
                    b=b,
                    epsilon=epsilon,
                    lowercase=lowercase,
                    tokenizer=tokenizer,
                    stopwords=stopwords,
                    extra_stopwords=extra_stopwords,
                    stemmer=stemmer,
                )

        return super().__new__(cls)

    def __init__(
        self,
        corpus: List[str],
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
        lowercase: bool = True,
        variant: Optional[str] = None,
        delta: Optional[float] = None,
        tokenizer=None,
        stopwords=None,
        extra_stopwords=None,
        stemmer: Optional[Callable[[str], str]] = None,
    ) -> None:
        if not isinstance(self, BM25):
            return
        if hasattr(self, "_core"):
            return
        if not corpus:
            raise ValueError("corpus must contain at least one document.")

        self._corpus = list(corpus)
        encoded_corpus, self._query_tokenizer, self._token_encoder = _prepare_text_corpus(
            corpus,
            tokenizer=tokenizer,
            lowercase=lowercase,
            stopwords=stopwords,
            extra_stopwords=extra_stopwords,
            stemmer=stemmer,
        )
        self._core = _BM25Core(
            encoded_corpus,
            k1,
            b,
            epsilon,
            lowercase if self._query_tokenizer is None else False,
        )

    @property
    def k1(self) -> float:
        return self._core.k1

    @property
    def b(self) -> float:
        return self._core.b

    @property
    def epsilon(self) -> float:
        return self._core.epsilon

    @property
    def corpus_size(self) -> int:
        return self._core.corpus_size

    @property
    def avg_doc_length(self) -> float:
        return self._core.avg_doc_length

    def get_scores(self, query: str) -> List[float]:
        return self._core.get_scores(self._encode_query(query))

    def get_top_n(self, query: str, n: int = 5) -> List[Tuple[float, int]]:
        return self._core.get_top_n(self._encode_query(query), n)

    def get_top_n_docs(self, query: str, n: int = 5) -> List[str]:
        return self._top_n_docs_from_corpus(query, n)

    def __repr__(self) -> str:
        return (
            f"BM25(corpus_size={self.corpus_size}, "
            f"avg_doc_length={self.avg_doc_length:.2f}, "
            f"k1={self.k1}, b={self.b})"
        )


class BM25L(_TokenizerSupportMixin):
    def __init__(
        self,
        corpus: List[str],
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 0.5,
        epsilon: float = 0.25,
        lowercase: bool = True,
        tokenizer=None,
        stopwords=None,
        extra_stopwords=None,
        stemmer: Optional[Callable[[str], str]] = None,
    ) -> None:
        if not corpus:
            raise ValueError("corpus must contain at least one document.")

        self._corpus = list(corpus)
        encoded_corpus, self._query_tokenizer, self._token_encoder = _prepare_text_corpus(
            corpus,
            tokenizer=tokenizer,
            lowercase=lowercase,
            stopwords=stopwords,
            extra_stopwords=extra_stopwords,
            stemmer=stemmer,
        )
        self._core = _BM25LCore(
            encoded_corpus,
            k1,
            b,
            delta,
            epsilon,
            lowercase if self._query_tokenizer is None else False,
        )

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
    def epsilon(self) -> float:
        return self._core.epsilon

    @property
    def corpus_size(self) -> int:
        return self._core.corpus_size

    @property
    def avg_doc_length(self) -> float:
        return self._core.avg_doc_length

    def get_scores(self, query: str) -> List[float]:
        return self._core.get_scores(self._encode_query(query))

    def get_top_n(self, query: str, n: int = 5) -> List[Tuple[float, int]]:
        return self._core.get_top_n(self._encode_query(query), n)

    def get_top_n_docs(self, query: str, n: int = 5) -> List[str]:
        return self._top_n_docs_from_corpus(query, n)

    def __repr__(self) -> str:
        return (
            f"BM25L(corpus_size={self.corpus_size}, "
            f"avg_doc_length={self.avg_doc_length:.2f}, "
            f"k1={self.k1}, b={self.b}, delta={self.delta})"
        )


class BM25Plus(_TokenizerSupportMixin):
    def __init__(
        self,
        corpus: List[str],
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 1.0,
        epsilon: float = 0.25,
        lowercase: bool = True,
        tokenizer=None,
        stopwords=None,
        extra_stopwords=None,
        stemmer: Optional[Callable[[str], str]] = None,
    ) -> None:
        if not corpus:
            raise ValueError("corpus must contain at least one document.")

        self._corpus = list(corpus)
        encoded_corpus, self._query_tokenizer, self._token_encoder = _prepare_text_corpus(
            corpus,
            tokenizer=tokenizer,
            lowercase=lowercase,
            stopwords=stopwords,
            extra_stopwords=extra_stopwords,
            stemmer=stemmer,
        )
        self._core = _BM25PlusCore(
            encoded_corpus,
            k1,
            b,
            delta,
            epsilon,
            lowercase if self._query_tokenizer is None else False,
        )

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
    def epsilon(self) -> float:
        return self._core.epsilon

    @property
    def corpus_size(self) -> int:
        return self._core.corpus_size

    @property
    def avg_doc_length(self) -> float:
        return self._core.avg_doc_length

    def get_scores(self, query: str) -> List[float]:
        return self._core.get_scores(self._encode_query(query))

    def get_top_n(self, query: str, n: int = 5) -> List[Tuple[float, int]]:
        return self._core.get_top_n(self._encode_query(query), n)

    def get_top_n_docs(self, query: str, n: int = 5) -> List[str]:
        return self._top_n_docs_from_corpus(query, n)

    def __repr__(self) -> str:
        return (
            f"BM25Plus(corpus_size={self.corpus_size}, "
            f"avg_doc_length={self.avg_doc_length:.2f}, "
            f"k1={self.k1}, b={self.b}, delta={self.delta})"
        )


class BM25Adpt(_TokenizerSupportMixin):
    def __init__(
        self,
        corpus: List[str],
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
        lowercase: bool = True,
        tokenizer=None,
        stopwords=None,
        extra_stopwords=None,
        stemmer: Optional[Callable[[str], str]] = None,
    ) -> None:
        if not corpus:
            raise ValueError("corpus must contain at least one document.")

        self._corpus = list(corpus)
        encoded_corpus, self._query_tokenizer, self._token_encoder = _prepare_text_corpus(
            corpus,
            tokenizer=tokenizer,
            lowercase=lowercase,
            stopwords=stopwords,
            extra_stopwords=extra_stopwords,
            stemmer=stemmer,
        )
        self._core = _BM25AdptCore(
            encoded_corpus,
            k1,
            b,
            epsilon,
            lowercase if self._query_tokenizer is None else False,
        )

    @property
    def k1(self) -> float:
        return self._core.k1

    @property
    def b(self) -> float:
        return self._core.b

    @property
    def epsilon(self) -> float:
        return self._core.epsilon

    @property
    def corpus_size(self) -> int:
        return self._core.corpus_size

    @property
    def avg_doc_length(self) -> float:
        return self._core.avg_doc_length

    def get_scores(self, query: str) -> List[float]:
        return self._core.get_scores(self._encode_query(query))

    def get_top_n(self, query: str, n: int = 5) -> List[Tuple[float, int]]:
        return self._core.get_top_n(self._encode_query(query), n)

    def get_top_n_docs(self, query: str, n: int = 5) -> List[str]:
        return self._top_n_docs_from_corpus(query, n)

    def __repr__(self) -> str:
        return (
            f"BM25Adpt(corpus_size={self.corpus_size}, "
            f"avg_doc_length={self.avg_doc_length:.2f}, "
            f"k1_base={self.k1}, b={self.b})"
        )


class BM25F(_TokenizerSupportMixin):
    def __init__(
        self,
        corpus: List[Dict[str, str]],
        field_weights: Dict[str, float],
        k1: float = 1.5,
        epsilon: float = 0.25,
        lowercase: bool = True,
        tokenizer=None,
        stopwords=None,
        extra_stopwords=None,
        stemmer: Optional[Callable[[str], str]] = None,
    ) -> None:
        if not corpus:
            raise ValueError("corpus must contain at least one document.")

        self._corpus = list(corpus)
        encoded_corpus, self._query_tokenizer, self._token_encoder = _prepare_field_corpus(
            corpus,
            tokenizer=tokenizer,
            lowercase=lowercase,
            stopwords=stopwords,
            extra_stopwords=extra_stopwords,
            stemmer=stemmer,
        )
        self._core = _BM25FCore(
            encoded_corpus,
            field_weights,
            k1,
            epsilon,
            lowercase if self._query_tokenizer is None else False,
        )

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
        self._core.set_field_b(field, b)

    def get_scores(self, query: str) -> List[float]:
        return self._core.get_scores(self._encode_query(query))

    def get_top_n(self, query: str, n: int = 5) -> List[Tuple[float, int]]:
        return self._core.get_top_n(self._encode_query(query), n)

    def __repr__(self) -> str:
        return f"BM25F(corpus_size={self.corpus_size}, k1={self.k1})"
