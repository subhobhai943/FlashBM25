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

import json
import os
import struct
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

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
    StopwordSpec,
    Tokenizer,
    TokenizerSpec,
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

_PERSISTENCE_MAGIC = b"FBM25PY\x00"
_PERSISTENCE_VERSION = 1


def _coerce_documents(documents: Iterable[str], *, source: str) -> List[str]:
    if isinstance(documents, str):
        raise TypeError(f"{source} must be an iterable of strings, not a single string.")

    docs = list(documents)
    for doc in docs:
        if not isinstance(doc, str):
            raise TypeError(f"{source} must contain only strings.")
    return docs


def _write_u32(handle, value: int) -> None:
    handle.write(struct.pack("<I", value))


def _write_u64(handle, value: int) -> None:
    handle.write(struct.pack("<Q", value))


def _read_exact(handle, size: int) -> bytes:
    payload = handle.read(size)
    if len(payload) != size:
        raise ValueError("Unexpected end of file while reading a persisted BM25 index.")
    return payload


def _read_u32(handle) -> int:
    return struct.unpack("<I", _read_exact(handle, 4))[0]


def _read_u64(handle) -> int:
    return struct.unpack("<Q", _read_exact(handle, 8))[0]


def _write_string(handle, value: str) -> None:
    payload = value.encode("utf-8")
    _write_u64(handle, len(payload))
    handle.write(payload)


def _read_string(handle) -> str:
    payload = _read_exact(handle, _read_u64(handle))
    return payload.decode("utf-8")


def _serialize_tokenizer_state(tokenizer: Tokenizer) -> Dict[str, Any]:
    if tokenizer.stemmer is not None:
        raise TypeError(
            "BM25.save does not support tokenizers that use a callable stemmer."
        )
    return tokenizer.to_state()


def _serialize_preprocess_state(
    *,
    tokenizer,
    lowercase: bool,
    stopwords,
    extra_stopwords,
    stemmer,
    query_tokenizer,
) -> Dict[str, Any]:
    if query_tokenizer is None:
        return {"kind": "core"}

    if tokenizer is None:
        if not isinstance(query_tokenizer, Tokenizer):
            raise TypeError("BM25.save could not serialize the active tokenizer pipeline.")
        return {
            "kind": "tokenizer",
            "state": _serialize_tokenizer_state(query_tokenizer),
        }

    if callable(tokenizer) and not isinstance(tokenizer, Tokenizer):
        raise TypeError(
            "BM25.save does not support arbitrary callable tokenizers because "
            "they cannot be reconstructed on load."
        )
    if stemmer is not None:
        raise TypeError(
            "BM25.save does not support arbitrary callable stemmers because "
            "they cannot be reconstructed on load."
        )

    if isinstance(tokenizer, str):
        base_state = {"kind": "builtin", "mode": tokenizer}
    elif isinstance(tokenizer, Tokenizer):
        base_state = {
            "kind": "tokenizer",
            "state": _serialize_tokenizer_state(tokenizer),
        }
    else:
        raise TypeError(
            "BM25.save only supports built-in tokenizer names or Tokenizer "
            "instances for persisted indices."
        )

    postprocess = Tokenizer(
        mode="regex",
        lowercase=lowercase,
        stopwords=stopwords,
        extra_stopwords=extra_stopwords,
    )
    return {
        "kind": "composed",
        "base": base_state,
        "postprocess": postprocess.to_state(),
    }


def _restore_base_tokenizer(state: Dict[str, Any]) -> Callable[[str], List[str]]:
    kind = state["kind"]
    if kind == "builtin":
        return Tokenizer(mode=state["mode"], lowercase=False)
    if kind == "tokenizer":
        return Tokenizer.from_state(state["state"])
    raise ValueError(f"Unsupported persisted tokenizer base kind: {kind!r}")


def _restore_query_tokenizer(state: Optional[Dict[str, Any]]):
    if state is None:
        return None

    kind = state["kind"]
    if kind == "core":
        return None
    if kind == "tokenizer":
        return Tokenizer.from_state(state["state"])
    if kind == "composed":
        base_tokenizer = _restore_base_tokenizer(state["base"])
        postprocessor = Tokenizer.from_state(state["postprocess"])

        def tokenize_text(text: str) -> List[str]:
            return postprocessor.process_tokens(base_tokenizer(text))

        return tokenize_text

    raise ValueError(f"Unsupported persisted tokenizer kind: {kind!r}")


def _prepare_text_corpus_from_state(
    corpus: Sequence[str],
    preprocess_state: Dict[str, Any],
):
    query_tokenizer = _restore_query_tokenizer(preprocess_state)
    if query_tokenizer is None:
        return list(corpus), None, None

    tokenized_corpus = [query_tokenizer(doc) for doc in corpus]
    token_encoder = _TokenEncoder()
    token_encoder.fit_many(tokenized_corpus)
    encoded_corpus = [token_encoder.encode_text(tokens) for tokens in tokenized_corpus]
    return encoded_corpus, query_tokenizer, token_encoder


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
    """Okapi BM25 ranking model backed by a C++ inverted index.

    ``BM25`` indexes a non-empty corpus of text documents and scores every
    document against a text query. When ``variant`` is provided, the constructor
    acts as a small factory and returns the requested variant class instead.

    Parameters
    ----------
    corpus:
        Non-empty sequence of document strings to index.
    k1:
        Term-frequency saturation parameter. Higher values keep rewarding
        repeated terms for longer.
    b:
        Document length normalization parameter. ``0`` disables length
        normalization and ``1`` applies full normalization.
    epsilon:
        Lower bound applied to IDF values to avoid negative scores.
    lowercase:
        Case-fold text before indexing and querying when using the default C++
        tokenizer. When a Python tokenizer is active, this is applied in the
        Python tokenizer pipeline.
    variant:
        Optional variant selector. Accepted values include ``"okapi"``,
        ``"l"``, ``"plus"``, and ``"adpt"`` plus common aliases.
    delta:
        Variant-specific additive lower bound used by BM25L and BM25Plus.
        Ignored by Okapi BM25 and BM25Adpt.
    tokenizer:
        Tokenizer configuration. Pass ``None`` for the default C++ tokenizer, a
        built-in tokenizer name, a :class:`Tokenizer`, or a callable that
        returns tokens.
    stopwords:
        Stopword configuration for the Python tokenizer pipeline.
    extra_stopwords:
        Additional stopwords to remove after ``stopwords`` are applied.
    stemmer:
        Optional callable that stems one token at a time.

    Returns
    -------
    BM25
        A BM25 instance, or a concrete variant instance when ``variant`` is set.

    Raises
    ------
    ValueError
        If ``corpus`` is empty or ``variant`` is unknown.
    TypeError
        If the corpus or tokenizer pipeline contains unsupported values.

    Notes
    -----
    Variant aliases map as follows: ``"okapi"`` and ``"bm25"`` return
    :class:`BM25`; ``"l"`` and ``"bm25l"`` return :class:`BM25L`; ``"plus"``,
    ``"bm25+"``, and ``"bm25plus"`` return :class:`BM25Plus`; ``"adpt"``,
    ``"adaptive"``, and ``"bm25adpt"`` return :class:`BM25Adpt`.
    """

    def __new__(
        cls,
        corpus: Sequence[str],
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
        lowercase: bool = True,
        variant: Optional[str] = None,
        delta: Optional[float] = None,
        tokenizer: TokenizerSpec = None,
        stopwords: StopwordSpec = None,
        extra_stopwords: Optional[Sequence[str]] = None,
        stemmer: Optional[Callable[[str], str]] = None,
    ) -> object:
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
        corpus: Sequence[str],
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
        lowercase: bool = True,
        variant: Optional[str] = None,
        delta: Optional[float] = None,
        tokenizer: TokenizerSpec = None,
        stopwords: StopwordSpec = None,
        extra_stopwords: Optional[Sequence[str]] = None,
        stemmer: Optional[Callable[[str], str]] = None,
    ) -> None:
        if not isinstance(self, BM25):
            return
        if hasattr(self, "_core"):
            return

        self._config = {
            "k1": float(k1),
            "b": float(b),
            "epsilon": float(epsilon),
            "lowercase": bool(lowercase),
        }
        self._preprocess_args = {
            "tokenizer": tokenizer,
            "lowercase": lowercase,
            "stopwords": stopwords,
            "extra_stopwords": extra_stopwords,
            "stemmer": stemmer,
        }
        self._preprocess_state = None
        self._corpus = _coerce_documents(corpus, source="corpus")
        if not self._corpus:
            raise ValueError("corpus must contain at least one document.")
        self._rebuild_core_from_corpus()

    def _prepare_encoded_corpus(self):
        if self._preprocess_args is not None:
            return _prepare_text_corpus(self._corpus, **self._preprocess_args)
        return _prepare_text_corpus_from_state(self._corpus, self._preprocess_state)

    def _rebuild_core_from_corpus(self) -> None:
        encoded_corpus, self._query_tokenizer, self._token_encoder = self._prepare_encoded_corpus()
        self._core = _BM25Core(
            encoded_corpus,
            self._config["k1"],
            self._config["b"],
            self._config["epsilon"],
            self._config["lowercase"] if self._query_tokenizer is None else False,
        )

    def _ensure_persistable_preprocess_state(self) -> Dict[str, Any]:
        if self._preprocess_state is not None:
            return self._preprocess_state
        if self._preprocess_args is None:
            raise TypeError("BM25 tokenizer state is unavailable for persistence.")

        self._preprocess_state = _serialize_preprocess_state(
            tokenizer=self._preprocess_args["tokenizer"],
            lowercase=self._preprocess_args["lowercase"],
            stopwords=self._preprocess_args["stopwords"],
            extra_stopwords=self._preprocess_args["extra_stopwords"],
            stemmer=self._preprocess_args["stemmer"],
            query_tokenizer=self._query_tokenizer,
        )
        return self._preprocess_state

    def save(self, path: Union[os.PathLike[str], str]) -> None:
        """Persist this index and its reconstructable Python state.

        Parameters
        ----------
        path:
            Destination file path. Parent directories must already exist.

        Raises
        ------
        TypeError
            If the active tokenizer pipeline includes arbitrary Python callables
            that cannot be reconstructed by :meth:`load`.
        """
        preprocess_state = self._ensure_persistable_preprocess_state()
        payload = {
            "variant": "okapi",
            "config": self._config,
            "preprocess": preprocess_state,
        }
        payload_bytes = json.dumps(
            payload,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        core_bytes = bytes(self._core.dumps())
        token_encoder_state = (
            self._token_encoder.to_state() if self._token_encoder is not None else {}
        )

        with open(os.fspath(path), "wb") as handle:
            handle.write(_PERSISTENCE_MAGIC)
            _write_u32(handle, _PERSISTENCE_VERSION)
            _write_u64(handle, len(payload_bytes))
            _write_u64(handle, len(core_bytes))
            _write_u64(handle, len(self._corpus))
            _write_u64(handle, len(token_encoder_state))
            handle.write(payload_bytes)
            handle.write(core_bytes)
            for document in self._corpus:
                _write_string(handle, document)
            for token, surrogate in sorted(token_encoder_state.items()):
                _write_string(handle, token)
                _write_string(handle, surrogate)

    @classmethod
    def load(cls, path: Union[os.PathLike[str], str]) -> "BM25":
        """Load an index saved with :meth:`save`.

        Parameters
        ----------
        path:
            Path to a persisted ``.fbm25`` file.

        Returns
        -------
        BM25
            Reconstructed Okapi BM25 index with corpus, scoring parameters, and
            tokenizer state restored.

        Raises
        ------
        ValueError
            If the file is not a supported FlashBM25 persistence file.
        """
        with open(os.fspath(path), "rb") as handle:
            magic = _read_exact(handle, len(_PERSISTENCE_MAGIC))
            if magic != _PERSISTENCE_MAGIC:
                raise ValueError("Invalid FlashBM25 persistence file.")

            version = _read_u32(handle)
            if version != _PERSISTENCE_VERSION:
                raise ValueError(
                    f"Unsupported FlashBM25 persistence version: {version}."
                )

            payload_len = _read_u64(handle)
            core_len = _read_u64(handle)
            corpus_len = _read_u64(handle)
            mapping_len = _read_u64(handle)

            payload = json.loads(_read_exact(handle, payload_len).decode("utf-8"))
            core_bytes = _read_exact(handle, core_len)
            corpus = [_read_string(handle) for _ in range(corpus_len)]
            token_mapping = {
                _read_string(handle): _read_string(handle)
                for _ in range(mapping_len)
            }

        if payload.get("variant") != "okapi":
            raise ValueError(
                f"BM25.load only supports okapi indices, got {payload.get('variant')!r}."
            )

        instance = object.__new__(cls)
        instance._core = _BM25Core.loads(core_bytes)
        instance._corpus = corpus
        instance._config = {
            "k1": float(instance._core.k1),
            "b": float(instance._core.b),
            "epsilon": float(instance._core.epsilon),
            "lowercase": bool(
                payload.get("config", {}).get("lowercase", instance._core.lowercase)
            ),
        }
        instance._preprocess_args = None
        instance._preprocess_state = payload.get("preprocess", {"kind": "core"})
        instance._query_tokenizer = _restore_query_tokenizer(instance._preprocess_state)
        instance._token_encoder = (
            None
            if instance._query_tokenizer is None
            else _TokenEncoder.from_state(token_mapping)
        )
        return instance

    def add_documents(self, new_docs: Sequence[str]) -> None:
        """Append documents to the index.

        Parameters
        ----------
        new_docs:
            Sequence of document strings to append. An empty sequence is a
            no-op.

        Raises
        ------
        TypeError
            If ``new_docs`` is a single string or contains non-string values.
        """
        documents = _coerce_documents(new_docs, source="new_docs")
        if not documents:
            return

        encoded_documents = documents
        if self._query_tokenizer is not None:
            tokenized_documents = [self._query_tokenizer(doc) for doc in documents]
            if self._token_encoder is None:
                self._token_encoder = _TokenEncoder()
            self._token_encoder.fit_many(tokenized_documents)
            encoded_documents = [
                self._token_encoder.encode_text(tokens)
                for tokens in tokenized_documents
            ]

        self._core.add_documents(encoded_documents)
        self._corpus.extend(documents)

    def remove_document(self, doc_id: int) -> None:
        """Remove one document by zero-based document id.

        Parameters
        ----------
        doc_id:
            Position of the document to remove from the current corpus.

        Raises
        ------
        IndexError
            If ``doc_id`` is outside the current corpus range.
        """
        if not 0 <= doc_id < len(self._corpus):
            raise IndexError("doc_id out of range.")

        del self._corpus[doc_id]
        self._rebuild_core_from_corpus()

    @property
    def k1(self) -> float:
        """Term-frequency saturation parameter used by the model."""
        return self._core.k1

    @property
    def b(self) -> float:
        """Document length normalization parameter used by the model."""
        return self._core.b

    @property
    def epsilon(self) -> float:
        """Lower bound applied to IDF values."""
        return self._core.epsilon

    @property
    def corpus_size(self) -> int:
        """Number of documents currently indexed."""
        return self._core.corpus_size

    @property
    def avg_doc_length(self) -> float:
        """Average indexed document length measured in tokens."""
        return self._core.avg_doc_length

    def get_scores(self, query: str) -> List[float]:
        """Score every indexed document against ``query``.

        Parameters
        ----------
        query:
            Query text to tokenize and score.

        Returns
        -------
        list[float]
            One BM25 score per document, in corpus order.
        """
        return self._core.get_scores(self._encode_query(query))

    def get_top_n(self, query: str, n: int = 5) -> List[Tuple[float, int]]:
        """Return the highest-scoring document ids for ``query``.

        Parameters
        ----------
        query:
            Query text to tokenize and score.
        n:
            Maximum number of results to return.

        Returns
        -------
        list[tuple[float, int]]
            ``(score, doc_id)`` pairs sorted from highest to lowest score.
        """
        return self._core.get_top_n(self._encode_query(query), n)

    def get_top_n_docs(self, query: str, n: int = 5) -> List[str]:
        """Return the highest-scoring document strings for ``query``.

        Parameters
        ----------
        query:
            Query text to tokenize and score.
        n:
            Maximum number of documents to return.

        Returns
        -------
        list[str]
            Original corpus documents sorted by descending BM25 score.
        """
        return self._top_n_docs_from_corpus(query, n)

    def __repr__(self) -> str:
        return (
            f"BM25(corpus_size={self.corpus_size}, "
            f"avg_doc_length={self.avg_doc_length:.2f}, "
            f"k1={self.k1}, b={self.b})"
        )


class BM25L(_TokenizerSupportMixin):
    """BM25L ranking model with a lower-bounded length-normalized TF term.

    BM25L reduces the tendency of classic BM25 to over-penalize long documents.

    Parameters
    ----------
    corpus:
        Non-empty sequence of document strings to index.
    k1:
        Term-frequency saturation parameter.
    b:
        Document length normalization parameter.
    delta:
        Additive lower bound for the normalized term-frequency component.
    epsilon:
        Lower bound applied to IDF values.
    lowercase:
        Case-fold text before indexing and querying.
    tokenizer:
        Tokenizer configuration or callable for the Python tokenizer pipeline.
    stopwords:
        Stopword configuration for the Python tokenizer pipeline.
    extra_stopwords:
        Additional stopwords to remove after ``stopwords`` are applied.
    stemmer:
        Optional callable that stems one token at a time.
    """

    def __init__(
        self,
        corpus: Sequence[str],
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 0.5,
        epsilon: float = 0.25,
        lowercase: bool = True,
        tokenizer: TokenizerSpec = None,
        stopwords: StopwordSpec = None,
        extra_stopwords: Optional[Sequence[str]] = None,
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
        """Term-frequency saturation parameter used by the model."""
        return self._core.k1

    @property
    def b(self) -> float:
        """Document length normalization parameter used by the model."""
        return self._core.b

    @property
    def delta(self) -> float:
        """Lower bound added to the normalized term-frequency component."""
        return self._core.delta

    @property
    def epsilon(self) -> float:
        """Lower bound applied to IDF values."""
        return self._core.epsilon

    @property
    def corpus_size(self) -> int:
        """Number of documents currently indexed."""
        return self._core.corpus_size

    @property
    def avg_doc_length(self) -> float:
        """Average indexed document length measured in tokens."""
        return self._core.avg_doc_length

    def get_scores(self, query: str) -> List[float]:
        """Score every indexed document against ``query``.

        Parameters
        ----------
        query:
            Query text to tokenize and score.

        Returns
        -------
        list[float]
            One BM25L score per document, in corpus order.
        """
        return self._core.get_scores(self._encode_query(query))

    def get_top_n(self, query: str, n: int = 5) -> List[Tuple[float, int]]:
        """Return the highest-scoring document ids for ``query``.

        Parameters
        ----------
        query:
            Query text to tokenize and score.
        n:
            Maximum number of results to return.

        Returns
        -------
        list[tuple[float, int]]
            ``(score, doc_id)`` pairs sorted from highest to lowest score.
        """
        return self._core.get_top_n(self._encode_query(query), n)

    def get_top_n_docs(self, query: str, n: int = 5) -> List[str]:
        """Return the highest-scoring document strings for ``query``.

        Parameters
        ----------
        query:
            Query text to tokenize and score.
        n:
            Maximum number of documents to return.

        Returns
        -------
        list[str]
            Original corpus documents sorted by descending BM25L score.
        """
        return self._top_n_docs_from_corpus(query, n)

    def __repr__(self) -> str:
        return (
            f"BM25L(corpus_size={self.corpus_size}, "
            f"avg_doc_length={self.avg_doc_length:.2f}, "
            f"k1={self.k1}, b={self.b}, delta={self.delta})"
        )


class BM25Plus(_TokenizerSupportMixin):
    """BM25Plus ranking model with a positive lower-bound TF contribution.

    BM25Plus adds ``delta`` to the matching-term contribution, which helps avoid
    over-penalizing longer documents that contain query terms.

    Parameters
    ----------
    corpus:
        Non-empty sequence of document strings to index.
    k1:
        Term-frequency saturation parameter.
    b:
        Document length normalization parameter.
    delta:
        Additive lower bound applied to matching term contributions.
    epsilon:
        Lower bound applied to IDF values.
    lowercase:
        Case-fold text before indexing and querying.
    tokenizer:
        Tokenizer configuration or callable for the Python tokenizer pipeline.
    stopwords:
        Stopword configuration for the Python tokenizer pipeline.
    extra_stopwords:
        Additional stopwords to remove after ``stopwords`` are applied.
    stemmer:
        Optional callable that stems one token at a time.
    """

    def __init__(
        self,
        corpus: Sequence[str],
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 1.0,
        epsilon: float = 0.25,
        lowercase: bool = True,
        tokenizer: TokenizerSpec = None,
        stopwords: StopwordSpec = None,
        extra_stopwords: Optional[Sequence[str]] = None,
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
        """Term-frequency saturation parameter used by the model."""
        return self._core.k1

    @property
    def b(self) -> float:
        """Document length normalization parameter used by the model."""
        return self._core.b

    @property
    def delta(self) -> float:
        """Positive lower-bound contribution for matching terms."""
        return self._core.delta

    @property
    def epsilon(self) -> float:
        """Lower bound applied to IDF values."""
        return self._core.epsilon

    @property
    def corpus_size(self) -> int:
        """Number of documents currently indexed."""
        return self._core.corpus_size

    @property
    def avg_doc_length(self) -> float:
        """Average indexed document length measured in tokens."""
        return self._core.avg_doc_length

    def get_scores(self, query: str) -> List[float]:
        """Score every indexed document against ``query``.

        Parameters
        ----------
        query:
            Query text to tokenize and score.

        Returns
        -------
        list[float]
            One BM25Plus score per document, in corpus order.
        """
        return self._core.get_scores(self._encode_query(query))

    def get_top_n(self, query: str, n: int = 5) -> List[Tuple[float, int]]:
        """Return the highest-scoring document ids for ``query``.

        Parameters
        ----------
        query:
            Query text to tokenize and score.
        n:
            Maximum number of results to return.

        Returns
        -------
        list[tuple[float, int]]
            ``(score, doc_id)`` pairs sorted from highest to lowest score.
        """
        return self._core.get_top_n(self._encode_query(query), n)

    def get_top_n_docs(self, query: str, n: int = 5) -> List[str]:
        """Return the highest-scoring document strings for ``query``.

        Parameters
        ----------
        query:
            Query text to tokenize and score.
        n:
            Maximum number of documents to return.

        Returns
        -------
        list[str]
            Original corpus documents sorted by descending BM25Plus score.
        """
        return self._top_n_docs_from_corpus(query, n)

    def __repr__(self) -> str:
        return (
            f"BM25Plus(corpus_size={self.corpus_size}, "
            f"avg_doc_length={self.avg_doc_length:.2f}, "
            f"k1={self.k1}, b={self.b}, delta={self.delta})"
        )


class BM25Adpt(_TokenizerSupportMixin):
    """BM25Adpt ranking model with adaptive per-term saturation.

    BM25Adpt starts from a base ``k1`` value and adapts saturation behavior per
    query term using corpus term-frequency statistics.

    Parameters
    ----------
    corpus:
        Non-empty sequence of document strings to index.
    k1:
        Base term-frequency saturation parameter.
    b:
        Document length normalization parameter.
    epsilon:
        Lower bound applied to IDF values.
    lowercase:
        Case-fold text before indexing and querying.
    tokenizer:
        Tokenizer configuration or callable for the Python tokenizer pipeline.
    stopwords:
        Stopword configuration for the Python tokenizer pipeline.
    extra_stopwords:
        Additional stopwords to remove after ``stopwords`` are applied.
    stemmer:
        Optional callable that stems one token at a time.
    """

    def __init__(
        self,
        corpus: Sequence[str],
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
        lowercase: bool = True,
        tokenizer: TokenizerSpec = None,
        stopwords: StopwordSpec = None,
        extra_stopwords: Optional[Sequence[str]] = None,
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
        """Base term-frequency saturation parameter used by the model."""
        return self._core.k1

    @property
    def b(self) -> float:
        """Document length normalization parameter used by the model."""
        return self._core.b

    @property
    def epsilon(self) -> float:
        """Lower bound applied to IDF values."""
        return self._core.epsilon

    @property
    def corpus_size(self) -> int:
        """Number of documents currently indexed."""
        return self._core.corpus_size

    @property
    def avg_doc_length(self) -> float:
        """Average indexed document length measured in tokens."""
        return self._core.avg_doc_length

    def get_scores(self, query: str) -> List[float]:
        """Score every indexed document against ``query``.

        Parameters
        ----------
        query:
            Query text to tokenize and score.

        Returns
        -------
        list[float]
            One BM25Adpt score per document, in corpus order.
        """
        return self._core.get_scores(self._encode_query(query))

    def get_top_n(self, query: str, n: int = 5) -> List[Tuple[float, int]]:
        """Return the highest-scoring document ids for ``query``.

        Parameters
        ----------
        query:
            Query text to tokenize and score.
        n:
            Maximum number of results to return.

        Returns
        -------
        list[tuple[float, int]]
            ``(score, doc_id)`` pairs sorted from highest to lowest score.
        """
        return self._core.get_top_n(self._encode_query(query), n)

    def get_top_n_docs(self, query: str, n: int = 5) -> List[str]:
        """Return the highest-scoring document strings for ``query``.

        Parameters
        ----------
        query:
            Query text to tokenize and score.
        n:
            Maximum number of documents to return.

        Returns
        -------
        list[str]
            Original corpus documents sorted by descending BM25Adpt score.
        """
        return self._top_n_docs_from_corpus(query, n)

    def __repr__(self) -> str:
        return (
            f"BM25Adpt(corpus_size={self.corpus_size}, "
            f"avg_doc_length={self.avg_doc_length:.2f}, "
            f"k1_base={self.k1}, b={self.b})"
        )


class BM25F(_TokenizerSupportMixin):
    """Field-weighted BM25 ranking model.

    ``BM25F`` accepts documents represented as dictionaries of field name to
    field text. Field weights boost or dampen each field's contribution to the
    final score.

    Parameters
    ----------
    corpus:
        Non-empty sequence of field dictionaries. Each dictionary maps field
        names to string values.
    field_weights:
        Mapping of field name to score multiplier, such as
        ``{"title": 2.0, "body": 1.0}``.
    k1:
        Term-frequency saturation parameter.
    epsilon:
        Lower bound applied to IDF values.
    lowercase:
        Case-fold field text before indexing and querying.
    tokenizer:
        Tokenizer configuration or callable for the Python tokenizer pipeline.
    stopwords:
        Stopword configuration for the Python tokenizer pipeline.
    extra_stopwords:
        Additional stopwords to remove after ``stopwords`` are applied.
    stemmer:
        Optional callable that stems one token at a time.

    Notes
    -----
    This is the Phase 1 BM25F skeleton. The public shape is available now, while
    deeper field-specific scoring controls are planned for the roadmap's Phase
    3 work.
    """

    def __init__(
        self,
        corpus: Sequence[Dict[str, str]],
        field_weights: Dict[str, float],
        k1: float = 1.5,
        epsilon: float = 0.25,
        lowercase: bool = True,
        tokenizer: TokenizerSpec = None,
        stopwords: StopwordSpec = None,
        extra_stopwords: Optional[Sequence[str]] = None,
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
        """Term-frequency saturation parameter used by the model."""
        return self._core.k1

    @property
    def epsilon(self) -> float:
        """Lower bound applied to IDF values."""
        return self._core.epsilon

    @property
    def corpus_size(self) -> int:
        """Number of fielded documents currently indexed."""
        return self._core.corpus_size

    def set_field_b(self, field: str, b: float) -> None:
        """Set the length-normalization value for one field.

        Parameters
        ----------
        field:
            Field name to configure.
        b:
            Field-specific length normalization parameter.
        """
        self._core.set_field_b(field, b)

    def get_scores(self, query: str) -> List[float]:
        """Score every indexed fielded document against ``query``.

        Parameters
        ----------
        query:
            Query text to tokenize and score.

        Returns
        -------
        list[float]
            One BM25F score per fielded document, in corpus order.
        """
        return self._core.get_scores(self._encode_query(query))

    def get_top_n(self, query: str, n: int = 5) -> List[Tuple[float, int]]:
        """Return the highest-scoring fielded document ids for ``query``.

        Parameters
        ----------
        query:
            Query text to tokenize and score.
        n:
            Maximum number of results to return.

        Returns
        -------
        list[tuple[float, int]]
            ``(score, doc_id)`` pairs sorted from highest to lowest score.
        """
        return self._core.get_top_n(self._encode_query(query), n)

    def __repr__(self) -> str:
        return f"BM25F(corpus_size={self.corpus_size}, k1={self.k1})"
