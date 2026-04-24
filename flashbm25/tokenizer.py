"""Tokenization helpers for the public FlashBM25 Python API."""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable, List, Optional, Sequence, Union


ENGLISH_STOPWORDS = frozenset(
    {
        "a", "about", "above", "after", "again", "against", "all", "am",
        "an", "and", "any", "are", "aren't", "as", "at", "be", "because",
        "been", "before", "being", "below", "between", "both", "but", "by",
        "can't", "cannot", "could", "couldn't", "did", "didn't", "do",
        "does", "doesn't", "doing", "don't", "down", "during", "each",
        "few", "for", "from", "further", "get", "got", "had", "hadn't",
        "has", "hasn't", "have", "haven't", "having", "he", "he'd",
        "he'll", "he's", "her", "here", "here's", "hers", "herself", "him",
        "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've",
        "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
        "let's", "me", "more", "most", "mustn't", "my", "myself", "no",
        "nor", "not", "of", "off", "on", "once", "only", "or", "other",
        "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
        "shan't", "she", "she'd", "she'll", "she's", "should",
        "shouldn't", "so", "some", "such", "than", "that", "that's", "the",
        "their", "theirs", "them", "themselves", "then", "there",
        "there's", "these", "they", "they'd", "they'll", "they're",
        "they've", "this", "those", "through", "to", "too", "under",
        "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll",
        "we're", "we've", "were", "weren't", "what", "what's", "when",
        "when's", "where", "where's", "which", "while", "who", "who's",
        "whom", "why", "why's", "will", "with", "won't", "would",
        "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your",
        "yours", "yourself", "yourselves",
    }
)

_DEFAULT_REGEX_PATTERN = r"[0-9A-Za-z_]+"
_UNICODE_WORD_PATTERN = re.compile(r"\w+", flags=re.UNICODE)

StopwordSpec = Optional[Union[bool, str, Sequence[str]]]
TokenizerSpec = Optional[Union[str, "Tokenizer", Callable[[str], Iterable[str]]]]


def _coerce_tokens(tokens: Iterable[str], *, source: str) -> List[str]:
    if isinstance(tokens, str):
        raise TypeError(f"{source} must return an iterable of strings, not a single string.")

    try:
        token_list = list(tokens)
    except TypeError as exc:
        raise TypeError(f"{source} must return an iterable of strings.") from exc

    for token in token_list:
        if not isinstance(token, str):
            raise TypeError(f"{source} must return only strings.")

    return token_list


def _normalize_stopwords(
    stopwords: StopwordSpec,
    extra_stopwords: Optional[Sequence[str]],
    *,
    lowercase: bool,
) -> frozenset[str]:
    words = set()

    if stopwords not in (None, False):
        if stopwords is True or stopwords == "english":
            words.update(ENGLISH_STOPWORDS)
        elif isinstance(stopwords, str):
            raise ValueError("stopwords must be True, 'english', or a sequence of strings.")
        else:
            words.update(_coerce_tokens(stopwords, source="stopwords"))

    if extra_stopwords is not None:
        words.update(_coerce_tokens(extra_stopwords, source="extra_stopwords"))

    if lowercase:
        return frozenset(word.casefold() for word in words)

    return frozenset(words)


class Tokenizer:
    """Configurable tokenizer used by the FlashBM25 Python wrapper.

    The tokenizer can split text with one of the built-in modes, normalize
    tokens, remove stopwords, and optionally apply a caller-provided stemmer.
    It is used for both corpus documents and incoming queries when passed to a
    ranking class such as :class:`flashbm25.BM25`.

    Parameters
    ----------
    mode:
        Token splitting strategy. ``"regex"`` uses ``pattern``,
        ``"unicode_word"`` uses Python's Unicode word matcher, and
        ``"whitespace"`` splits on whitespace only.
    pattern:
        Regular expression used when ``mode="regex"``. When omitted, the
        tokenizer matches ASCII word-like tokens.
    lowercase:
        Case-fold tokens before stopword filtering and stemming.
    stopwords:
        Stopword configuration. Use ``True`` or ``"english"`` for the built-in
        English set, pass a sequence of strings for a custom set, or leave as
        ``None``/``False`` to disable stopword filtering.
    extra_stopwords:
        Additional words to remove after applying ``stopwords``.
    stemmer:
        Optional callable that receives one token and returns its stemmed form.
        FlashBM25 does not depend on any specific stemming library.
    min_token_len:
        Drop tokens shorter than this many characters after normalization.
    """

    _VALID_MODES = frozenset({"regex", "unicode_word", "whitespace"})

    def __init__(
        self,
        mode: str = "regex",
        *,
        pattern: Optional[str] = None,
        lowercase: bool = True,
        stopwords: StopwordSpec = None,
        extra_stopwords: Optional[Sequence[str]] = None,
        stemmer: Optional[Callable[[str], str]] = None,
        min_token_len: int = 1,
    ) -> None:
        if mode not in self._VALID_MODES:
            raise ValueError(
                f"Unknown tokenizer mode {mode!r}. "
                "Valid modes: 'regex', 'unicode_word', 'whitespace'."
            )
        if pattern is not None and mode != "regex":
            raise ValueError("pattern is only supported when mode='regex'.")
        if min_token_len < 1:
            raise ValueError("min_token_len must be at least 1.")
        if stemmer is not None and not callable(stemmer):
            raise TypeError("stemmer must be callable.")

        self.mode = mode
        self.pattern = pattern or _DEFAULT_REGEX_PATTERN
        self.lowercase = lowercase
        self.stemmer = stemmer
        self.min_token_len = min_token_len
        self.stopwords = _normalize_stopwords(
            stopwords,
            extra_stopwords,
            lowercase=lowercase,
        )
        self._regex = re.compile(self.pattern) if mode == "regex" else None

    def _split(self, text: str) -> List[str]:
        if not isinstance(text, str):
            raise TypeError("Tokenizer input must be a string.")

        if self.mode == "whitespace":
            return text.split()
        if self.mode == "regex":
            return self._regex.findall(text)
        return _UNICODE_WORD_PATTERN.findall(text)

    def process_tokens(self, tokens: Iterable[str]) -> List[str]:
        """Normalize, filter, and stem pre-tokenized input.

        Parameters
        ----------
        tokens:
            Iterable of string tokens produced by a tokenizer.

        Returns
        -------
        list[str]
            Tokens after case-folding, stopword removal, stemming, and minimum
            length filtering.
        """
        processed: List[str] = []
        for token in _coerce_tokens(tokens, source="tokenizer"):
            if self.lowercase:
                token = token.casefold()

            if len(token) < self.min_token_len or token in self.stopwords:
                continue

            if self.stemmer is not None:
                token = self.stemmer(token)
                if not isinstance(token, str):
                    raise TypeError("stemmer must return strings.")
                if self.lowercase:
                    token = token.casefold()
                if len(token) < self.min_token_len or token in self.stopwords or not token:
                    continue

            processed.append(token)

        return processed

    def tokenize(self, text: str) -> List[str]:
        """Split and process a text string.

        Parameters
        ----------
        text:
            Input document or query text.

        Returns
        -------
        list[str]
            Processed tokens ready to be indexed or encoded for query scoring.
        """
        return self.process_tokens(self._split(text))

    def __call__(self, text: str) -> List[str]:
        """Tokenize ``text``.

        Parameters
        ----------
        text:
            Input document or query text.

        Returns
        -------
        list[str]
            Processed tokens.
        """
        return self.tokenize(text)

    def __repr__(self) -> str:
        return (
            f"Tokenizer(mode={self.mode!r}, lowercase={self.lowercase}, "
            f"stopwords={len(self.stopwords)}, stemmer={self.stemmer is not None})"
        )

    def to_state(self) -> dict[str, Any]:
        """Serialize tokenizer settings to JSON-compatible state.

        Returns
        -------
        dict[str, Any]
            Reconstructable tokenizer configuration.

        Raises
        ------
        TypeError
            If the tokenizer uses a callable stemmer, which cannot be restored
            automatically from persisted state.
        """
        if self.stemmer is not None:
            raise TypeError(
                "Tokenizer state cannot be serialized when it uses a callable stemmer."
            )

        return {
            "mode": self.mode,
            "pattern": self.pattern,
            "lowercase": self.lowercase,
            "stopwords": sorted(self.stopwords),
            "min_token_len": self.min_token_len,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "Tokenizer":
        """Create a tokenizer from :meth:`to_state` output.

        Parameters
        ----------
        state:
            Serialized tokenizer configuration.

        Returns
        -------
        Tokenizer
            Reconstructed tokenizer instance.
        """
        return cls(
            mode=state["mode"],
            pattern=state.get("pattern"),
            lowercase=state.get("lowercase", True),
            stopwords=state.get("stopwords"),
            min_token_len=state.get("min_token_len", 1),
        )


def _build_tokenizer_callable(
    tokenizer: TokenizerSpec,
    *,
    lowercase: bool,
    stopwords: StopwordSpec = None,
    extra_stopwords: Optional[Sequence[str]] = None,
    stemmer: Optional[Callable[[str], str]] = None,
) -> Optional[Callable[[str], List[str]]]:
    needs_python_tokenizer = (
        tokenizer is not None
        or stopwords not in (None, False)
        or extra_stopwords is not None
        or stemmer is not None
    )
    if not needs_python_tokenizer:
        return None

    if tokenizer is None:
        return Tokenizer(
            mode="regex",
            lowercase=lowercase,
            stopwords=stopwords,
            extra_stopwords=extra_stopwords,
            stemmer=stemmer,
        )

    if isinstance(tokenizer, str):
        base_tokenizer: Callable[[str], Iterable[str]] = Tokenizer(mode=tokenizer, lowercase=False)
    elif isinstance(tokenizer, Tokenizer):
        base_tokenizer = tokenizer
    elif callable(tokenizer):
        base_tokenizer = tokenizer
    else:
        raise TypeError(
            "tokenizer must be None, a Tokenizer, a built-in tokenizer name, or a callable."
        )

    postprocessor = Tokenizer(
        mode="regex",
        lowercase=lowercase,
        stopwords=stopwords,
        extra_stopwords=extra_stopwords,
        stemmer=stemmer,
    )

    def tokenize_text(text: str) -> List[str]:
        return postprocessor.process_tokens(base_tokenizer(text))

    return tokenize_text


class _TokenEncoder:
    def __init__(self) -> None:
        self._token_to_surrogate: dict[str, str] = {}

    def fit_many(self, documents: Iterable[Iterable[str]]) -> None:
        for document in documents:
            for token in document:
                self._token_to_surrogate.setdefault(
                    token,
                    f"fbm25tok_{len(self._token_to_surrogate):x}",
                )

    def encode_text(self, tokens: Iterable[str]) -> str:
        encoded_tokens = []
        for token in tokens:
            surrogate = self._token_to_surrogate.get(token)
            if surrogate is not None:
                encoded_tokens.append(surrogate)
        return " ".join(encoded_tokens)

    def to_state(self) -> dict[str, str]:
        return dict(self._token_to_surrogate)

    @classmethod
    def from_state(cls, state: dict[str, str]) -> "_TokenEncoder":
        encoder = cls()
        encoder._token_to_surrogate = dict(state)
        return encoder
