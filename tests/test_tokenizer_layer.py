"""
Tokenizer-layer tests for FlashBM25.
"""

import numpy as np
import pytest

try:
    from flashbm25 import BM25, BM25F, BM25Plus, Tokenizer
    HAS_EXT = True
except ImportError:
    HAS_EXT = False


skip_no_ext = pytest.mark.skipif(not HAS_EXT, reason="C++ extension not built")


UNICODE_CORPUS = [
    "naïve café customers love croissants",
    "plain ascii diner food",
]


@skip_no_ext
def test_tokenizer_whitespace_mode():
    tokenizer = Tokenizer(mode="whitespace")
    assert tokenizer("Keep punctuation, please!") == ["keep", "punctuation,", "please!"]


@skip_no_ext
def test_tokenizer_regex_mode():
    tokenizer = Tokenizer(mode="regex", pattern=r"[A-Za-z]+")
    assert tokenizer("Token-123 split!") == ["token", "split"]


@skip_no_ext
def test_tokenizer_unicode_word_mode():
    tokenizer = Tokenizer(mode="unicode_word")
    tokens = tokenizer("naïve café 東京")
    assert "naïve" in tokens
    assert "café" in tokens
    assert "東京" in tokens


@skip_no_ext
def test_tokenizer_stopwords_and_extra_stopwords():
    tokenizer = Tokenizer(
        mode="regex",
        stopwords="english",
        extra_stopwords=["flashbm25"],
    )
    assert tokenizer("The FlashBM25 fox jumps") == ["fox", "jumps"]


@skip_no_ext
def test_tokenizer_stemmer_hook():
    stemmer = lambda token: {"running": "run", "runs": "run"}.get(token, token)
    tokenizer = Tokenizer(mode="regex", stemmer=stemmer)
    assert tokenizer("running runs runner") == ["run", "run", "runner"]


@skip_no_ext
def test_bm25_accepts_builtin_tokenizer_name():
    bm25 = BM25(UNICODE_CORPUS, tokenizer="unicode_word")
    scores = bm25.get_scores("café")
    assert scores[0] > 0.0
    assert scores[1] == 0.0


@skip_no_ext
def test_bm25_accepts_callable_tokenizer():
    def hyphen_tokenizer(text: str):
        return text.replace("-", " ").split()

    bm25 = BM25(
        ["state-of-the-art retrieval", "basic search"],
        tokenizer=hyphen_tokenizer,
        stopwords="english",
    )
    scores = bm25.get_scores("art retrieval")
    assert scores[0] > scores[1]


@skip_no_ext
def test_bm25_stopwords_and_stemmer_apply_with_default_python_tokenizer():
    stemmer = lambda token: {"running": "run", "runs": "run"}.get(token, token)
    bm25 = BM25(
        ["the running fox", "the sleepy dog"],
        stopwords="english",
        stemmer=stemmer,
    )
    scores = bm25.get_scores("run")
    assert scores[0] > scores[1]


@skip_no_ext
def test_bm25_persistence_with_builtin_tokenizer(tmp_path):
    bm25 = BM25(UNICODE_CORPUS, tokenizer="unicode_word")
    path = tmp_path / "unicode_word.fbm25"

    expected_scores = bm25.get_scores("cafÃ©")
    bm25.save(path)
    loaded = BM25.load(path)

    np.testing.assert_allclose(loaded.get_scores("cafÃ©"), expected_scores)


@skip_no_ext
def test_bm25_add_documents_with_python_tokenizer_updates_encoder():
    bm25 = BM25(["plain ascii diner food"], tokenizer="unicode_word")
    bm25.add_documents(["naÃ¯ve cafÃ© customers love croissants"])

    scores = bm25.get_scores("croissants")
    assert scores[0] == 0.0
    assert scores[1] > 0.0


@skip_no_ext
def test_bm25_save_rejects_callable_tokenizer(tmp_path):
    def hyphen_tokenizer(text: str):
        return text.replace("-", " ").split()

    bm25 = BM25(
        ["state-of-the-art retrieval", "basic search"],
        tokenizer=hyphen_tokenizer,
    )

    with pytest.raises(TypeError, match="callable tokenizers"):
        bm25.save(tmp_path / "callable-tokenizer.fbm25")


@skip_no_ext
def test_bm25_save_rejects_callable_stemmer(tmp_path):
    stemmer = lambda token: {"running": "run", "runs": "run"}.get(token, token)
    bm25 = BM25(
        ["the running fox", "the sleepy dog"],
        stopwords="english",
        stemmer=stemmer,
    )

    with pytest.raises(TypeError, match="callable stemmer"):
        bm25.save(tmp_path / "callable-stemmer.fbm25")


@skip_no_ext
def test_variant_with_tokenizer_support():
    bm25 = BM25Plus(UNICODE_CORPUS, tokenizer=Tokenizer(mode="unicode_word"))
    scores = bm25.get_scores("croissants")
    assert scores[0] > 0.0
    assert scores[1] == 0.0


@skip_no_ext
def test_bm25f_tokenizer_support():
    corpus = [
        {"title": "naïve café", "body": "fresh croissants daily"},
        {"title": "plain diner", "body": "comfort food"},
    ]
    bm25f = BM25F(corpus, field_weights={"title": 2.0, "body": 1.0}, tokenizer="unicode_word")
    scores = bm25f.get_scores("café")
    assert scores[0] > scores[1]
