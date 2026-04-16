"""
pytest test suite for FlashBM25.
Run with:  pytest tests/ -v
"""
import pytest

try:
    from flashbm25 import BM25
    HAS_EXT = True
except ImportError:
    HAS_EXT = False

skip_no_ext = pytest.mark.skipif(not HAS_EXT, reason="C++ extension not built")

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "a fast red fox leaped across the sleeping hound",
    "machine learning is a subfield of artificial intelligence",
    "deep learning models require large amounts of training data",
    "information retrieval systems rank documents by relevance",
    "BM25 is a bag of words retrieval function used in search engines",
]


# ── Construction ──────────────────────────────────────────────────────────────

@skip_no_ext
def test_construction_basic():
    bm25 = BM25(CORPUS)
    assert bm25.corpus_size == len(CORPUS)
    assert bm25.avg_doc_length > 0


@skip_no_ext
def test_construction_empty_corpus_raises():
    with pytest.raises(ValueError):
        BM25([])


@skip_no_ext
def test_custom_params():
    bm25 = BM25(CORPUS, k1=1.2, b=0.5, epsilon=0.1)
    assert bm25.k1      == pytest.approx(1.2)
    assert bm25.b       == pytest.approx(0.5)
    assert bm25.epsilon == pytest.approx(0.1)


# ── get_scores ────────────────────────────────────────────────────────────────

@skip_no_ext
def test_get_scores_returns_correct_length():
    bm25   = BM25(CORPUS)
    scores = bm25.get_scores("fox")
    assert len(scores) == len(CORPUS)


@skip_no_ext
def test_get_scores_non_negative():
    bm25 = BM25(CORPUS)
    for s in bm25.get_scores("retrieval search ranking"):
        assert s >= 0.0


@skip_no_ext
def test_get_scores_unknown_query_all_zeros():
    bm25   = BM25(CORPUS)
    scores = bm25.get_scores("xyzzy_never_appears_12345")
    assert all(s == 0.0 for s in scores)


@skip_no_ext
def test_get_scores_relevant_doc_scores_highest():
    bm25   = BM25(CORPUS)
    scores = bm25.get_scores("BM25 retrieval search")
    best   = scores.index(max(scores))
    assert best == 5


# ── get_top_n ─────────────────────────────────────────────────────────────────

@skip_no_ext
def test_get_top_n_length():
    bm25 = BM25(CORPUS)
    top  = bm25.get_top_n("fox", n=3)
    assert len(top) == 3


@skip_no_ext
def test_get_top_n_sorted_descending():
    bm25   = BM25(CORPUS)
    top    = bm25.get_top_n("machine learning deep", n=4)
    scores = [s for s, _ in top]
    assert scores == sorted(scores, reverse=True)


@skip_no_ext
def test_get_top_n_large_n_clamped():
    bm25 = BM25(CORPUS)
    top  = bm25.get_top_n("fox", n=1000)
    assert len(top) == len(CORPUS)


# ── get_top_n_docs ────────────────────────────────────────────────────────────

@skip_no_ext
def test_get_top_n_docs_returns_strings():
    bm25 = BM25(CORPUS)
    docs = bm25.get_top_n_docs("fox", n=2)
    assert len(docs) == 2
    assert all(isinstance(d, str) for d in docs)


@skip_no_ext
def test_get_top_n_docs_first_result_relevant():
    bm25 = BM25(CORPUS)
    docs = bm25.get_top_n_docs("BM25 retrieval", n=1)
    assert "BM25" in docs[0] or "retrieval" in docs[0].lower()


# ── repr ──────────────────────────────────────────────────────────────────────

@skip_no_ext
def test_repr():
    bm25 = BM25(CORPUS)
    r = repr(bm25)
    assert "BM25" in r
    assert "corpus_size" in r
