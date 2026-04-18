"""
pytest test suite for FlashBM25.
Run with:  pytest tests/ -v
"""
import pytest

try:
    from flashbm25 import BM25, BM25L, BM25Plus, BM25Adpt, BM25F
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


# ══════════════════════════════════════════════════════════════════════════════
#  BM25 (Okapi) — original tests
# ══════════════════════════════════════════════════════════════════════════════

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


@skip_no_ext
def test_repr():
    bm25 = BM25(CORPUS)
    r = repr(bm25)
    assert "BM25" in r
    assert "corpus_size" in r


@skip_no_ext
def test_save_and_load_roundtrip(tmp_path):
    bm25 = BM25(CORPUS)
    path = tmp_path / "roundtrip.fbm25"

    expected_scores = bm25.get_scores("BM25 retrieval search")
    expected_docs = bm25.get_top_n_docs("fox", n=2)

    bm25.save(path)
    loaded = BM25.load(path)

    assert loaded.corpus_size == bm25.corpus_size
    assert loaded.avg_doc_length == pytest.approx(bm25.avg_doc_length)
    assert loaded.get_scores("BM25 retrieval search") == pytest.approx(expected_scores)
    assert loaded.get_top_n_docs("fox", n=2) == expected_docs


@skip_no_ext
def test_add_documents_updates_index_incrementally():
    bm25 = BM25(CORPUS[:2])
    assert all(score == 0.0 for score in bm25.get_scores("retrieval"))

    bm25.add_documents(
        [
            "retrieval systems rank documents",
            "bm25 retrieval models power search",
        ]
    )

    assert bm25.corpus_size == 4
    scores = bm25.get_scores("retrieval")
    assert len(scores) == 4
    assert max(scores[2:]) > 0.0


@skip_no_ext
def test_remove_document_rebuilds_remaining_corpus():
    bm25 = BM25(CORPUS)
    bm25.remove_document(5)

    assert bm25.corpus_size == len(CORPUS) - 1
    assert bm25.get_top_n_docs("retrieval rank documents", n=1) == [CORPUS[4]]


# ══════════════════════════════════════════════════════════════════════════════
#  Variant factory — `variant` parameter
# ══════════════════════════════════════════════════════════════════════════════

@skip_no_ext
def test_variant_okapi_default():
    bm25 = BM25(CORPUS)
    assert isinstance(bm25, BM25)
    assert bm25.corpus_size == len(CORPUS)


@skip_no_ext
def test_variant_okapi_explicit():
    bm25 = BM25(CORPUS, variant="okapi")
    assert isinstance(bm25, BM25)


@skip_no_ext
def test_variant_l():
    bm25 = BM25(CORPUS, variant="l")
    assert isinstance(bm25, BM25L)
    assert bm25.corpus_size == len(CORPUS)


@skip_no_ext
def test_variant_plus():
    bm25 = BM25(CORPUS, variant="plus")
    assert isinstance(bm25, BM25Plus)
    assert bm25.corpus_size == len(CORPUS)


@skip_no_ext
def test_variant_adpt():
    bm25 = BM25(CORPUS, variant="adpt")
    assert isinstance(bm25, BM25Adpt)
    assert bm25.corpus_size == len(CORPUS)


@skip_no_ext
def test_variant_case_insensitive():
    bm25 = BM25(CORPUS, variant="PLUS")
    assert isinstance(bm25, BM25Plus)


@skip_no_ext
def test_variant_invalid_raises():
    with pytest.raises(ValueError, match="Unknown variant"):
        BM25(CORPUS, variant="invalid_variant")


@skip_no_ext
def test_variant_l_with_custom_delta():
    bm25 = BM25(CORPUS, variant="l", delta=0.3)
    assert isinstance(bm25, BM25L)
    assert bm25.delta == pytest.approx(0.3)


@skip_no_ext
def test_variant_plus_with_custom_delta():
    bm25 = BM25(CORPUS, variant="plus", delta=2.0)
    assert isinstance(bm25, BM25Plus)
    assert bm25.delta == pytest.approx(2.0)


# ══════════════════════════════════════════════════════════════════════════════
#  BM25L
# ══════════════════════════════════════════════════════════════════════════════

@skip_no_ext
def test_bm25l_construction():
    bm25 = BM25L(CORPUS)
    assert bm25.corpus_size == len(CORPUS)
    assert bm25.avg_doc_length > 0
    assert bm25.delta == pytest.approx(0.5)


@skip_no_ext
def test_bm25l_empty_corpus_raises():
    with pytest.raises(ValueError):
        BM25L([])


@skip_no_ext
def test_bm25l_scores_correct_length():
    bm25   = BM25L(CORPUS)
    scores = bm25.get_scores("fox")
    assert len(scores) == len(CORPUS)


@skip_no_ext
def test_bm25l_scores_non_negative():
    bm25 = BM25L(CORPUS)
    for s in bm25.get_scores("retrieval search ranking"):
        assert s >= 0.0


@skip_no_ext
def test_bm25l_top_n_sorted_descending():
    bm25   = BM25L(CORPUS)
    top    = bm25.get_top_n("machine learning deep", n=4)
    scores = [s for s, _ in top]
    assert scores == sorted(scores, reverse=True)


@skip_no_ext
def test_bm25l_top_n_docs():
    bm25 = BM25L(CORPUS)
    docs = bm25.get_top_n_docs("fox", n=2)
    assert len(docs) == 2
    assert all(isinstance(d, str) for d in docs)


@skip_no_ext
def test_bm25l_repr():
    bm25 = BM25L(CORPUS)
    r = repr(bm25)
    assert "BM25L" in r


# ══════════════════════════════════════════════════════════════════════════════
#  BM25Plus
# ══════════════════════════════════════════════════════════════════════════════

@skip_no_ext
def test_bm25plus_construction():
    bm25 = BM25Plus(CORPUS)
    assert bm25.corpus_size == len(CORPUS)
    assert bm25.avg_doc_length > 0
    assert bm25.delta == pytest.approx(1.0)


@skip_no_ext
def test_bm25plus_empty_corpus_raises():
    with pytest.raises(ValueError):
        BM25Plus([])


@skip_no_ext
def test_bm25plus_scores_correct_length():
    bm25   = BM25Plus(CORPUS)
    scores = bm25.get_scores("fox")
    assert len(scores) == len(CORPUS)


@skip_no_ext
def test_bm25plus_scores_non_negative():
    bm25 = BM25Plus(CORPUS)
    for s in bm25.get_scores("retrieval search ranking"):
        assert s >= 0.0


@skip_no_ext
def test_bm25plus_scores_higher_than_okapi():
    """BM25+ should give higher scores due to the delta lower bound."""
    okapi = BM25(CORPUS)
    plus  = BM25Plus(CORPUS)
    okapi_scores = okapi.get_scores("fox")
    plus_scores  = plus.get_scores("fox")
    # For documents that match, BM25+ should score >= BM25 Okapi
    for o, p in zip(okapi_scores, plus_scores):
        if o > 0:
            assert p >= o


@skip_no_ext
def test_bm25plus_top_n_sorted_descending():
    bm25   = BM25Plus(CORPUS)
    top    = bm25.get_top_n("machine learning deep", n=4)
    scores = [s for s, _ in top]
    assert scores == sorted(scores, reverse=True)


@skip_no_ext
def test_bm25plus_top_n_docs():
    bm25 = BM25Plus(CORPUS)
    docs = bm25.get_top_n_docs("fox", n=2)
    assert len(docs) == 2
    assert all(isinstance(d, str) for d in docs)


@skip_no_ext
def test_bm25plus_repr():
    bm25 = BM25Plus(CORPUS)
    r = repr(bm25)
    assert "BM25Plus" in r


# ══════════════════════════════════════════════════════════════════════════════
#  BM25Adpt
# ══════════════════════════════════════════════════════════════════════════════

@skip_no_ext
def test_bm25adpt_construction():
    bm25 = BM25Adpt(CORPUS)
    assert bm25.corpus_size == len(CORPUS)
    assert bm25.avg_doc_length > 0
    assert bm25.k1 == pytest.approx(1.5)


@skip_no_ext
def test_bm25adpt_empty_corpus_raises():
    with pytest.raises(ValueError):
        BM25Adpt([])


@skip_no_ext
def test_bm25adpt_scores_correct_length():
    bm25   = BM25Adpt(CORPUS)
    scores = bm25.get_scores("fox")
    assert len(scores) == len(CORPUS)


@skip_no_ext
def test_bm25adpt_scores_non_negative():
    bm25 = BM25Adpt(CORPUS)
    for s in bm25.get_scores("retrieval search ranking"):
        assert s >= 0.0


@skip_no_ext
def test_bm25adpt_unknown_query_all_zeros():
    bm25   = BM25Adpt(CORPUS)
    scores = bm25.get_scores("xyzzy_never_appears_12345")
    assert all(s == 0.0 for s in scores)


@skip_no_ext
def test_bm25adpt_top_n_sorted_descending():
    bm25   = BM25Adpt(CORPUS)
    top    = bm25.get_top_n("machine learning deep", n=4)
    scores = [s for s, _ in top]
    assert scores == sorted(scores, reverse=True)


@skip_no_ext
def test_bm25adpt_top_n_docs():
    bm25 = BM25Adpt(CORPUS)
    docs = bm25.get_top_n_docs("fox", n=2)
    assert len(docs) == 2
    assert all(isinstance(d, str) for d in docs)


@skip_no_ext
def test_bm25adpt_relevant_doc_scores_highest():
    bm25   = BM25Adpt(CORPUS)
    scores = bm25.get_scores("BM25 retrieval search")
    best   = scores.index(max(scores))
    assert best == 5


@skip_no_ext
def test_bm25adpt_repr():
    bm25 = BM25Adpt(CORPUS)
    r = repr(bm25)
    assert "BM25Adpt" in r


# ══════════════════════════════════════════════════════════════════════════════
#  BM25F (skeleton)
# ══════════════════════════════════════════════════════════════════════════════

FIELD_CORPUS = [
    {"title": "BM25 algorithm overview", "body": "BM25 is a bag of words retrieval function"},
    {"title": "deep learning models",    "body": "deep learning requires large amounts of data"},
    {"title": "the quick brown fox",     "body": "the fox jumps over the lazy dog"},
    {"title": "information retrieval",   "body": "IR systems rank documents by relevance scores"},
]


@skip_no_ext
def test_bm25f_construction():
    bm25f = BM25F(FIELD_CORPUS, field_weights={"title": 2.0, "body": 1.0})
    assert bm25f.corpus_size == len(FIELD_CORPUS)


@skip_no_ext
def test_bm25f_empty_corpus_raises():
    with pytest.raises(ValueError):
        BM25F([], field_weights={"title": 1.0})


@skip_no_ext
def test_bm25f_scores_correct_length():
    bm25f  = BM25F(FIELD_CORPUS, field_weights={"title": 2.0, "body": 1.0})
    scores = bm25f.get_scores("BM25")
    assert len(scores) == len(FIELD_CORPUS)


@skip_no_ext
def test_bm25f_scores_non_negative():
    bm25f = BM25F(FIELD_CORPUS, field_weights={"title": 2.0, "body": 1.0})
    for s in bm25f.get_scores("retrieval ranking"):
        assert s >= 0.0


@skip_no_ext
def test_bm25f_title_boost():
    """A term in the title (weight 2.0) should boost the score vs body-only."""
    bm25f  = BM25F(FIELD_CORPUS, field_weights={"title": 2.0, "body": 1.0})
    scores = bm25f.get_scores("BM25")
    # Doc 0 has "BM25" in both title and body — should score highest
    best = scores.index(max(scores))
    assert best == 0


@skip_no_ext
def test_bm25f_top_n_sorted_descending():
    bm25f = BM25F(FIELD_CORPUS, field_weights={"title": 2.0, "body": 1.0})
    top   = bm25f.get_top_n("deep learning", n=3)
    scores = [s for s, _ in top]
    assert scores == sorted(scores, reverse=True)


@skip_no_ext
def test_bm25f_set_field_b():
    bm25f = BM25F(FIELD_CORPUS, field_weights={"title": 2.0, "body": 1.0})
    # Should not raise
    bm25f.set_field_b("title", 0.5)
    bm25f.set_field_b("body", 0.3)


@skip_no_ext
def test_bm25f_repr():
    bm25f = BM25F(FIELD_CORPUS, field_weights={"title": 2.0, "body": 1.0})
    r = repr(bm25f)
    assert "BM25F" in r


# ══════════════════════════════════════════════════════════════════════════════
#  Cross-variant: all variants produce valid scores for the same query
# ══════════════════════════════════════════════════════════════════════════════

@skip_no_ext
def test_all_variants_produce_scores():
    """Every variant should produce a valid score vector for the same query."""
    query = "machine learning retrieval"
    variants = [
        BM25(CORPUS),
        BM25L(CORPUS),
        BM25Plus(CORPUS),
        BM25Adpt(CORPUS),
    ]
    for v in variants:
        scores = v.get_scores(query)
        assert len(scores) == len(CORPUS), f"{type(v).__name__} returned wrong length"
        assert all(s >= 0.0 for s in scores), f"{type(v).__name__} has negative scores"
