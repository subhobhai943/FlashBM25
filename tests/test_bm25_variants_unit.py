"""Focused unit tests for BM25L, BM25Plus, and BM25Adpt."""

import pytest

try:
    from flashbm25 import BM25
    from flashbm25 import BM25Adpt
    from flashbm25 import BM25L
    from flashbm25 import BM25Plus

    HAS_EXT = True
except ImportError:
    HAS_EXT = False


skip_no_ext = pytest.mark.skipif(not HAS_EXT, reason="C++ extension not built")

CORPUS = [
    "saturn has bright rings made of ice and rock particles",
    "jupiter is the largest planet in the solar system",
    "mars is often called the red planet",
    "saturn rings rings rings are visible with a telescope",
]

VARIANT_FACTORY_CASES = [
    pytest.param("l", BM25L, {"delta": 0.5}, id="bm25l"),
    pytest.param("plus", BM25Plus, {"delta": 1.0}, id="bm25plus"),
    pytest.param("adpt", BM25Adpt, {}, id="bm25adpt"),
]

VARIANT_BUILDERS = [
    pytest.param(BM25L, id="bm25l"),
    pytest.param(BM25Plus, id="bm25plus"),
    pytest.param(BM25Adpt, id="bm25adpt"),
]


@skip_no_ext
@pytest.mark.parametrize("variant_name,expected_type,extra_kwargs", VARIANT_FACTORY_CASES)
def test_variant_factory_returns_expected_type(variant_name, expected_type, extra_kwargs):
    model = BM25(CORPUS, variant=variant_name, **extra_kwargs)
    assert isinstance(model, expected_type)
    assert model.corpus_size == len(CORPUS)


@skip_no_ext
@pytest.mark.parametrize("variant_cls", VARIANT_BUILDERS)
def test_variants_return_zero_for_unknown_query(variant_cls):
    model = variant_cls(CORPUS)
    assert all(score == 0.0 for score in model.get_scores("nonexistent_token_123"))


@skip_no_ext
@pytest.mark.parametrize("variant_cls", VARIANT_BUILDERS)
def test_variants_rank_high_tf_document_higher(variant_cls):
    model = variant_cls(CORPUS)
    scores = model.get_scores("saturn rings")

    assert scores[3] > scores[0]
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]


@skip_no_ext
@pytest.mark.parametrize("variant_cls", VARIANT_BUILDERS)
def test_variants_top_n_docs_align_with_top_n_indices(variant_cls):
    model = variant_cls(CORPUS)
    top_pairs = model.get_top_n("saturn rings", n=3)
    top_docs = model.get_top_n_docs("saturn rings", n=3)

    resolved_docs = [CORPUS[index] for _, index in top_pairs]
    assert top_docs == resolved_docs
