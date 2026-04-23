"""Property-based tests for BM25 scoring invariants."""

import math

import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

try:
    from flashbm25 import BM25
    from flashbm25 import BM25Adpt
    from flashbm25 import BM25L
    from flashbm25 import BM25Plus

    HAS_EXT = True
except ImportError:
    HAS_EXT = False


skip_no_ext = pytest.mark.skipif(not HAS_EXT, reason="C++ extension not built")

VARIANT_KEYS = ("okapi", "l", "plus", "adpt")


def _build_variant(corpus, variant_key):
    if variant_key == "okapi":
        return BM25(corpus, k1=1.5, b=0.75, epsilon=0.25)
    if variant_key == "l":
        return BM25L(corpus, k1=1.5, b=0.75, delta=0.5, epsilon=0.25)
    if variant_key == "plus":
        return BM25Plus(corpus, k1=1.5, b=0.75, delta=1.0, epsilon=0.25)
    if variant_key == "adpt":
        return BM25Adpt(corpus, k1=1.5, b=0.75, epsilon=0.25)
    raise ValueError(f"Unsupported variant key: {variant_key!r}")


def _score_scale(model, variant_key):
    if variant_key in {"l", "plus"}:
        return 1.0 + model.delta
    return 1.0


def _expected_idf(num_docs, df, epsilon):
    raw = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)
    return max(epsilon, raw)


@st.composite
def _df_triplets(draw):
    num_docs = draw(st.integers(min_value=3, max_value=12))
    df_common = draw(st.integers(min_value=1, max_value=num_docs))
    df_rare = draw(st.integers(min_value=1, max_value=df_common))
    return num_docs, df_rare, df_common


@skip_no_ext
@settings(max_examples=60, deadline=None)
@given(
    variant_key=st.sampled_from(VARIANT_KEYS),
    term_frequencies=st.lists(
        st.integers(min_value=1, max_value=20),
        min_size=3,
        max_size=8,
        unique=True,
    ),
)
def test_score_monotonicity_with_term_frequency(variant_key, term_frequencies):
    fixed_doc_length = 24
    sorted_frequencies = sorted(term_frequencies)
    corpus = []

    for doc_index, tf in enumerate(sorted_frequencies):
        padding = [f"pad_{doc_index}_{j}" for j in range(fixed_doc_length - tf)]
        corpus.append(" ".join((["signal"] * tf) + padding))

    model = _build_variant(corpus, variant_key)
    scores = model.get_scores("signal")

    for left, right in zip(scores, scores[1:]):
        assert left <= right + 1e-9


@skip_no_ext
@settings(max_examples=60, deadline=None)
@given(variant_key=st.sampled_from(VARIANT_KEYS), triplet=_df_triplets())
def test_idf_bounds_match_expected_formula(variant_key, triplet):
    num_docs, df_rare, df_common = triplet
    doc_length = 8
    corpus = []

    for doc_index in range(num_docs):
        tokens = [f"filler_{doc_index}_{j}" for j in range(doc_length - 2)]

        if doc_index < df_common:
            tokens.append("common_term")
        if doc_index < df_rare:
            tokens.append("rare_term")

        while len(tokens) < doc_length:
            tokens.append(f"pad_{doc_index}_{len(tokens)}")

        corpus.append(" ".join(tokens))

    model = _build_variant(corpus, variant_key)
    score_scale = _score_scale(model, variant_key)

    rare_scores = model.get_scores("rare_term")
    common_scores = model.get_scores("common_term")
    unseen_scores = model.get_scores("term_that_is_not_present")

    expected_rare = _expected_idf(num_docs, df_rare, model.epsilon) * score_scale
    expected_common = _expected_idf(num_docs, df_common, model.epsilon) * score_scale

    assert rare_scores[0] == pytest.approx(expected_rare, rel=1e-6, abs=1e-9)
    assert common_scores[0] == pytest.approx(expected_common, rel=1e-6, abs=1e-9)
    assert rare_scores[0] + 1e-9 >= common_scores[0]
    assert common_scores[0] >= (model.epsilon * score_scale) - 1e-9
    assert all(score == 0.0 for score in unseen_scores)
