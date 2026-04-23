"""Integration tests against curated TREC/MS-MARCO query-document pairs."""

import json
from pathlib import Path

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

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "trec_ms_marco_pairs.json"
PAIR_FIXTURES = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _build_okapi(corpus):
    return BM25(corpus)


def _build_bm25l(corpus):
    return BM25L(corpus)


def _build_bm25plus(corpus):
    return BM25Plus(corpus)


def _build_bm25adpt(corpus):
    return BM25Adpt(corpus)


VARIANT_BUILDERS = [
    pytest.param("okapi", _build_okapi, id="okapi"),
    pytest.param("bm25l", _build_bm25l, id="bm25l"),
    pytest.param("bm25plus", _build_bm25plus, id="bm25plus"),
    pytest.param("bm25adpt", _build_bm25adpt, id="bm25adpt"),
]


def _scenario_id(scenario):
    return f"{scenario['dataset']}-{scenario['query_id']}"


@skip_no_ext
@pytest.mark.parametrize("scenario", PAIR_FIXTURES, ids=_scenario_id)
@pytest.mark.parametrize("variant_name,builder", VARIANT_BUILDERS)
def test_known_pairs_retrieve_expected_documents(scenario, variant_name, builder):
    corpus = [doc["text"] for doc in scenario["documents"]]
    doc_ids = [doc["doc_id"] for doc in scenario["documents"]]
    relevant_doc_ids = set(scenario["relevant_doc_ids"])

    model = builder(corpus)
    scores = model.get_scores(scenario["query"])

    top_n = min(3, len(corpus))
    top_indices = [doc_index for _, doc_index in model.get_top_n(scenario["query"], n=top_n)]
    top_doc_ids = {doc_ids[index] for index in top_indices}

    assert top_doc_ids.intersection(relevant_doc_ids), (
        f"{variant_name} failed for {scenario['dataset']}:{scenario['query_id']}"
    )

    for relevant_doc_id in relevant_doc_ids:
        relevant_index = doc_ids.index(relevant_doc_id)
        assert scores[relevant_index] > 0.0
