#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Use the canonical header directly to avoid the ODR violation that
// occurred when both "bm25.hpp" and "core/bm25.hpp" were pulled into
// the same translation unit through transitive includes.
#include "core/bm25.hpp"
#include "bm25_variants.hpp"
#include "bm25f.hpp"
#include "inverted_index.hpp"
#include "hybrid_rrf.hpp"

namespace py = pybind11;
using namespace flashbm25;

PYBIND11_MODULE(_flashbm25, m) {
    m.doc() = "FlashBM25 — High-performance retrieval engine (C/C++ core)";

    // ── BM25 ─────────────────────────────────────────────────────────────────────────────
    py::class_<BM25>(m, "BM25")
        .def(py::init<const std::vector<std::string>&, float, float, float, bool>(),
             py::arg("corpus"),
             py::arg("k1")       = 1.5f,
             py::arg("b")        = 0.75f,
             py::arg("epsilon")  = 0.25f,
             py::arg("lowercase") = true,
             R"doc(Classic Okapi BM25 index.

Parameters
----------
corpus : list[str]
k1 : float  — term saturation (default 1.5)
b  : float  — length normalization (default 0.75)
epsilon : float — IDF floor (default 0.25)
lowercase : bool (default True))doc")
        .def("get_scores",   &BM25::get_scores,   py::arg("query"))
        .def("get_top_n",    &BM25::get_top_n,    py::arg("query"), py::arg("n") = 5)
        .def("get_top_n_docs", &BM25::get_top_n_docs,
             py::arg("corpus"), py::arg("query"), py::arg("n") = 5)
        .def("add_documents", &BM25::add_documents, py::arg("documents"),
             "Append new documents without rebuilding the full index.")
        .def("save", &BM25::save, py::arg("path"),
             "Persist the BM25 core state to disk.")
        .def("dumps", [](const BM25& bm25) {
            return py::bytes(bm25.dumps());
        }, "Serialize the BM25 core state to bytes.")
        .def_static("load", &BM25::load, py::arg("path"),
             "Load a persisted BM25 core state from disk.")
        .def_static("loads", [](py::bytes payload) {
            return BM25::loads(static_cast<std::string>(payload));
        }, py::arg("payload"),
             "Deserialize a BM25 core state from bytes.")
        .def_property_readonly("corpus_size",    &BM25::corpus_size)
        .def_property_readonly("avg_doc_length", &BM25::average_doc_length)
        .def_readonly("k1", &BM25::k1)
        .def_readonly("b",  &BM25::b)
        .def_readonly("epsilon", &BM25::epsilon)
        .def_readonly("lowercase", &BM25::lowercase)
        .def("__repr__", [](const BM25& s) {
            return "<BM25 corpus_size=" + std::to_string(s.corpus_size()) +
                   " avgdl=" + std::to_string(s.average_doc_length()) + ">";
        });

    // ── BM25+ ────────────────────────────────────────────────────────────────────────────
    py::class_<BM25Plus>(m, "BM25Plus")
        .def(py::init<const std::vector<std::string>&, float, float, float, float, bool>(),
             py::arg("corpus"),
             py::arg("k1")      = 1.5f,
             py::arg("b")       = 0.75f,
             py::arg("delta")   = 1.0f,
             py::arg("epsilon") = 0.25f,
             py::arg("lowercase") = true,
             R"doc(BM25+ (Lv & Zhai, 2011) — adds lower-bound delta to TF component.

delta > 0 (default 1.0) ensures every matching term contributes positively,
fixing the over-penalisation problem in classic BM25.)doc")
        .def("get_scores", &BM25Plus::get_scores, py::arg("query"))
        .def("get_top_n",  &BM25Plus::get_top_n,  py::arg("query"), py::arg("n") = 5)
        .def_property_readonly("corpus_size",    &BM25Plus::corpus_size)
        .def_property_readonly("avg_doc_length", &BM25Plus::average_doc_length)
        .def_readonly("k1",    &BM25Plus::k1)
        .def_readonly("b",     &BM25Plus::b)
        .def_readonly("delta", &BM25Plus::delta)
        .def_readonly("epsilon", &BM25Plus::epsilon)
        .def("__repr__", [](const BM25Plus& s) {
            return "<BM25Plus corpus_size=" + std::to_string(s.corpus_size()) +
                   " delta=" + std::to_string(s.delta) + ">";
        });

    // ── BM25L ────────────────────────────────────────────────────────────────────────────
    py::class_<BM25L>(m, "BM25L")
        .def(py::init<const std::vector<std::string>&, float, float, float, float, bool>(),
             py::arg("corpus"),
             py::arg("k1")      = 1.5f,
             py::arg("b")       = 0.75f,
             py::arg("delta")   = 0.5f,
             py::arg("epsilon") = 0.25f,
             py::arg("lowercase") = true,
             R"doc(BM25L (Lv & Zhai, 2011) — normalises TF before length-normalisation.

Reduces over-penalisation of long documents; delta recommended in [0.0, 0.5].)doc")
        .def("get_scores", &BM25L::get_scores, py::arg("query"))
        .def("get_top_n",  &BM25L::get_top_n,  py::arg("query"), py::arg("n") = 5)
        .def_property_readonly("corpus_size",    &BM25L::corpus_size)
        .def_property_readonly("avg_doc_length", &BM25L::average_doc_length)
        .def_readonly("k1",    &BM25L::k1)
        .def_readonly("b",     &BM25L::b)
        .def_readonly("delta", &BM25L::delta)
        .def_readonly("epsilon", &BM25L::epsilon)
        .def("__repr__", [](const BM25L& s) {
            return "<BM25L corpus_size=" + std::to_string(s.corpus_size()) +
                   " delta=" + std::to_string(s.delta) + ">";
        });

    // ── BM25Adpt ─────────────────────────────────────────────────────────────────────────────
    py::class_<BM25Adpt>(m, "BM25Adpt")
        .def(py::init<const std::vector<std::string>&, float, float, float, bool>(),
             py::arg("corpus"),
             py::arg("k1")      = 1.5f,
             py::arg("b")       = 0.75f,
             py::arg("epsilon") = 0.25f,
             py::arg("lowercase") = true,
             R"doc(BM25Adpt (Lv & Zhai, 2011) — adaptive k1 per query term.

k1 is automatically tuned per term based on the term's average TF across
the corpus.  High-frequency terms get higher k1 (slower saturation),
rare terms get lower k1 (faster saturation).)doc")
        .def("get_scores", &BM25Adpt::get_scores, py::arg("query"))
        .def("get_top_n",  &BM25Adpt::get_top_n,  py::arg("query"), py::arg("n") = 5)
        .def_property_readonly("corpus_size",    &BM25Adpt::corpus_size)
        .def_property_readonly("avg_doc_length", &BM25Adpt::average_doc_length)
        .def_readonly("k1", &BM25Adpt::k1)
        .def_readonly("b",  &BM25Adpt::b)
        .def_readonly("epsilon", &BM25Adpt::epsilon)
        .def("__repr__", [](const BM25Adpt& s) {
            return "<BM25Adpt corpus_size=" + std::to_string(s.corpus_size()) +
                   " k1_base=" + std::to_string(s.k1) + ">";
        });

    // ── BM25F ──────────────────────────────────────────────────────────────────────────────
    py::class_<BM25F>(m, "BM25F")
        .def(py::init<const std::vector<FieldDoc>&,
                      const std::unordered_map<std::string, double>&,
                      float, float, bool>(),
             py::arg("corpus"),
             py::arg("field_weights"),
             py::arg("k1")       = 1.5f,
             py::arg("epsilon")  = 0.25f,
             py::arg("lowercase") = true,
             R"doc(BM25F (Robertson et al., 2004) — field-weighted multi-field scoring.

Each document is a dict mapping field names to text.  field_weights maps
field names to their boost multipliers.

Example
-------
corpus = [{"title": "BM25 intro", "body": "BM25 is ..."}, ...]
bm25f = BM25F(corpus, field_weights={"title": 2.0, "body": 1.0})
scores = bm25f.get_scores("BM25"))doc")
        .def("get_scores",   &BM25F::get_scores,   py::arg("query"))
        .def("get_top_n",    &BM25F::get_top_n,    py::arg("query"), py::arg("n") = 5)
        .def("set_field_b",  &BM25F::set_field_b,  py::arg("field"), py::arg("b"))
        .def_property_readonly("corpus_size", &BM25F::corpus_size)
        .def_readonly("k1", &BM25F::k1)
        .def_readonly("epsilon", &BM25F::epsilon)
        .def("__repr__", [](const BM25F& s) {
            return "<BM25F corpus_size=" + std::to_string(s.corpus_size()) + ">";
        });

    // ── InvertedIndex ──────────────────────────────────────────────────────────────────────────
    py::class_<InvertedIndex>(m, "InvertedIndex")
        .def(py::init<>(),
             R"doc(Low-level inverted index with on-disk serialization.

Use add_document() to build incrementally, then finalize() before querying.
Save/load with .save(path) / .load(path).)doc")
        .def("add_document", &InvertedIndex::add_document,
             py::arg("doc_id"), py::arg("tokens"),
             "Add a tokenized document to the index.")
        .def("finalize", &InvertedIndex::finalize,
             "Compute avgdl. Must be called after all documents are added.")
        .def("save", &InvertedIndex::save, py::arg("path"),
             "Serialize the index to a binary .fbidx file.")
        .def("load", &InvertedIndex::load, py::arg("path"),
             "Load an index from a .fbidx file.")
        .def_readonly("num_docs",   &InvertedIndex::num_docs)
        .def_readonly("avgdl",      &InvertedIndex::avgdl)
        .def("__repr__", [](const InvertedIndex& idx) {
            return "<InvertedIndex num_docs=" + std::to_string(idx.num_docs) +
                   " avgdl=" + std::to_string(idx.avgdl) + ">";
        });

    // ── RRFScorer ──────────────────────────────────────────────────────────────────────────
    py::class_<RRFScorer>(m, "RRFScorer")
        .def(py::init<double>(), py::arg("k") = 60.0,
             R"doc(Reciprocal Rank Fusion scorer (Cormack et al., 2009).

Fuses multiple ranked lists without score normalisation.

Usage
-----
rrf = RRFScorer(k=60)
rrf.add_ranking(bm25.get_top_n(query, 100))
rrf.add_ranking(bm25plus.get_top_n(query, 100))
results = rrf.fuse(top_n=10))doc")
        .def("add_ranking", &RRFScorer::add_ranking, py::arg("ranked"),
             "Add a ranked list (list of (score, doc_id) pairs).")
        .def("fuse",  &RRFScorer::fuse, py::arg("top_n") = 10,
             "Return top-n fused (rrf_score, doc_id) pairs.")
        .def("reset", &RRFScorer::reset, "Clear all stored rankings.")
        .def_property_readonly("k", &RRFScorer::k)
        .def("__repr__", [](const RRFScorer& r) {
            return "<RRFScorer k=" + std::to_string(r.k()) + ">";
        });
}
