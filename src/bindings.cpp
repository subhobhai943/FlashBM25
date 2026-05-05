#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

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

namespace {

struct TopNRecord {
    float score;
    std::uint32_t doc_id;
};

static_assert(sizeof(TopNRecord) == sizeof(float) + sizeof(std::uint32_t),
              "TopNRecord must match the NumPy structured dtype layout.");

py::array_t<float> scores_to_numpy(std::vector<double>&& scores) {
    auto buffer = std::make_unique<std::vector<float>>();
    buffer->reserve(scores.size());
    for (double score : scores) {
        buffer->push_back(static_cast<float>(score));
    }

    auto* raw = buffer.get();
    py::capsule owner(buffer.release(), [](void* ptr) {
        delete static_cast<std::vector<float>*>(ptr);
    });

    return py::array_t<float>(
        {static_cast<py::ssize_t>(raw->size())},
        {static_cast<py::ssize_t>(sizeof(float))},
        raw->data(),
        owner
    );
}

py::dtype top_n_dtype() {
    py::module_ numpy = py::module_::import("numpy");
    py::list fields;
    fields.append(py::make_tuple("score", "f4"));
    fields.append(py::make_tuple("doc_id", "u4"));
    return py::reinterpret_borrow<py::dtype>(numpy.attr("dtype")(fields));
}

py::array top_n_to_numpy(std::vector<std::pair<double, std::size_t>>&& ranked) {
    auto buffer = std::make_unique<std::vector<TopNRecord>>();
    buffer->reserve(ranked.size());

    constexpr auto max_doc_id = std::numeric_limits<std::uint32_t>::max();
    for (const auto& [score, doc_id] : ranked) {
        if (doc_id > max_doc_id) {
            throw std::overflow_error("doc_id does not fit in uint32.");
        }
        buffer->push_back({static_cast<float>(score), static_cast<std::uint32_t>(doc_id)});
    }

    auto* raw = buffer.get();
    py::capsule owner(buffer.release(), [](void* ptr) {
        delete static_cast<std::vector<TopNRecord>*>(ptr);
    });

    return py::array(
        top_n_dtype(),
        {static_cast<py::ssize_t>(raw->size())},
        {static_cast<py::ssize_t>(sizeof(TopNRecord))},
        raw->data(),
        owner
    );
}

bool has_top_n_fields(py::handle ranked) {
    if (!py::hasattr(ranked, "dtype")) {
        return false;
    }

    py::object names_obj = py::reinterpret_borrow<py::object>(ranked).attr("dtype").attr("names");
    if (names_obj.is_none()) {
        return false;
    }

    bool has_score = false;
    bool has_doc_id = false;
    for (py::handle name : py::reinterpret_borrow<py::tuple>(names_obj)) {
        const auto field_name = py::cast<std::string>(name);
        has_score = has_score || field_name == "score";
        has_doc_id = has_doc_id || field_name == "doc_id";
    }
    return has_score && has_doc_id;
}

std::vector<std::pair<double, std::size_t>> ranking_from_python(py::handle ranked) {
    if (!has_top_n_fields(ranked)) {
        return py::reinterpret_borrow<py::object>(ranked)
            .cast<std::vector<std::pair<double, std::size_t>>>();
    }

    py::object ranked_obj = py::reinterpret_borrow<py::object>(ranked);
    py::object scores = ranked_obj.attr("__getitem__")("score");
    py::object doc_ids = ranked_obj.attr("__getitem__")("doc_id");
    const py::ssize_t length = py::len(ranked_obj);

    std::vector<std::pair<double, std::size_t>> converted;
    converted.reserve(static_cast<std::size_t>(length));
    for (py::ssize_t i = 0; i < length; ++i) {
        const auto index = py::int_(i);
        converted.emplace_back(
            scores.attr("__getitem__")(index).cast<double>(),
            doc_ids.attr("__getitem__")(index).cast<std::size_t>()
        );
    }
    return converted;
}

template <typename Model>
py::array_t<float> get_scores_array(const Model& model, const std::string& query) {
    std::vector<double> scores;
    {
        py::gil_scoped_release release;
        scores = model.get_scores(query);
    }
    return scores_to_numpy(std::move(scores));
}

template <typename Model>
py::array get_top_n_array(const Model& model, const std::string& query, std::size_t n) {
    std::vector<std::pair<double, std::size_t>> ranked;
    {
        py::gil_scoped_release release;
        ranked = model.get_top_n(query, n);
    }
    return top_n_to_numpy(std::move(ranked));
}

} // namespace

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
        .def("get_scores",   &get_scores_array<BM25>,   py::arg("query"))
        .def("get_top_n",    &get_top_n_array<BM25>,    py::arg("query"), py::arg("n") = 5)
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
        .def("get_scores", &get_scores_array<BM25Plus>, py::arg("query"))
        .def("get_top_n",  &get_top_n_array<BM25Plus>,  py::arg("query"), py::arg("n") = 5)
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
        .def("get_scores", &get_scores_array<BM25L>, py::arg("query"))
        .def("get_top_n",  &get_top_n_array<BM25L>,  py::arg("query"), py::arg("n") = 5)
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
        .def("get_scores", &get_scores_array<BM25Adpt>, py::arg("query"))
        .def("get_top_n",  &get_top_n_array<BM25Adpt>,  py::arg("query"), py::arg("n") = 5)
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
        .def("get_scores",   &get_scores_array<BM25F>,   py::arg("query"))
        .def("get_top_n",    &get_top_n_array<BM25F>,    py::arg("query"), py::arg("n") = 5)
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
        .def("add_ranking", [](RRFScorer& scorer, py::object ranked) {
            scorer.add_ranking(ranking_from_python(ranked));
        }, py::arg("ranked"),
             "Add a ranked list or structured NumPy top-n record array.")
        .def("fuse",  &RRFScorer::fuse, py::arg("top_n") = 10,
             "Return top-n fused (rrf_score, doc_id) pairs.")
        .def("reset", &RRFScorer::reset, "Clear all stored rankings.")
        .def_property_readonly("k", &RRFScorer::k)
        .def("__repr__", [](const RRFScorer& r) {
            return "<RRFScorer k=" + std::to_string(r.k()) + ">";
        });
}
