#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "bm25.hpp"

namespace py = pybind11;
using namespace flashbm25;

PYBIND11_MODULE(_flashbm25, m) {
    m.doc() = "FlashBM25 — High-performance BM25 retrieval engine (C++ core)";

    py::class_<BM25>(m, "BM25")
        .def(py::init<const std::vector<std::string>&, float, float, float, bool>(),
             py::arg("corpus"),
             py::arg("k1")      = 1.5f,
             py::arg("b")       = 0.75f,
             py::arg("epsilon") = 0.25f,
             py::arg("lowercase") = true,
             R"doc(
Build a BM25 index over a corpus of documents.

Parameters
----------
corpus : list[str]
    List of plain-text documents to index.
k1 : float, optional
    Term saturation parameter (default 1.5).
b : float, optional
    Length normalisation parameter (default 0.75).
epsilon : float, optional
    IDF floor to prevent negative scores (default 0.25).
lowercase : bool, optional
    Normalise text to lowercase before indexing (default True).
)doc")

        .def("get_scores", &BM25::get_scores, py::arg("query"),
             R"doc(
Return BM25 scores for every document in the corpus.

Parameters
----------
query : str

Returns
-------
list[float]
    Score for each document (same order as the original corpus).
)doc")

        .def("get_top_n", &BM25::get_top_n,
             py::arg("query"), py::arg("n") = 5,
             R"doc(
Return the top-n (score, doc_index) pairs, sorted by score descending.

Parameters
----------
query : str
n : int, optional
    Number of results to return (default 5).

Returns
-------
list[tuple[float, int]]
)doc")

        .def("get_top_n_docs", &BM25::get_top_n_docs,
             py::arg("corpus"), py::arg("query"), py::arg("n") = 5,
             R"doc(
Return the top-n documents from *corpus* sorted by score descending.

Parameters
----------
corpus : list[str]
    The same corpus used to build the index.
query : str
n : int, optional
    Number of results to return (default 5).

Returns
-------
list[str]
)doc")

        .def_property_readonly("corpus_size",    &BM25::corpus_size)
        .def_property_readonly("avg_doc_length", &BM25::average_doc_length)
        .def_readonly("k1",      &BM25::k1)
        .def_readonly("b",       &BM25::b)
        .def_readonly("epsilon", &BM25::epsilon)

        .def("__repr__", [](const BM25& self) {
            return "<BM25 corpus_size=" + std::to_string(self.corpus_size()) +
                   " avgdl=" + std::to_string(self.average_doc_length()) + ">";
        });
}
