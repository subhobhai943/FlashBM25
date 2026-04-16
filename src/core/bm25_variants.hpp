#pragma once
// BM25+ and BM25L — drop-in replacements for classic BM25.
//
// BM25+  (Lv & Zhai, 2011) — adds a lower-bound delta to the TF component so
//         that a term that appears at least once in a document always gets
//         a positive contribution, fixing the "over-penalization" problem.
//
// BM25L  (Lv & Zhai, 2011) — normalizes the average TF before applying the
//         length-normalization so that the effective TF for long docs is not
//         artificially depressed.

#include <string>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <numeric>
#include "bm25.hpp"  // reuses tokenize() and the same index structures

namespace flashbm25 {

// ─── BM25+ ────────────────────────────────────────────────────────────────────
// delta > 0 (typically 1.0) prevents TF from reaching 0 when tf is very small.
class BM25Plus {
public:
    float  k1;
    float  b;
    float  delta;     // lower-bound addition (Lv & Zhai recommend 1.0)
    float  epsilon;
    bool   lowercase;

    std::size_t num_docs;
    double      avgdl;

    std::vector<double>                                                doc_len;
    std::unordered_map<std::string, std::unordered_map<std::size_t, double>> tf_index;
    std::unordered_map<std::string, double>                            idf_cache;

    explicit BM25Plus(
        const std::vector<std::string>& corpus,
        float k1_    = 1.5f,
        float b_     = 0.75f,
        float delta_ = 1.0f,
        float eps_   = 0.25f,
        bool  lower  = true
    );

    std::vector<double>                         get_scores(const std::string& query) const;
    std::vector<std::pair<double, std::size_t>> get_top_n (const std::string& query, std::size_t n = 5) const;

    std::size_t corpus_size()        const { return num_docs; }
    double      average_doc_length() const { return avgdl; }

private:
    double _idf(const std::string& term) const;
    void   _build_index(const std::vector<std::string>& corpus);
    void   _build_idf();
};

// ─── BM25L ────────────────────────────────────────────────────────────────────
// ctf_hat is the normalized TF; delta is added to the normalised score
// before multiplication with IDF, further reducing the penalty on long docs.
class BM25L {
public:
    float  k1;
    float  b;
    float  delta;     // recommended range 0.0–0.5
    float  epsilon;
    bool   lowercase;

    std::size_t num_docs;
    double      avgdl;

    std::vector<double>                                                doc_len;
    std::unordered_map<std::string, std::unordered_map<std::size_t, double>> tf_index;
    std::unordered_map<std::string, double>                            idf_cache;

    explicit BM25L(
        const std::vector<std::string>& corpus,
        float k1_    = 1.5f,
        float b_     = 0.75f,
        float delta_ = 0.5f,
        float eps_   = 0.25f,
        bool  lower  = true
    );

    std::vector<double>                         get_scores(const std::string& query) const;
    std::vector<std::pair<double, std::size_t>> get_top_n (const std::string& query, std::size_t n = 5) const;

    std::size_t corpus_size()        const { return num_docs; }
    double      average_doc_length() const { return avgdl; }

private:
    double _idf(const std::string& term) const;
    void   _build_index(const std::vector<std::string>& corpus);
    void   _build_idf();
};

} // namespace flashbm25
