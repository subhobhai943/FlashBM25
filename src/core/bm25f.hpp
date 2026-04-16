#pragma once
// BM25F — Field-weighted BM25 scoring (Robertson et al., 2004)
//
// Scores documents that have multiple named fields (e.g. title, body, tags).
// Each field has its own boost weight and optional per-field length normalization
// parameter (b_f).  Term frequencies are combined across fields using weighted
// sums before the standard BM25 saturation function is applied.
//
// This is a SKELETON for Phase 1 — full implementation lands in Phase 3.

#include <string>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <numeric>
#include "bm25.hpp"  // tokenize()

namespace flashbm25 {

// A single document is represented as a mapping: field_name -> field_text.
using FieldDoc = std::unordered_map<std::string, std::string>;

// Per-field configuration: weight (boost) and length normalization (b).
struct FieldParams {
    double weight = 1.0;  // boost multiplier for this field
    double b      = 0.75; // length normalization for this field
};

class BM25F {
public:
    float  k1;
    float  epsilon;
    bool   lowercase;

    std::size_t num_docs;

    // Per-field average document lengths
    std::unordered_map<std::string, double> field_avgdl;

    // Per-field per-document lengths
    std::unordered_map<std::string, std::vector<double>> field_doc_len;

    // Combined weighted TF index: term -> {doc_id -> weighted_tf}
    std::unordered_map<std::string, std::unordered_map<std::size_t, double>> tf_index;

    // IDF cache
    std::unordered_map<std::string, double> idf_cache;

    // Field configuration
    std::unordered_map<std::string, FieldParams> field_params;

    explicit BM25F(
        const std::vector<FieldDoc>& corpus,
        const std::unordered_map<std::string, double>& field_weights,
        float k1_      = 1.5f,
        float epsilon_  = 0.25f,
        bool  lower     = true
    );

    std::vector<double>                         get_scores(const std::string& query) const;
    std::vector<std::pair<double, std::size_t>> get_top_n (const std::string& query, std::size_t n = 5) const;

    // Set per-field b parameter (default 0.75 for all fields)
    void set_field_b(const std::string& field, double b_val);

    std::size_t corpus_size() const { return num_docs; }

private:
    double _idf(const std::string& term) const;
    void   _build_index(const std::vector<FieldDoc>& corpus);
    void   _build_idf();
};

} // namespace flashbm25
