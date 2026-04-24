#pragma once

// BM25+ and BM25L are drop-in replacements for classic BM25.
//
// BM25+ (Lv & Zhai, 2011) adds a lower-bound delta to the TF component so a
// term that appears at least once in a document always contributes positively.
//
// BM25L (Lv & Zhai, 2011) normalizes average TF before applying length
// normalization so long documents are not over-penalized.

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "bm25.hpp"
#include "simd_scoring.hpp"

namespace flashbm25 {

class BM25Plus {
public:
    float k1;
    float b;
    float delta;
    float epsilon;
    bool lowercase;

    std::size_t num_docs;
    double avgdl;

    std::vector<double> doc_len;
    TfIndex tf_index;
    PostingsIndex postings_index;
    std::unordered_map<std::string, double> idf_cache;

    explicit BM25Plus(const std::vector<std::string>& corpus,
                      float k1_ = 1.5f,
                      float b_ = 0.75f,
                      float delta_ = 1.0f,
                      float eps_ = 0.25f,
                      bool lower = true);

    std::vector<double> get_scores(const std::string& query) const;
    std::vector<std::pair<double, std::size_t>> get_top_n(const std::string& query,
                                                          std::size_t n = 5) const;

    std::size_t corpus_size() const {
        return num_docs;
    }

    double average_doc_length() const {
        return avgdl;
    }

private:
    double _idf(const std::string& term) const;
    void _build_index(const std::vector<std::string>& corpus);
    void _build_idf();
    void _build_postings_index();
};

class BM25L {
public:
    float k1;
    float b;
    float delta;
    float epsilon;
    bool lowercase;

    std::size_t num_docs;
    double avgdl;

    std::vector<double> doc_len;
    TfIndex tf_index;
    PostingsIndex postings_index;
    std::unordered_map<std::string, double> idf_cache;

    explicit BM25L(const std::vector<std::string>& corpus,
                   float k1_ = 1.5f,
                   float b_ = 0.75f,
                   float delta_ = 0.5f,
                   float eps_ = 0.25f,
                   bool lower = true);

    std::vector<double> get_scores(const std::string& query) const;
    std::vector<std::pair<double, std::size_t>> get_top_n(const std::string& query,
                                                          std::size_t n = 5) const;

    std::size_t corpus_size() const {
        return num_docs;
    }

    double average_doc_length() const {
        return avgdl;
    }

private:
    double _idf(const std::string& term) const;
    void _build_index(const std::vector<std::string>& corpus);
    void _build_idf();
    void _build_postings_index();
};

class BM25Adpt {
public:
    float k1;
    float b;
    float epsilon;
    bool lowercase;

    std::size_t num_docs;
    double avgdl;

    std::vector<double> doc_len;
    TfIndex tf_index;
    PostingsIndex postings_index;
    std::unordered_map<std::string, double> idf_cache;
    std::unordered_map<std::string, double> k1_cache;

    explicit BM25Adpt(const std::vector<std::string>& corpus,
                      float k1_ = 1.5f,
                      float b_ = 0.75f,
                      float eps_ = 0.25f,
                      bool lower = true);

    std::vector<double> get_scores(const std::string& query) const;
    std::vector<std::pair<double, std::size_t>> get_top_n(const std::string& query,
                                                          std::size_t n = 5) const;

    std::size_t corpus_size() const {
        return num_docs;
    }

    double average_doc_length() const {
        return avgdl;
    }

private:
    double _idf(const std::string& term) const;
    double _adaptive_k1(const std::string& term) const;
    void _build_index(const std::vector<std::string>& corpus);
    void _build_idf();
    void _build_postings_index();
    void _build_adaptive_k1();
};

} // namespace flashbm25
