#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace flashbm25 {

struct Posting {
    std::size_t doc_id;
    double tf;
};

using TfIndex = std::unordered_map<std::string, std::unordered_map<std::size_t, double>>;
using PostingList = std::vector<Posting>;
using PostingsIndex = std::unordered_map<std::string, PostingList>;

void rebuild_postings_index(const TfIndex& tf_index, PostingsIndex& postings_index);

void score_okapi_postings(const PostingList& postings,
                          const std::vector<double>& doc_len,
                          double avgdl,
                          double k1,
                          double b,
                          double term_weight,
                          std::vector<double>& scores);

void score_bm25plus_postings(const PostingList& postings,
                             const std::vector<double>& doc_len,
                             double avgdl,
                             double k1,
                             double b,
                             double delta,
                             double term_weight,
                             std::vector<double>& scores);

void score_bm25l_postings(const PostingList& postings,
                          const std::vector<double>& doc_len,
                          double avgdl,
                          double k1,
                          double b,
                          double delta,
                          double term_weight,
                          std::vector<double>& scores);

const char* active_simd_backend_name() noexcept;

} // namespace flashbm25
