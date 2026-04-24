#include "simd_scoring.hpp"

#include <immintrin.h>

namespace flashbm25 {
namespace {

constexpr std::size_t kAvx2Lanes = 4;

inline void load_posting_batch(const PostingList& postings,
                               std::size_t offset,
                               long long (&doc_ids)[kAvx2Lanes],
                               double (&tf_values)[kAvx2Lanes]) {
    for (std::size_t lane = 0; lane < kAvx2Lanes; ++lane) {
        const auto& posting = postings[offset + lane];
        doc_ids[lane] = static_cast<long long>(posting.doc_id);
        tf_values[lane] = posting.tf;
    }
}

inline void scatter_add(const long long (&doc_ids)[kAvx2Lanes],
                        const double (&contrib)[kAvx2Lanes],
                        std::vector<double>& scores) {
    for (std::size_t lane = 0; lane < kAvx2Lanes; ++lane) {
        scores[static_cast<std::size_t>(doc_ids[lane])] += contrib[lane];
    }
}

inline void score_okapi_tail(const PostingList& postings,
                             const std::vector<double>& doc_len,
                             std::size_t offset,
                             double avgdl,
                             double k1,
                             double b,
                             double term_weight,
                             std::vector<double>& scores) {
    const double k1_plus_one = k1 + 1.0;
    const double one_minus_b = 1.0 - b;
    const double b_over_avgdl = b / avgdl;

    for (std::size_t index = offset; index < postings.size(); ++index) {
        const auto& posting = postings[index];
        const double tf = posting.tf;
        const double norm = one_minus_b + b_over_avgdl * doc_len[posting.doc_id];
        const double tf_adj = (tf * k1_plus_one) / (tf + k1 * norm);
        scores[posting.doc_id] += term_weight * tf_adj;
    }
}

inline void score_bm25plus_tail(const PostingList& postings,
                                const std::vector<double>& doc_len,
                                std::size_t offset,
                                double avgdl,
                                double k1,
                                double b,
                                double delta,
                                double term_weight,
                                std::vector<double>& scores) {
    const double k1_plus_one = k1 + 1.0;
    const double one_minus_b = 1.0 - b;
    const double b_over_avgdl = b / avgdl;

    for (std::size_t index = offset; index < postings.size(); ++index) {
        const auto& posting = postings[index];
        const double tf = posting.tf;
        const double norm = one_minus_b + b_over_avgdl * doc_len[posting.doc_id];
        const double tf_adj = (tf * k1_plus_one) / (tf + k1 * norm) + delta;
        scores[posting.doc_id] += term_weight * tf_adj;
    }
}

inline void score_bm25l_tail(const PostingList& postings,
                             const std::vector<double>& doc_len,
                             std::size_t offset,
                             double avgdl,
                             double k1,
                             double b,
                             double delta,
                             double term_weight,
                             std::vector<double>& scores) {
    const double k1_plus_one = k1 + 1.0;
    const double one_minus_b = 1.0 - b;
    const double b_over_avgdl = b / avgdl;

    for (std::size_t index = offset; index < postings.size(); ++index) {
        const auto& posting = postings[index];
        const double tf = posting.tf;
        const double norm = one_minus_b + b_over_avgdl * doc_len[posting.doc_id];
        const double c_tilde = tf / norm;
        const double tf_adj = (k1_plus_one * c_tilde) / (k1 + c_tilde) + delta;
        scores[posting.doc_id] += term_weight * tf_adj;
    }
}

} // namespace

void score_okapi_x86_avx2(const PostingList& postings,
                          const std::vector<double>& doc_len,
                          double avgdl,
                          double k1,
                          double b,
                          double term_weight,
                          std::vector<double>& scores) {
    const __m256d k1_vec = _mm256_set1_pd(k1);
    const __m256d k1_plus_one_vec = _mm256_set1_pd(k1 + 1.0);
    const __m256d one_minus_b_vec = _mm256_set1_pd(1.0 - b);
    const __m256d b_over_avgdl_vec = _mm256_set1_pd(b / avgdl);
    const __m256d term_weight_vec = _mm256_set1_pd(term_weight);

    std::size_t index = 0;
    const std::size_t vectorized_end = postings.size() - (postings.size() % kAvx2Lanes);
    for (; index < vectorized_end; index += kAvx2Lanes) {
        long long doc_ids[kAvx2Lanes];
        double tf_values[kAvx2Lanes];
        double contrib[kAvx2Lanes];
        load_posting_batch(postings, index, doc_ids, tf_values);

        const __m256i doc_id_vec =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(doc_ids));
        const __m256d tf_vec = _mm256_loadu_pd(tf_values);
        const __m256d dl_vec = _mm256_i64gather_pd(doc_len.data(), doc_id_vec, 8);
        const __m256d norm_vec =
            _mm256_add_pd(one_minus_b_vec, _mm256_mul_pd(b_over_avgdl_vec, dl_vec));
        const __m256d numerator_vec = _mm256_mul_pd(tf_vec, k1_plus_one_vec);
        const __m256d denominator_vec =
            _mm256_add_pd(tf_vec, _mm256_mul_pd(k1_vec, norm_vec));
        const __m256d contrib_vec =
            _mm256_mul_pd(term_weight_vec, _mm256_div_pd(numerator_vec, denominator_vec));

        _mm256_storeu_pd(contrib, contrib_vec);
        scatter_add(doc_ids, contrib, scores);
    }

    score_okapi_tail(postings, doc_len, index, avgdl, k1, b, term_weight, scores);
}

void score_bm25plus_x86_avx2(const PostingList& postings,
                             const std::vector<double>& doc_len,
                             double avgdl,
                             double k1,
                             double b,
                             double delta,
                             double term_weight,
                             std::vector<double>& scores) {
    const __m256d k1_vec = _mm256_set1_pd(k1);
    const __m256d k1_plus_one_vec = _mm256_set1_pd(k1 + 1.0);
    const __m256d one_minus_b_vec = _mm256_set1_pd(1.0 - b);
    const __m256d b_over_avgdl_vec = _mm256_set1_pd(b / avgdl);
    const __m256d delta_vec = _mm256_set1_pd(delta);
    const __m256d term_weight_vec = _mm256_set1_pd(term_weight);

    std::size_t index = 0;
    const std::size_t vectorized_end = postings.size() - (postings.size() % kAvx2Lanes);
    for (; index < vectorized_end; index += kAvx2Lanes) {
        long long doc_ids[kAvx2Lanes];
        double tf_values[kAvx2Lanes];
        double contrib[kAvx2Lanes];
        load_posting_batch(postings, index, doc_ids, tf_values);

        const __m256i doc_id_vec =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(doc_ids));
        const __m256d tf_vec = _mm256_loadu_pd(tf_values);
        const __m256d dl_vec = _mm256_i64gather_pd(doc_len.data(), doc_id_vec, 8);
        const __m256d norm_vec =
            _mm256_add_pd(one_minus_b_vec, _mm256_mul_pd(b_over_avgdl_vec, dl_vec));
        const __m256d numerator_vec = _mm256_mul_pd(tf_vec, k1_plus_one_vec);
        const __m256d denominator_vec =
            _mm256_add_pd(tf_vec, _mm256_mul_pd(k1_vec, norm_vec));
        const __m256d tf_adj_vec =
            _mm256_add_pd(_mm256_div_pd(numerator_vec, denominator_vec), delta_vec);
        const __m256d contrib_vec = _mm256_mul_pd(term_weight_vec, tf_adj_vec);

        _mm256_storeu_pd(contrib, contrib_vec);
        scatter_add(doc_ids, contrib, scores);
    }

    score_bm25plus_tail(postings, doc_len, index, avgdl, k1, b, delta, term_weight, scores);
}

void score_bm25l_x86_avx2(const PostingList& postings,
                          const std::vector<double>& doc_len,
                          double avgdl,
                          double k1,
                          double b,
                          double delta,
                          double term_weight,
                          std::vector<double>& scores) {
    const __m256d k1_vec = _mm256_set1_pd(k1);
    const __m256d k1_plus_one_vec = _mm256_set1_pd(k1 + 1.0);
    const __m256d one_minus_b_vec = _mm256_set1_pd(1.0 - b);
    const __m256d b_over_avgdl_vec = _mm256_set1_pd(b / avgdl);
    const __m256d delta_vec = _mm256_set1_pd(delta);
    const __m256d term_weight_vec = _mm256_set1_pd(term_weight);

    std::size_t index = 0;
    const std::size_t vectorized_end = postings.size() - (postings.size() % kAvx2Lanes);
    for (; index < vectorized_end; index += kAvx2Lanes) {
        long long doc_ids[kAvx2Lanes];
        double tf_values[kAvx2Lanes];
        double contrib[kAvx2Lanes];
        load_posting_batch(postings, index, doc_ids, tf_values);

        const __m256i doc_id_vec =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(doc_ids));
        const __m256d tf_vec = _mm256_loadu_pd(tf_values);
        const __m256d dl_vec = _mm256_i64gather_pd(doc_len.data(), doc_id_vec, 8);
        const __m256d norm_vec =
            _mm256_add_pd(one_minus_b_vec, _mm256_mul_pd(b_over_avgdl_vec, dl_vec));
        const __m256d c_tilde_vec = _mm256_div_pd(tf_vec, norm_vec);
        const __m256d numerator_vec = _mm256_mul_pd(k1_plus_one_vec, c_tilde_vec);
        const __m256d denominator_vec = _mm256_add_pd(k1_vec, c_tilde_vec);
        const __m256d tf_adj_vec =
            _mm256_add_pd(_mm256_div_pd(numerator_vec, denominator_vec), delta_vec);
        const __m256d contrib_vec = _mm256_mul_pd(term_weight_vec, tf_adj_vec);

        _mm256_storeu_pd(contrib, contrib_vec);
        scatter_add(doc_ids, contrib, scores);
    }

    score_bm25l_tail(postings, doc_len, index, avgdl, k1, b, delta, term_weight, scores);
}

} // namespace flashbm25
