#include "simd_scoring.hpp"

#include <arm_neon.h>

namespace flashbm25 {
namespace {

constexpr std::size_t kNeonLanes = 2;

inline void load_posting_batch(const PostingList& postings,
                               const std::vector<double>& doc_len,
                               std::size_t offset,
                               std::size_t (&doc_ids)[kNeonLanes],
                               double (&tf_values)[kNeonLanes],
                               double (&doc_len_values)[kNeonLanes]) {
    for (std::size_t lane = 0; lane < kNeonLanes; ++lane) {
        const auto& posting = postings[offset + lane];
        doc_ids[lane] = posting.doc_id;
        tf_values[lane] = posting.tf;
        doc_len_values[lane] = doc_len[posting.doc_id];
    }
}

inline void scatter_add(const std::size_t (&doc_ids)[kNeonLanes],
                        const double (&contrib)[kNeonLanes],
                        std::vector<double>& scores) {
    for (std::size_t lane = 0; lane < kNeonLanes; ++lane) {
        scores[doc_ids[lane]] += contrib[lane];
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

void score_okapi_neon(const PostingList& postings,
                      const std::vector<double>& doc_len,
                      double avgdl,
                      double k1,
                      double b,
                      double term_weight,
                      std::vector<double>& scores) {
    const float64x2_t k1_vec = vdupq_n_f64(k1);
    const float64x2_t k1_plus_one_vec = vdupq_n_f64(k1 + 1.0);
    const float64x2_t one_minus_b_vec = vdupq_n_f64(1.0 - b);
    const float64x2_t b_over_avgdl_vec = vdupq_n_f64(b / avgdl);
    const float64x2_t term_weight_vec = vdupq_n_f64(term_weight);

    std::size_t index = 0;
    const std::size_t vectorized_end = postings.size() - (postings.size() % kNeonLanes);
    for (; index < vectorized_end; index += kNeonLanes) {
        std::size_t doc_ids[kNeonLanes];
        double tf_values[kNeonLanes];
        double doc_len_values[kNeonLanes];
        double contrib[kNeonLanes];
        load_posting_batch(postings, doc_len, index, doc_ids, tf_values, doc_len_values);

        const float64x2_t tf_vec = vld1q_f64(tf_values);
        const float64x2_t dl_vec = vld1q_f64(doc_len_values);
        const float64x2_t norm_vec =
            vaddq_f64(one_minus_b_vec, vmulq_f64(b_over_avgdl_vec, dl_vec));
        const float64x2_t numerator_vec = vmulq_f64(tf_vec, k1_plus_one_vec);
        const float64x2_t denominator_vec = vaddq_f64(tf_vec, vmulq_f64(k1_vec, norm_vec));
        const float64x2_t contrib_vec =
            vmulq_f64(term_weight_vec, vdivq_f64(numerator_vec, denominator_vec));

        vst1q_f64(contrib, contrib_vec);
        scatter_add(doc_ids, contrib, scores);
    }

    score_okapi_tail(postings, doc_len, index, avgdl, k1, b, term_weight, scores);
}

void score_bm25plus_neon(const PostingList& postings,
                         const std::vector<double>& doc_len,
                         double avgdl,
                         double k1,
                         double b,
                         double delta,
                         double term_weight,
                         std::vector<double>& scores) {
    const float64x2_t k1_vec = vdupq_n_f64(k1);
    const float64x2_t k1_plus_one_vec = vdupq_n_f64(k1 + 1.0);
    const float64x2_t one_minus_b_vec = vdupq_n_f64(1.0 - b);
    const float64x2_t b_over_avgdl_vec = vdupq_n_f64(b / avgdl);
    const float64x2_t delta_vec = vdupq_n_f64(delta);
    const float64x2_t term_weight_vec = vdupq_n_f64(term_weight);

    std::size_t index = 0;
    const std::size_t vectorized_end = postings.size() - (postings.size() % kNeonLanes);
    for (; index < vectorized_end; index += kNeonLanes) {
        std::size_t doc_ids[kNeonLanes];
        double tf_values[kNeonLanes];
        double doc_len_values[kNeonLanes];
        double contrib[kNeonLanes];
        load_posting_batch(postings, doc_len, index, doc_ids, tf_values, doc_len_values);

        const float64x2_t tf_vec = vld1q_f64(tf_values);
        const float64x2_t dl_vec = vld1q_f64(doc_len_values);
        const float64x2_t norm_vec =
            vaddq_f64(one_minus_b_vec, vmulq_f64(b_over_avgdl_vec, dl_vec));
        const float64x2_t numerator_vec = vmulq_f64(tf_vec, k1_plus_one_vec);
        const float64x2_t denominator_vec = vaddq_f64(tf_vec, vmulq_f64(k1_vec, norm_vec));
        const float64x2_t tf_adj_vec =
            vaddq_f64(vdivq_f64(numerator_vec, denominator_vec), delta_vec);
        const float64x2_t contrib_vec = vmulq_f64(term_weight_vec, tf_adj_vec);

        vst1q_f64(contrib, contrib_vec);
        scatter_add(doc_ids, contrib, scores);
    }

    score_bm25plus_tail(postings, doc_len, index, avgdl, k1, b, delta, term_weight, scores);
}

void score_bm25l_neon(const PostingList& postings,
                      const std::vector<double>& doc_len,
                      double avgdl,
                      double k1,
                      double b,
                      double delta,
                      double term_weight,
                      std::vector<double>& scores) {
    const float64x2_t k1_vec = vdupq_n_f64(k1);
    const float64x2_t k1_plus_one_vec = vdupq_n_f64(k1 + 1.0);
    const float64x2_t one_minus_b_vec = vdupq_n_f64(1.0 - b);
    const float64x2_t b_over_avgdl_vec = vdupq_n_f64(b / avgdl);
    const float64x2_t delta_vec = vdupq_n_f64(delta);
    const float64x2_t term_weight_vec = vdupq_n_f64(term_weight);

    std::size_t index = 0;
    const std::size_t vectorized_end = postings.size() - (postings.size() % kNeonLanes);
    for (; index < vectorized_end; index += kNeonLanes) {
        std::size_t doc_ids[kNeonLanes];
        double tf_values[kNeonLanes];
        double doc_len_values[kNeonLanes];
        double contrib[kNeonLanes];
        load_posting_batch(postings, doc_len, index, doc_ids, tf_values, doc_len_values);

        const float64x2_t tf_vec = vld1q_f64(tf_values);
        const float64x2_t dl_vec = vld1q_f64(doc_len_values);
        const float64x2_t norm_vec =
            vaddq_f64(one_minus_b_vec, vmulq_f64(b_over_avgdl_vec, dl_vec));
        const float64x2_t c_tilde_vec = vdivq_f64(tf_vec, norm_vec);
        const float64x2_t numerator_vec = vmulq_f64(k1_plus_one_vec, c_tilde_vec);
        const float64x2_t denominator_vec = vaddq_f64(k1_vec, c_tilde_vec);
        const float64x2_t tf_adj_vec =
            vaddq_f64(vdivq_f64(numerator_vec, denominator_vec), delta_vec);
        const float64x2_t contrib_vec = vmulq_f64(term_weight_vec, tf_adj_vec);

        vst1q_f64(contrib, contrib_vec);
        scatter_add(doc_ids, contrib, scores);
    }

    score_bm25l_tail(postings, doc_len, index, avgdl, k1, b, delta, term_weight, scores);
}

} // namespace flashbm25
