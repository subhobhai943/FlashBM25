#pragma once
/*
 * Reciprocal Rank Fusion (RRF) — Cormack et al. (2009)
 *
 * Fuses N ranked result lists (each a vector of doc_id ordered by score)
 * into a single ranked list without any score normalization.
 *
 * Formula:  RRF(d) = sum_i 1 / (k + rank_i(d))
 * where k = 60 (Cormack's constant) and rank_i is 1-based.
 *
 * Usage:
 *   RRFScorer rrf;
 *   rrf.add_ranking(bm25_results);       // vector<pair<double,size_t>>
 *   rrf.add_ranking(bm25plus_results);
 *   auto fused = rrf.fuse(top_n);
 */

#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cstddef>

namespace flashbm25 {

class RRFScorer {
public:
    explicit RRFScorer(double k = 60.0) : k_(k) {}

    /* Add one ranked list. Each pair is (score, doc_id); order matters. */
    void add_ranking(const std::vector<std::pair<double, std::size_t>>& ranked);

    /* Return the top-n fused results as (rrf_score, doc_id) pairs. */
    std::vector<std::pair<double, std::size_t>> fuse(std::size_t top_n = 10) const;

    /* Reset all stored rankings. */
    void reset();

    double k() const { return k_; }

private:
    double k_;
    std::unordered_map<std::size_t, double> rrf_scores_;
};

} // namespace flashbm25
