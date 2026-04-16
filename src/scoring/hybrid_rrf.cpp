#include "hybrid_rrf.hpp"

namespace flashbm25 {

void RRFScorer::add_ranking(
        const std::vector<std::pair<double, std::size_t>>& ranked) {
    for (std::size_t rank = 0; rank < ranked.size(); ++rank) {
        std::size_t doc_id = ranked[rank].second;
        // rank is 0-based, formula uses 1-based -> (k + rank + 1)
        rrf_scores_[doc_id] += 1.0 / (k_ + static_cast<double>(rank + 1));
    }
}

std::vector<std::pair<double, std::size_t>>
RRFScorer::fuse(std::size_t top_n) const {
    std::vector<std::pair<double, std::size_t>> results;
    results.reserve(rrf_scores_.size());
    for (const auto& [doc_id, score] : rrf_scores_)
        results.emplace_back(score, doc_id);

    top_n = std::min(top_n, results.size());
    std::partial_sort(results.begin(),
                      results.begin() + static_cast<std::ptrdiff_t>(top_n),
                      results.end(),
                      [](const auto& a, const auto& b){ return a.first > b.first; });
    results.resize(top_n);
    return results;
}

void RRFScorer::reset() {
    rrf_scores_.clear();
}

} // namespace flashbm25
