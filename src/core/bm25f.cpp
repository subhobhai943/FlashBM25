#include "bm25f.hpp"

namespace flashbm25 {

// ═══════════════════════════════════════════════════════════════════════════════
// BM25F — Field-weighted BM25
// ═══════════════════════════════════════════════════════════════════════════════

BM25F::BM25F(const std::vector<FieldDoc>& corpus,
             const std::unordered_map<std::string, double>& field_weights,
             float k1_, float epsilon_, bool lower)
    : k1(k1_), epsilon(epsilon_), lowercase(lower),
      num_docs(corpus.size())
{
    // Initialize field params from the weight map
    for (const auto& [field, weight] : field_weights) {
        field_params[field] = FieldParams{weight, 0.75};
    }

    _build_index(corpus);
    _build_idf();
}

void BM25F::_build_index(const std::vector<FieldDoc>& corpus) {
    // First pass: discover all fields and compute per-field doc lengths
    for (const auto& [field, _] : field_params) {
        field_doc_len[field].resize(num_docs, 0.0);
        field_avgdl[field] = 0.0;
    }

    for (std::size_t i = 0; i < num_docs; ++i) {
        const auto& doc = corpus[i];

        for (const auto& [field, params] : field_params) {
            auto it = doc.find(field);
            if (it == doc.end()) continue;

            auto tokens = tokenize(it->second, lowercase);
            double len  = static_cast<double>(tokens.size());
            field_doc_len[field][i] = len;
            field_avgdl[field] += len;

            // Compute raw TF per field
            std::unordered_map<std::string, double> freq;
            for (const auto& t : tokens) freq[t] += 1.0;

            // Accumulate weighted TF across fields
            double w = params.weight;
            double b_f = params.b;
            // Note: we normalize per-field TF later in get_scores for full BM25F.
            // For the skeleton, we store raw weighted TF.
            for (auto& [term, cnt] : freq) {
                tf_index[term][i] += w * cnt;
            }
        }
    }

    // Compute per-field avgdl
    for (auto& [field, total] : field_avgdl) {
        total = (num_docs > 0) ? total / static_cast<double>(num_docs) : 0.0;
    }
}

void BM25F::_build_idf() {
    for (auto& [term, postings] : tf_index) {
        double df  = static_cast<double>(postings.size());
        double val = std::log((static_cast<double>(num_docs) - df + 0.5) /
                              (df + 0.5) + 1.0);
        idf_cache[term] = std::max(static_cast<double>(epsilon), val);
    }
}

double BM25F::_idf(const std::string& term) const {
    auto it = idf_cache.find(term);
    return (it == idf_cache.end()) ? 0.0 : it->second;
}

void BM25F::set_field_b(const std::string& field, double b_val) {
    auto it = field_params.find(field);
    if (it != field_params.end()) {
        it->second.b = b_val;
    }
}

std::vector<double> BM25F::get_scores(const std::string& query) const {
    std::vector<double> scores(num_docs, 0.0);
    for (const auto& term : tokenize(query, lowercase)) {
        double idf_val = _idf(term);
        if (idf_val == 0.0) continue;
        auto it = tf_index.find(term);
        if (it == tf_index.end()) continue;
        for (const auto& [doc_id, weighted_tf] : it->second) {
            // BM25F saturation using combined weighted TF
            double tf_adj = (weighted_tf * (k1 + 1.0)) / (weighted_tf + k1);
            scores[doc_id] += idf_val * tf_adj;
        }
    }
    return scores;
}

std::vector<std::pair<double, std::size_t>>
BM25F::get_top_n(const std::string& query, std::size_t n) const {
    auto scores = get_scores(query);
    std::vector<std::size_t> idx(num_docs);
    std::iota(idx.begin(), idx.end(), 0);
    n = std::min(n, num_docs);
    std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(n), idx.end(),
        [&scores](std::size_t a, std::size_t b_) { return scores[a] > scores[b_]; });
    std::vector<std::pair<double, std::size_t>> result;
    result.reserve(n);
    for (std::size_t i = 0; i < n; ++i) result.emplace_back(scores[idx[i]], idx[i]);
    return result;
}

} // namespace flashbm25
