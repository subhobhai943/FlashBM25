#include "bm25_variants.hpp"

namespace flashbm25 {

// ═══════════════════════════════════════════════════════════════════════════════
// BM25+
// ═══════════════════════════════════════════════════════════════════════════════

BM25Plus::BM25Plus(const std::vector<std::string>& corpus,
                   float k1_, float b_, float delta_, float eps_, bool lower)
    : k1(k1_), b(b_), delta(delta_), epsilon(eps_), lowercase(lower),
      num_docs(corpus.size()), avgdl(0.0)
{
    _build_index(corpus);
    _build_idf();
}

void BM25Plus::_build_index(const std::vector<std::string>& corpus) {
    doc_len.reserve(num_docs);
    double total = 0.0;
    for (std::size_t i = 0; i < num_docs; ++i) {
        auto tokens = tokenize(corpus[i], lowercase);
        double len  = static_cast<double>(tokens.size());
        doc_len.push_back(len);
        total += len;
        std::unordered_map<std::string, double> freq;
        for (const auto& t : tokens) freq[t] += 1.0;
        for (auto& [term, cnt] : freq) tf_index[term][i] = cnt;
    }
    avgdl = (num_docs > 0) ? total / static_cast<double>(num_docs) : 0.0;
}

void BM25Plus::_build_idf() {
    for (auto& [term, postings] : tf_index) {
        double df  = static_cast<double>(postings.size());
        double val = std::log((static_cast<double>(num_docs) - df + 0.5) /
                              (df + 0.5) + 1.0);
        idf_cache[term] = std::max(static_cast<double>(epsilon), val);
    }
}

double BM25Plus::_idf(const std::string& term) const {
    auto it = idf_cache.find(term);
    return (it == idf_cache.end()) ? 0.0 : it->second;
}

std::vector<double> BM25Plus::get_scores(const std::string& query) const {
    std::vector<double> scores(num_docs, 0.0);
    for (const auto& term : tokenize(query, lowercase)) {
        double idf_val = _idf(term);
        if (idf_val == 0.0) continue;
        auto it = tf_index.find(term);
        if (it == tf_index.end()) continue;
        for (const auto& [doc_id, tf] : it->second) {
            double dl    = doc_len[doc_id];
            double norm  = 1.0 - b + b * (dl / avgdl);
            // BM25+ adds delta to the TF component
            double tf_bm25plus = (tf * (k1 + 1.0)) / (tf + k1 * norm) + delta;
            scores[doc_id] += idf_val * tf_bm25plus;
        }
    }
    return scores;
}

std::vector<std::pair<double, std::size_t>>
BM25Plus::get_top_n(const std::string& query, std::size_t n) const {
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

// ═══════════════════════════════════════════════════════════════════════════════
// BM25L
// ═══════════════════════════════════════════════════════════════════════════════

BM25L::BM25L(const std::vector<std::string>& corpus,
             float k1_, float b_, float delta_, float eps_, bool lower)
    : k1(k1_), b(b_), delta(delta_), epsilon(eps_), lowercase(lower),
      num_docs(corpus.size()), avgdl(0.0)
{
    _build_index(corpus);
    _build_idf();
}

void BM25L::_build_index(const std::vector<std::string>& corpus) {
    doc_len.reserve(num_docs);
    double total = 0.0;
    for (std::size_t i = 0; i < num_docs; ++i) {
        auto tokens = tokenize(corpus[i], lowercase);
        double len  = static_cast<double>(tokens.size());
        doc_len.push_back(len);
        total += len;
        std::unordered_map<std::string, double> freq;
        for (const auto& t : tokens) freq[t] += 1.0;
        for (auto& [term, cnt] : freq) tf_index[term][i] = cnt;
    }
    avgdl = (num_docs > 0) ? total / static_cast<double>(num_docs) : 0.0;
}

void BM25L::_build_idf() {
    for (auto& [term, postings] : tf_index) {
        double df  = static_cast<double>(postings.size());
        double val = std::log((static_cast<double>(num_docs) - df + 0.5) /
                              (df + 0.5) + 1.0);
        idf_cache[term] = std::max(static_cast<double>(epsilon), val);
    }
}

double BM25L::_idf(const std::string& term) const {
    auto it = idf_cache.find(term);
    return (it == idf_cache.end()) ? 0.0 : it->second;
}

std::vector<double> BM25L::get_scores(const std::string& query) const {
    std::vector<double> scores(num_docs, 0.0);
    for (const auto& term : tokenize(query, lowercase)) {
        double idf_val = _idf(term);
        if (idf_val == 0.0) continue;
        auto it = tf_index.find(term);
        if (it == tf_index.end()) continue;
        for (const auto& [doc_id, tf] : it->second) {
            double dl      = doc_len[doc_id];
            double c_tilde = tf / (1.0 - b + b * (dl / avgdl)); // normalized TF
            // BM25L formula: ((k1+1)*c_tilde) / (k1 + c_tilde) + delta
            double tf_bm25l = ((k1 + 1.0) * c_tilde) / (k1 + c_tilde) + delta;
            scores[doc_id] += idf_val * tf_bm25l;
        }
    }
    return scores;
}

std::vector<std::pair<double, std::size_t>>
BM25L::get_top_n(const std::string& query, std::size_t n) const {
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

// ═══════════════════════════════════════════════════════════════════════════════
// BM25Adpt — Adaptive k1 per term
// ═══════════════════════════════════════════════════════════════════════════════

BM25Adpt::BM25Adpt(const std::vector<std::string>& corpus,
                   float k1_, float b_, float eps_, bool lower)
    : k1(k1_), b(b_), epsilon(eps_), lowercase(lower),
      num_docs(corpus.size()), avgdl(0.0)
{
    _build_index(corpus);
    _build_idf();
    _build_adaptive_k1();
}

void BM25Adpt::_build_index(const std::vector<std::string>& corpus) {
    doc_len.reserve(num_docs);
    double total = 0.0;
    for (std::size_t i = 0; i < num_docs; ++i) {
        auto tokens = tokenize(corpus[i], lowercase);
        double len  = static_cast<double>(tokens.size());
        doc_len.push_back(len);
        total += len;
        std::unordered_map<std::string, double> freq;
        for (const auto& t : tokens) freq[t] += 1.0;
        for (auto& [term, cnt] : freq) tf_index[term][i] = cnt;
    }
    avgdl = (num_docs > 0) ? total / static_cast<double>(num_docs) : 0.0;
}

void BM25Adpt::_build_idf() {
    for (auto& [term, postings] : tf_index) {
        double df  = static_cast<double>(postings.size());
        double val = std::log((static_cast<double>(num_docs) - df + 0.5) /
                              (df + 0.5) + 1.0);
        idf_cache[term] = std::max(static_cast<double>(epsilon), val);
    }
}

void BM25Adpt::_build_adaptive_k1() {
    // For each term, compute mean TF across documents that contain it,
    // then set k1_t = k1 * log(1 + mean_tf_t).
    for (const auto& [term, postings] : tf_index) {
        double sum_tf = 0.0;
        for (const auto& [doc_id, tf] : postings) {
            sum_tf += tf;
        }
        double mean_tf = sum_tf / static_cast<double>(postings.size());
        k1_cache[term] = k1 * std::log(1.0 + mean_tf);
    }
}

double BM25Adpt::_idf(const std::string& term) const {
    auto it = idf_cache.find(term);
    return (it == idf_cache.end()) ? 0.0 : it->second;
}

double BM25Adpt::_adaptive_k1(const std::string& term) const {
    auto it = k1_cache.find(term);
    return (it == k1_cache.end()) ? static_cast<double>(k1) : it->second;
}

std::vector<double> BM25Adpt::get_scores(const std::string& query) const {
    std::vector<double> scores(num_docs, 0.0);
    for (const auto& term : tokenize(query, lowercase)) {
        double idf_val = _idf(term);
        if (idf_val == 0.0) continue;
        auto it = tf_index.find(term);
        if (it == tf_index.end()) continue;
        double k1_t = _adaptive_k1(term);
        for (const auto& [doc_id, tf] : it->second) {
            double dl   = doc_len[doc_id];
            double norm = 1.0 - b + b * (dl / avgdl);
            double tf_adj = (tf * (k1_t + 1.0)) / (tf + k1_t * norm);
            scores[doc_id] += idf_val * tf_adj;
        }
    }
    return scores;
}

std::vector<std::pair<double, std::size_t>>
BM25Adpt::get_top_n(const std::string& query, std::size_t n) const {
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
