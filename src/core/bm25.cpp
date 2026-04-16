#include "bm25.hpp"

namespace flashbm25 {

BM25::BM25(const std::vector<std::string>& corpus,
           float k1_, float b_, float epsilon_, bool lower)
    : k1(k1_), b(b_), epsilon(epsilon_), lowercase(lower),
      num_docs(corpus.size()), avgdl(0.0)
{
    _build_index(corpus);
    _build_idf();
}

void BM25::_build_index(const std::vector<std::string>& corpus) {
    tokenized_corpus.reserve(num_docs);
    doc_len.reserve(num_docs);
    double total_len = 0.0;

    for (std::size_t i = 0; i < num_docs; ++i) {
        auto tokens = tokenize(corpus[i], lowercase);
        double len  = static_cast<double>(tokens.size());
        doc_len.push_back(len);
        total_len += len;

        std::unordered_map<std::string, double> freq;
        for (const auto& t : tokens) freq[t] += 1.0;
        for (auto& [term, cnt] : freq) tf_index[term][i] = cnt;

        tokenized_corpus.push_back(std::move(tokens));
    }
    avgdl = (num_docs > 0) ? (total_len / static_cast<double>(num_docs)) : 0.0;
}

void BM25::_build_idf() {
    for (auto& [term, postings] : tf_index) {
        double df  = static_cast<double>(postings.size());
        double val = std::log((static_cast<double>(num_docs) - df + 0.5) / (df + 0.5) + 1.0);
        idf_cache[term] = std::max(static_cast<double>(epsilon), val);
    }
}

double BM25::_idf(const std::string& term) const {
    auto it = idf_cache.find(term);
    return (it == idf_cache.end()) ? 0.0 : it->second;
}

std::vector<double> BM25::get_scores(const std::string& query) const {
    std::vector<double> scores(num_docs, 0.0);
    for (const auto& term : tokenize(query, lowercase)) {
        double idf_val = _idf(term);
        if (idf_val == 0.0) continue;
        auto it = tf_index.find(term);
        if (it == tf_index.end()) continue;
        for (const auto& [doc_id, tf] : it->second) {
            double dl   = doc_len[doc_id];
            double norm = 1.0 - b + b * (dl / avgdl);
            scores[doc_id] += idf_val * (tf * (k1 + 1.0)) / (tf + k1 * norm);
        }
    }
    return scores;
}

std::vector<std::pair<double, std::size_t>>
BM25::get_top_n(const std::string& query, std::size_t n) const {
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

std::vector<std::string>
BM25::get_top_n_docs(const std::vector<std::string>& corpus,
                     const std::string& query, std::size_t n) const {
    auto top = get_top_n(query, n);
    std::vector<std::string> docs;
    docs.reserve(top.size());
    for (auto& [score, idx] : top)
        if (idx < corpus.size()) docs.push_back(corpus[idx]);
    return docs;
}

} // namespace flashbm25
