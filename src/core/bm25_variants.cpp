#include "bm25_variants.hpp"

namespace flashbm25 {
namespace {

std::unordered_map<std::string, double>
build_query_term_weights(const std::vector<std::string>& tokens,
                         const std::unordered_map<std::string, double>& idf_cache) {
    std::unordered_map<std::string, double> term_weights;
    term_weights.reserve(tokens.size());

    for (const auto& term : tokens) {
        const auto idf_it = idf_cache.find(term);
        if (idf_it == idf_cache.end() || idf_it->second == 0.0) {
            continue;
        }
        term_weights[term] += idf_it->second;
    }

    return term_weights;
}

} // namespace

BM25Plus::BM25Plus(const std::vector<std::string>& corpus,
                   float k1_,
                   float b_,
                   float delta_,
                   float eps_,
                   bool lower)
    : k1(k1_), b(b_), delta(delta_), epsilon(eps_), lowercase(lower),
      num_docs(corpus.size()), avgdl(0.0) {
    _build_index(corpus);
    _build_idf();
    _build_postings_index();
}

void BM25Plus::_build_index(const std::vector<std::string>& corpus) {
    doc_len.reserve(num_docs);
    double total = 0.0;

    for (std::size_t i = 0; i < num_docs; ++i) {
        auto tokens = tokenize(corpus[i], lowercase);
        const double len = static_cast<double>(tokens.size());
        doc_len.push_back(len);
        total += len;

        std::unordered_map<std::string, double> freq;
        for (const auto& token : tokens) {
            freq[token] += 1.0;
        }
        for (auto& [term, count] : freq) {
            tf_index[term][i] = count;
        }
    }

    avgdl = (num_docs > 0) ? total / static_cast<double>(num_docs) : 0.0;
}

void BM25Plus::_build_idf() {
    for (auto& [term, postings] : tf_index) {
        const double df = static_cast<double>(postings.size());
        const double value =
            std::log((static_cast<double>(num_docs) - df + 0.5) / (df + 0.5) + 1.0);
        idf_cache[term] = std::max(static_cast<double>(epsilon), value);
    }
}

void BM25Plus::_build_postings_index() {
    rebuild_postings_index(tf_index, postings_index);
}

double BM25Plus::_idf(const std::string& term) const {
    auto it = idf_cache.find(term);
    return (it == idf_cache.end()) ? 0.0 : it->second;
}

std::vector<double> BM25Plus::get_scores(const std::string& query) const {
    std::vector<double> scores(num_docs, 0.0);

    const auto term_weights = build_query_term_weights(tokenize(query, lowercase), idf_cache);
    for (const auto& [term, term_weight] : term_weights) {
        auto it = postings_index.find(term);
        if (it == postings_index.end()) {
            continue;
        }
        score_bm25plus_postings(it->second, doc_len, avgdl, k1, b, delta, term_weight, scores);
    }
    return scores;
}

std::vector<std::pair<double, std::size_t>> BM25Plus::get_top_n(const std::string& query,
                                                                std::size_t n) const {
    auto scores = get_scores(query);
    std::vector<std::size_t> idx(num_docs);
    std::iota(idx.begin(), idx.end(), 0);

    n = std::min(n, num_docs);
    std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(n), idx.end(),
                      [&scores](std::size_t lhs, std::size_t rhs) {
                          return scores[lhs] > scores[rhs];
                      });

    std::vector<std::pair<double, std::size_t>> result;
    result.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        result.emplace_back(scores[idx[i]], idx[i]);
    }
    return result;
}

BM25L::BM25L(const std::vector<std::string>& corpus,
             float k1_,
             float b_,
             float delta_,
             float eps_,
             bool lower)
    : k1(k1_), b(b_), delta(delta_), epsilon(eps_), lowercase(lower),
      num_docs(corpus.size()), avgdl(0.0) {
    _build_index(corpus);
    _build_idf();
    _build_postings_index();
}

void BM25L::_build_index(const std::vector<std::string>& corpus) {
    doc_len.reserve(num_docs);
    double total = 0.0;

    for (std::size_t i = 0; i < num_docs; ++i) {
        auto tokens = tokenize(corpus[i], lowercase);
        const double len = static_cast<double>(tokens.size());
        doc_len.push_back(len);
        total += len;

        std::unordered_map<std::string, double> freq;
        for (const auto& token : tokens) {
            freq[token] += 1.0;
        }
        for (auto& [term, count] : freq) {
            tf_index[term][i] = count;
        }
    }

    avgdl = (num_docs > 0) ? total / static_cast<double>(num_docs) : 0.0;
}

void BM25L::_build_idf() {
    for (auto& [term, postings] : tf_index) {
        const double df = static_cast<double>(postings.size());
        const double value =
            std::log((static_cast<double>(num_docs) - df + 0.5) / (df + 0.5) + 1.0);
        idf_cache[term] = std::max(static_cast<double>(epsilon), value);
    }
}

void BM25L::_build_postings_index() {
    rebuild_postings_index(tf_index, postings_index);
}

double BM25L::_idf(const std::string& term) const {
    auto it = idf_cache.find(term);
    return (it == idf_cache.end()) ? 0.0 : it->second;
}

std::vector<double> BM25L::get_scores(const std::string& query) const {
    std::vector<double> scores(num_docs, 0.0);

    const auto term_weights = build_query_term_weights(tokenize(query, lowercase), idf_cache);
    for (const auto& [term, term_weight] : term_weights) {
        auto it = postings_index.find(term);
        if (it == postings_index.end()) {
            continue;
        }
        score_bm25l_postings(it->second, doc_len, avgdl, k1, b, delta, term_weight, scores);
    }
    return scores;
}

std::vector<std::pair<double, std::size_t>> BM25L::get_top_n(const std::string& query,
                                                             std::size_t n) const {
    auto scores = get_scores(query);
    std::vector<std::size_t> idx(num_docs);
    std::iota(idx.begin(), idx.end(), 0);

    n = std::min(n, num_docs);
    std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(n), idx.end(),
                      [&scores](std::size_t lhs, std::size_t rhs) {
                          return scores[lhs] > scores[rhs];
                      });

    std::vector<std::pair<double, std::size_t>> result;
    result.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        result.emplace_back(scores[idx[i]], idx[i]);
    }
    return result;
}

BM25Adpt::BM25Adpt(const std::vector<std::string>& corpus,
                   float k1_,
                   float b_,
                   float eps_,
                   bool lower)
    : k1(k1_), b(b_), epsilon(eps_), lowercase(lower), num_docs(corpus.size()),
      avgdl(0.0) {
    _build_index(corpus);
    _build_idf();
    _build_postings_index();
    _build_adaptive_k1();
}

void BM25Adpt::_build_index(const std::vector<std::string>& corpus) {
    doc_len.reserve(num_docs);
    double total = 0.0;

    for (std::size_t i = 0; i < num_docs; ++i) {
        auto tokens = tokenize(corpus[i], lowercase);
        const double len = static_cast<double>(tokens.size());
        doc_len.push_back(len);
        total += len;

        std::unordered_map<std::string, double> freq;
        for (const auto& token : tokens) {
            freq[token] += 1.0;
        }
        for (auto& [term, count] : freq) {
            tf_index[term][i] = count;
        }
    }

    avgdl = (num_docs > 0) ? total / static_cast<double>(num_docs) : 0.0;
}

void BM25Adpt::_build_idf() {
    for (auto& [term, postings] : tf_index) {
        const double df = static_cast<double>(postings.size());
        const double value =
            std::log((static_cast<double>(num_docs) - df + 0.5) / (df + 0.5) + 1.0);
        idf_cache[term] = std::max(static_cast<double>(epsilon), value);
    }
}

void BM25Adpt::_build_postings_index() {
    rebuild_postings_index(tf_index, postings_index);
}

void BM25Adpt::_build_adaptive_k1() {
    for (const auto& [term, postings] : tf_index) {
        double sum_tf = 0.0;
        for (const auto& posting : postings) {
            sum_tf += posting.second;
        }
        const double mean_tf = sum_tf / static_cast<double>(postings.size());
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

    const auto term_weights = build_query_term_weights(tokenize(query, lowercase), idf_cache);
    for (const auto& [term, term_weight] : term_weights) {
        auto it = postings_index.find(term);
        if (it == postings_index.end()) {
            continue;
        }
        score_okapi_postings(it->second, doc_len, avgdl, _adaptive_k1(term), b, term_weight,
                             scores);
    }
    return scores;
}

std::vector<std::pair<double, std::size_t>> BM25Adpt::get_top_n(const std::string& query,
                                                                std::size_t n) const {
    auto scores = get_scores(query);
    std::vector<std::size_t> idx(num_docs);
    std::iota(idx.begin(), idx.end(), 0);

    n = std::min(n, num_docs);
    std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(n), idx.end(),
                      [&scores](std::size_t lhs, std::size_t rhs) {
                          return scores[lhs] > scores[rhs];
                      });

    std::vector<std::pair<double, std::size_t>> result;
    result.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        result.emplace_back(scores[idx[i]], idx[i]);
    }
    return result;
}

} // namespace flashbm25
