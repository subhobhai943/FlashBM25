#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <cctype>
#include <numeric>

namespace flashbm25 {

// ─── Tokenizer ───────────────────────────────────────────────────────────────
inline std::vector<std::string> tokenize(const std::string& text, bool lowercase = true) {
    std::vector<std::string> tokens;
    std::string token;
    for (char c : text) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_') {
            token += lowercase ? static_cast<char>(std::tolower(static_cast<unsigned char>(c))) : c;
        } else {
            if (!token.empty()) {
                tokens.push_back(std::move(token));
                token.clear();
            }
        }
    }
    if (!token.empty()) tokens.push_back(std::move(token));
    return tokens;
}

// ─── BM25 Core ───────────────────────────────────────────────────────────────
class BM25 {
public:
    // Parameters
    float k1;
    float b;
    float epsilon;
    bool  lowercase;

    // Corpus stats
    std::size_t num_docs;
    double      avgdl;

    // Per-doc data
    std::vector<std::vector<std::string>>              tokenized_corpus;
    std::vector<double>                                doc_len;

    // Inverted index: term -> {docid -> tf}
    std::unordered_map<std::string, std::unordered_map<std::size_t, double>> tf_index;

    // IDF cache
    std::unordered_map<std::string, double> idf_cache;

    // ── Constructor ──────────────────────────────────────────────────────────
    explicit BM25(
        const std::vector<std::string>& corpus,
        float k1_      = 1.5f,
        float b_       = 0.75f,
        float epsilon_ = 0.25f,
        bool  lower    = true
    );

    // ── Query API ────────────────────────────────────────────────────────────
    /// Returns a score for each document in the corpus.
    std::vector<double> get_scores(const std::string& query) const;

    /// Returns the top-n (score, doc_index) pairs, sorted descending.
    std::vector<std::pair<double, std::size_t>> get_top_n(
        const std::string& query,
        std::size_t n = 5
    ) const;

    /// Returns the top-n original documents (sorted descending by score).
    std::vector<std::string> get_top_n_docs(
        const std::vector<std::string>& corpus,
        const std::string& query,
        std::size_t n = 5
    ) const;

    // ── Corpus info ──────────────────────────────────────────────────────────
    std::size_t corpus_size() const { return num_docs; }
    double      average_doc_length() const { return avgdl; }

private:
    double _idf(const std::string& term) const;
    void   _build_index(const std::vector<std::string>& corpus);
    void   _build_idf();
};

} // namespace flashbm25
