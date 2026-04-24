#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iosfwd>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "simd_scoring.hpp"

namespace flashbm25 {

// Simple inline tokenizer used by BM25 when no external tokenizer is provided.
inline std::vector<std::string> tokenize(const std::string& text, bool lowercase = true) {
    std::vector<std::string> tokens;
    std::string token;
    for (char c : text) {
        const auto ch = static_cast<unsigned char>(c);
        if (std::isalnum(ch) || c == '_') {
            token += lowercase ? static_cast<char>(std::tolower(ch)) : c;
        } else if (!token.empty()) {
            tokens.push_back(std::move(token));
            token.clear();
        }
    }
    if (!token.empty()) {
        tokens.push_back(std::move(token));
    }
    return tokens;
}

class BM25 {
public:
    float k1;
    float b;
    float epsilon;
    bool lowercase;

    std::size_t num_docs;
    double avgdl;

    std::vector<std::vector<std::string>> tokenized_corpus;
    std::vector<double> doc_len;
    TfIndex tf_index;
    PostingsIndex postings_index;
    std::unordered_map<std::string, double> idf_cache;

    explicit BM25(const std::vector<std::string>& corpus,
                  float k1_ = 1.5f,
                  float b_ = 0.75f,
                  float epsilon_ = 0.25f,
                  bool lower = true);

    std::vector<double> get_scores(const std::string& query) const;
    std::vector<std::pair<double, std::size_t>> get_top_n(const std::string& query,
                                                          std::size_t n = 5) const;
    std::vector<std::string> get_top_n_docs(const std::vector<std::string>& corpus,
                                            const std::string& query,
                                            std::size_t n = 5) const;
    void add_documents(const std::vector<std::string>& documents);
    void save(const std::string& path) const;
    std::string dumps() const;

    static BM25 load(const std::string& path);
    static BM25 loads(const std::string& data);

    std::size_t corpus_size() const {
        return num_docs;
    }

    double average_doc_length() const {
        return avgdl;
    }

private:
    double _idf(const std::string& term) const;
    void _build_index(const std::vector<std::string>& corpus);
    void _build_idf();
    void _build_postings_index();
    void _write_serialized(std::ostream& out) const;
    static BM25 _read_serialized(std::istream& in);
};

} // namespace flashbm25
