#include "bm25.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {

constexpr std::uint32_t BM25_SERIALIZATION_VERSION = 1;
constexpr char BM25_SERIALIZATION_MAGIC[] = "FBM25\0\0\0";

void write_u32(std::ostream& out, std::uint32_t value) {
    char buffer[4];
    for (int i = 0; i < 4; ++i) {
        buffer[i] = static_cast<char>((value >> (8 * i)) & 0xFF);
    }
    out.write(buffer, 4);
}

void write_u64(std::ostream& out, std::uint64_t value) {
    char buffer[8];
    for (int i = 0; i < 8; ++i) {
        buffer[i] = static_cast<char>((value >> (8 * i)) & 0xFF);
    }
    out.write(buffer, 8);
}

void write_f32(std::ostream& out, float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    write_u32(out, bits);
}

void write_f64(std::ostream& out, double value) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    write_u64(out, bits);
}

std::uint32_t read_u32(std::istream& in) {
    unsigned char buffer[4];
    in.read(reinterpret_cast<char*>(buffer), 4);
    if (!in) {
        throw std::runtime_error("BM25::load failed while reading uint32.");
    }
    return static_cast<std::uint32_t>(buffer[0]) |
           (static_cast<std::uint32_t>(buffer[1]) << 8) |
           (static_cast<std::uint32_t>(buffer[2]) << 16) |
           (static_cast<std::uint32_t>(buffer[3]) << 24);
}

std::uint64_t read_u64(std::istream& in) {
    unsigned char buffer[8];
    in.read(reinterpret_cast<char*>(buffer), 8);
    if (!in) {
        throw std::runtime_error("BM25::load failed while reading uint64.");
    }

    std::uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<std::uint64_t>(buffer[i]) << (8 * i);
    }
    return value;
}

float read_f32(std::istream& in) {
    const auto bits = read_u32(in);
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

double read_f64(std::istream& in) {
    const auto bits = read_u64(in);
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

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

namespace flashbm25 {

BM25::BM25(const std::vector<std::string>& corpus, float k1_, float b_, float epsilon_,
           bool lower)
    : k1(k1_), b(b_), epsilon(epsilon_), lowercase(lower), num_docs(corpus.size()),
      avgdl(0.0) {
    _build_index(corpus);
    _build_idf();
    _build_postings_index();
}

void BM25::_build_index(const std::vector<std::string>& corpus) {
    tokenized_corpus.reserve(num_docs);
    doc_len.reserve(num_docs);
    double total_len = 0.0;

    for (std::size_t i = 0; i < num_docs; ++i) {
        auto tokens = tokenize(corpus[i], lowercase);
        const double len = static_cast<double>(tokens.size());
        doc_len.push_back(len);
        total_len += len;

        std::unordered_map<std::string, double> freq;
        for (const auto& token : tokens) {
            freq[token] += 1.0;
        }
        for (auto& [term, count] : freq) {
            tf_index[term][i] = count;
        }

        tokenized_corpus.push_back(std::move(tokens));
    }

    avgdl = (num_docs > 0) ? (total_len / static_cast<double>(num_docs)) : 0.0;
}

void BM25::_build_idf() {
    idf_cache.clear();
    for (auto& [term, postings] : tf_index) {
        const double df = static_cast<double>(postings.size());
        const double value =
            std::log((static_cast<double>(num_docs) - df + 0.5) / (df + 0.5) + 1.0);
        idf_cache[term] = std::max(static_cast<double>(epsilon), value);
    }
}

void BM25::_build_postings_index() {
    rebuild_postings_index(tf_index, postings_index);
}

double BM25::_idf(const std::string& term) const {
    auto it = idf_cache.find(term);
    return (it == idf_cache.end()) ? 0.0 : it->second;
}

std::vector<double> BM25::get_scores(const std::string& query) const {
    std::vector<double> scores(num_docs, 0.0);

    const auto term_weights = build_query_term_weights(tokenize(query, lowercase), idf_cache);
    for (const auto& [term, term_weight] : term_weights) {
        auto it = postings_index.find(term);
        if (it == postings_index.end()) {
            continue;
        }
        score_okapi_postings(it->second, doc_len, avgdl, k1, b, term_weight, scores);
    }
    return scores;
}

std::vector<std::pair<double, std::size_t>> BM25::get_top_n(const std::string& query,
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

std::vector<std::string> BM25::get_top_n_docs(const std::vector<std::string>& corpus,
                                              const std::string& query,
                                              std::size_t n) const {
    auto top = get_top_n(query, n);
    std::vector<std::string> docs;
    docs.reserve(top.size());
    for (auto& [score, idx] : top) {
        if (idx < corpus.size()) {
            docs.push_back(corpus[idx]);
        }
    }
    return docs;
}

void BM25::add_documents(const std::vector<std::string>& documents) {
    if (documents.empty()) {
        return;
    }

    tokenized_corpus.reserve(tokenized_corpus.size() + documents.size());
    doc_len.reserve(doc_len.size() + documents.size());

    double total_len = avgdl * static_cast<double>(num_docs);
    for (const auto& document : documents) {
        auto tokens = tokenize(document, lowercase);
        const std::size_t doc_id = doc_len.size();
        const double length = static_cast<double>(tokens.size());

        doc_len.push_back(length);
        total_len += length;

        std::unordered_map<std::string, double> freq;
        for (const auto& term : tokens) {
            freq[term] += 1.0;
        }
        for (const auto& [term, count] : freq) {
            tf_index[term][doc_id] = count;
        }

        tokenized_corpus.push_back(std::move(tokens));
    }

    num_docs = doc_len.size();
    avgdl = (num_docs > 0) ? (total_len / static_cast<double>(num_docs)) : 0.0;
    _build_idf();
    _build_postings_index();
}

void BM25::_write_serialized(std::ostream& out) const {
    out.write(BM25_SERIALIZATION_MAGIC, 8);
    if (!out) {
        throw std::runtime_error("BM25::save failed while writing file header.");
    }

    write_u32(out, BM25_SERIALIZATION_VERSION);
    write_f32(out, k1);
    write_f32(out, b);
    write_f32(out, epsilon);
    write_u32(out, lowercase ? 1u : 0u);
    write_u64(out, static_cast<std::uint64_t>(num_docs));
    write_u64(out, static_cast<std::uint64_t>(tf_index.size()));

    for (const auto& [term, postings] : tf_index) {
        write_u32(out, static_cast<std::uint32_t>(term.size()));
        out.write(term.data(), static_cast<std::streamsize>(term.size()));
        write_u64(out, static_cast<std::uint64_t>(postings.size()));
        for (const auto& [doc_id, tf] : postings) {
            write_u64(out, static_cast<std::uint64_t>(doc_id));
            write_f64(out, tf);
        }
    }

    write_u64(out, static_cast<std::uint64_t>(doc_len.size()));
    for (double length : doc_len) {
        write_f64(out, length);
    }
    write_f64(out, avgdl);

    if (!out) {
        throw std::runtime_error("BM25::save failed while writing serialized state.");
    }
}

BM25 BM25::_read_serialized(std::istream& in) {
    char magic[8];
    in.read(magic, 8);
    if (!in) {
        throw std::runtime_error("BM25::load failed while reading file header.");
    }
    if (std::memcmp(magic, BM25_SERIALIZATION_MAGIC, 8) != 0) {
        throw std::runtime_error("BM25::load received invalid magic bytes.");
    }

    const auto version = read_u32(in);
    if (version != BM25_SERIALIZATION_VERSION) {
        throw std::runtime_error("BM25::load received an unsupported serialization version.");
    }

    const auto loaded_k1 = read_f32(in);
    const auto loaded_b = read_f32(in);
    const auto loaded_epsilon = read_f32(in);
    const auto loaded_lowercase = read_u32(in) != 0;

    BM25 bm25(std::vector<std::string>{}, loaded_k1, loaded_b, loaded_epsilon,
              loaded_lowercase);
    bm25.num_docs = static_cast<std::size_t>(read_u64(in));
    const auto num_terms = read_u64(in);
    bm25.tf_index.clear();
    bm25.tf_index.reserve(static_cast<std::size_t>(num_terms));
    bm25.idf_cache.clear();
    bm25.tokenized_corpus.clear();

    for (std::uint64_t term_index = 0; term_index < num_terms; ++term_index) {
        const auto term_length = read_u32(in);
        std::string term(term_length, '\0');
        if (term_length > 0) {
            in.read(&term[0], static_cast<std::streamsize>(term_length));
            if (!in) {
                throw std::runtime_error("BM25::load failed while reading a term.");
            }
        }

        const auto postings_count = read_u64(in);
        auto& postings = bm25.tf_index[term];
        postings.reserve(static_cast<std::size_t>(postings_count));
        for (std::uint64_t posting_index = 0; posting_index < postings_count; ++posting_index) {
            postings.emplace(static_cast<std::size_t>(read_u64(in)), read_f64(in));
        }
    }

    const auto doc_count = read_u64(in);
    bm25.doc_len.resize(static_cast<std::size_t>(doc_count));
    for (std::size_t index = 0; index < bm25.doc_len.size(); ++index) {
        bm25.doc_len[index] = read_f64(in);
    }
    bm25.avgdl = read_f64(in);

    if (bm25.doc_len.size() != bm25.num_docs) {
        throw std::runtime_error("BM25::load found inconsistent document metadata.");
    }

    bm25._build_idf();
    bm25._build_postings_index();
    return bm25;
}

std::string BM25::dumps() const {
    std::ostringstream out(std::ios::binary | std::ios::out);
    _write_serialized(out);
    return out.str();
}

void BM25::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("BM25::save could not open path: " + path);
    }
    _write_serialized(out);
}

BM25 BM25::loads(const std::string& data) {
    std::istringstream in(data, std::ios::binary | std::ios::in);
    return _read_serialized(in);
}

BM25 BM25::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("BM25::load could not open path: " + path);
    }
    return _read_serialized(in);
}

} // namespace flashbm25
