#include "inverted_index.hpp"
#include <unordered_map>
#include <numeric>

#include <cstdint>   // uint16_t, uint32_t, uint64_t
#include <cstring>   // std::memcpy, std::memcmp
#include <stdexcept> // std::runtime_error

namespace flashbm25 {

void InvertedIndex::add_document(std::size_t doc_id,
                                  const std::vector<std::string>& tokens) {
    // Extend doc_lengths if needed
    if (doc_id >= doc_lengths.size())
        doc_lengths.resize(doc_id + 1, 0.0f);

    doc_lengths[doc_id] = static_cast<float>(tokens.size());
    num_docs = doc_lengths.size();

    std::unordered_map<std::string, float> freq;
    for (const auto& t : tokens) freq[t] += 1.0f;

    for (auto& [term, tf] : freq)
        index[term].push_back({doc_id, tf});
}

void InvertedIndex::finalize() {
    num_docs = doc_lengths.size();
    double total = 0.0;
    for (float l : doc_lengths) total += static_cast<double>(l);
    avgdl = (num_docs > 0) ? total / static_cast<double>(num_docs) : 0.0;
}

const std::vector<Posting>* InvertedIndex::lookup(const std::string& term) const {
    auto it = index.find(term);
    return (it == index.end()) ? nullptr : &it->second;
}

// ─── Write helpers (little-endian) ───────────────────────────────────────────────────────────────
void InvertedIndex::write_u16(std::ofstream& f, uint16_t v) {
    char buf[2] = { (char)(v & 0xFF), (char)((v >> 8) & 0xFF) };
    f.write(buf, 2);
}
void InvertedIndex::write_u32(std::ofstream& f, uint32_t v) {
    char buf[4];
    for (int i = 0; i < 4; ++i) buf[i] = (char)((v >> (8*i)) & 0xFF);
    f.write(buf, 4);
}
void InvertedIndex::write_u64(std::ofstream& f, uint64_t v) {
    char buf[8];
    for (int i = 0; i < 8; ++i) buf[i] = (char)((v >> (8*i)) & 0xFF);
    f.write(buf, 8);
}
void InvertedIndex::write_f32(std::ofstream& f, float v) {
    uint32_t bits; std::memcpy(&bits, &v, 4);
    write_u32(f, bits);
}
void InvertedIndex::write_f64(std::ofstream& f, double v) {
    uint64_t bits; std::memcpy(&bits, &v, 8);
    write_u64(f, bits);
}

// ─── Read helpers ───────────────────────────────────────────────────────────────────────────────
uint16_t InvertedIndex::read_u16(std::ifstream& f) {
    unsigned char b[2]; f.read((char*)b, 2);
    return (uint16_t)b[0] | ((uint16_t)b[1] << 8);
}
uint32_t InvertedIndex::read_u32(std::ifstream& f) {
    unsigned char b[4]; f.read((char*)b, 4);
    return (uint32_t)b[0] | ((uint32_t)b[1]<<8) | ((uint32_t)b[2]<<16) | ((uint32_t)b[3]<<24);
}
uint64_t InvertedIndex::read_u64(std::ifstream& f) {
    unsigned char b[8]; f.read((char*)b, 8);
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= ((uint64_t)b[i] << (8*i));
    return v;
}
float InvertedIndex::read_f32(std::ifstream& f) {
    uint32_t bits = read_u32(f); float v; std::memcpy(&v, &bits, 4); return v;
}
double InvertedIndex::read_f64(std::ifstream& f) {
    uint64_t bits = read_u64(f); double v; std::memcpy(&v, &bits, 8); return v;
}

// ─── Save ───────────────────────────────────────────────────────────────────────────────────────
void InvertedIndex::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("InvertedIndex::save — cannot open: " + path);

    f.write(MAGIC, 8);
    write_u32(f, VERSION);
    write_u64(f, static_cast<uint64_t>(num_docs));
    write_u64(f, static_cast<uint64_t>(index.size()));

    for (const auto& [term, postings] : index) {
        uint16_t tlen = static_cast<uint16_t>(term.size());
        write_u16(f, tlen);
        f.write(term.data(), tlen);
        write_u32(f, static_cast<uint32_t>(postings.size()));
        for (const auto& p : postings) {
            write_u64(f, static_cast<uint64_t>(p.doc_id));
            write_f32(f, p.tf);
        }
    }

    for (float dl : doc_lengths) write_f32(f, dl);
    write_f64(f, avgdl);
}

// ─── Load ───────────────────────────────────────────────────────────────────────────────────────
void InvertedIndex::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("InvertedIndex::load — cannot open: " + path);

    char magic[8]; f.read(magic, 8);
    if (std::memcmp(magic, MAGIC, 8) != 0)
        throw std::runtime_error("InvertedIndex::load — invalid magic bytes");

    uint32_t ver = read_u32(f);
    if (ver != VERSION)
        throw std::runtime_error("InvertedIndex::load — unsupported version");

    num_docs = static_cast<std::size_t>(read_u64(f));
    uint64_t num_terms = read_u64(f);

    index.clear();
    index.reserve(num_terms);

    for (uint64_t t = 0; t < num_terms; ++t) {
        uint16_t tlen = read_u16(f);
        std::string term(tlen, '\0');
        f.read(&term[0], tlen);

        uint32_t pcount = read_u32(f);
        std::vector<Posting> postings(pcount);
        for (uint32_t p = 0; p < pcount; ++p) {
            postings[p].doc_id = static_cast<std::size_t>(read_u64(f));
            postings[p].tf     = read_f32(f);
        }
        index.emplace(std::move(term), std::move(postings));
    }

    doc_lengths.resize(num_docs);
    for (std::size_t i = 0; i < num_docs; ++i) doc_lengths[i] = read_f32(f);

    avgdl = read_f64(f);
}

} // namespace flashbm25
