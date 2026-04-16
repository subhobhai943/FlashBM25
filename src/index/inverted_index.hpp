#pragma once
/*
 * InvertedIndex — a standalone inverted index with on-disk persistence.
 *
 * Layout (binary format, little-endian):
 *   [magic: 8 bytes "FBIDX\0\0\0"]
 *   [version: uint32]
 *   [num_docs: uint64]
 *   [num_terms: uint64]
 *   for each term:
 *     [term_len: uint16][term bytes]
 *     [postings_count: uint32]
 *     for each posting: [doc_id: uint64][tf: float32]
 *   [doc_lengths: num_docs x float32]
 *   [avgdl: float64]
 */

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <stdexcept>
#include <cstring>

namespace flashbm25 {

struct Posting {
    std::size_t doc_id;
    float       tf;
};

class InvertedIndex {
public:
    std::unordered_map<std::string, std::vector<Posting>> index;
    std::vector<float>   doc_lengths;
    double               avgdl    = 0.0;
    std::size_t          num_docs = 0;

    InvertedIndex() = default;

    /* Add a single document (already tokenized). */
    void add_document(std::size_t doc_id, const std::vector<std::string>& tokens);

    /* Finalize avgdl — call once after all documents are added. */
    void finalize();

    /* Lookup: returns nullptr if term not found. */
    const std::vector<Posting>* lookup(const std::string& term) const;

    /* ── Serialization ──────────────────────────────────────────────── */
    void save(const std::string& path) const;
    void load(const std::string& path);

private:
    static constexpr uint32_t VERSION = 1;
    static constexpr char     MAGIC[] = "FBIDX\0\0\0";

    static void write_u16(std::ofstream& f, uint16_t v);
    static void write_u32(std::ofstream& f, uint32_t v);
    static void write_u64(std::ofstream& f, uint64_t v);
    static void write_f32(std::ofstream& f, float v);
    static void write_f64(std::ofstream& f, double v);

    static uint16_t read_u16(std::ifstream& f);
    static uint32_t read_u32(std::ifstream& f);
    static uint64_t read_u64(std::ifstream& f);
    static float    read_f32(std::ifstream& f);
    static double   read_f64(std::ifstream& f);
};

} // namespace flashbm25
