#pragma once
/*
 * compressed_index.hpp — Inverted index with delta-coded doc IDs
 *                        and VarInt-encoded term frequencies.
 *
 * Each posting list is stored as a contiguous byte blob:
 *   [count: VarInt][gap_0: VarInt][tf_bits_0: VarInt] ...
 *
 * This typically reduces posting-list memory by 50-70% compared to
 * storing (uint64, float32) pairs in a plain vector.
 *
 * Binary file format (version 2):
 *   [magic 8B][version u32][num_docs u64][num_terms u64]
 *   for each term (sorted):
 *     [term_len u16][term bytes]
 *     [blob_len u32][compressed blob bytes]
 *   [doc_lengths: num_docs × f32]
 *   [avgdl f64]
 */

#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <cstdint>

#include "postings_list.hpp"
#include "varint.hpp"

namespace flashbm25 {

struct CompressedPostingList {
    std::vector<uint8_t> data;  // delta+VarInt encoded blob
    std::size_t          count = 0;

    /* Decode to SortedPostingList on demand. */
    SortedPostingList decompress() const {
        std::size_t pos = 0;
        auto decoded = varint::decode_delta_list(data, pos);
        SortedPostingList result;
        result.reserve(decoded.size());
        for (auto& dp : decoded)
            result.push_back({dp.doc_id, dp.tf});
        return result;
    }
};

class CompressedIndex {
public:
    std::vector<std::string>          terms_;
    std::vector<CompressedPostingList> postings_;
    std::vector<float>                doc_lengths_;
    double                            avgdl_    = 0.0;
    std::size_t                       num_docs_ = 0;

    CompressedIndex() = default;

    /* Build from a SortedPostingsIndex. */
    explicit CompressedIndex(const SortedPostingsIndex& src,
                             const std::vector<float>& doc_lengths,
                             double avgdl,
                             std::size_t num_docs)
        : doc_lengths_(doc_lengths), avgdl_(avgdl), num_docs_(num_docs)
    {
        terms_    = src.terms();
        const auto& raw_postings = src.postings();
        postings_.resize(terms_.size());
        for (std::size_t i = 0; i < terms_.size(); ++i) {
            const auto& pl = raw_postings[i];
            std::vector<std::size_t> doc_ids;
            std::vector<float>       tfs;
            doc_ids.reserve(pl.size());
            tfs.reserve(pl.size());
            for (const auto& p : pl) {
                doc_ids.push_back(p.doc_id);
                tfs.push_back(p.tf);
            }
            varint::encode_delta_list(doc_ids, tfs, postings_[i].data);
            postings_[i].count = pl.size();
        }
    }

    /* O(log V) lookup — decompresses on the fly. */
    SortedPostingList lookup(const std::string& term) const {
        auto it = std::lower_bound(terms_.begin(), terms_.end(), term);
        if (it == terms_.end() || *it != term) return {};
        return postings_[static_cast<std::size_t>(it - terms_.begin())].decompress();
    }

    std::size_t num_terms()  const noexcept { return terms_.size(); }
    std::size_t num_docs()   const noexcept { return num_docs_; }
    double      avg_dl()     const noexcept { return avgdl_; }

    /* Bytes occupied by compressed posting lists (excluding term strings). */
    std::size_t compressed_bytes() const noexcept {
        std::size_t n = 0;
        for (const auto& cpl : postings_) n += cpl.data.size();
        return n;
    }

    /* ── Serialisation ──────────────────────────────────────────────── */
    void save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f) throw std::runtime_error("CompressedIndex::save — cannot open: " + path);
        _write(f);
    }

    void load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("CompressedIndex::load — cannot open: " + path);
        _read(f);
    }

private:
    static constexpr uint32_t VERSION = 2;
    static constexpr char     MAGIC[] = "FBCIDX\0\0";  // 8 bytes

    // ── little-endian I/O helpers ──────────────────────────────────────
    static void w_u16(std::ofstream& f, uint16_t v) {
        char b[2] = {(char)(v & 0xFF), (char)((v >> 8) & 0xFF)}; f.write(b, 2);
    }
    static void w_u32(std::ofstream& f, uint32_t v) {
        char b[4]; for(int i=0;i<4;++i) b[i]=(char)((v>>(8*i))&0xFF); f.write(b,4);
    }
    static void w_u64(std::ofstream& f, uint64_t v) {
        char b[8]; for(int i=0;i<8;++i) b[i]=(char)((v>>(8*i))&0xFF); f.write(b,8);
    }
    static void w_f32(std::ofstream& f, float v) {
        uint32_t bits; std::memcpy(&bits,&v,4); w_u32(f,bits);
    }
    static void w_f64(std::ofstream& f, double v) {
        uint64_t bits; std::memcpy(&bits,&v,8); w_u64(f,bits);
    }
    static uint16_t r_u16(std::ifstream& f) {
        unsigned char b[2]; f.read((char*)b,2);
        return (uint16_t)b[0]|((uint16_t)b[1]<<8);
    }
    static uint32_t r_u32(std::ifstream& f) {
        unsigned char b[4]; f.read((char*)b,4);
        return (uint32_t)b[0]|((uint32_t)b[1]<<8)|((uint32_t)b[2]<<16)|((uint32_t)b[3]<<24);
    }
    static uint64_t r_u64(std::ifstream& f) {
        unsigned char b[8]; f.read((char*)b,8);
        uint64_t v=0; for(int i=0;i<8;++i) v|=((uint64_t)b[i]<<(8*i)); return v;
    }
    static float  r_f32(std::ifstream& f) { uint32_t bits=r_u32(f); float  v; std::memcpy(&v,&bits,4); return v; }
    static double r_f64(std::ifstream& f) { uint64_t bits=r_u64(f); double v; std::memcpy(&v,&bits,8); return v; }

    void _write(std::ofstream& f) const {
        f.write(MAGIC, 8);
        w_u32(f, VERSION);
        w_u64(f, static_cast<uint64_t>(num_docs_));
        w_u64(f, static_cast<uint64_t>(terms_.size()));
        for (std::size_t i = 0; i < terms_.size(); ++i) {
            const auto& term = terms_[i];
            w_u16(f, static_cast<uint16_t>(term.size()));
            f.write(term.data(), static_cast<std::streamsize>(term.size()));
            const auto& blob = postings_[i].data;
            w_u32(f, static_cast<uint32_t>(blob.size()));
            f.write(reinterpret_cast<const char*>(blob.data()),
                    static_cast<std::streamsize>(blob.size()));
        }
        for (float dl : doc_lengths_) w_f32(f, dl);
        w_f64(f, avgdl_);
    }

    void _read(std::ifstream& f) {
        char magic[8]; f.read(magic, 8);
        if (std::memcmp(magic, MAGIC, 8) != 0)
            throw std::runtime_error("CompressedIndex::load — invalid magic");
        uint32_t ver = r_u32(f);
        if (ver != VERSION)
            throw std::runtime_error("CompressedIndex::load — unsupported version");
        num_docs_ = static_cast<std::size_t>(r_u64(f));
        uint64_t nt = r_u64(f);
        terms_.resize(static_cast<std::size_t>(nt));
        postings_.resize(static_cast<std::size_t>(nt));
        for (uint64_t i = 0; i < nt; ++i) {
            uint16_t tlen = r_u16(f);
            terms_[i].resize(tlen);
            f.read(&terms_[i][0], tlen);
            uint32_t blen = r_u32(f);
            postings_[i].data.resize(blen);
            f.read(reinterpret_cast<char*>(postings_[i].data.data()),
                   static_cast<std::streamsize>(blen));
        }
        doc_lengths_.resize(num_docs_);
        for (std::size_t i = 0; i < num_docs_; ++i) doc_lengths_[i] = r_f32(f);
        avgdl_ = r_f64(f);
    }
};

} // namespace flashbm25
