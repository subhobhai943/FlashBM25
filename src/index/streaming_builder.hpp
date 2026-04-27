#pragma once
/*
 * streaming_builder.hpp — On-disk streaming index builder.
 *
 * Designed for corpora that are too large to index in a single in-memory
 * pass.  Documents are ingested in configurable chunks; each chunk is
 * sorted and flushed to a temporary shard file.  When build() is called
 * the shards are merged into a single CompressedIndex file.
 *
 * Usage
 * ─────
 *   StreamingIndexBuilder builder("output.fbcidx", /*chunk_size=*\/ 50'000);
 *   for (auto& doc : huge_corpus)
 *       builder.add_document(doc_id++, tokenize(doc));
 *   builder.build();   // merge shards → final index
 *
 * Temporary shard files are written to `tmp_dir` (default: system temp)
 * and cleaned up automatically after a successful build.
 */

#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <filesystem>
#include <map>

#include "postings_list.hpp"
#include "varint.hpp"
#include "compressed_index.hpp"

namespace flashbm25 {

class StreamingIndexBuilder {
public:
    /*
     * @param output_path   Final .fbcidx file to write.
     * @param chunk_size    Number of documents per in-memory chunk.
     * @param tmp_dir       Directory for temporary shard files.
     *                      Defaults to output_path's parent directory.
     */
    explicit StreamingIndexBuilder(std::string output_path,
                                   std::size_t chunk_size = 100'000,
                                   std::string tmp_dir    = "")
        : output_path_(std::move(output_path))
        , chunk_size_(chunk_size)
        , tmp_dir_(tmp_dir.empty()
                       ? std::filesystem::path(output_path_).parent_path().string()
                       : std::move(tmp_dir))
    {}

    ~StreamingIndexBuilder() { cleanup_shards(); }

    /* Add one tokenised document.  doc_id must be monotonically increasing. */
    void add_document(std::size_t doc_id,
                      const std::vector<std::string>& tokens) {
        if (doc_id >= doc_lengths_.size())
            doc_lengths_.resize(doc_id + 1, 0.0f);
        doc_lengths_[doc_id] = static_cast<float>(tokens.size());
        ++num_docs_;

        std::map<std::string, float> freq;
        for (const auto& t : tokens) freq[t] += 1.0f;
        for (auto& [term, tf] : freq)
            current_chunk_[term].push_back({doc_id, tf});

        if (num_docs_ % chunk_size_ == 0) flush_chunk();
    }

    /*
     * Flush remaining documents, merge all shards, write the final
     * CompressedIndex to output_path_.  Returns the path written.
     */
    std::string build() {
        if (!current_chunk_.empty()) flush_chunk();
        merge_shards();
        return output_path_;
    }

    std::size_t num_docs()   const noexcept { return num_docs_; }
    std::size_t num_shards() const noexcept { return shard_paths_.size(); }

private:
    std::string  output_path_;
    std::size_t  chunk_size_;
    std::string  tmp_dir_;
    std::size_t  num_docs_ = 0;

    std::vector<float>        doc_lengths_;
    std::vector<std::string>  shard_paths_;

    // current in-memory chunk: term → [(doc_id, tf)]
    std::map<std::string, std::vector<SortedPosting>> current_chunk_;

    /* ── Shard I/O (simple binary: term_len u16, term, count u32, entries) ── */

    static void w_u16(std::ofstream& f, uint16_t v) {
        char b[2]={(char)(v&0xFF),(char)((v>>8)&0xFF)}; f.write(b,2);
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
    static float r_f32(std::ifstream& f) {
        uint32_t bits=r_u32(f); float v; std::memcpy(&v,&bits,4); return v;
    }

    /* Write current_chunk_ to a numbered shard file. */
    void flush_chunk() {
        std::string shard_path = tmp_dir_ + "/flashbm25_shard_"
                                 + std::to_string(shard_paths_.size()) + ".tmp";
        std::ofstream f(shard_path, std::ios::binary | std::ios::trunc);
        if (!f) throw std::runtime_error("StreamingIndexBuilder: cannot write shard: " + shard_path);

        w_u64(f, static_cast<uint64_t>(current_chunk_.size()));
        for (const auto& [term, pl] : current_chunk_) {
            w_u16(f, static_cast<uint16_t>(term.size()));
            f.write(term.data(), static_cast<std::streamsize>(term.size()));
            w_u32(f, static_cast<uint32_t>(pl.size()));
            for (const auto& p : pl) {
                w_u64(f, static_cast<uint64_t>(p.doc_id));
                w_f32(f, p.tf);
            }
        }
        shard_paths_.push_back(shard_path);
        current_chunk_.clear();
    }

    /* Read a shard back into a map<term, SortedPostingList>. */
    static std::map<std::string, std::vector<SortedPosting>>
    read_shard(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("StreamingIndexBuilder: cannot read shard: " + path);
        std::map<std::string, std::vector<SortedPosting>> out;
        uint64_t nt = r_u64(f);
        for (uint64_t i = 0; i < nt; ++i) {
            uint16_t tlen = r_u16(f);
            std::string term(tlen, '\0');
            f.read(&term[0], tlen);
            uint32_t cnt = r_u32(f);
            auto& pl = out[term];
            pl.reserve(cnt);
            for (uint32_t j = 0; j < cnt; ++j) {
                std::size_t did = static_cast<std::size_t>(r_u64(f));
                float tf = r_f32(f);
                pl.push_back({did, tf});
            }
        }
        return out;
    }

    /* Merge all shard files into a single CompressedIndex on disk. */
    void merge_shards() {
        // Accumulate all term→postings from all shards
        std::map<std::string, std::vector<SortedPosting>> merged;
        for (const auto& sp : shard_paths_) {
            auto shard = read_shard(sp);
            for (auto& [term, pl] : shard) {
                auto& dst = merged[term];
                dst.insert(dst.end(), pl.begin(), pl.end());
            }
        }

        // Build a SortedPostingsIndex from the merged data
        SortedPostingsIndex spi;
        for (auto& [term, pl] : merged) {
            std::sort(pl.begin(), pl.end(),
                      [](const SortedPosting& a, const SortedPosting& b){
                          return a.doc_id < b.doc_id;
                      });
            for (const auto& p : pl)
                spi.insert(term, p.doc_id, p.tf);
        }
        spi.finalize();

        // Compute avgdl
        double total = 0.0;
        for (float l : doc_lengths_) total += static_cast<double>(l);
        double avgdl = (num_docs_ > 0) ? total / static_cast<double>(num_docs_) : 0.0;

        // Compress and save
        CompressedIndex ci(spi, doc_lengths_, avgdl, num_docs_);
        ci.save(output_path_);
    }

    void cleanup_shards() noexcept {
        for (const auto& sp : shard_paths_) {
            std::error_code ec;
            std::filesystem::remove(sp, ec);  // ignore errors
        }
        shard_paths_.clear();
    }
};

} // namespace flashbm25
