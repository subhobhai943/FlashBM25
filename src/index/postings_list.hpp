#pragma once
/*
 * postings_list.hpp — Compact sorted postings list.
 *
 * Replaces std::unordered_map<string, vector<Posting>> with a flat
 * sorted structure:
 *   - Terms stored in a sorted std::vector<std::string>
 *   - Corresponding PostingList stored in a parallel vector
 *   - Lookup via binary-search in O(log V) instead of hash-table overhead
 *
 * Benefits vs unordered_map:
 *   - ~40% lower per-entry memory (no linked-list buckets / load-factor slack)
 *   - Better cache locality during sequential term iteration
 *   - Deterministic serialisation order (no bucket ordering)
 */

#include <algorithm>
#include <string>
#include <vector>
#include <utility>
#include <stdexcept>
#include <cstddef>

namespace flashbm25 {

struct SortedPosting {
    std::size_t doc_id;  // kept sorted within each PostingList
    float       tf;
};

using SortedPostingList = std::vector<SortedPosting>;

/*
 * SortedPostingsIndex — term → sorted postings, backed by two parallel vectors.
 *
 * After all documents are added call finalize() to sort both the term
 * vector and the per-term postings by doc_id.  After finalize() the
 * index is immutable and lookup() is valid.
 */
class SortedPostingsIndex {
public:
    SortedPostingsIndex() = default;

    /* Add (or extend) the postings for `term` with a single entry. */
    void insert(const std::string& term, std::size_t doc_id, float tf) {
        auto it = find_term_mutable(term);
        if (it == terms_.end() || *it != term) {
            auto idx = static_cast<std::size_t>(it - terms_.begin());
            terms_.insert(it, term);
            postings_.insert(postings_.begin() + static_cast<std::ptrdiff_t>(idx),
                             SortedPostingList{});
            postings_[idx].push_back({doc_id, tf});
        } else {
            auto idx = static_cast<std::size_t>(it - terms_.begin());
            postings_[idx].push_back({doc_id, tf});
        }
    }

    /* Sort each per-term postings list by doc_id (call once after ingestion). */
    void finalize() {
        for (auto& pl : postings_) {
            std::sort(pl.begin(), pl.end(),
                      [](const SortedPosting& a, const SortedPosting& b) {
                          return a.doc_id < b.doc_id;
                      });
        }
    }

    /* O(log V) lookup. Returns nullptr when term not found. */
    const SortedPostingList* lookup(const std::string& term) const {
        auto it = std::lower_bound(terms_.begin(), terms_.end(), term);
        if (it == terms_.end() || *it != term) return nullptr;
        return &postings_[static_cast<std::size_t>(it - terms_.begin())];
    }

    std::size_t num_terms()    const noexcept { return terms_.size(); }
    bool        empty()        const noexcept { return terms_.empty(); }

    /* Iterators for serialisation. */
    const std::vector<std::string>&       terms()    const noexcept { return terms_; }
    const std::vector<SortedPostingList>& postings() const noexcept { return postings_; }

    /* Approximate heap memory in bytes. */
    std::size_t memory_bytes() const noexcept {
        std::size_t n = 0;
        for (const auto& t : terms_)   n += t.size() + sizeof(std::string);
        for (const auto& pl : postings_) n += pl.size() * sizeof(SortedPosting)
                                             + sizeof(SortedPostingList);
        return n;
    }

private:
    std::vector<std::string>       terms_;
    std::vector<SortedPostingList> postings_;

    /* Lower-bound iterator into terms_ — used by insert(). */
    std::vector<std::string>::iterator find_term_mutable(const std::string& t) {
        return std::lower_bound(terms_.begin(), terms_.end(), t);
    }
};

} // namespace flashbm25
