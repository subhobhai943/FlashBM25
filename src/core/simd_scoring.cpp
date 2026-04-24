#include "simd_scoring.hpp"

#include <algorithm>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#define FLASHBM25_X86_TARGET 1
#endif

namespace flashbm25 {
namespace {

using OkapiKernel = void (*)(const PostingList&,
                             const std::vector<double>&,
                             double,
                             double,
                             double,
                             double,
                             std::vector<double>&);
using PlusKernel = void (*)(const PostingList&,
                            const std::vector<double>&,
                            double,
                            double,
                            double,
                            double,
                            double,
                            std::vector<double>&);
using LKernel = PlusKernel;

struct ScoringBackend {
    const char* name;
    OkapiKernel okapi;
    PlusKernel plus;
    LKernel l;
};

void score_okapi_scalar(const PostingList& postings,
                        const std::vector<double>& doc_len,
                        double avgdl,
                        double k1,
                        double b,
                        double term_weight,
                        std::vector<double>& scores) {
    const double k1_plus_one = k1 + 1.0;
    const double one_minus_b = 1.0 - b;
    const double b_over_avgdl = b / avgdl;

    for (const auto& posting : postings) {
        const double tf = posting.tf;
        const double norm = one_minus_b + b_over_avgdl * doc_len[posting.doc_id];
        const double tf_adj = (tf * k1_plus_one) / (tf + k1 * norm);
        scores[posting.doc_id] += term_weight * tf_adj;
    }
}

void score_bm25plus_scalar(const PostingList& postings,
                           const std::vector<double>& doc_len,
                           double avgdl,
                           double k1,
                           double b,
                           double delta,
                           double term_weight,
                           std::vector<double>& scores) {
    const double k1_plus_one = k1 + 1.0;
    const double one_minus_b = 1.0 - b;
    const double b_over_avgdl = b / avgdl;

    for (const auto& posting : postings) {
        const double tf = posting.tf;
        const double norm = one_minus_b + b_over_avgdl * doc_len[posting.doc_id];
        const double tf_adj = (tf * k1_plus_one) / (tf + k1 * norm) + delta;
        scores[posting.doc_id] += term_weight * tf_adj;
    }
}

void score_bm25l_scalar(const PostingList& postings,
                        const std::vector<double>& doc_len,
                        double avgdl,
                        double k1,
                        double b,
                        double delta,
                        double term_weight,
                        std::vector<double>& scores) {
    const double k1_plus_one = k1 + 1.0;
    const double one_minus_b = 1.0 - b;
    const double b_over_avgdl = b / avgdl;

    for (const auto& posting : postings) {
        const double tf = posting.tf;
        const double norm = one_minus_b + b_over_avgdl * doc_len[posting.doc_id];
        const double c_tilde = tf / norm;
        const double tf_adj = (k1_plus_one * c_tilde) / (k1 + c_tilde) + delta;
        scores[posting.doc_id] += term_weight * tf_adj;
    }
}

#if defined(FLASHBM25_BUILD_X86_SIMD) && defined(FLASHBM25_X86_TARGET)
void score_okapi_x86_avx2(const PostingList& postings,
                          const std::vector<double>& doc_len,
                          double avgdl,
                          double k1,
                          double b,
                          double term_weight,
                          std::vector<double>& scores);
void score_bm25plus_x86_avx2(const PostingList& postings,
                             const std::vector<double>& doc_len,
                             double avgdl,
                             double k1,
                             double b,
                             double delta,
                             double term_weight,
                             std::vector<double>& scores);
void score_bm25l_x86_avx2(const PostingList& postings,
                          const std::vector<double>& doc_len,
                          double avgdl,
                          double k1,
                          double b,
                          double delta,
                          double term_weight,
                          std::vector<double>& scores);
void score_okapi_x86_sse42(const PostingList& postings,
                           const std::vector<double>& doc_len,
                           double avgdl,
                           double k1,
                           double b,
                           double term_weight,
                           std::vector<double>& scores);
void score_bm25plus_x86_sse42(const PostingList& postings,
                              const std::vector<double>& doc_len,
                              double avgdl,
                              double k1,
                              double b,
                              double delta,
                              double term_weight,
                              std::vector<double>& scores);
void score_bm25l_x86_sse42(const PostingList& postings,
                           const std::vector<double>& doc_len,
                           double avgdl,
                           double k1,
                           double b,
                           double delta,
                           double term_weight,
                           std::vector<double>& scores);

bool cpu_supports_avx2() noexcept {
#if defined(_MSC_VER)
    int regs[4] = {0, 0, 0, 0};
    __cpuid(regs, 0);
    if (regs[0] < 7) {
        return false;
    }

    __cpuid(regs, 1);
    const bool osxsave = (regs[2] & (1 << 27)) != 0;
    const bool avx = (regs[2] & (1 << 28)) != 0;
    if (!osxsave || !avx) {
        return false;
    }

    const unsigned long long xcr0 = _xgetbv(0);
    if ((xcr0 & 0x6) != 0x6) {
        return false;
    }

    __cpuidex(regs, 7, 0);
    return (regs[1] & (1 << 5)) != 0;
#elif defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}

bool cpu_supports_sse42() noexcept {
#if defined(_MSC_VER)
    int regs[4] = {0, 0, 0, 0};
    __cpuid(regs, 1);
    return (regs[2] & (1 << 20)) != 0;
#elif defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("sse4.2");
#else
    return false;
#endif
}
#endif

#if defined(FLASHBM25_BUILD_NEON)
void score_okapi_neon(const PostingList& postings,
                      const std::vector<double>& doc_len,
                      double avgdl,
                      double k1,
                      double b,
                      double term_weight,
                      std::vector<double>& scores);
void score_bm25plus_neon(const PostingList& postings,
                         const std::vector<double>& doc_len,
                         double avgdl,
                         double k1,
                         double b,
                         double delta,
                         double term_weight,
                         std::vector<double>& scores);
void score_bm25l_neon(const PostingList& postings,
                      const std::vector<double>& doc_len,
                      double avgdl,
                      double k1,
                      double b,
                      double delta,
                      double term_weight,
                      std::vector<double>& scores);
#endif

ScoringBackend select_backend() noexcept {
#if defined(FLASHBM25_BUILD_X86_SIMD) && defined(FLASHBM25_X86_TARGET)
    if (cpu_supports_avx2()) {
        return {"avx2", score_okapi_x86_avx2, score_bm25plus_x86_avx2, score_bm25l_x86_avx2};
    }
    if (cpu_supports_sse42()) {
        return {"sse4.2", score_okapi_x86_sse42, score_bm25plus_x86_sse42,
                score_bm25l_x86_sse42};
    }
#endif

#if defined(FLASHBM25_BUILD_NEON)
    return {"neon", score_okapi_neon, score_bm25plus_neon, score_bm25l_neon};
#else
    return {"scalar", score_okapi_scalar, score_bm25plus_scalar, score_bm25l_scalar};
#endif
}

const ScoringBackend& active_backend() noexcept {
    static const ScoringBackend backend = select_backend();
    return backend;
}

} // namespace

void rebuild_postings_index(const TfIndex& tf_index, PostingsIndex& postings_index) {
    postings_index.clear();
    postings_index.reserve(tf_index.size());

    for (const auto& term_entry : tf_index) {
        const auto& term = term_entry.first;
        const auto& postings = term_entry.second;
        auto& dense_postings = postings_index[term];
        dense_postings.reserve(postings.size());

        for (const auto& posting : postings) {
            dense_postings.push_back(Posting{posting.first, posting.second});
        }

        std::sort(dense_postings.begin(), dense_postings.end(),
                  [](const Posting& lhs, const Posting& rhs) {
                      return lhs.doc_id < rhs.doc_id;
                  });
    }
}

void score_okapi_postings(const PostingList& postings,
                          const std::vector<double>& doc_len,
                          double avgdl,
                          double k1,
                          double b,
                          double term_weight,
                          std::vector<double>& scores) {
    active_backend().okapi(postings, doc_len, avgdl, k1, b, term_weight, scores);
}

void score_bm25plus_postings(const PostingList& postings,
                             const std::vector<double>& doc_len,
                             double avgdl,
                             double k1,
                             double b,
                             double delta,
                             double term_weight,
                             std::vector<double>& scores) {
    active_backend().plus(postings, doc_len, avgdl, k1, b, delta, term_weight, scores);
}

void score_bm25l_postings(const PostingList& postings,
                          const std::vector<double>& doc_len,
                          double avgdl,
                          double k1,
                          double b,
                          double delta,
                          double term_weight,
                          std::vector<double>& scores) {
    active_backend().l(postings, doc_len, avgdl, k1, b, delta, term_weight, scores);
}

const char* active_simd_backend_name() noexcept {
    return active_backend().name;
}

} // namespace flashbm25
