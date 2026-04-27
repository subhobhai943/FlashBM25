#pragma once
/*
 * varint.hpp — Variable-length integer encoding (LEB128 / VarInt)
 *              + gap/delta coding helpers for doc-ID compression.
 *
 * Encoding scheme
 * ───────────────
 * Each byte stores 7 payload bits in the low 7 bits.
 * The high bit (0x80) is the "more bytes follow" continuation flag.
 * Values 0-127 encode in 1 byte; 128-16383 in 2 bytes, etc.
 *
 * Delta coding
 * ────────────
 * Doc IDs within a postings list are stored as *gaps* (deltas) from the
 * previous doc ID (first delta == first doc ID).  This keeps values small
 * and improves VarInt compression ratios.
 *
 * Usage
 * ─────
 *   std::vector<uint8_t> buf;
 *   varint::encode(42,  buf);   // appends bytes
 *   varint::encode(300, buf);
 *
 *   std::size_t pos = 0;
 *   uint64_t a = varint::decode(buf, pos);
 *   uint64_t b = varint::decode(buf, pos);
 */

#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace flashbm25 {
namespace varint {

/* Append the VarInt encoding of `v` to `buf`. */
inline void encode(uint64_t v, std::vector<uint8_t>& buf) {
    do {
        uint8_t byte = static_cast<uint8_t>(v & 0x7F);
        v >>= 7;
        if (v != 0) byte |= 0x80;
        buf.push_back(byte);
    } while (v != 0);
}

/* Decode one VarInt from `buf` starting at `pos`; advances `pos`. */
inline uint64_t decode(const std::vector<uint8_t>& buf, std::size_t& pos) {
    uint64_t result = 0;
    int      shift  = 0;
    while (pos < buf.size()) {
        uint8_t byte = buf[pos++];
        result |= static_cast<uint64_t>(byte & 0x7F) << shift;
        if ((byte & 0x80) == 0) return result;
        shift += 7;
        if (shift >= 64)
            throw std::runtime_error("varint::decode — overflow (>64 bits)");
    }
    throw std::runtime_error("varint::decode — unexpected end of buffer");
}

/* Same but reading from a raw byte pointer range [begin, end). */
inline uint64_t decode_ptr(const uint8_t* begin, const uint8_t* end,
                           const uint8_t*& cur) {
    uint64_t result = 0;
    int      shift  = 0;
    while (cur < end) {
        uint8_t byte = *cur++;
        result |= static_cast<uint64_t>(byte & 0x7F) << shift;
        if ((byte & 0x80) == 0) return result;
        shift += 7;
        if (shift >= 64)
            throw std::runtime_error("varint::decode_ptr — overflow");
    }
    throw std::runtime_error("varint::decode_ptr — unexpected end of buffer");
}

/*
 * encode_delta_list — encode a sorted list of doc IDs as delta-coded VarInts.
 * `tfs` must be the same length as `doc_ids`; TF is stored as raw float32.
 */
inline void encode_delta_list(const std::vector<std::size_t>& doc_ids,
                              const std::vector<float>&       tfs,
                              std::vector<uint8_t>&           out) {
    // number of entries
    encode(static_cast<uint64_t>(doc_ids.size()), out);
    std::size_t prev = 0;
    for (std::size_t i = 0; i < doc_ids.size(); ++i) {
        encode(static_cast<uint64_t>(doc_ids[i] - prev), out);  // gap
        prev = doc_ids[i];
        // store TF as raw 4-byte float
        uint32_t bits;
        static_assert(sizeof(float) == 4, "float must be 4 bytes");
        __builtin_memcpy(&bits, &tfs[i], 4);
        encode(static_cast<uint64_t>(bits), out);  // piggyback float bits
    }
}

/*
 * decode_delta_list — decode a list previously encoded by encode_delta_list.
 * Returns the number of entries decoded.
 */
struct DecodedPosting { std::size_t doc_id; float tf; };

inline std::vector<DecodedPosting>
decode_delta_list(const std::vector<uint8_t>& buf, std::size_t& pos) {
    uint64_t count = decode(buf, pos);
    std::vector<DecodedPosting> result;
    result.reserve(static_cast<std::size_t>(count));
    std::size_t prev = 0;
    for (uint64_t i = 0; i < count; ++i) {
        uint64_t gap  = decode(buf, pos);
        uint64_t bits = decode(buf, pos);
        float tf;
        uint32_t bits32 = static_cast<uint32_t>(bits);
        __builtin_memcpy(&tf, &bits32, 4);
        prev += static_cast<std::size_t>(gap);
        result.push_back({prev, tf});
    }
    return result;
}

} // namespace varint
} // namespace flashbm25
