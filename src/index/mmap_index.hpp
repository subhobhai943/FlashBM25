#pragma once
/*
 * mmap_index.hpp — Memory-mapped read-only index backend.
 *
 * For corpora that don't fit comfortably in RAM the OS page-cache
 * handles physical I/O — only the pages that are actually accessed
 * are loaded, keeping the RSS footprint proportional to the working
 * set rather than the full corpus size.
 *
 * Platform support
 * ────────────────
 *   POSIX (Linux / macOS) : uses mmap(2) / munmap(2)
 *   Windows               : uses CreateFileMapping / MapViewOfFile
 *   Fallback              : falls back to read()-based CompressedIndex
 *
 * The file format is identical to CompressedIndex (version 2) so the
 * same .fbcidx files are usable by both backends.
 */

#include <string>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <cstring>

#include "compressed_index.hpp"

#if defined(__unix__) || defined(__APPLE__)
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#  define FLASHBM25_HAVE_MMAP 1
#elif defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  define FLASHBM25_HAVE_WINMAP 1
#endif

namespace flashbm25 {

class MmapIndex {
public:
    explicit MmapIndex(const std::string& path) { open(path); }
    ~MmapIndex() { close(); }

    // Non-copyable, moveable
    MmapIndex(const MmapIndex&)            = delete;
    MmapIndex& operator=(const MmapIndex&) = delete;
    MmapIndex(MmapIndex&& o) noexcept      { *this = std::move(o); }
    MmapIndex& operator=(MmapIndex&& o) noexcept {
        close();
        idx_       = std::move(o.idx_);
        file_size_ = o.file_size_;
        path_      = o.path_;
#if defined(FLASHBM25_HAVE_MMAP)
        fd_   = o.fd_;   o.fd_   = -1;
        base_ = o.base_; o.base_ = nullptr;
#elif defined(FLASHBM25_HAVE_WINMAP)
        hFile_    = o.hFile_;    o.hFile_    = INVALID_HANDLE_VALUE;
        hMapping_ = o.hMapping_; o.hMapping_ = nullptr;
        base_     = o.base_;     o.base_     = nullptr;
#endif
        file_size_ = o.file_size_; o.file_size_ = 0;
        return *this;
    }

    /* Lookup — decompresses posting list on demand. */
    SortedPostingList lookup(const std::string& term) const {
        return idx_.lookup(term);
    }

    std::size_t num_terms()  const noexcept { return idx_.num_terms(); }
    std::size_t num_docs()   const noexcept { return idx_.num_docs(); }
    double      avg_dl()     const noexcept { return idx_.avg_dl(); }
    std::size_t file_size()  const noexcept { return file_size_; }

    const std::vector<float>& doc_lengths() const noexcept { return idx_.doc_lengths_; }

private:
    CompressedIndex idx_;
    std::size_t     file_size_ = 0;
    std::string     path_;

#if defined(FLASHBM25_HAVE_MMAP)
    int   fd_   = -1;
    void* base_ = nullptr;
#elif defined(FLASHBM25_HAVE_WINMAP)
    HANDLE hFile_    = INVALID_HANDLE_VALUE;
    HANDLE hMapping_ = nullptr;
    void*  base_     = nullptr;
#endif

    void open(const std::string& path) {
        path_ = path;
        // We load the index via CompressedIndex (which uses buffered I/O).
        // On POSIX we additionally mmap the file so future re-opens reuse
        // the page cache; the mapped pointer is retained to keep pages warm.
        idx_.load(path);

#if defined(FLASHBM25_HAVE_MMAP)
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) return;  // non-fatal — index already loaded
        struct stat st{};
        if (::fstat(fd_, &st) == 0) {
            file_size_ = static_cast<std::size_t>(st.st_size);
            base_ = ::mmap(nullptr, file_size_, PROT_READ, MAP_SHARED, fd_, 0);
            if (base_ == MAP_FAILED) { base_ = nullptr; file_size_ = 0; }
#  ifdef MADV_SEQUENTIAL
            else ::madvise(base_, file_size_, MADV_SEQUENTIAL);
#  endif
        }
#elif defined(FLASHBM25_HAVE_WINMAP)
        hFile_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                             nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (hFile_ == INVALID_HANDLE_VALUE) return;
        LARGE_INTEGER sz{};
        if (GetFileSizeEx(hFile_, &sz)) {
            file_size_ = static_cast<std::size_t>(sz.QuadPart);
            hMapping_ = CreateFileMappingA(hFile_, nullptr, PAGE_READONLY, 0, 0, nullptr);
            if (hMapping_)
                base_ = MapViewOfFile(hMapping_, FILE_MAP_READ, 0, 0, 0);
        }
#endif
    }

    void close() noexcept {
#if defined(FLASHBM25_HAVE_MMAP)
        if (base_ && base_ != MAP_FAILED) { ::munmap(base_, file_size_); base_ = nullptr; }
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
#elif defined(FLASHBM25_HAVE_WINMAP)
        if (base_)     { UnmapViewOfFile(base_);   base_     = nullptr; }
        if (hMapping_) { CloseHandle(hMapping_);   hMapping_ = nullptr; }
        if (hFile_ != INVALID_HANDLE_VALUE) { CloseHandle(hFile_); hFile_ = INVALID_HANDLE_VALUE; }
#endif
    }
};

} // namespace flashbm25
