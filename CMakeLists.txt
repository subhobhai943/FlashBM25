cmake_minimum_required(VERSION 3.15)
project(flashbm25 VERSION 0.2.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD   11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Build type ─────────────────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Use portable optimization flags; avoid -march=native which can cause
# cross-compilation and CI failures (especially on macOS/Windows runners).
add_compile_options(
  $<$<CONFIG:Release>:-O3>
  $<$<CONFIG:Release>:-DNDEBUG>
)

# ── pybind11 (fetched automatically if not found) ───────────────────────────────────
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.1
  )
  FetchContent_MakeAvailable(pybind11)
endif()

# ── Optional: OpenMP for parallel scoring ───────────────────────────────────────────────
find_package(OpenMP QUIET)

# ── Pure-C tokenizer + Porter stemmer (callable from C, C++, Rust, WASM) ────────────
add_library(flashbm25_tokenizer STATIC
  src/tokenizer/stemmer.c
  src/tokenizer/tokenizer.c
)
target_include_directories(flashbm25_tokenizer PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/tokenizer
)
set_target_properties(flashbm25_tokenizer PROPERTIES
  POSITION_INDEPENDENT_CODE ON   # needed for linking into a shared pybind module
)

# ── C++ core: BM25, BM25+, BM25L, InvertedIndex, RRF ─────────────────────────────
set(FLASHBM25_CORE_SOURCES
  src/core/bm25.cpp
  src/core/bm25_variants.cpp
  src/core/bm25f.cpp
  src/core/simd_scoring.cpp
  src/index/inverted_index.cpp
  src/scoring/hybrid_rrf.cpp
)

set(FLASHBM25_SYSTEM_PROCESSOR "${CMAKE_SYSTEM_PROCESSOR}")
if(APPLE AND CMAKE_OSX_ARCHITECTURES)
  list(LENGTH CMAKE_OSX_ARCHITECTURES FLASHBM25_OSX_ARCH_COUNT)
  if(FLASHBM25_OSX_ARCH_COUNT EQUAL 1)
    list(GET CMAKE_OSX_ARCHITECTURES 0 FLASHBM25_SYSTEM_PROCESSOR)
  else()
    set(FLASHBM25_SYSTEM_PROCESSOR "universal")
  endif()
endif()
string(TOLOWER "${FLASHBM25_SYSTEM_PROCESSOR}" FLASHBM25_SYSTEM_PROCESSOR_LOWER)

set(FLASHBM25_X86_SIMD_SOURCES)
if(FLASHBM25_SYSTEM_PROCESSOR_LOWER MATCHES "^(x86_64|amd64|x64|i[3-6]86)$")
  list(APPEND FLASHBM25_X86_SIMD_SOURCES
    src/core/simd_scoring_x86_avx2.cpp
    src/core/simd_scoring_x86_sse42.cpp
  )
endif()

set(FLASHBM25_NEON_SOURCES)
if(FLASHBM25_SYSTEM_PROCESSOR_LOWER MATCHES "^(aarch64|arm64)$")
  list(APPEND FLASHBM25_NEON_SOURCES
    src/core/simd_scoring_neon.cpp
  )
endif()

add_library(flashbm25_core STATIC
  ${FLASHBM25_CORE_SOURCES}
  ${FLASHBM25_X86_SIMD_SOURCES}
  ${FLASHBM25_NEON_SOURCES}
)
target_include_directories(flashbm25_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/core
  ${CMAKE_CURRENT_SOURCE_DIR}/src/index
  ${CMAKE_CURRENT_SOURCE_DIR}/src/scoring
  ${CMAKE_CURRENT_SOURCE_DIR}/src/tokenizer
)
set_target_properties(flashbm25_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(flashbm25_core PUBLIC flashbm25_tokenizer)

if(FLASHBM25_X86_SIMD_SOURCES)
  target_compile_definitions(flashbm25_core PRIVATE FLASHBM25_BUILD_X86_SIMD)
  if(MSVC)
    set_source_files_properties(src/core/simd_scoring_x86_avx2.cpp PROPERTIES
      COMPILE_OPTIONS "/arch:AVX2"
    )
  else()
    set_source_files_properties(src/core/simd_scoring_x86_avx2.cpp PROPERTIES
      COMPILE_OPTIONS "-mavx2"
    )
    set_source_files_properties(src/core/simd_scoring_x86_sse42.cpp PROPERTIES
      COMPILE_OPTIONS "-msse4.2"
    )
  endif()
endif()

if(FLASHBM25_NEON_SOURCES)
  target_compile_definitions(flashbm25_core PRIVATE FLASHBM25_BUILD_NEON)
endif()

if(OpenMP_CXX_FOUND)
  target_link_libraries(flashbm25_core PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(flashbm25_core PUBLIC FLASHBM25_USE_OMP)
endif()

# ── Python extension ─────────────────────────────────────────────────────────────────────
pybind11_add_module(_flashbm25
  src/bindings.cpp
)
target_link_libraries(_flashbm25 PRIVATE flashbm25_core)

# Install the extension next to the Python package
install(TARGETS _flashbm25 DESTINATION flashbm25)

# ── Optional: install static libs for C/Rust consumers ─────────────────────────────────
install(TARGETS flashbm25_core flashbm25_tokenizer
  ARCHIVE DESTINATION lib
)
install(FILES
  src/tokenizer/tokenizer.h
  src/tokenizer/stemmer.h
  DESTINATION include/flashbm25
)
