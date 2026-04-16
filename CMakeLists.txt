cmake_minimum_required(VERSION 3.15)
project(flashbm25 VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Optimisation flags ────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(
  $<$<CONFIG:Release>:-O3>
  $<$<CONFIG:Release>:-march=native>
  $<$<CONFIG:Release>:-DNDEBUG>
)

# ── pybind11 (fetched automatically if not found) ────────────────────────────
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

# ── Optional: OpenMP for parallel scoring ────────────────────────────────────
find_package(OpenMP QUIET)

# ── Extension module ─────────────────────────────────────────────────────────
pybind11_add_module(_flashbm25
  src/bindings.cpp
  src/bm25.cpp
)

target_include_directories(_flashbm25 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

if(OpenMP_CXX_FOUND)
  target_link_libraries(_flashbm25 PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(_flashbm25 PRIVATE FLASHBM25_USE_OMP)
endif()

# Install the extension next to the Python package
install(TARGETS _flashbm25 DESTINATION flashbm25)
