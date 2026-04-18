#pragma once
// This file is a forwarding header kept for backward compatibility.
// The canonical BM25 definition lives in src/core/bm25.hpp.
// All consumers (bindings.cpp, tests, etc.) should prefer including
// "core/bm25.hpp" directly; this header simply re-exports it so that
// any existing include of "bm25.hpp" still resolves without causing
// duplicate / ODR violations.
#include "core/bm25.hpp"
