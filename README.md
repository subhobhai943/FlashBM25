<div align="center">

# ⚡ FlashBM25

**High-performance BM25 information-retrieval library — C++ engine, Python API**

[![PyPI version](https://img.shields.io/pypi/v/flashbm25.svg)](https://pypi.org/project/flashbm25/)
[![Python](https://img.shields.io/pypi/pyversions/flashbm25.svg)](https://pypi.org/project/flashbm25/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![CI](https://github.com/subhobhai943/FlashBM25/actions/workflows/ci.yml/badge.svg)](https://github.com/subhobhai943/FlashBM25/actions/workflows/ci.yml)

</div>

---

FlashBM25 implements the [Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25) ranking algorithm in
**C++17** and exposes it to Python via **pybind11**. It is designed to be a drop-in replacement for
[rank\_bm25](https://github.com/dorianbrown/rank_bm25) with significantly faster index build and
query times, especially on large corpora.

## Features

- 🚀 **C++ core** — compiled extension via pybind11, no Python loops in the hot path  
- 📦 **Zero runtime dependencies** — pure C++17 standard library  
- 🔧 **Tunable** — `k1`, `b`, `epsilon` exposed as constructor arguments  
- 🧵 **OpenMP-ready** — automatically uses multi-core parallelism when OpenMP is available  
- 🐍 **Pythonic API** — thin wrapper with full type hints  

## Installation

```bash
pip install flashbm25
```

Pre-built wheels are provided for **Linux, macOS, and Windows** on Python 3.9–3.12.

To build from source (requires a C++17 compiler and CMake ≥ 3.15):

```bash
git clone https://github.com/subhobhai943/FlashBM25.git
cd FlashBM25
pip install -e ".[dev]" --no-build-isolation
```

## Quick Start

```python
from flashbm25 import BM25

corpus = [
    "the quick brown fox jumps over the lazy dog",
    "a fast red fox leaped across the sleeping hound",
    "machine learning is a subfield of artificial intelligence",
    "deep learning models require large amounts of training data",
    "BM25 is a bag-of-words retrieval function used in search engines",
]

bm25 = BM25(corpus)

# Score every document against a query
scores = bm25.get_scores("fox jumps")
print(scores)  # [1.43, 0.72, 0.0, 0.0, 0.0]

# Get the top-3 (score, doc_index) pairs
print(bm25.get_top_n("fox jumps", n=3))
# [(1.43, 0), (0.72, 1), (0.0, 2)]

# Get the actual top-3 documents
print(bm25.get_top_n_docs("fox jumps", n=3))
```

## Documentation

The documentation source lives in [`docs/`](docs/) and is configured for Read the Docs via
[`.readthedocs.yaml`](.readthedocs.yaml). The API reference is generated with Sphinx autodoc from
the public Python docstrings.

## API Reference

### `BM25(corpus, k1=1.5, b=0.75, epsilon=0.25, lowercase=True)`

| Parameter   | Type        | Default | Description                                 |
|-------------|-------------|---------|---------------------------------------------|
| `corpus`    | `list[str]` | —       | Documents to index                          |
| `k1`        | `float`     | `1.5`   | Term saturation parameter                   |
| `b`         | `float`     | `0.75`  | Length normalisation parameter              |
| `epsilon`   | `float`     | `0.25`  | IDF floor (prevents negative scores)        |
| `lowercase` | `bool`      | `True`  | Normalise text to lowercase before indexing |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_scores(query)` | `list[float]` | BM25 score for every doc |
| `get_top_n(query, n=5)` | `list[tuple[float, int]]` | Top-n `(score, index)` pairs |
| `get_top_n_docs(query, n=5)` | `list[str]` | Top-n document strings |

### Properties

| Property | Description |
|----------|-------------|
| `corpus_size` | Number of indexed documents |
| `avg_doc_length` | Average document length in tokens |
| `k1`, `b`, `epsilon` | BM25 hyperparameters |

## Benchmarks

```
Corpus: 10 000 documents, 60 tokens each
Query:  "quick brown fox machine learning"

                         FlashBM25 (C++)   rank_bm25 (Python)   Speedup
  Index build            ~12 ms            ~340 ms              ~28x
  Top-10 query           ~0.4 ms           ~8 ms                ~20x
```

*Results vary by hardware. Run `python benchmarks/bench_flashbm25.py` to reproduce.*

## Development

```bash
git clone https://github.com/subhobhai943/FlashBM25.git
cd FlashBM25
pip install -e ".[dev]" --no-build-isolation
pytest tests/ -v
python benchmarks/bench_flashbm25.py
```

## Releasing to PyPI

Releases are automated. Push a version tag to trigger the build-and-publish pipeline:

```bash
git tag v0.1.0
git push origin v0.1.0
```

Wheels are built for Linux (manylinux), macOS, and Windows via `cibuildwheel`, then uploaded
to PyPI using **Trusted Publishing** (OIDC — no API tokens required).

See the [Trusted Publishing setup guide](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
before your first release.

## Contributing

Pull requests are welcome! Please open an issue first to discuss major changes.

## License

Apache 2.0 — see [LICENSE](LICENSE).
