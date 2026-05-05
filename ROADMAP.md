# ⚡ FlashBM25 — Development Roadmap

> **Vision:** Grow FlashBM25 from a fast BM25 drop-in replacement into a full-featured, production-grade
> information-retrieval library — the TensorFlow of lexical search.

---

## Current State (Baseline)

| Area | Status |
|---|---|
| BM25Okapi core (C++17 + pybind11) | ✅ Done |
| Python thin wrapper (`BM25` class) | ✅ Done |
| `get_scores`, `get_top_n`, `get_top_n_docs` | ✅ Done |
| OpenMP parallelism (optional) | ✅ Done |
| CI / cibuildwheel wheels (Linux, macOS, Windows) | ✅ Done |
| PyPI Trusted Publishing | ✅ Done |
| Basic unit tests | ✅ Done |

---

## Phase 1 — Solid Foundation  
**Goal:** Harden the existing core, fill obvious gaps, and establish the engineering discipline
that will support every future phase.

### 1.1 — Algorithm Completeness
- [x] Add `BM25L` variant (length-penalised lower-bound IDF)
- [x] Add `BM25Plus` variant (lower-bound term frequency)
- [x] Add `BM25Adpt` variant (adaptive `k1` per term)
- [x] Expose a `variant` parameter on the top-level `BM25` factory so users can switch variants with one argument
- [x] Add `BM25F` skeleton (field-weighted multi-field scoring — full impl in Phase 3)

### 1.2 — Tokenizer Layer
- [x] Expose `Tokenizer` as a first-class Python class (currently hidden inside C++)
- [x] Add pluggable tokenizer support: users can pass any `Callable[[str], list[str]]`
- [x] Built-in tokenizer options: `whitespace` (current), `regex`, `unicode_word`
- [x] Stopword list support (built-in English list + custom list injection)
- [x] Stemmer hook — accept any callable (e.g., `nltk.PorterStemmer().stem`) without a hard dependency

### 1.3 — Index Persistence
- [x] `bm25.save(path)` — serialize index to a compact binary format (MessagePack or custom)
- [x] `BM25.load(path)` — deserialize and reconstruct without re-indexing
- [x] Incremental add: `bm25.add_documents(new_docs)` — append docs without full rebuild
- [x] Incremental delete: `bm25.remove_document(doc_id)` — mark-and-rebuild strategy

### 1.4 — Testing & Quality Gates
- [x] Expand unit tests to cover all three BM25 variants
- [x] Add property-based tests (Hypothesis) for score monotonicity and IDF bounds
- [x] Add integration tests against known TREC / MS-MARCO query-document pairs
- [x] Set up coverage reporting (≥ 90 % line coverage gate in CI)
- [x] Add a linting / formatting step (`clang-format` for C++, `ruff` for Python)

### 1.5 — Documentation Baseline
- [x] Set up Sphinx + `sphinx.ext.autodoc` for Python API docs
- [x] Write narrative "Getting Started" guide
- [x] Document every public class, method, and parameter with type hints + docstrings
- [x] Configure Read the Docs publishing via `.readthedocs.yaml`

**Exit criteria:** All existing tests pass, three BM25 variants available, index save/load works,
docs site configured for Read the Docs.

---

## Phase 2 — Performance & Scale  
**Goal:** Make FlashBM25 the fastest pure-CPU BM25 library at any corpus size — from 10 K docs
to 100 M docs.

### 2.1 — SIMD & CPU Optimisations
- [x] Port inner-product loop to AVX2 intrinsics with SSE4.2 fallback
- [x] Investigate NEON path for Apple Silicon / ARM
- [x] Profile with `perf` / VTune and eliminate top-3 hotspots
- [x] Benchmark against Whoosh, Tantivy (via Python bindings), Elasticsearch BM25

### 2.2 — Parallel & Async Query
- [x] Batch query API: `bm25.get_scores_batch(queries: list[str]) -> np.ndarray`
- [x] Thread-pool executor for batch queries (configurable `n_jobs`)
- [x] Async interface: `await bm25.aget_scores(query)` using `asyncio.to_thread`
- [x] Release the GIL in all hot-path pybind11 calls

### 2.3 — Memory-Efficient Index Structures
- [x] Switch inverted index from `std::unordered_map` to a compact sorted postings list
- [x] Compressed posting lists (delta-coded doc IDs + VarInt TF encoding)
- [x] Memory-mapped index for corpora that don't fit in RAM (`mmap` backend)
- [x] On-disk streaming index builder for very large corpora (chunked ingestion)

### 2.4 — NumPy / SciPy Integration
- [x] `get_scores` returns `np.ndarray` (float32) instead of `list[float]` — zero-copy via buffer protocol
- [x] Sparse score matrix: `get_scores_batch` returns `scipy.sparse.csr_matrix` for large batch × corpus results
- [x] `get_top_n` returns structured NumPy record array `(score: f32, doc_id: u32)`

### 2.5 — Benchmarking Suite
- [ ] Automate benchmark runs in CI on every release tag
- [ ] Add benchmark result tracking (JSON artefact per run, plotted in docs)
- [ ] Publish a public leaderboard page comparing FlashBM25 to alternatives

**Exit criteria:** 50× speedup vs. rank_bm25 on 1 M-document corpus; batch query throughput
> 10 K queries/sec on commodity hardware; memory usage < 2 GB for 1 M × 100-token corpus.

---

## Phase 3 — Rich Feature Set  
**Goal:** Match and exceed the feature set of Elasticsearch / OpenSearch BM25 — making FlashBM25
a standalone search engine core.

### 3.1 — Multi-Field Search (`BM25F`)
- [ ] Complete `BM25F` implementation: weighted scoring across named fields (title, body, tags, …)
- [ ] Per-field `b` and `k1` parameters
- [ ] Python API: `BM25F(corpus_dicts, fields={"title": 2.0, "body": 1.0})`
- [ ] Boost formulas: per-field score combination strategies (sum, max, reciprocal-rank fusion)

### 3.2 — Filtering & Faceting
- [ ] Metadata store: attach arbitrary `dict` payload to each document at index time
- [ ] Pre-filter: `bm25.get_scores(query, filter={"category": "tech"})` — bitset-accelerated
- [ ] Post-filter: `bm25.get_top_n(query, n=10, where=lambda doc: doc["year"] > 2020)`
- [ ] Facet aggregation: `bm25.facets(query, field="category")` — returns top-k facet counts

### 3.3 — Query Language
- [ ] Boolean queries: `AND`, `OR`, `NOT` operators via a simple query parser
- [ ] Phrase queries: `"machine learning"` exact-phrase scoring
- [ ] Proximity / slop queries: `"fox dog"~3` (within 3 tokens)
- [ ] Wildcard / prefix queries: `"machin*"` (with inverted index support)
- [ ] Expose query AST as a Python object so users can build queries programmatically

### 3.4 — Highlighting & Snippets
- [ ] `bm25.highlight(query, doc_index)` — returns the document with matched terms wrapped in `<mark>`
- [ ] `bm25.snippet(query, doc_index, context=50)` — returns best-matching passage (± 50 chars)
- [ ] Custom highlight formatter: accept a `Callable[[str], str]` for custom markup

### 3.5 — Evaluation & Metrics
- [ ] Built-in IR evaluation: `flashbm25.evaluate(queries, qrels, k=[1,5,10])`
- [ ] Metrics: MRR, MAP, nDCG@k, Recall@k, Precision@k
- [ ] BEIR benchmark integration helper
- [ ] Automatic tuning: `bm25.tune(queries, qrels)` — grid-search `k1` / `b` for best nDCG

**Exit criteria:** Multi-field search, boolean queries, faceting, and highlighting all working;
nDCG@10 on MS-MARCO passage dev ≥ 0.185 (competitive with Elasticsearch default BM25).

---

## Phase 4 — Ecosystem & Integrations  
**Goal:** Make FlashBM25 the default BM25 backend for the Python ML/NLP ecosystem.

### 4.1 — Hugging Face / Datasets Integration
- [ ] `flashbm25.from_dataset(hf_dataset, text_column="text")` — index directly from a HF Dataset
- [ ] Arrow / Parquet ingestion: index from a `pyarrow.Table` column without materialising Python strings
- [ ] Push to Hugging Face Hub: serialised index artefacts uploadable as HF datasets

### 4.2 — LangChain / LlamaIndex Retrievers
- [ ] `FlashBM25Retriever` for LangChain (`BaseRetriever` subclass)
- [ ] `FlashBM25Retriever` for LlamaIndex (`BaseRetriever` subclass)
- [ ] Hybrid retriever helper: `HybridRetriever(dense_retriever, bm25, alpha=0.5)` (RRF or linear combination)
- [ ] Publish integration packages as optional extras: `pip install flashbm25[langchain]`

### 4.3 — REST / gRPC Server
- [ ] `flashbm25-server` CLI: spins up a FastAPI server exposing the index over HTTP
- [ ] Endpoints: `POST /index`, `GET /search`, `DELETE /document/{id}`, `GET /health`
- [ ] gRPC service definition (`.proto`) for high-throughput production use
- [ ] Docker image: `ghcr.io/subhobhai943/flashbm25-server`

### 4.4 — Sparse Vector Export
- [ ] `bm25.to_sparse_vectors(queries)` — export BM25 scores as sparse vectors for vector-DB ingestion
- [ ] Pinecone / Weaviate / Qdrant sparse vector format helpers
- [ ] SPLADE-style learned sparse retrieval compatibility layer

### 4.5 — Language Support
- [ ] Japanese tokenizer integration (MeCab / Sudachi hook)
- [ ] Chinese tokenizer integration (jieba hook)
- [ ] Arabic, Korean, Thai — whitespace-free language detection + segmenter hooks
- [ ] Unicode normalisation pipeline (NFC, NFKC, case-folding) built-in

**Exit criteria:** LangChain and LlamaIndex integrations published and tested;
REST server Docker image available; Hugging Face Datasets ingestion works end-to-end.

---

## Phase 5 — Production Hardening & Enterprise Features  
**Goal:** Make FlashBM25 safe to run in production at scale — reliability, observability,
security, and governance.

### 5.1 — Observability
- [ ] Structured logging via `logging` module (log level, query time, corpus size)
- [ ] OpenTelemetry tracing: spans for index build, query, and serialisation
- [ ] Prometheus metrics endpoint for the server (`/metrics`)
- [ ] Query latency histogram, error rate, and throughput dashboards (Grafana template)

### 5.2 — Fault Tolerance & Safety
- [ ] Atomic index writes (write to temp file, then rename) to prevent corruption on crash
- [ ] Index checksumming (SHA-256 of serialised bytes) with verification on load
- [ ] Thread-safe concurrent reads + exclusive write lock (reader-writer lock in C++)
- [ ] Input validation and sanitisation: reject or truncate documents > configurable max length

### 5.3 — Versioning & Migration
- [ ] Index format versioning with forward-compatibility guarantees
- [ ] `flashbm25 migrate --from v1 --to v2 index.fbm25` CLI migration tool
- [ ] Deprecation policy: 2-minor-version grace period for any public API change

### 5.4 — Cloud & Distributed
- [ ] Sharded index: split corpus across N shards, merge scores at query time
- [ ] Distributed query coordinator (pure Python, no Hadoop/Spark dependency)
- [ ] AWS S3 / GCS / Azure Blob index backend: `BM25.load("s3://bucket/index.fbm25")`
- [ ] Kubernetes operator (Helm chart) for running `flashbm25-server` at scale

### 5.5 — Security
- [ ] Fuzz-test the C++ parser and tokeniser with libFuzzer / AFL++
- [ ] SAST scan in CI (CodeQL for C++, Bandit for Python)
- [ ] Signed release artefacts (Sigstore / cosign)
- [ ] SBOM (Software Bill of Materials) generated on every release

**Exit criteria:** Zero known CVEs; atomic index writes verified; distributed sharded queries
work across 4 nodes; OpenTelemetry traces visible in Jaeger.

---

## Phase 6 — Developer Experience & Community  
**Goal:** Make FlashBM25 the most pleasant IR library to use and contribute to.

### 6.1 — Ergonomics
- [ ] `BM25.from_texts(texts)`, `BM25.from_jsonl(path)`, `BM25.from_csv(path, column)` constructors
- [ ] Interactive `bm25.explain(query, doc_index)` — prints per-term IDF, TF, and score breakdown
- [ ] Rich `__repr__` showing corpus size, avg doc length, variant, and top-5 vocabulary terms
- [ ] Progress bar (tqdm integration) for large index builds

### 6.2 — Full Documentation Site
- [ ] Conceptual guides: "How BM25 works", "Choosing a variant", "Tuning k1 and b"
- [ ] Tutorials: semantic search hybrid, RAG pipeline, BEIR evaluation
- [ ] API reference (auto-generated from docstrings)
- [ ] Changelog maintained as `CHANGELOG.md` following Keep a Changelog format
- [ ] Translation: Simplified Chinese and Hindi (community-contributed)

### 6.3 — Tooling
- [ ] `flashbm25` CLI: `index`, `search`, `evaluate`, `migrate`, `bench` sub-commands
- [ ] VS Code extension: syntax highlighting and snippets for `.fbm25` index manifest files
- [ ] Jupyter notebook examples published in `examples/` directory
- [ ] `flashbm25.testing` module: fixtures and helpers for library consumers writing their own tests

### 6.4 — Community Infrastructure
- [ ] `CONTRIBUTING.md` with dev-environment setup, coding standards, and PR checklist
- [ ] Issue templates: bug report, feature request, performance regression
- [ ] GitHub Discussions enabled for Q&A
- [ ] Monthly release cadence; semver strictly enforced
- [ ] Discord / Slack community channel

**Exit criteria:** CLI fully functional; docs site with tutorials live; 5+ community contributors.

---

## Milestone Summary

| Phase | Name | Key Deliverable | Target |
|---|---|---|---|
| **1** | Solid Foundation | 3 BM25 variants + save/load + docs | v0.2.0 |
| **2** | Performance & Scale | AVX2 SIMD + batch API + NumPy output | v0.4.0 |
| **3** | Rich Feature Set | BM25F + boolean queries + eval metrics | v0.6.0 |
| **4** | Ecosystem | LangChain/LlamaIndex + REST server | v0.8.0 |
| **5** | Production Hardening | Distributed + observability + security | v1.0.0 |
| **6** | DX & Community | CLI + full docs + community infra | v1.2.0 |

---

## Guiding Principles

1. **C++ for speed, Python for ergonomics** — the hot path always lives in C++; the Python layer stays thin and expressive.
2. **Zero mandatory runtime dependencies** — every integration (HF, LangChain, etc.) is an optional extra.
3. **Correctness before optimisation** — new features ship with tests and docs before performance tuning.
4. **Semver strictly** — breaking changes only in major versions; deprecation warnings precede removal.
5. **Benchmarks are first-class citizens** — every performance claim is reproducible via `flashbm25 bench`.

---

*Last updated: April 2026 — maintained by [@subhobhai943](https://github.com/subhobhai943)*
