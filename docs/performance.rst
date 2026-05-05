Performance
===========

SIMD Scoring
------------

FlashBM25 keeps the public index format portable while using CPU-specific
scoring kernels at query time. During index construction, each term's sparse
``unordered_map`` postings are mirrored into a sorted contiguous postings list.
The serialized format still writes the canonical map, so existing indexes remain
compatible.

At runtime the C++ core selects the fastest available backend:

* ``avx2`` on x86/x86_64 CPUs with AVX2 and OS YMM register support.
* ``sse4.2`` on x86/x86_64 CPUs without AVX2 but with SSE4.2.
* ``neon`` on ARM64, including Apple Silicon.
* ``scalar`` everywhere else.

The SIMD kernels vectorize the BM25 term-frequency normalization for Okapi
BM25, BM25+, BM25L, and BM25Adpt. Scores are accumulated in document order and
exposed to Python as float32 NumPy arrays; top-n results are exposed as
structured ``(score, doc_id)`` NumPy records.

Hotspot Work
------------

The Phase 2.1 CPU work targets the three hottest query-time costs:

* Sparse postings traversal now walks contiguous sorted postings instead of
  iterating hash-table buckets.
* Duplicate query terms are aggregated into one weighted term contribution, so
  repeated terms do not rescan the same postings list.
* The per-posting BM25 arithmetic is dispatched to AVX2, SSE4.2, or NEON
  kernels when the host CPU supports them.

Profiling
---------

Use release builds when profiling. On Linux, ``perf`` can capture instruction
and cache behavior around the benchmark harness:

.. code-block:: bash

   perf stat -d python benchmarks/bench_competitors.py --docs 100000 --runs 10
   perf record -g python benchmarks/bench_competitors.py --docs 100000 --runs 10
   perf report

On Intel platforms, VTune's Hotspots analysis is useful for checking whether the
active backend spends most of its time in the SIMD scoring kernels:

.. code-block:: bash

   vtune -collect hotspots -- python benchmarks/bench_competitors.py --docs 100000

Benchmarks
----------

The optional competitor benchmark compares FlashBM25 with rank_bm25, Whoosh,
Tantivy, and Elasticsearch when those packages or services are available:

.. code-block:: bash

   python benchmarks/bench_competitors.py --docs 10000 --runs 5

Set ``ELASTICSEARCH_URL`` to include Elasticsearch. Missing optional packages are
reported as skipped rather than treated as benchmark failures.
