"""
Benchmark FlashBM25 against rank_bm25 (pure Python).
Run with:  python benchmarks/bench_flashbm25.py
"""
import time
import random
import string


def random_doc(word_count: int = 60) -> str:
    words = [
        "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 10)))
        for _ in range(word_count)
    ]
    return " ".join(words)


def build_corpus(n: int = 10_000, wc: int = 60):
    return [random_doc(wc) for _ in range(n)]


def timeit(fn, label: str, runs: int = 5) -> float:
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times)
    print(f"  {label:<30} avg={avg * 1000:.2f} ms  (over {runs} runs)")
    return avg


def main():
    random.seed(42)
    N = 10_000
    print(f"\n=== FlashBM25 Benchmark  (corpus={N} docs) ===\n")

    corpus = build_corpus(N)
    query  = "quick brown fox machine learning"

    results = {}

    # ── FlashBM25 ────────────────────────────────────────────────────────────
    try:
        from flashbm25 import BM25 as FlashBM25

        def flash_index():
            FlashBM25(corpus)

        flash_bm25 = FlashBM25(corpus)

        def flash_query():
            flash_bm25.get_top_n(query, n=10)

        print("FlashBM25 (C++):")
        results["flash_index"] = timeit(flash_index, "index build")
        results["flash_query"] = timeit(flash_query, "top-10 query")
    except ImportError:
        print("  FlashBM25 not installed — skipping.\n")

    # ── rank_bm25 (pure Python) ───────────────────────────────────────────────
    try:
        from rank_bm25 import BM25Okapi

        tokenised = [d.split() for d in corpus]

        def rank_index():
            BM25Okapi(tokenised)

        rank = BM25Okapi(tokenised)

        def rank_query():
            q      = query.split()
            scores = rank.get_scores(q)
            return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]

        print("\nrank_bm25 (pure Python):")
        results["rank_index"] = timeit(rank_index, "index build")
        results["rank_query"] = timeit(rank_query, "top-10 query")
    except ImportError:
        print("\n  rank_bm25 not installed — pip install rank_bm25")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Speedup summary ─────────────────────────────────────")
    if "flash_index" in results and "rank_index" in results:
        print(f"  Index build  speedup: {results['rank_index'] / results['flash_index']:.1f}x")
    if "flash_query" in results and "rank_query" in results:
        print(f"  Query        speedup: {results['rank_query'] / results['flash_query']:.1f}x")
    print()


if __name__ == "__main__":
    main()
