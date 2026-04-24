"""
Benchmark FlashBM25 against optional lexical-search competitors.

The competitor packages are intentionally optional. Install the ``bench`` extra
or the individual packages you want to compare, then run:

    python benchmarks/bench_competitors.py --docs 10000 --runs 5

Set ``ELASTICSEARCH_URL`` to include Elasticsearch in the run.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import string
import time
from typing import Callable, Dict, List, Optional


QUERY = "quick brown fox machine learning"
QUERY_TERMS = QUERY.split()


def random_word(rng: random.Random) -> str:
    return "".join(rng.choices(string.ascii_lowercase, k=rng.randint(4, 10)))


def build_corpus(doc_count: int, words_per_doc: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    vocabulary = [random_word(rng) for _ in range(2048)]
    corpus = []

    for doc_id in range(doc_count):
        words = [rng.choice(vocabulary) for _ in range(words_per_doc)]
        if doc_id % 3 == 0:
            words.extend(QUERY_TERMS[:2])
        if doc_id % 11 == 0:
            words.extend(QUERY_TERMS[2:])
        rng.shuffle(words)
        corpus.append(" ".join(words))

    return corpus


def measure(label: str, fn: Callable[[], object], runs: int) -> float:
    durations = []
    for _ in range(runs):
        started = time.perf_counter()
        fn()
        durations.append(time.perf_counter() - started)

    avg = sum(durations) / len(durations)
    print(f"  {label:<24} {avg * 1000:>10.3f} ms")
    return avg


def skip(name: str, reason: str) -> None:
    print(f"\n{name}: skipped ({reason})")


def bench_flashbm25(corpus: List[str], query: str, runs: int) -> Dict[str, float]:
    from flashbm25 import BM25

    try:
        from flashbm25._flashbm25 import _cpu_backend

        backend = _cpu_backend()
    except Exception:  # pragma: no cover - best-effort introspection for benchmarks
        backend = "unknown"

    print(f"\nFlashBM25 ({backend}):")
    results = {"index": measure("index build", lambda: BM25(corpus), runs)}
    index = BM25(corpus)
    results["query"] = measure("top-10 query", lambda: index.get_top_n(query, n=10), runs)
    return results


def bench_rank_bm25(corpus: List[str], query: str, runs: int) -> Optional[Dict[str, float]]:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        skip("rank_bm25", "install rank_bm25")
        return None

    tokenized = [doc.split() for doc in corpus]

    print("\nrank_bm25:")
    results = {"index": measure("index build", lambda: BM25Okapi(tokenized), runs)}
    index = BM25Okapi(tokenized)
    query_terms = query.split()

    def search() -> List[int]:
        scores = index.get_scores(query_terms)
        return sorted(range(len(scores)), key=lambda doc_id: scores[doc_id], reverse=True)[:10]

    results["query"] = measure("top-10 query", search, runs)
    return results


def bench_whoosh(corpus: List[str], query: str, runs: int) -> Optional[Dict[str, float]]:
    try:
        from whoosh.fields import ID, Schema, TEXT
        from whoosh.filedb.filestore import RamStorage
        from whoosh.qparser import QueryParser
    except ImportError:
        skip("Whoosh", "install whoosh")
        return None

    schema = Schema(doc_id=ID(stored=True), body=TEXT)

    def build_index():
        storage = RamStorage()
        index = storage.create_index(schema)
        writer = index.writer()
        for doc_id, document in enumerate(corpus):
            writer.add_document(doc_id=str(doc_id), body=document)
        writer.commit()
        return index

    print("\nWhoosh:")
    results = {"index": measure("index build", build_index, runs)}
    index = build_index()
    parser = QueryParser("body", schema=index.schema)
    parsed_query = parser.parse(query)

    def search() -> List[str]:
        with index.searcher() as searcher:
            return [hit["doc_id"] for hit in searcher.search(parsed_query, limit=10)]

    results["query"] = measure("top-10 query", search, runs)
    return results


def bench_tantivy(corpus: List[str], query: str, runs: int) -> Optional[Dict[str, float]]:
    try:
        import tantivy
    except ImportError:
        skip("Tantivy", "install tantivy")
        return None

    try:
        schema_builder = tantivy.SchemaBuilder()
        schema_builder.add_text_field("body", stored=True)
        schema = schema_builder.build()
    except Exception as exc:
        skip("Tantivy", f"unsupported Python binding API: {exc}")
        return None

    def build_index():
        index = tantivy.Index(schema)
        writer = index.writer()
        for document in corpus:
            writer.add_document(tantivy.Document(body=document))
        writer.commit()
        index.reload()
        return index

    print("\nTantivy:")
    try:
        results = {"index": measure("index build", build_index, runs)}
        index = build_index()
        query_parser = tantivy.QueryParser.for_index(index, ["body"])
        parsed_query = query_parser.parse_query(query)
        searcher = index.searcher()

        def search():
            return searcher.search(parsed_query, 10).hits

        results["query"] = measure("top-10 query", search, runs)
        return results
    except Exception as exc:
        skip("Tantivy", f"benchmark failed: {exc}")
        return None


def bench_elasticsearch(corpus: List[str], query: str, runs: int) -> Optional[Dict[str, float]]:
    url = os.environ.get("ELASTICSEARCH_URL")
    if not url:
        skip("Elasticsearch", "set ELASTICSEARCH_URL")
        return None

    try:
        from elasticsearch import Elasticsearch
    except ImportError:
        skip("Elasticsearch", "install elasticsearch")
        return None

    client = Elasticsearch(url)
    index_name = "flashbm25-phase-2-1-bench"

    def build_index() -> str:
        if client.indices.exists(index=index_name):
            client.indices.delete(index=index_name)
        client.indices.create(
            index=index_name,
            mappings={"properties": {"body": {"type": "text"}}},
        )
        operations = []
        for doc_id, document in enumerate(corpus):
            operations.append({"index": {"_index": index_name, "_id": str(doc_id)}})
            operations.append({"body": document})
        client.bulk(operations=operations, refresh=True)
        return index_name

    print("\nElasticsearch:")
    try:
        results = {"index": measure("index build", build_index, runs)}
        build_index()

        def search():
            return client.search(
                index=index_name,
                query={"match": {"body": query}},
                size=10,
            )

        results["query"] = measure("top-10 query", search, runs)
        return results
    except Exception as exc:
        skip("Elasticsearch", f"benchmark failed: {exc}")
        return None
    finally:
        try:
            if client.indices.exists(index=index_name):
                client.indices.delete(index=index_name)
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=int, default=10_000)
    parser.add_argument("--words", type=int, default=60)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true", help="print machine-readable results")
    args = parser.parse_args()

    corpus = build_corpus(args.docs, args.words, args.seed)
    results = {
        "flashbm25": bench_flashbm25(corpus, QUERY, args.runs),
        "rank_bm25": bench_rank_bm25(corpus, QUERY, args.runs),
        "whoosh": bench_whoosh(corpus, QUERY, args.runs),
        "tantivy": bench_tantivy(corpus, QUERY, args.runs),
        "elasticsearch": bench_elasticsearch(corpus, QUERY, args.runs),
    }

    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
