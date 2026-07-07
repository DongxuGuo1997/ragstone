#!/usr/bin/env python3
"""Corpus-scale benchmark: where do the shipped defaults stop holding?

Every quality number in this repo was measured at 63–231 chunks. This
benchmark measures the *code's* scaling behavior — index build time,
ensemble query latency, memory — at 1k/10k/100k chunks, using synthetic
1536-dim unit vectors so no embedding API is involved: embedding cost is
linear and known (~$0.02 per 1M tokens); the unknowns are FAISS's flat
index, BM25 tokenization, the ensemble merge, and embedded Qdrant's
full-scan local mode. Those are what this measures.

    python evals/bench_scale.py --sizes 1000 10000 100000
    python evals/bench_scale.py --backends faiss          # skip qdrant

Results feed Experiment 16 in EXPERIMENTS.md.
"""

import argparse
import hashlib
import resource
import statistics
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402

DIM = 1536  # text-embedding-3-small's dimensionality
QUERIES = 50


class SyntheticEmbeddings(Embeddings):
    """Deterministic pseudo-random unit vectors, seeded per text.

    Realistic in dimensionality and distribution shape, free of API
    calls, and stable across runs — hash-seeded, no RNG state.
    """

    def _vector(self, text: str) -> List[float]:
        seed = hashlib.sha256(text.encode()).digest()
        raw: List[float] = []
        counter = 0
        while len(raw) < DIM:
            block = hashlib.sha256(seed + struct.pack("I", counter)).digest()
            raw.extend(b / 255.0 - 0.5 for b in block)
            counter += 1
        vec = raw[:DIM]
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vector(text)


def make_corpus(n_chunks: int) -> List[Document]:
    """Synthetic chunks shaped like the real ones (~500 chars, distinct)."""
    body = (
        "covers routine care of the panel and its companion inverter. "
        "The rated output is {i} watts with a conversion efficiency of "
        "twenty two percent, and the expected degradation is half a "
        "percent per year. Clean the panels every six months using a "
        "soft brush and deionized water; torque the mounting bolts to "
        "the specified value and inspect the grounding strap annually. "
    )
    return [
        Document(
            page_content=f"Product {i} maintenance guide " + body.format(i=i),
            metadata={"source": f"doc_{i % (max(n_chunks // 4, 1))}.md"},
        )
        for i in range(n_chunks)
    ]


def rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss units differ: bytes on macOS, kilobytes on Linux.
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return usage / divisor


def bench_backend(backend: str, docs: List[Document], embeddings) -> dict:
    from ragstone.config.settings import get_config
    from ragstone.rag.vector_db import create_vector_store_proxy

    get_config().database.embed_cache_enabled = False  # measure raw path

    kwargs = {}
    if backend == "qdrant":
        kwargs["path"] = tempfile.mkdtemp(prefix="bench_qdrant_")
    proxy = create_vector_store_proxy(backend, **kwargs)

    build_start = time.perf_counter()
    proxy.create_db(docs=docs, embeddings=embeddings)
    build_s = time.perf_counter() - build_start

    # Query through the retriever interface the pipeline actually uses.
    retriever = proxy.db.as_retriever(search_kwargs={"k": 4})
    latencies = []
    for i in range(QUERIES):
        query = f"what is the rated output of product {i * 7}?"
        start = time.perf_counter()
        retriever.invoke(query)
        latencies.append((time.perf_counter() - start) * 1000)

    proxy.cleanup()
    return {
        "build_s": round(build_s, 2),
        "query_p50_ms": round(statistics.median(latencies), 1),
        "query_p95_ms": round(statistics.quantiles(latencies, n=20)[18], 1),
        "rss_mb": round(rss_mb(), 0),
    }


def bench_bm25(docs: List[Document]) -> dict:
    """BM25 build + query — the ensemble's lexical half."""
    from langchain_community.retrievers import BM25Retriever

    build_start = time.perf_counter()
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = 4
    build_s = time.perf_counter() - build_start

    latencies = []
    for i in range(QUERIES):
        start = time.perf_counter()
        retriever.invoke(f"rated output of product {i * 7}")
        latencies.append((time.perf_counter() - start) * 1000)
    return {
        "build_s": round(build_s, 2),
        "query_p50_ms": round(statistics.median(latencies), 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[1_000, 10_000, 100_000]
    )
    parser.add_argument("--backends", nargs="+", default=["faiss", "qdrant"])
    args = parser.parse_args()

    embeddings = SyntheticEmbeddings()
    print("| chunks | backend | build (s) | query p50 (ms) | p95 (ms) | RSS (MB) |")
    print("|---|---|---|---|---|---|")
    for size in args.sizes:
        docs = make_corpus(size)
        for backend in args.backends:
            result = bench_backend(backend, docs, embeddings)
            print(
                f"| {size:,} | {backend} | {result['build_s']} "
                f"| {result['query_p50_ms']} | {result['query_p95_ms']} "
                f"| {result['rss_mb']:.0f} |"
            )
        bm25 = bench_bm25(docs)
        print(
            f"| {size:,} | bm25 | {bm25['build_s']} "
            f"| {bm25['query_p50_ms']} | — | — |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
