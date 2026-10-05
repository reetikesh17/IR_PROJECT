"""
demo_tfidf.py
=============
Demonstration script for TF-IDF retrieval system on the 20 Newsgroups dataset.

Author : Reetikesh Choudhury
Module : TF-IDF Retrieval (own component)

Usage:
    python notebooks/demo_tfidf.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Allow imports from src/
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tfidf import build_tfidf
from utils import PARQUET_FILE


def run_demo() -> None:
    print("=" * 70)
    print("  20 NEWSGROUPS IR — TF-IDF RETRIEVAL SYSTEM DEMO")
    print("=" * 70)

    # 1. Load common processed dataset
    if not PARQUET_FILE.exists():
        print(f"Error: Processed dataset not found at '{PARQUET_FILE}'.")
        print("Please run: python src/build_dataset.py")
        sys.exit(1)

    print(f"\n[1/3] Loading processed dataset from: {PARQUET_FILE.name}")
    t0 = time.perf_counter()
    df = pd.read_parquet(PARQUET_FILE)
    load_time = time.perf_counter() - t0
    print(f"      Loaded {len(df):,} documents in {load_time:.2f}s")

    # 2. Build TF-IDF search index
    print("\n[2/3] Building TF-IDF index over corpus ...")
    t0 = time.perf_counter()
    searcher = build_tfidf(df)
    build_time = time.perf_counter() - t0
    print(f"      Index built in {build_time:.2f}s  "
          f"(Vocab size: {searcher.vocab_size:,} terms)")

    # 3. Demonstration queries from various 20 Newsgroups topics
    sample_queries = [
        "space shuttle launch moon orbit",
        "3d graphics rendering algorithm ray tracing",
        "public key encryption pgp cryptography security",
        "motorcycle helmet riding safety gears",
        "christian church faith jesus religion",
        "windows driver installation system setup",
        "baseball pitcher home run inning game",
    ]

    print("\n[3/3] Executing sample search queries:")
    print("=" * 70)

    for query_idx, query in enumerate(sample_queries, start=1):
        t_start = time.perf_counter()
        results = searcher.search(query, top_k=5)
        search_time_ms = (time.perf_counter() - t_start) * 1000

        print(f"\nQuery #{query_idx}: \"{query}\"  [{search_time_ms:.2f} ms]")
        print("-" * 70)
        print(f"{'Rank':<6} {'doc_id':<8} {'Score':<10} {'Category':<28} {'Snippet'}")
        print("-" * 70)

        if not results:
            print("  (No matching documents found)")
            continue

        for res in results:
            doc_id = res["doc_id"]
            rank = res["rank"]
            score = res["score"]
            doc = searcher.get_document(doc_id)
            category = doc["category"] if doc else "N/A"
            raw_text = doc["text"] if doc else ""
            snippet = raw_text.replace("\n", " ").strip()[:45] + "..." if raw_text else ""

            print(f"{rank:<6} {doc_id:<8} {score:<10.4f} {category:<28} {snippet}")

    print("\n" + "=" * 70)
    print("  DEMO COMPLETE [OK]")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
