"""
build_dataset.py
================
Reproducible pipeline that generates the common processed dataset.

Author : Reetikesh Choudhury
Module : Dataset Build Pipeline (own component)

This script is the single entry point for creating the shared artifact that
all teammates (Srijan – BM25, Vidur – Semantic Search / RRF) consume.
Running it always produces an identical result because:

  • ``data_loader.load_dataset()`` is deterministic (fixed category order,
    consistent tarball iteration).
  • ``preprocessing.preprocess_documents()`` is deterministic (module-level
    PorterStemmer + frozenset stopwords, no randomness).
  • The 80/20 train/test split uses ``random_state=42`` and ``stratify=labels``
    so results are fully reproducible across runs and machines.

Data Flow (correct order – no leakage)
---------------------------------------
  1.  Load all documents from ``twenty+newsgroups.zip``
  2.  Basic text cleaning / preprocessing  → adds ``clean_text`` to each record
  3.  Create stratified 80/20 train / test split (``train_test_split``)
         • test_size  = 0.20
         • random_state = 42
         • stratify   = category labels
  4.  Assign ``split`` key (``"train"`` / ``"test"``) to each record
  5.  Serialise to parquet

The TF-IDF vectorizer (step 4 in the broader IR pipeline) is fitted ONLY
on training documents after this parquet is built.  The test documents are
kept entirely separate until evaluation time.

Output
------
``data/processed/processed_documents.parquet``
    Apache Parquet file readable by pandas, PyArrow, Polars, etc.
    Columns: doc_id, text, clean_text, label, category, split
    ~19,997 rows (no unstructured docs; split by sklearn).

``data/processed/metadata.json``
    Lightweight JSON committed to Git that describes the artifact so
    teammates can verify compatibility without loading the full parquet.

``data/processed/README.md``
    Human-readable guide for teammates to load the dataset.

Usage
-----
    python src/build_dataset.py                       # default settings
    python src/build_dataset.py --archive /path/to/twenty+newsgroups.zip
    python src/build_dataset.py --out-dir /custom/dir
    python src/build_dataset.py --verify-only         # reload & check, no rebuild
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Make src/ importable when run as a script.
# ---------------------------------------------------------------------------
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_loader import CATEGORIES, load_dataset
from preprocessing import preprocess_documents
from utils import (
    ARCHIVE_PATH,
    DATA_README_FILE,
    METADATA_FILE,
    PARQUET_FILE,
    PREPROCESSING_VERSION,
    PROCESSED_COLUMNS,
    PROCESSED_DATA_DIR,
)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def build_processed_dataset(
    archive_path: Optional[str | Path] = None,
    *,
    out_dir: Optional[str | Path] = None,
    verbose: bool = True,
) -> Path:
    """
    Run the full pipeline and write the processed parquet file.

    Pipeline
    --------
    1. Load all documents from the archive (no split yet).
    2. Preprocess all documents (adds ``clean_text``).
    3. Create stratified 80/20 split with ``train_test_split``.
    4. Assign ``split`` column (``"train"`` / ``"test"``).
    5. Serialise to parquet + write metadata.json + README.md.

    Parameters
    ----------
    archive_path : str or Path, optional
        Path to ``twenty+newsgroups.zip``.
        Defaults to ``utils.ARCHIVE_PATH``.
    out_dir : str or Path, optional
        Directory to write output files.
        Defaults to ``utils.PROCESSED_DATA_DIR``.
    verbose : bool, default True
        Print progress messages.

    Returns
    -------
    Path
        Absolute path to the written parquet file.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    archive = Path(archive_path) if archive_path else ARCHIVE_PATH
    out = Path(out_dir) if out_dir else PROCESSED_DATA_DIR
    out.mkdir(parents=True, exist_ok=True)

    parquet_out  = out / "processed_documents.parquet"
    metadata_out = out / "metadata.json"
    readme_out   = out / "README.md"

    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Step 1 – Load raw corpus (all documents, no split key yet)
    # ------------------------------------------------------------------
    if verbose:
        print(f"[1/4] Loading corpus from {archive} ...", flush=True)

    docs = load_dataset(
        archive_path=archive,
        strip_headers=True,   # remove From:/Subject:/etc. before preprocessing
    )
    n_loaded = len(docs)
    if verbose:
        from collections import Counter
        cat_counts = Counter(d["category"] for d in docs)
        print(f"     Loaded {n_loaded:,} documents across {len(cat_counts)} categories")

    # ------------------------------------------------------------------
    # Step 2 – Preprocess ALL documents (clean_text added)
    # NOTE: No vectorizer is fitted here.  Preprocessing is purely
    # token-level (header removal, lowercase, stopwords, stemming) and
    # does NOT look at corpus-level statistics, so it is safe to apply
    # to all documents before splitting.
    # ------------------------------------------------------------------
    if verbose:
        print("[2/4] Preprocessing ...", flush=True)

    processed = preprocess_documents(docs, verbose=verbose)

    # Sanity check: doc_ids must be unique and contiguous 0-based
    ids = [d["doc_id"] for d in processed]
    assert len(ids) == len(set(ids)), "doc_ids are not unique after preprocessing"
    assert sorted(ids) == list(range(len(ids))), "doc_ids are not contiguous 0-based"

    # ------------------------------------------------------------------
    # Step 3 – Create stratified 80/20 train/test split
    # This happens AFTER preprocessing but BEFORE any TF-IDF fitting.
    # The vectorizer will be fitted only on the train subset (in tfidf.py).
    # ------------------------------------------------------------------
    if verbose:
        print("[3/4] Creating stratified 80/20 train/test split ...", flush=True)

    labels = [d["label"] for d in processed]

    train_docs, test_docs = train_test_split(
        processed,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )

    n_train = len(train_docs)
    n_test  = len(test_docs)

    if verbose:
        print(f"     Total documents   : {n_loaded:,}")
        print(f"     Training documents: {n_train:,}  "
              f"({100 * n_train / n_loaded:.1f}%)")
        print(f"     Testing documents : {n_test:,}  "
              f"({100 * n_test / n_loaded:.1f}%)")

        # Print category distribution to verify stratification
        from collections import Counter
        train_cats = Counter(d["category"] for d in train_docs)
        test_cats  = Counter(d["category"] for d in test_docs)
        print()
        print(f"     {'Category':<35} {'Train':>6}  {'Test':>6}  {'Total':>6}")
        print(f"     {'-'*35} {'-'*6}  {'-'*6}  {'-'*6}")
        for cat in CATEGORIES:
            tr = train_cats.get(cat, 0)
            te = test_cats.get(cat, 0)
            print(f"     {cat:<35} {tr:>6}  {te:>6}  {tr+te:>6}")

    # Assign the ``split`` key and re-index doc_ids contiguously
    # (train first, then test — stable ordering for downstream consumers)
    for doc in train_docs:
        doc["split"] = "train"
    for doc in test_docs:
        doc["split"] = "test"

    # Merge back and re-index: train docs first, then test docs
    all_docs = train_docs + test_docs
    for new_id, doc in enumerate(all_docs):
        doc["doc_id"] = new_id

    # ------------------------------------------------------------------
    # Step 4 – Serialise to parquet
    # ------------------------------------------------------------------
    if verbose:
        print(f"[4/4] Writing parquet -> {parquet_out} ...", flush=True)

    df = pd.DataFrame(all_docs, columns=PROCESSED_COLUMNS)

    # Enforce dtypes for space-efficiency and portability
    df["doc_id"]     = df["doc_id"].astype("int32")
    df["label"]      = df["label"].astype("int8")
    df["category"]   = df["category"].astype("category")
    df["split"]      = df["split"].astype("category")
    df["text"]       = df["text"].astype("string")
    df["clean_text"] = df["clean_text"].astype("string")

    df.to_parquet(parquet_out, index=False, engine="pyarrow", compression="snappy")

    elapsed = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Write metadata.json  (committed to Git)
    # ------------------------------------------------------------------
    empty_clean = int((df["clean_text"].fillna("").str.strip() == "").sum())
    split_info = (
        df.groupby("split", observed=True)["doc_id"]
        .count()
        .to_dict()
    )
    cat_info = (
        df.groupby("category", observed=True)["doc_id"]
        .count()
        .to_dict()
    )
    avg_clean_tokens = float(
        df["clean_text"].fillna("").apply(lambda t: len(t.split()) if t else 0).mean()
    )

    metadata = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "archive_used": str(archive),
        "num_documents": int(len(df)),
        "num_categories": int(df["category"].nunique()),
        "split_method": "sklearn.model_selection.train_test_split",
        "split_params": {
            "test_size": 0.20,
            "random_state": 42,
            "stratify": "label",
        },
        "empty_clean_text_count": empty_clean,
        "avg_clean_tokens": round(avg_clean_tokens, 2),
        "columns": PROCESSED_COLUMNS,
        "dtypes": {
            "doc_id":     "int32",
            "text":       "string",
            "clean_text": "string",
            "label":      "int8",
            "category":   "category",
            "split":      "category",
        },
        "split_counts": {k: int(v) for k, v in split_info.items()},
        "category_counts": {k: int(v) for k, v in cat_info.items()},
        "categories": CATEGORIES,
        "parquet_file": parquet_out.name,
        "parquet_size_bytes": parquet_out.stat().st_size,
        "build_duration_s": round(elapsed, 2),
    }

    with open(metadata_out, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Write README.md  (committed to Git)
    # ------------------------------------------------------------------
    _write_data_readme(readme_out, metadata)

    if verbose:
        print()
        print("=" * 60)
        print("  BUILD COMPLETE")
        print(f"  Total documents : {len(df):,}")
        print(f"  Categories      : {df['category'].nunique()}")
        train_n = split_info.get("train", 0)
        test_n  = split_info.get("test",  0)
        print(f"  Train           : {train_n:,}  "
              f"({100 * train_n / len(df):.1f}%)")
        print(f"  Test            : {test_n:,}  "
              f"({100 * test_n / len(df):.1f}%)")
        print(f"  Split method    : stratified train_test_split "
              f"(test_size=0.20, random_state=42)")
        print(f"  Empty clean     : {empty_clean:,}")
        print(f"  Avg tokens      : {avg_clean_tokens:.1f}")
        print(f"  Parquet         : {parquet_out}  "
              f"({parquet_out.stat().st_size / 1_048_576:.1f} MB)")
        print(f"  Metadata        : {metadata_out}")
        print(f"  Elapsed         : {elapsed:.1f}s")
        print("=" * 60)
        print()
        print("IMPORTANT – Data Leakage Note:")
        print("  The TF-IDF vectorizer must be fitted ONLY on training")
        print("  documents (split == 'train').  Use the train subset when")
        print("  calling build_tfidf() in tfidf.py:")
        print("    df = load_processed_dataset()")
        print("    train_df = df[df['split'] == 'train']")
        print("    searcher = build_tfidf(train_df)")

    return parquet_out


# ---------------------------------------------------------------------------
# Verification helper (usable by teammates without rebuilding)
# ---------------------------------------------------------------------------

def verify_processed_dataset(
    parquet_path: Optional[str | Path] = None,
    *,
    verbose: bool = True,
) -> dict:
    """
    Load the parquet file and run integrity checks.

    Returns a dict with check results.  Raises ``AssertionError`` if any
    critical invariant is violated.

    Teammates can call this to confirm their environment can read the file::

        python -c "
        import sys; sys.path.insert(0, 'src')
        from build_dataset import verify_processed_dataset
        verify_processed_dataset()
        "
    """
    import pandas as pd

    if parquet_path is not None:
        path = Path(parquet_path)
    else:
        path = PARQUET_FILE

    if not path.exists():
        raise FileNotFoundError(
            f"Parquet file not found at '{path}'.\n"
            "Run 'python src/build_dataset.py' first."
        )

    if verbose:
        print(f"Verifying {path} ...")

    df = pd.read_parquet(path, engine="pyarrow")
    results: dict = {}

    # Check 1 – required columns
    required = set(PROCESSED_COLUMNS)
    missing = required - set(df.columns)
    results["columns_ok"] = len(missing) == 0
    assert not missing, f"Missing columns: {missing}"

    # Check 2 – row count > 0
    results["has_rows"] = len(df) > 0
    assert len(df) > 0, "Empty parquet file"

    # Check 3 – doc_id unique and contiguous 0-based
    ids = df["doc_id"].tolist()
    results["doc_ids_unique"] = len(ids) == len(set(ids))
    results["doc_ids_contiguous"] = sorted(ids) == list(range(len(ids)))
    assert results["doc_ids_unique"],    "doc_ids are not unique"
    assert results["doc_ids_contiguous"], "doc_ids are not contiguous 0-based"

    # Check 4 – no nulls in key columns
    for col in ["doc_id", "label", "category", "split"]:
        null_n = int(df[col].isnull().sum())
        results[f"no_null_{col}"] = null_n == 0
        assert null_n == 0, f"Column '{col}' has {null_n} null values"

    # Check 5 – all 20 categories present
    cats = set(df["category"].unique())
    results["all_categories_present"] = cats == set(CATEGORIES)
    assert cats == set(CATEGORIES), f"Missing categories: {set(CATEGORIES) - cats}"

    # Check 6 – labels in [0, 19]
    label_min = int(df["label"].min())
    label_max = int(df["label"].max())
    results["labels_in_range"] = label_min == 0 and label_max == 19
    assert label_min == 0 and label_max == 19, \
        f"Labels out of range: min={label_min}, max={label_max}"

    # Check 7 – split values are only "train" / "test"
    split_vals = set(df["split"].unique())
    results["valid_splits"] = split_vals <= {"train", "test"}
    assert split_vals <= {"train", "test"}, f"Unexpected split values: {split_vals}"

    # Check 8 – 80/20 ratio (allow ±2% tolerance)
    n = len(df)
    train_n = int((df["split"] == "train").sum())
    test_n  = int((df["split"] == "test").sum())
    train_pct = train_n / n
    test_pct  = test_n  / n
    results["train_pct"] = round(train_pct * 100, 2)
    results["test_pct"]  = round(test_pct  * 100, 2)
    assert 0.78 <= train_pct <= 0.82, \
        f"Training percentage {train_pct:.1%} outside expected 78–82% band"
    assert 0.18 <= test_pct  <= 0.22, \
        f"Testing percentage {test_pct:.1%} outside expected 18–22% band"

    if verbose:
        print(f"  Total documents : {n:,}")
        print(f"  Train           : {train_n:,}  ({train_pct:.1%})")
        print(f"  Test            : {test_n:,}  ({test_pct:.1%})")
        print(f"  Categories      : {len(cats)}")
        print("  All checks PASSED.")

    results["total"] = n
    results["train_n"] = train_n
    results["test_n"]  = test_n
    return results


# ---------------------------------------------------------------------------
# README writer
# ---------------------------------------------------------------------------

def _write_data_readme(path: Path, metadata: dict) -> None:
    """Write a human-readable README.md for the data/processed/ directory."""
    train_n = metadata["split_counts"].get("train", 0)
    test_n  = metadata["split_counts"].get("test",  0)
    total   = metadata["num_documents"]

    lines = [
        "# Processed Dataset",
        "",
        "This directory contains the shared processed dataset for the",
        "20 Newsgroups IR project.",
        "",
        "## Files",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `processed_documents.parquet` | Full preprocessed corpus (Snappy-compressed Parquet) |",
        "| `metadata.json` | Dataset statistics and build parameters |",
        "| `README.md` | This file |",
        "",
        "## Dataset Statistics",
        "",
        f"| Stat | Value |",
        f"|------|-------|",
        f"| Total documents | {total:,} |",
        f"| Training documents | {train_n:,} (~80%) |",
        f"| Testing documents | {test_n:,} (~20%) |",
        f"| Categories | {metadata['num_categories']} |",
        f"| Avg clean tokens | {metadata['avg_clean_tokens']} |",
        f"| Generated at | {metadata['generated_at']} |",
        "",
        "## Split Method",
        "",
        "The train/test split is created by `build_dataset.py` using:",
        "```python",
        "from sklearn.model_selection import train_test_split",
        "train_docs, test_docs = train_test_split(",
        "    processed,",
        "    test_size=0.20,",
        "    random_state=42,",
        "    stratify=labels,",
        ")",
        "```",
        "",
        "## Loading the Dataset",
        "",
        "```python",
        "import sys",
        "sys.path.insert(0, 'src')",
        "from data_loader import load_processed_dataset",
        "",
        "df = load_processed_dataset()",
        "train_df = df[df['split'] == 'train']",
        "test_df  = df[df['split'] == 'test']",
        "```",
        "",
        "## Columns",
        "",
        "| Column | Type | Description |",
        "|--------|------|-------------|",
        "| `doc_id` | int32 | Globally unique 0-based document ID |",
        "| `text` | string | Raw document body (headers stripped) |",
        "| `clean_text` | string | Preprocessed text (stemmed, stopwords removed) |",
        "| `label` | int8 | Integer category label (0–19) |",
        "| `category` | category | Newsgroup name |",
        "| `split` | category | `\"train\"` or `\"test\"` |",
        "",
        "## Important: No Data Leakage",
        "",
        "The TF-IDF vectorizer (and any other data-dependent feature extractor)",
        "**must be fitted only on training documents**.  Always filter by split",
        "before building an index:",
        "",
        "```python",
        "from tfidf import build_tfidf",
        "train_df = df[df['split'] == 'train']",
        "searcher = build_tfidf(train_df)  # fit on train only",
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the processed 20 Newsgroups dataset parquet artifact.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--archive",
        default=None,
        help="Path to twenty+newsgroups.zip.  "
             "Defaults to NEWSGROUPS_ARCHIVE_PATH env var or "
             "20-newsgroups-ir/twenty+newsgroups.zip.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for parquet + metadata.  "
             "Defaults to data/processed/.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip the build and only verify the existing parquet file.",
    )
    args = parser.parse_args()

    if args.verify_only:
        verify_processed_dataset(verbose=True)
    else:
        build_processed_dataset(
            archive_path=args.archive,
            out_dir=args.out_dir,
            verbose=True,
        )


if __name__ == "__main__":
    main()
