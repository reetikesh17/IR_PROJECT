"""
build_dataset.py
================
Reproducible pipeline that generates the common processed dataset.

Author : Reetikesh Choudhury
Module : Dataset Build Pipeline (own component)

This script is the single entry point for creating the shared artifact that
all teammates (Srijan – BM25, Vidur – Semantic Search / RRF) consume.
Running it always produces an identical result on the same machine because:

  • ``data_loader.load_dataset()`` is deterministic (sorted category order,
    fixed regex parsing).
  • ``preprocessing.preprocess_documents()`` is deterministic (module-level
    PorterStemmer + frozenset stopwords, no randomness).
  • ``doc_id`` values are re-indexed to a contiguous 0-based range that is
    stable across categories within a single run, and the same ordering is
    reproduced on every run because the category list order never changes.

Output
------
``data/processed/processed_documents.parquet``
    Apache Parquet file readable by pandas, PyArrow, Polars, etc.
    Columns: doc_id, text, clean_text, label, category, split
    ~37 500 rows (train + test; unstructured excluded by default).

``data/processed/metadata.json``
    Lightweight JSON committed to Git that describes the artifact so
    teammates can verify compatibility without loading the full parquet.

``data/processed/README.md``
    Human-readable guide for teammates to load the dataset.

Usage
-----
    python src/build_dataset.py                       # default settings
    python src/build_dataset.py --archive /path/to/archive.zip
    python src/build_dataset.py --include-unstructured
    python src/build_dataset.py --splits train        # train only
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
    splits: Optional[List[str]] = None,
    include_unstructured: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Run the full pipeline and write the processed parquet file.

    Parameters
    ----------
    archive_path : str or Path, optional
        Path to ``archive.zip``. Defaults to ``utils.ARCHIVE_PATH``.
    out_dir : str or Path, optional
        Directory to write output files. Defaults to ``utils.PROCESSED_DATA_DIR``.
    splits : list of str, optional
        Which splits to include. Defaults to ``["train", "test"]``.
    include_unstructured : bool, default False
        Include the ~1 600 header-less documents.
    verbose : bool, default True
        Print progress messages.

    Returns
    -------
    Path
        Absolute path to the written parquet file.
    """
    import pandas as pd

    archive = Path(archive_path) if archive_path else ARCHIVE_PATH
    out = Path(out_dir) if out_dir else PROCESSED_DATA_DIR
    out.mkdir(parents=True, exist_ok=True)

    parquet_out  = out / "processed_documents.parquet"
    metadata_out = out / "metadata.json"
    readme_out   = out / "README.md"

    target_splits = splits if splits is not None else ["train", "test"]

    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Step 1 – Load raw corpus
    # ------------------------------------------------------------------
    if verbose:
        print(f"[1/3] Loading corpus from {archive} …", flush=True)

    docs = load_dataset(
        archive_path=archive,
        splits=target_splits,
        include_unstructured=include_unstructured,
        strip_headers=True,   # remove From:/Subject: before preprocessing
    )
    n_loaded = len(docs)
    if verbose:
        split_counts = {}
        for d in docs:
            split_counts[d["split"]] = split_counts.get(d["split"], 0) + 1
        print(f"     Loaded {n_loaded:,} documents  "
              f"({', '.join(f'{v:,} {k}' for k, v in sorted(split_counts.items()))})")

    # ------------------------------------------------------------------
    # Step 2 – Preprocess
    # ------------------------------------------------------------------
    if verbose:
        print("[2/3] Preprocessing …", flush=True)

    processed = preprocess_documents(docs, verbose=verbose)

    # Sanity check: doc_ids must be unique and contiguous 0-based
    ids = [d["doc_id"] for d in processed]
    assert len(ids) == len(set(ids)), "doc_ids are not unique after preprocessing"
    assert sorted(ids) == list(range(len(ids))), "doc_ids are not contiguous 0-based"

    # ------------------------------------------------------------------
    # Step 3 – Serialise to parquet
    # ------------------------------------------------------------------
    if verbose:
        print(f"[3/3] Writing parquet → {parquet_out} …", flush=True)

    df = pd.DataFrame(processed, columns=PROCESSED_COLUMNS)

    # Enforce dtypes for space-efficiency and portability
    df["doc_id"]   = df["doc_id"].astype("int32")
    df["label"]    = df["label"].astype("int8")
    df["category"] = df["category"].astype("category")
    df["split"]    = df["split"].astype("category")
    df["text"]     = df["text"].astype("string")
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
        "splits_included": sorted(target_splits),
        "include_unstructured": include_unstructured,
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
        print(f"  Documents   : {len(df):,}")
        print(f"  Categories  : {df['category'].nunique()}")
        train_n = split_info.get("train", 0)
        test_n  = split_info.get("test", 0)
        print(f"  Train / Test: {train_n:,} / {test_n:,}")
        print(f"  Empty clean : {empty_clean:,}")
        print(f"  Avg tokens  : {avg_clean_tokens:.1f}")
        print(f"  Parquet     : {parquet_out}  "
              f"({parquet_out.stat().st_size / 1_048_576:.1f} MB)")
        print(f"  Metadata    : {metadata_out}")
        print(f"  Elapsed     : {elapsed:.1f}s")
        print("=" * 60)

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

    Teammates can call this to confirm their environment can read the file:

        python -c "
        import sys; sys.path.insert(0, 'src')
        from build_dataset import verify_processed_dataset
        verify_processed_dataset()
        "
    """
    import pandas as pd

    path = Path(parquet_path) if parquet_path else PARQUET_FILE

    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at '{path}'.\n"
            "Run:  python src/build_dataset.py"
        )

    df = pd.read_parquet(path, engine="pyarrow")
    report: dict = {}

    # 1. Required columns
    missing_cols = set(PROCESSED_COLUMNS) - set(df.columns)
    report["missing_columns"] = list(missing_cols)
    assert not missing_cols, f"Missing columns: {missing_cols}"

    # 2. Row count
    report["num_rows"] = len(df)
    assert len(df) > 0, "Parquet file is empty"

    # 3. doc_id uniqueness and contiguity
    ids = df["doc_id"].tolist()
    report["doc_id_unique"] = len(ids) == len(set(ids))
    assert report["doc_id_unique"], "doc_ids are not unique"
    report["doc_id_contiguous"] = sorted(ids) == list(range(len(ids)))
    assert report["doc_id_contiguous"], "doc_ids are not contiguous 0-based"

    # 4. No null doc_id / label / category / split
    for col in ("doc_id", "label", "category", "split"):
        null_count = int(df[col].isna().sum())
        report[f"null_{col}"] = null_count
        assert null_count == 0, f"Column '{col}' has {null_count} nulls"

    # 5. No empty raw text
    empty_text = int((df["text"].fillna("").str.strip() == "").sum())
    report["empty_text_count"] = empty_text
    assert empty_text == 0, f"{empty_text} rows have empty 'text'"

    # 6. All 20 categories present
    found_cats = set(df["category"].unique())
    report["num_categories"] = len(found_cats)
    assert len(found_cats) == 20, f"Expected 20 categories, got {len(found_cats)}"

    # 7. Labels in [0, 19]
    label_range_ok = bool(df["label"].between(0, 19).all())
    report["labels_in_range"] = label_range_ok
    assert label_range_ok, "Some labels are outside [0, 19]"

    # 8. label ↔ category consistency
    from data_loader import CATEGORIES as _CATS
    bad_label = df[df.apply(
        lambda r: _CATS[r["label"]] != r["category"], axis=1
    )]
    report["label_category_mismatches"] = len(bad_label)
    assert len(bad_label) == 0, f"{len(bad_label)} label/category mismatches"

    # 9. clean_text is string
    assert df["clean_text"].dtype == object or str(df["clean_text"].dtype) in (
        "string", "StringDtype"
    ), "clean_text column is not string type"
    report["clean_text_dtype_ok"] = True

    # 10. Splits are only valid values
    valid_splits = {"train", "test", "unstructured"}
    bad_splits = set(df["split"].unique()) - valid_splits
    report["invalid_splits"] = list(bad_splits)
    assert not bad_splits, f"Unknown split values: {bad_splits}"

    report["all_checks_passed"] = True

    if verbose:
        print("=" * 50)
        print("  PROCESSED DATASET VERIFICATION")
        print(f"  File     : {path}")
        print(f"  Rows     : {report['num_rows']:,}")
        print(f"  Cols     : {list(df.columns)}")
        print(f"  Categories: {report['num_categories']}")
        splits_found = df["split"].value_counts().to_dict()
        for sp, cnt in sorted(splits_found.items()):
            print(f"  {sp:<15}: {cnt:,}")
        print(f"  doc_id unique     : {report['doc_id_unique']}")
        print(f"  doc_id contiguous : {report['doc_id_contiguous']}")
        print(f"  label/cat OK      : {report['label_category_mismatches'] == 0}")
        print(f"  empty text        : {report['empty_text_count']}")
        print("  ALL CHECKS PASSED ✓")
        print("=" * 50)

    return report


# ---------------------------------------------------------------------------
# README writer
# ---------------------------------------------------------------------------

def _write_data_readme(path: Path, meta: dict) -> None:
    lines = [
        "# 20 Newsgroups — Processed Dataset",
        "",
        "> **This directory is generated automatically.**  "
        "Do NOT commit `processed_documents.parquet` to Git.",
        "> Only `metadata.json` and `README.md` are tracked.",
        "> To regenerate, run: `python src/build_dataset.py`",
        "",
        "## Files",
        "",
        "| File | In Git | Description |",
        "|---|---|---|",
        "| `processed_documents.parquet` | ❌ | Full processed corpus (~37 500 rows) |",
        "| `metadata.json` | ✅ | Dataset description and stats |",
        "| `README.md` | ✅ | This file |",
        "",
        "## Quick-start (pandas)",
        "",
        "```python",
        "import pandas as pd",
        "",
        "df = pd.read_parquet('data/processed/processed_documents.parquet')",
        "print(df.shape)          # (~37500, 6)",
        "print(df.columns.tolist())",
        "# ['doc_id', 'text', 'clean_text', 'label', 'category', 'split']",
        "```",
        "",
        "## Quick-start (PyArrow)",
        "",
        "```python",
        "import pyarrow.parquet as pq",
        "",
        "table = pq.read_table('data/processed/processed_documents.parquet')",
        "# Filter to train split only:",
        "import pyarrow.compute as pc",
        "train = table.filter(pc.equal(table['split'], 'train'))",
        "```",
        "",
        "## Schema",
        "",
        "| Column | Type | Description |",
        "|---|---|---|",
        "| `doc_id` | int32 | Globally unique 0-based integer |",
        "| `text` | string | Raw document body (email headers stripped) |",
        "| `clean_text` | string | Preprocessed text: lowercase stemmed tokens |",
        "| `label` | int8 | Numeric class label 0–19 |",
        "| `category` | category | Newsgroup name e.g. `alt.atheism` |",
        "| `split` | category | `train` or `test` |",
        "",
        "## Dataset statistics",
        "",
        f"| Statistic | Value |",
        f"|---|---|",
        f"| Total documents | {meta['num_documents']:,} |",
        f"| Categories | {meta['num_categories']} |",
        f"| Train documents | {meta['split_counts'].get('train', 0):,} |",
        f"| Test documents | {meta['split_counts'].get('test', 0):,} |",
        f"| Empty clean_text | {meta['empty_clean_text_count']:,} |",
        f"| Avg clean tokens | {meta['avg_clean_tokens']} |",
        f"| Preprocessing version | `{meta['preprocessing_version']}` |",
        f"| Generated at | {meta['generated_at']} |",
        "",
        "## Preprocessing pipeline",
        "",
        "Implemented in `src/preprocessing.py`.  Steps applied in order:",
        "",
        "1. Residual header / metadata removal",
        "2. Lowercase",
        "3. Tokenisation (NLTK `word_tokenize`)",
        "4. Stopword removal (NLTK English)",
        "5. Stemming (Porter Stemmer)",
        "6. Drop non-alpha / length < 2 tokens",
        "",
        "## Reproducing the dataset",
        "",
        "```bash",
        "# From the 20-newsgroups-ir/ directory:",
        "python src/build_dataset.py",
        "",
        "# With a custom archive path:",
        "python src/build_dataset.py --archive /path/to/archive.zip",
        "",
        "# Verify an existing file without rebuilding:",
        "python src/build_dataset.py --verify-only",
        "```",
        "",
        "The `NEWSGROUPS_ARCHIVE_PATH` environment variable can also be set.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build the common processed dataset (parquet + metadata).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--archive", metavar="PATH", default=None,
                   help="Path to archive.zip")
    p.add_argument("--out-dir", metavar="DIR", default=None,
                   help="Output directory (default: data/processed/)")
    p.add_argument("--splits", nargs="+", default=None,
                   choices=["train", "test", "unstructured"],
                   metavar="SPLIT",
                   help="Splits to include (default: train test)")
    p.add_argument("--include-unstructured", action="store_true", default=False,
                   help="Also include ~1 600 header-less documents")
    p.add_argument("--verify-only", action="store_true", default=False,
                   help="Only verify an existing parquet; do not rebuild")
    p.add_argument("--quiet", action="store_true", default=False,
                   help="Suppress progress output")
    return p


def main(argv: list | None = None) -> None:
    args = _build_parser().parse_args(argv)
    verbose = not args.quiet

    if args.verify_only:
        out = Path(args.out_dir) if args.out_dir else PROCESSED_DATA_DIR
        verify_processed_dataset(out / "processed_documents.parquet", verbose=verbose)
        return

    parquet_path = build_processed_dataset(
        archive_path=args.archive,
        out_dir=args.out_dir,
        splits=args.splits,
        include_unstructured=args.include_unstructured,
        verbose=verbose,
    )
    verify_processed_dataset(parquet_path, verbose=verbose)


if __name__ == "__main__":
    main()
