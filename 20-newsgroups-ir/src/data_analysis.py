"""
data_analysis.py
================
Dataset analysis module for the 20 Newsgroups IR project.

Author : Reetikesh Choudhury
Module : Dataset Analysis (own component)

Overview
--------
Computes and reports the following statistics from the real local dataset:

  1.  Total number of documents (train + test combined)
  2.  Number of categories
  3.  Number of documents per category
  4.  Training document count
  5.  Testing document count
  6.  Average document length (in characters, after header stripping)
  7.  Minimum document length
  8.  Maximum document length
  9.  Number of empty / null documents (pre- and post-filter)
  10. Duplicate documents (exact-text matches across the corpus)

Public API
----------
  analyse_dataset(archive_path=None, *, include_unstructured=False)
      → AnalysisResult  (dataclass with all 10 statistics + per-category breakdown)

  save_results(result, results_dir)
      → saves dataset_stats.json  and  per_category_stats.csv  to results_dir

Running as a script
-------------------
  python src/data_analysis.py
  python src/data_analysis.py --archive /path/to/archive.zip
  python src/data_analysis.py --include-unstructured
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Allow running as a script from any working directory.
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from data_loader import CATEGORIES, load_dataset  # noqa: E402

_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CategoryStats:
    """Per-category breakdown of document counts and lengths."""
    category: str
    label: int
    train_count: int
    test_count: int
    unstructured_count: int
    total_count: int
    avg_length: float
    min_length: int
    max_length: int
    duplicate_count: int   # exact duplicates within this category


@dataclass
class AnalysisResult:
    """
    All analysis statistics for the 20 Newsgroups corpus.

    Attributes
    ----------
    total_documents      : total docs in the loaded corpus
    num_categories       : number of distinct newsgroup categories
    train_count          : docs with split == 'train'
    test_count           : docs with split == 'test'
    unstructured_count   : docs with split == 'unstructured' (0 unless requested)
    avg_length           : mean character length of document text
    min_length           : shortest document (characters)
    max_length           : longest document (characters)
    empty_null_count     : docs with empty or whitespace-only text (always 0
                           after load_dataset filters them, but reported for
                           transparency)
    duplicate_count      : number of documents whose text is an exact match of
                           another document in the corpus
    duplicate_pairs      : list of (doc_id_a, doc_id_b, category, split) tuples
                           for the first 100 duplicate pairs found
    per_category         : list of CategoryStats, one per newsgroup
    generated_at         : ISO-8601 timestamp of when analysis was run
    archive_path         : path to the source archive
    analysis_duration_s  : wall-clock seconds taken to run the analysis
    """
    total_documents: int = 0
    num_categories: int = 0
    train_count: int = 0
    test_count: int = 0
    unstructured_count: int = 0
    avg_length: float = 0.0
    min_length: int = 0
    max_length: int = 0
    empty_null_count: int = 0
    duplicate_count: int = 0
    expected_twin_count: int = 0   # train/test mirrors – expected by dataset design
    duplicate_pairs: List[dict] = field(default_factory=list)
    per_category: List[CategoryStats] = field(default_factory=list)
    generated_at: str = ""
    archive_path: str = ""
    analysis_duration_s: float = 0.0


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyse_dataset(
    archive_path: Optional[str | Path] = None,
    *,
    include_unstructured: bool = False,
) -> AnalysisResult:
    """
    Load the corpus and compute all 10 required statistics.

    Parameters
    ----------
    archive_path : str or Path, optional
        Path to ``archive.zip``. Falls back to the default loader resolution
        (``NEWSGROUPS_ARCHIVE_PATH`` env var, then ``../../archive.zip``).
    include_unstructured : bool, default False
        When True, also load the ~1 600 docs that have no structured headers.

    Returns
    -------
    AnalysisResult
        Dataclass containing every statistic. Suitable for JSON serialisation
        via ``dataclasses.asdict()``.
    """
    import datetime

    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Load corpus
    # ------------------------------------------------------------------
    docs = load_dataset(
        archive_path=archive_path,
        include_unstructured=include_unstructured,
    )

    result = AnalysisResult()
    result.generated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    result.archive_path = str(
        Path(archive_path).resolve() if archive_path
        else Path(__file__).resolve().parents[3] / "archive.zip"
    )

    # ------------------------------------------------------------------
    # 1. Total documents
    # ------------------------------------------------------------------
    result.total_documents = len(docs)

    # ------------------------------------------------------------------
    # 2. Number of categories
    # ------------------------------------------------------------------
    result.num_categories = len({d["category"] for d in docs})

    # ------------------------------------------------------------------
    # 4 & 5. Train / test / unstructured counts
    # ------------------------------------------------------------------
    split_counter: Dict[str, int] = collections.Counter(d["split"] for d in docs)
    result.train_count        = split_counter.get("train", 0)
    result.test_count         = split_counter.get("test", 0)
    result.unstructured_count = split_counter.get("unstructured", 0)

    # ------------------------------------------------------------------
    # 9. Empty / null documents
    #    load_dataset already filters empty bodies, so this is 0 for the
    #    returned corpus. We count here for transparency / test coverage.
    # ------------------------------------------------------------------
    result.empty_null_count = sum(
        1 for d in docs if not d.get("text") or not d["text"].strip()
    )

    # ------------------------------------------------------------------
    # 6 / 7 / 8. Document lengths (character count of text field)
    # ------------------------------------------------------------------
    lengths = [len(d["text"]) for d in docs]

    result.avg_length = round(sum(lengths) / len(lengths), 2) if lengths else 0.0
    result.min_length = min(lengths) if lengths else 0
    result.max_length = max(lengths) if lengths else 0

    # ------------------------------------------------------------------
    # 10. Duplicate documents
    #     The dataset is deliberately structured so that every train document
    #     has an identical twin in the test split (same category, same text,
    #     different split tag).  Those expected train/test twins are NOT
    #     interesting duplicates.
    #
    #     We report two numbers:
    #       duplicate_count        – docs involved in ANY exact-text match
    #       unexpected_dup_count   – docs duplicated within the SAME split, or
    #                                across DIFFERENT categories (genuinely odd)
    #
    #     Strategy: hash text → group doc_ids → for each group check whether
    #     the pair is purely a train↔test mirror (expected) or something else.
    # ------------------------------------------------------------------
    text_to_docs: Dict[int, List[dict]] = collections.defaultdict(list)
    for d in docs:
        text_to_docs[hash(d["text"])].append(d)

    dup_pairs: List[dict] = []
    total_dup_docs: set[int] = set()
    unexpected_dup_docs: set[int] = set()

    for group in text_to_docs.values():
        if len(group) < 2:
            continue

        # Hash collision guard: keep only docs whose text truly matches group[0].
        ref_text = group[0]["text"]
        group = [d for d in group if d["text"] == ref_text]
        if len(group) < 2:
            continue

        # Classify the group.
        # An "expected" pair is exactly 2 docs: one train, one test, same category.
        def _is_expected_twin_pair(grp: List[dict]) -> bool:
            if len(grp) != 2:
                return False
            a, b = grp
            return (
                a["category"] == b["category"]
                and {a["split"], b["split"]} == {"train", "test"}
            )

        is_expected = _is_expected_twin_pair(group)

        for d in group:
            total_dup_docs.add(d["doc_id"])
            if not is_expected:
                unexpected_dup_docs.add(d["doc_id"])

        if not is_expected:
            # Record unexpected pairs (capped at 100).
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    if len(dup_pairs) >= 100:
                        break
                    a, b = group[i], group[j]
                    dup_pairs.append({
                        "doc_id_a": a["doc_id"],
                        "doc_id_b": b["doc_id"],
                        "category_a": a["category"],
                        "category_b": b["category"],
                        "split_a": a["split"],
                        "split_b": b["split"],
                        "text_length": len(a["text"]),
                        "note": (
                            "same-split duplicate"
                            if a["split"] == b["split"]
                            else "cross-category duplicate"
                        ),
                    })
                if len(dup_pairs) >= 100:
                    break

    result.duplicate_count = len(unexpected_dup_docs)
    result.duplicate_pairs = dup_pairs
    result.expected_twin_count = len(total_dup_docs) - len(unexpected_dup_docs)

    # ------------------------------------------------------------------
    # 3. Per-category breakdown (counts + lengths)
    # ------------------------------------------------------------------
    cat_docs: Dict[str, List[dict]] = collections.defaultdict(list)
    for d in docs:
        cat_docs[d["category"]].append(d)

    for cat in CATEGORIES:
        if cat not in cat_docs:
            continue
        cdocs = cat_docs[cat]
        clengths = [len(d["text"]) for d in cdocs]
        csplit   = collections.Counter(d["split"] for d in cdocs)

        # Unexpected duplicates within this category only.
        # A same-category train/test pair with identical text is EXPECTED
        # (by dataset design); we only flag extras beyond that expected pair.
        cat_text_groups: Dict[int, List[dict]] = collections.defaultdict(list)
        for d in cdocs:
            cat_text_groups[hash(d["text"])].append(d)
        cat_dup_docs: set[int] = set()
        for grp in cat_text_groups.values():
            if len(grp) < 2:
                continue
            ref_text = grp[0]["text"]
            grp = [d for d in grp if d["text"] == ref_text]
            if len(grp) < 2:
                continue
            # Expected: exactly one train + one test with same text in same category
            splits_in_grp = [d["split"] for d in grp]
            if len(grp) == 2 and set(splits_in_grp) == {"train", "test"}:
                continue  # this is the normal train/test twin – not a dup
            cat_dup_docs.update(d["doc_id"] for d in grp)

        result.per_category.append(CategoryStats(
            category          = cat,
            label             = CATEGORIES.index(cat),
            train_count       = csplit.get("train", 0),
            test_count        = csplit.get("test", 0),
            unstructured_count= csplit.get("unstructured", 0),
            total_count       = len(cdocs),
            avg_length        = round(sum(clengths) / len(clengths), 2),
            min_length        = min(clengths),
            max_length        = max(clengths),
            duplicate_count   = len(cat_dup_docs),
        ))

    result.analysis_duration_s = round(time.perf_counter() - t0, 3)
    return result


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_results(
    result: AnalysisResult,
    results_dir: Optional[str | Path] = None,
) -> tuple[Path, Path]:
    """
    Write analysis output to disk.

    Writes two files:

    * ``dataset_stats.json``    – full AnalysisResult as JSON (all 10 stats)
    * ``per_category_stats.csv`` – one row per category with counts & lengths

    Parameters
    ----------
    result      : AnalysisResult returned by analyse_dataset()
    results_dir : directory to write into; defaults to ``results/``

    Returns
    -------
    (json_path, csv_path) : absolute Paths of the two written files
    """
    out_dir = Path(results_dir) if results_dir else _RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- JSON -----------------------------------------------------------
    # Convert dataclass → plain dict, excluding the verbose duplicate_pairs
    # list from the top-level summary (it's already inside the dict).
    raw = asdict(result)
    json_path = out_dir / "dataset_stats.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(raw, fh, indent=2, ensure_ascii=False)

    # --- CSV ------------------------------------------------------------
    csv_path = out_dir / "per_category_stats.csv"
    fieldnames = [
        "category", "label",
        "train_count", "test_count", "unstructured_count", "total_count",
        "avg_length", "min_length", "max_length", "duplicate_count",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for cs in result.per_category:
            writer.writerow({k: getattr(cs, k) for k in fieldnames})

    return json_path, csv_path


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def print_report(result: AnalysisResult) -> None:
    """Print a human-readable summary to stdout."""
    sep = "=" * 62
    print(sep)
    print("  20 NEWSGROUPS — DATASET ANALYSIS REPORT")
    print(f"  Generated : {result.generated_at}")
    print(f"  Archive   : {result.archive_path}")
    print(sep)

    print(f"\n{'OVERALL STATISTICS':}")
    print(f"  {'1. Total documents':<38} {result.total_documents:>8,}")
    print(f"  {'2. Number of categories':<38} {result.num_categories:>8,}")
    print(f"  {'4. Training documents':<38} {result.train_count:>8,}")
    print(f"  {'5. Testing documents':<38} {result.test_count:>8,}")
    if result.unstructured_count:
        print(f"  {'   Unstructured documents':<38} {result.unstructured_count:>8,}")
    print(f"  {'6. Average document length (chars)':<38} {result.avg_length:>8,.1f}")
    print(f"  {'7. Minimum document length (chars)':<38} {result.min_length:>8,}")
    print(f"  {'8. Maximum document length (chars)':<38} {result.max_length:>8,}")
    print(f"  {'9. Empty/null documents':<38} {result.empty_null_count:>8,}")
    twin_note = ""
    if result.expected_twin_count:
        twin_note = f"  ({result.expected_twin_count:,} expected train/test twins excluded)"
    print(f"  {'10.Unexpected duplicate documents':<38} {result.duplicate_count:>8,}{twin_note}")

    print(f"\n{'3. DOCUMENTS PER CATEGORY':}")
    print(f"  {'Category':<35} {'Train':>7} {'Test':>7} {'Total':>7} "
          f"{'Avg len':>9} {'Dups':>6}")
    print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*7} {'-'*9} {'-'*6}")
    for cs in result.per_category:
        print(
            f"  {cs.category:<35} {cs.train_count:>7,} {cs.test_count:>7,} "
            f"{cs.total_count:>7,} {cs.avg_length:>9,.1f} {cs.duplicate_count:>6,}"
        )

    print(f"\n  Analysis completed in {result.analysis_duration_s:.2f}s")
    print(sep)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analyse the 20 Newsgroups dataset and save results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--archive",
        metavar="PATH",
        default=None,
        help="Path to archive.zip (default: auto-detected from repo layout or "
             "NEWSGROUPS_ARCHIVE_PATH env var)",
    )
    p.add_argument(
        "--results-dir",
        metavar="DIR",
        default=str(_RESULTS_DIR),
        help="Directory to write dataset_stats.json and per_category_stats.csv",
    )
    p.add_argument(
        "--include-unstructured",
        action="store_true",
        default=False,
        help="Also analyse the ~1 600 header-less documents",
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        default=False,
        help="Print the report but do not write files to disk",
    )
    return p


def main(argv: Optional[list] = None) -> AnalysisResult:
    args = _build_parser().parse_args(argv)

    print("Loading dataset …", flush=True)
    result = analyse_dataset(
        archive_path=args.archive,
        include_unstructured=args.include_unstructured,
    )

    print_report(result)

    if not args.no_save:
        json_path, csv_path = save_results(result, args.results_dir)
        print(f"\nSaved: {json_path}")
        print(f"Saved: {csv_path}")

    return result


if __name__ == "__main__":
    main()
