"""
Comprehensive artifact verification script.
Deleted after use — results are shown in stdout.
"""
import sys, json, hashlib, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pyarrow.parquet as pq
from data_loader import CATEGORIES, load_dataset
from utils import (
    ARCHIVE_PATH, METADATA_FILE, PARQUET_FILE,
    PREPROCESSING_VERSION, PROCESSED_COLUMNS,
)

SEP = "=" * 64
PASS = "  PASS"
FAIL = "  FAIL !!!"

failures = []

def check(label, condition, detail=""):
    if condition:
        print(f"{PASS}  {label}")
    else:
        print(f"{FAIL}  {label}  {detail}")
        failures.append(label)

# ── 1. Files exist ──────────────────────────────────────────────────────────
print(SEP)
print("  1. FILE EXISTENCE")
print(SEP)
check("processed_documents.parquet exists", PARQUET_FILE.exists())
check("metadata.json exists",              METADATA_FILE.exists())
check("README.md exists",                  (PARQUET_FILE.parent / "README.md").exists())

# ── 2. Load and basic shape ─────────────────────────────────────────────────
print()
print(SEP)
print("  2. PARQUET LOAD + SHAPE")
print(SEP)
df = pd.read_parquet(PARQUET_FILE, engine="pyarrow")
print(f"  Shape  : {df.shape}")
print(f"  Columns: {df.columns.tolist()}")
check("Row count > 0",               len(df) > 0)
check("Exactly 6 columns",           len(df.columns) == 6)
check("All required columns present",
      set(PROCESSED_COLUMNS) == set(df.columns),
      f"found={set(df.columns)}")
check("Row count in expected range",
      37_000 <= len(df) <= 38_000,
      f"got {len(df)}")

# ── 3. Column dtypes ────────────────────────────────────────────────────────
print()
print(SEP)
print("  3. COLUMN DTYPES")
print(SEP)
print(f"  dtypes:\n{df.dtypes.to_string()}")
check("doc_id   is int32",    str(df['doc_id'].dtype)    == 'int32')
check("label    is int8",     str(df['label'].dtype)     == 'int8')
check("category is category", str(df['category'].dtype)  == 'category')
check("split    is category", str(df['split'].dtype)     == 'category')
check("text     is string/object",
      str(df['text'].dtype) in ('string', 'object', 'StringDtype'))
check("clean_text is string/object",
      str(df['clean_text'].dtype) in ('string', 'object', 'StringDtype'))

# ── 4. doc_id integrity ─────────────────────────────────────────────────────
print()
print(SEP)
print("  4. doc_id INTEGRITY")
print(SEP)
ids = df["doc_id"].tolist()
check("doc_ids are unique",      len(ids) == len(set(ids)))
check("doc_ids start at 0",      min(ids) == 0)
check("doc_ids end at N-1",      max(ids) == len(df) - 1)
check("doc_ids are contiguous",  sorted(ids) == list(range(len(df))))
check("no null doc_ids",         int(df["doc_id"].isna().sum()) == 0)

# ── 5. Split correctness ────────────────────────────────────────────────────
print()
print(SEP)
print("  5. SPLIT CORRECTNESS")
print(SEP)
split_counts = df["split"].value_counts().to_dict()
print(f"  Split counts: {dict(sorted(split_counts.items()))}")
check("train split present",          "train" in split_counts)
check("test  split present",          "test"  in split_counts)
check("no unexpected split values",
      set(split_counts.keys()) <= {"train", "test", "unstructured"})
check("train count in expected range",
      18_600 <= split_counts.get("train", 0) <= 19_000)
check("test count in expected range",
      18_600 <= split_counts.get("test", 0) <= 19_000)
check("train == test count",
      split_counts.get("train", 0) == split_counts.get("test", 0))

# ── 6. Category & label integrity ──────────────────────────────────────────
print()
print(SEP)
print("  6. CATEGORY & LABEL INTEGRITY")
print(SEP)
found_cats = set(df["category"].unique())
check("exactly 20 categories",     len(found_cats) == 20)
check("all CATEGORIES present",    found_cats == set(CATEGORIES))
check("label range 0-19",          bool(df["label"].between(0, 19).all()))
check("no null categories",        int(df["category"].isna().sum()) == 0)
check("no null labels",            int(df["label"].isna().sum()) == 0)
bad_label = df[df.apply(lambda r: CATEGORIES[r["label"]] != r["category"], axis=1)]
check("label <-> category consistent",
      len(bad_label) == 0,
      f"{len(bad_label)} mismatches")

# ── 7. Text field integrity ─────────────────────────────────────────────────
print()
print(SEP)
print("  7. TEXT FIELD INTEGRITY")
print(SEP)
empty_text  = int((df["text"].fillna("").str.strip() == "").sum())
empty_clean = int((df["clean_text"].fillna("").str.strip() == "").sum())
print(f"  Empty 'text'       : {empty_text}")
print(f"  Empty 'clean_text' : {empty_clean}  (short posts with no body after preprocessing)")
check("no empty raw text",               empty_text == 0)
check("empty clean_text < 1% of corpus",
      empty_clean < int(len(df) * 0.01),
      f"got {empty_clean}")
check("clean_text tokens are lowercase",
      df["clean_text"].fillna("").str.match(r'^[a-z ]*$').mean() > 0.99)

clean_token_counts = df["clean_text"].fillna("").apply(
    lambda t: len(t.split()) if t.strip() else 0
)
avg_tokens = clean_token_counts.mean()
print(f"  Avg clean tokens  : {avg_tokens:.1f}")
print(f"  Min clean tokens  : {clean_token_counts.min()}")
print(f"  Max clean tokens  : {clean_token_counts.max():,}")
check("avg tokens in expected range", 80 <= avg_tokens <= 120)

# ── 8. Original text preserved ─────────────────────────────────────────────
print()
print(SEP)
print("  8. ORIGINAL TEXT PRESERVED")
print(SEP)
# The parquet text column must differ from clean_text (headers still present)
identical = (df["text"] == df["clean_text"]).sum()
print(f"  Rows where text == clean_text: {identical}")
check("raw text != clean_text for nearly all rows",
      identical < 10,
      f"got {identical} identical pairs")
# Sample: text should be longer than clean_text for most docs
longer = (df["text"].fillna("").str.len() > df["clean_text"].fillna("").str.len()).mean()
check("raw text longer than clean_text (>95%)", longer > 0.95,
      f"got {longer:.1%}")

# ── 9. PyArrow native read ──────────────────────────────────────────────────
print()
print(SEP)
print("  9. PYARROW NATIVE READ")
print(SEP)
table = pq.read_table(str(PARQUET_FILE))
check("PyArrow table rows match pandas",  table.num_rows == len(df))
check("PyArrow table cols match pandas",  table.num_columns == len(df.columns))
# Filter by split using PyArrow compute
import pyarrow.compute as pc
train_table = table.filter(pc.equal(table["split"], "train"))
check("PyArrow split filter works",
      train_table.num_rows == split_counts.get("train", 0))

# ── 10. Metadata.json integrity ─────────────────────────────────────────────
print()
print(SEP)
print("  10. METADATA.JSON INTEGRITY")
print(SEP)
with open(METADATA_FILE, encoding="utf-8") as f:
    meta = json.load(f)
print(f"  Keys: {list(meta.keys())}")
required_meta_keys = {
    "preprocessing_version", "generated_at", "num_documents",
    "num_categories", "splits_included", "columns",
    "split_counts", "category_counts", "categories",
    "parquet_file", "parquet_size_bytes",
}
missing_meta = required_meta_keys - meta.keys()
check("all required metadata keys present",
      not missing_meta, f"missing={missing_meta}")
check("metadata num_documents matches parquet",
      meta["num_documents"] == len(df))
check("metadata num_categories == 20",
      meta["num_categories"] == 20)
check("metadata preprocessing_version set",
      meta["preprocessing_version"] == PREPROCESSING_VERSION)
check("metadata columns match PROCESSED_COLUMNS",
      meta["columns"] == PROCESSED_COLUMNS)
check("metadata split_counts correct",
      meta["split_counts"].get("train") == split_counts.get("train") and
      meta["split_counts"].get("test")  == split_counts.get("test"))

# ── 11. Reproducibility ─────────────────────────────────────────────────────
print()
print(SEP)
print("  11. REPRODUCIBILITY  (reload + hash first 1000 rows)")
print(SEP)
# Hash a stable slice of the data so we can compare two runs
sample = df.head(1000)[["doc_id", "clean_text", "label", "category", "split"]]
h1 = hashlib.md5(sample.to_csv(index=False).encode()).hexdigest()

# Re-run the preprocessing pipeline on a small slice and compare
t0 = time.perf_counter()
docs_small = load_dataset(
    ARCHIVE_PATH,
    categories=["alt.atheism"],
    splits=["train"],
)
from preprocessing import preprocess_documents
proc1 = preprocess_documents(docs_small)
proc2 = preprocess_documents(docs_small)
elapsed = time.perf_counter() - t0
check("same input -> same clean_text (deterministic)",
      all(a["clean_text"] == b["clean_text"] for a, b in zip(proc1, proc2)))
print(f"  Determinism verified in {elapsed:.2f}s")

# ── 12. .gitignore safety ───────────────────────────────────────────────────
print()
print(SEP)
print("  12. .GITIGNORE SAFETY")
print(SEP)
import subprocess
repo_root = PARQUET_FILE.parents[3]
rel_parquet = str(PARQUET_FILE.relative_to(repo_root))
rel_archive = str(ARCHIVE_PATH.relative_to(repo_root)) if ARCHIVE_PATH.is_relative_to(repo_root) else "archive.zip"
result = subprocess.run(
    ["git", "check-ignore", "-v", rel_parquet, rel_archive],
    capture_output=True, text=True,
    cwd=str(repo_root),
)
output = result.stdout + result.stderr
print(f"  git check-ignore output:\n    {output.strip().replace(chr(10), chr(10)+'    ')}")
check("parquet is git-ignored",
      str(PARQUET_FILE.name) in output or "processed_documents" in output)
check("archive.zip is git-ignored",
      "archive.zip" in output or "*.zip" in output)

# ── Final summary ────────────────────────────────────────────────────────────
print()
print(SEP)
if failures:
    print(f"  RESULT: {len(failures)} CHECK(S) FAILED:")
    for f in failures:
        print(f"    - {f}")
else:
    print(f"  ALL CHECKS PASSED  ({sum(1 for _ in range(1)) + 30}+ assertions)")
print(SEP)
