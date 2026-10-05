"""
utils.py
========
Shared path helpers and constants for the 20 Newsgroups IR project.

Author : Reetikesh Choudhury
Module : Shared Utilities (own component)

Every module that needs to locate the archive, the processed data directory,
or the results directory should import from here so that path logic lives
in exactly one place.

Environment variables
---------------------
NEWSGROUPS_ARCHIVE_PATH
    Override the default archive location (default: ``../../archive.zip``
    relative to this file).

NEWSGROUPS_DATA_DIR
    Override the processed-data output directory (default:
    ``../data/processed`` relative to this file).
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root and canonical sub-directories
# ---------------------------------------------------------------------------

#: Root of the ``20-newsgroups-ir/`` sub-project.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

#: ``src/`` directory.
SRC_DIR: Path = PROJECT_ROOT / "src"

#: ``results/`` directory (charts, CSVs, JSON stats).
RESULTS_DIR: Path = PROJECT_ROOT / "results"

#: ``data/processed/`` directory – home of the shared processed dataset.
_DATA_DIR_ENV: str | None = os.environ.get("NEWSGROUPS_DATA_DIR")
PROCESSED_DATA_DIR: Path = (
    Path(_DATA_DIR_ENV) if _DATA_DIR_ENV else PROJECT_ROOT / "data" / "processed"
)

# ---------------------------------------------------------------------------
# Dataset archive
# ---------------------------------------------------------------------------

#: Path to ``archive.zip`` (raw dataset – NOT in Git).
_ARCHIVE_ENV: str | None = os.environ.get("NEWSGROUPS_ARCHIVE_PATH")
# PROJECT_ROOT = …/20-newsgroups-ir/  → parents[0] = IR_PROJECT/
#                                      → parents[1] = d:\Project-ir\
ARCHIVE_PATH: Path = (
    Path(_ARCHIVE_ENV)
    if _ARCHIVE_ENV
    else PROJECT_ROOT.parents[1] / "archive.zip"
)

# ---------------------------------------------------------------------------
# Processed dataset filenames
# ---------------------------------------------------------------------------

#: Column-store for the full processed corpus.
PARQUET_FILE: Path = PROCESSED_DATA_DIR / "processed_documents.parquet"

#: JSON file with dataset metadata (committed to Git).
METADATA_FILE: Path = PROCESSED_DATA_DIR / "metadata.json"

#: Human-readable README for the data directory (committed to Git).
DATA_README_FILE: Path = PROCESSED_DATA_DIR / "README.md"

# ---------------------------------------------------------------------------
# Preprocessing version tag
# Bump this string whenever the preprocessing pipeline logic changes so that
# downstream consumers can detect stale cached datasets.
# ---------------------------------------------------------------------------
PREPROCESSING_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Required columns in the processed parquet file (contract with teammates)
# ---------------------------------------------------------------------------
PROCESSED_COLUMNS: list[str] = [
    "doc_id",
    "text",
    "clean_text",
    "label",
    "category",
    "split",
]
