"""
data_loader.py
==============
Dataset loading module for the 20 Newsgroups IR project.

Author : Reetikesh Choudhury
Module : Dataset Loading (own component)

Overview
--------
Loads the 20 Newsgroups corpus from the local ``twenty+newsgroups.zip`` file.
Inside that ZIP is ``20_newsgroups.tar.gz``, which unpacks to a directory
tree of the form::

    20_newsgroups/
        alt.atheism/
            53366
            53367
            ...
        comp.graphics/
            ...

Each leaf file is a single raw news article (RFC-2822 formatted).

The full dataset contains ~19,997 documents across 20 categories.
No pre-assigned train/test split exists in the archive.  The 80/20
stratified split is created by ``build_dataset.py`` using
``sklearn.model_selection.train_test_split``.

The public API exposes:

* :func:`load_dataset`          – full corpus → ``list[dict]``
* :func:`load_dataframe`        – full corpus → ``pandas.DataFrame``
* :func:`load_processed_dataset`– loads the pre-built parquet artifact
* :func:`dataset_info`          – fast metadata scan (no full text load)

Each record returned by ``load_dataset`` has the keys:

    doc_id   (int)  – globally stable 0-based integer, unique in the list
    text     (str)  – raw document body (headers stripped by default)
    label    (int)  – zero-based integer label (same ordering as CATEGORIES)
    category (str)  – newsgroup name, e.g. ``"alt.atheism"``

Note: the ``split`` key (``"train"`` / ``"test"``) is NOT assigned here.
It is computed later by ``build_dataset.py`` using ``train_test_split``.
"""

from __future__ import annotations

import io
import os
import re
import tarfile
import zipfile
from pathlib import Path
from typing import Iterator, List, Optional

# ---------------------------------------------------------------------------
# Default archive location – can be overridden via NEWSGROUPS_ARCHIVE_PATH
# or by passing ``archive_path`` explicitly.
# ---------------------------------------------------------------------------
_DEFAULT_ARCHIVE = Path(__file__).resolve().parents[1] / "twenty+newsgroups.zip"

# Name of the tar.gz member inside the ZIP that contains the full dataset.
_FULL_DATASET_TARBALL = "20_newsgroups.tar.gz"

# Ordered list of the 20 newsgroup categories (matches subdirectory names).
CATEGORIES: List[str] = [
    "alt.atheism",
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.windows.x",
    "misc.forsale",
    "rec.autos",
    "rec.motorcycles",
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.crypt",
    "sci.electronics",
    "sci.med",
    "sci.space",
    "soc.religion.christian",
    "talk.politics.guns",
    "talk.politics.mideast",
    "talk.politics.misc",
    "talk.religion.misc",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_archive(archive_path: Optional[str | Path] = None) -> Path:
    """Return the resolved path to the archive ZIP, raising a clear error if missing."""
    if archive_path is not None:
        path = Path(archive_path)
    else:
        env_override = os.environ.get("NEWSGROUPS_ARCHIVE_PATH")
        path = Path(env_override) if env_override else _DEFAULT_ARCHIVE

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset archive not found at '{path}'.\n"
            "Set the NEWSGROUPS_ARCHIVE_PATH environment variable or pass "
            "`archive_path=` explicitly to load_dataset().\n"
            "Expected: twenty+newsgroups.zip (containing 20_newsgroups.tar.gz)"
        )
    return path


def _extract_body(raw_article: str, strip_headers: bool) -> str:
    """
    Return the text body of a raw news article.

    If ``strip_headers`` is True, the RFC-2822 header lines at the top of
    the article (``From:``, ``Newsgroups:``, ``Subject:``, ``Date:``,
    ``Message-ID:``, ``References:``, ``Lines:``, ``Organization:``,
    ``NNTP-Posting-Host:``, ``X-*:``, ``Path:``, etc.) are removed,
    leaving only the prose body.  The blank separator line between headers
    and body is also consumed.  An empty string is returned if nothing
    remains after stripping.
    """
    if not strip_headers:
        return raw_article.strip()

    lines = raw_article.split("\n")
    body_lines: List[str] = []
    in_header = True
    for line in lines:
        if in_header:
            if line.strip() == "":
                # Blank line terminates the header block.
                in_header = False
            elif re.match(r"^[\w\-]+:", line):
                # Standard header line – skip.
                continue
            else:
                # Non-header line before the blank separator → body starts now.
                in_header = False
                body_lines.append(line)
        else:
            body_lines.append(line)

    return "\n".join(body_lines).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(
    archive_path: Optional[str | Path] = None,
    *,
    categories: Optional[List[str]] = None,
    strip_headers: bool = True,
) -> List[dict]:
    """
    Load the 20 Newsgroups corpus from ``twenty+newsgroups.zip``.

    Reads ``20_newsgroups.tar.gz`` inside the ZIP, iterates over the
    directory-per-document layout, and returns one dict per document.

    Parameters
    ----------
    archive_path : str or Path, optional
        Path to ``twenty+newsgroups.zip``.  Defaults to the file located
        in the ``20-newsgroups-ir/`` project directory, or to the path
        given by the ``NEWSGROUPS_ARCHIVE_PATH`` environment variable.
    categories : list of str, optional
        Subset of category names to load.  ``None`` loads all 20.
    strip_headers : bool, default True
        Strip RFC-2822 header lines (From, Subject, Newsgroups, etc.) from
        each article, leaving only the prose body.

    Returns
    -------
    list of dict
        Each dict has keys: ``doc_id``, ``text``, ``label``, ``category``.
        ``doc_id`` is a 0-based integer, unique and contiguous in the
        returned list (re-indexed at the end so it is always 0-based).

    Notes
    -----
    * No ``split`` key is present – the train/test split is assigned by
      ``build_dataset.py`` using ``sklearn.model_selection.train_test_split``.
    * Documents with an empty body after header stripping are silently skipped.

    Examples
    --------
    >>> docs = load_dataset()
    >>> len(docs)
    19997          # may vary slightly by archive version
    >>> docs[0].keys()
    dict_keys(['doc_id', 'text', 'label', 'category'])
    """
    archive = _resolve_archive(archive_path)

    target_categories = set(categories) if categories is not None else set(CATEGORIES)
    unknown = target_categories - set(CATEGORIES)
    if unknown:
        raise ValueError(f"Unknown categories: {unknown}. Valid choices: {CATEGORIES}")

    records: List[dict] = []
    doc_id_counter = 0

    with zipfile.ZipFile(archive, "r") as outer_zip:
        # Open the inner tarball in streaming mode (no temp disk extraction).
        with outer_zip.open(_FULL_DATASET_TARBALL) as tar_fh:
            tar_bytes = tar_fh.read()

    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            # We only want leaf files: 20_newsgroups/<category>/<article_id>
            parts = member.name.split("/")
            if len(parts) != 3 or not parts[2]:
                continue  # skip directory entries and top-level tar root

            category = parts[1]
            if category not in target_categories:
                continue

            label = CATEGORIES.index(category)

            fh = tar.extractfile(member)
            if fh is None:
                continue

            raw = fh.read().decode("utf-8", errors="replace")
            text = _extract_body(raw, strip_headers)

            if not text:
                continue  # skip empty-body articles

            records.append(
                {
                    "doc_id": doc_id_counter,
                    "text": text,
                    "label": label,
                    "category": category,
                }
            )
            doc_id_counter += 1

    # Re-index doc_ids to a contiguous 0-based range in the final list.
    for new_id, record in enumerate(records):
        record["doc_id"] = new_id

    return records


def load_dataframe(
    archive_path: Optional[str | Path] = None,
    **kwargs,
):
    """
    Load the corpus as a ``pandas.DataFrame``.

    Columns: ``doc_id``, ``text``, ``label``, ``category``.

    Parameters
    ----------
    archive_path : str or Path, optional
        Passed through to :func:`load_dataset`.
    **kwargs
        Any other keyword argument accepted by :func:`load_dataset`.

    Returns
    -------
    pandas.DataFrame
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for load_dataframe(). "
            "Install it with: pip install pandas"
        ) from exc

    records = load_dataset(archive_path=archive_path, **kwargs)
    df = pd.DataFrame(records, columns=["doc_id", "text", "label", "category"])
    return df


def load_processed_dataset(parquet_path: Optional[str | Path] = None) -> "pd.DataFrame":
    """
    Load the common processed dataset parquet artifact.

    Columns: ``doc_id``, ``text``, ``clean_text``, ``label``, ``category``, ``split``.

    Parameters
    ----------
    parquet_path : str or Path, optional
        Path to ``processed_documents.parquet``.
        Defaults to ``utils.PARQUET_FILE``.

    Returns
    -------
    pandas.DataFrame
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for load_processed_dataset().") from exc

    if parquet_path is not None:
        path = Path(parquet_path)
    else:
        from utils import PARQUET_FILE
        path = PARQUET_FILE

    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset parquet file not found at '{path}'.\n"
            "Run 'python src/build_dataset.py' to generate it."
        )

    return pd.read_parquet(path, engine="pyarrow")


def dataset_info(archive_path: Optional[str | Path] = None) -> dict:
    """
    Return a summary dictionary describing the dataset without loading all text.

    Keys
    ----
    archive_path    : resolved path to the ZIP file
    num_categories  : number of categories found in the archive
    categories      : list of category names present
    counts          : dict mapping category → document count
    total_documents : total number of leaf documents across all categories

    Notes
    -----
    This is a fast scan that only inspects tar member names, not file
    contents.  Counts reflect the raw file count (before empty-body
    filtering applied during full :func:`load_dataset` calls).
    """
    archive = _resolve_archive(archive_path)
    counts: dict[str, int] = {}

    with zipfile.ZipFile(archive, "r") as outer_zip:
        with outer_zip.open(_FULL_DATASET_TARBALL) as tar_fh:
            tar_bytes = tar_fh.read()

    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            parts = member.name.split("/")
            if len(parts) == 3 and parts[2]:
                cat = parts[1]
                counts[cat] = counts.get(cat, 0) + 1

    categories_found = [c for c in CATEGORIES if c in counts]

    return {
        "archive_path": str(archive),
        "num_categories": len(categories_found),
        "categories": categories_found,
        "counts": {c: counts.get(c, 0) for c in categories_found},
        "total_documents": sum(counts.get(c, 0) for c in categories_found),
    }
