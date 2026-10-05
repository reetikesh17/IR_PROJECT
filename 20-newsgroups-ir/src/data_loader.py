"""
data_loader.py
==============
Dataset loading module for the 20 Newsgroups IR project.

Author : Reetikesh Choudhury
Module : Dataset Loading (own component)

Overview
--------
Loads the 20 Newsgroups corpus from the local ``archive.zip`` file.
Each of the 20 category ``.txt`` files inside the archive has two
structured sections separated by a header-case convention:

* **Train** docs  – header pair ``Newsgroup: <cat>`` / ``document_id: <n>``
  (lowercase ``d``)
* **Test** docs   – header pair ``Newsgroup: <cat>`` / ``Document_id: <n>``
  (uppercase ``D``)
* **Unstructured** docs – ``From:`` boundary only, no ``Newsgroup:`` /
  ``document_id:`` header; assigned ``split="unstructured"`` and a
  generated doc_id.

The public API exposes three functions:

* :func:`load_dataset`       – full corpus → ``list[dict]``
* :func:`load_split`         – one split  → ``list[dict]``
* :func:`load_dataframe`     – full corpus → ``pandas.DataFrame``

Each record has the keys:

    doc_id   (int)  – globally stable integer, unique across the whole corpus
    text     (str)  – raw document body (headers stripped by default)
    label    (int)  – zero-based integer label (same ordering as categories)
    category (str)  – newsgroup name, e.g. ``"alt.atheism"``
    split    (str)  – ``"train"`` | ``"test"`` | ``"unstructured"``
"""

from __future__ import annotations

import io
import os
import re
import zipfile
from pathlib import Path
from typing import Iterator, List, Optional

# ---------------------------------------------------------------------------
# Default archive location – can be overridden via environment variable
# NEWSGROUPS_ARCHIVE_PATH or by passing ``archive_path`` explicitly.
# ---------------------------------------------------------------------------
_DEFAULT_ARCHIVE = Path(__file__).resolve().parents[3] / "archive.zip"

# Ordered list of the 20 newsgroup categories (matches the .txt filenames).
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

# Regex that matches the structured doc header block at the START of a document.
# Captures:
#   group 1 – category name   (from ``Newsgroup:`` line)
#   group 2 – document id     (from ``document_id:`` or ``Document_id:`` line)
#   group 3 – split indicator (``'d'`` for train, ``'D'`` for test)
_STRUCTURED_HEADER_RE = re.compile(
    r"^Newsgroup:\s*(.+?)\s*\n"
    r"(d|D)ocument_id:\s*(\d+)\s*\n",
    re.MULTILINE,
)

# Regex that detects a bare ``From:`` line used as a document boundary in
# the unstructured (header-less) sections of some category files.
_BARE_FROM_RE = re.compile(r"^From: ", re.MULTILINE)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_archive(archive_path: Optional[str | Path] = None) -> Path:
    """Return the resolved path to archive.zip, raising a clear error if missing."""
    if archive_path is not None:
        path = Path(archive_path)
    else:
        env_override = os.environ.get("NEWSGROUPS_ARCHIVE_PATH")
        path = Path(env_override) if env_override else _DEFAULT_ARCHIVE

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset archive not found at '{path}'.\n"
            "Set the NEWSGROUPS_ARCHIVE_PATH environment variable or pass "
            "`archive_path=` explicitly to load_dataset()."
        )
    return path


def _extract_body(raw_block: str, strip_headers: bool) -> str:
    """
    Return the text body of a document block.

    If ``strip_headers`` is True the email-style header lines at the top of
    the block (``From:``, ``Subject:``, ``Newsgroup:``, ``document_id:``,
    ``Lines:``, ``Date:``, ``Organization:``, ``Message-ID:``, ``References:``,
    ``NNTP-Posting-Host:``, ``X-*:``, etc.) are removed, leaving only the
    prose body.  An empty string is returned if nothing remains.
    """
    if not strip_headers:
        return raw_block.strip()

    lines = raw_block.split("\n")
    # Skip contiguous header lines at the top (key: value pattern) and the
    # blank separator line that follows them in RFC-2822 style.
    body_lines: List[str] = []
    in_header = True
    for line in lines:
        if in_header:
            # A blank line ends the header section.
            if line.strip() == "":
                in_header = False
            elif re.match(r"^[\w\-]+:", line):
                # Header line – skip it.
                continue
            else:
                # Non-header content while still in "header" zone means the
                # header block was very short / absent; start collecting now.
                in_header = False
                body_lines.append(line)
        else:
            body_lines.append(line)

    return "\n".join(body_lines).strip()


def _iter_structured_docs(
    content: str,
    label: int,
    strip_headers: bool,
    doc_id_counter: list,
) -> Iterator[dict]:
    """
    Yield document dicts from the structured (Newsgroup: / document_id:) sections.

    ``doc_id_counter`` is a single-element list used as a mutable integer so
    the counter persists across calls without needing a class.
    """
    # Split the file content on each structured header block.
    # re.split with a capturing group keeps the captured text in the result list.
    parts = _STRUCTURED_HEADER_RE.split(content)
    # parts pattern: [pre_text, cat, d_or_D, raw_id, body, cat, d_or_D, raw_id, body, ...]
    # Index 0 is text before the first match (may contain unstructured docs).
    # After that every group of 4 is: category, case_indicator, raw_id, body_text.

    i = 1  # skip pre_text (index 0)
    while i + 3 < len(parts):
        category_name = parts[i].strip()
        case_indicator = parts[i + 1]   # 'd' = train, 'D' = test
        raw_id = parts[i + 2]
        body_raw = parts[i + 3]

        split = "train" if case_indicator == "d" else "test"
        text = _extract_body(body_raw, strip_headers)

        if text:  # skip empty-body documents
            yield {
                "doc_id": doc_id_counter[0],
                "text": text,
                "label": label,
                "category": category_name,
                "split": split,
                "_source_doc_id": int(raw_id),  # original ID from file (for debugging)
            }
            doc_id_counter[0] += 1

        i += 4


def _iter_unstructured_docs(
    content: str,
    category: str,
    label: int,
    strip_headers: bool,
    doc_id_counter: list,
) -> Iterator[dict]:
    """
    Yield document dicts from the unstructured (``From:``-delimited) sections.

    Only blocks that have NO preceding ``Newsgroup:`` header (within 3 lines)
    are treated as truly unstructured; the rest belong to structured sections
    and are handled by ``_iter_structured_docs``.
    """
    lines = content.split("\n")
    # Collect line indices of structured headers so we can exclude them.
    structured_lines: set[int] = set()
    for m in re.finditer(r"^(?:N|n)ewsgroup:", content, re.MULTILINE):
        line_no = content[: m.start()].count("\n")
        # Mark this line and the next 2 as belonging to a structured block.
        for offset in range(3):
            structured_lines.add(line_no + offset)

    # Find bare ``From:`` line indices that are NOT part of a structured block.
    from_indices: List[int] = []
    for m in re.finditer(_BARE_FROM_RE, content):
        line_no = content[: m.start()].count("\n")
        # Check if any of the 3 lines before this From: is a doc_id: line.
        context_start = max(0, line_no - 3)
        context = "\n".join(lines[context_start:line_no])
        if not re.search(r"(?:d|D)ocument_id:", context):
            from_indices.append(m.start())

    if not from_indices:
        return

    # Reconstruct document blocks between consecutive bare From: positions.
    for idx, start_pos in enumerate(from_indices):
        end_pos = from_indices[idx + 1] if idx + 1 < len(from_indices) else len(content)
        block = content[start_pos:end_pos]
        text = _extract_body(block, strip_headers)

        if text:
            yield {
                "doc_id": doc_id_counter[0],
                "text": text,
                "label": label,
                "category": category,
                "split": "unstructured",
                "_source_doc_id": None,
            }
            doc_id_counter[0] += 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(
    archive_path: Optional[str | Path] = None,
    *,
    categories: Optional[List[str]] = None,
    splits: Optional[List[str]] = None,
    strip_headers: bool = True,
    include_unstructured: bool = False,
) -> List[dict]:
    """
    Load the 20 Newsgroups corpus from the local ZIP archive.

    Parameters
    ----------
    archive_path : str or Path, optional
        Path to ``archive.zip``.  Defaults to the sibling directory of the
        repo root, or the ``NEWSGROUPS_ARCHIVE_PATH`` environment variable.
    categories : list of str, optional
        Subset of category names to load.  ``None`` loads all 20.
    splits : list of str, optional
        Subset of splits to include: any combination of ``"train"``,
        ``"test"``, ``"unstructured"``.  ``None`` loads ``train`` and ``test``
        (excludes unstructured by default; use ``include_unstructured=True``
        as a shorthand).
    strip_headers : bool, default True
        Strip email-style header lines (From, Subject, etc.) from the body.
    include_unstructured : bool, default False
        Convenience flag to include the small set of ``"unstructured"`` docs
        (those without ``Newsgroup:``/``document_id:`` headers).  Ignored if
        ``splits`` is provided explicitly.

    Returns
    -------
    list of dict
        Each dict has keys: ``doc_id``, ``text``, ``label``, ``category``,
        ``split``.  ``doc_id`` is a globally stable 0-based integer that is
        unique within the returned list.

    Examples
    --------
    >>> docs = load_dataset()
    >>> len(docs)
    37656
    >>> docs[0].keys()
    dict_keys(['doc_id', 'text', 'label', 'category', 'split'])
    """
    archive = _resolve_archive(archive_path)

    target_categories = categories if categories is not None else CATEGORIES
    # Validate category names.
    unknown = set(target_categories) - set(CATEGORIES)
    if unknown:
        raise ValueError(f"Unknown categories: {unknown}. Valid choices: {CATEGORIES}")

    if splits is not None:
        target_splits: set[str] = set(splits)
    else:
        target_splits = {"train", "test"}
        if include_unstructured:
            target_splits.add("unstructured")

    records: List[dict] = []
    # Shared counter so doc_ids are globally unique across categories.
    doc_id_counter = [0]

    with zipfile.ZipFile(archive, "r") as zf:
        available_files = {name for name in zf.namelist()}

        for category in target_categories:
            filename = f"{category}.txt"
            if filename not in available_files:
                # Gracefully skip missing categories instead of crashing.
                continue

            label = CATEGORIES.index(category)

            with zf.open(filename) as fh:
                content = fh.read().decode("utf-8", errors="replace")

            # --- Structured docs (train + test) ---
            for doc in _iter_structured_docs(
                content, label, strip_headers, doc_id_counter
            ):
                if doc["split"] in target_splits:
                    records.append(doc)
                else:
                    # Still advance the counter so doc_ids stay stable
                    # regardless of which splits are requested.
                    doc_id_counter[0] -= 1  # undo the increment
                    # We do NOT want to skip the counter entirely;
                    # re-use this slot for the next accepted doc.

            # --- Unstructured docs ---
            if "unstructured" in target_splits:
                for doc in _iter_unstructured_docs(
                    content, category, label, strip_headers, doc_id_counter
                ):
                    records.append(doc)

    # Re-index doc_ids to be contiguous 0-based integers in the final list.
    for new_id, record in enumerate(records):
        record["doc_id"] = new_id

    # Remove the internal debugging key before returning.
    for record in records:
        record.pop("_source_doc_id", None)

    return records


def load_split(
    split: str,
    archive_path: Optional[str | Path] = None,
    **kwargs,
) -> List[dict]:
    """
    Convenience wrapper to load a single split.

    Parameters
    ----------
    split : str
        One of ``"train"``, ``"test"``, or ``"unstructured"``.
    archive_path : str or Path, optional
        Passed through to :func:`load_dataset`.
    **kwargs
        Any other keyword argument accepted by :func:`load_dataset`.

    Returns
    -------
    list of dict
    """
    if split not in {"train", "test", "unstructured"}:
        raise ValueError(
            f"Invalid split '{split}'. Choose from 'train', 'test', 'unstructured'."
        )
    return load_dataset(archive_path=archive_path, splits=[split], **kwargs)


def load_dataframe(
    archive_path: Optional[str | Path] = None,
    **kwargs,
):
    """
    Load the corpus as a ``pandas.DataFrame``.

    Columns: ``doc_id``, ``text``, ``label``, ``category``, ``split``.

    Parameters
    ----------
    archive_path : str or Path, optional
        Passed through to :func:`load_dataset`.
    **kwargs
        Any other keyword argument accepted by :func:`load_dataset`.

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    ImportError
        If ``pandas`` is not installed.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for load_dataframe(). "
            "Install it with: pip install pandas"
        ) from exc

    records = load_dataset(archive_path=archive_path, **kwargs)
    df = pd.DataFrame(records, columns=["doc_id", "text", "label", "category", "split"])
    return df


def dataset_info(archive_path: Optional[str | Path] = None) -> dict:
    """
    Return a summary dictionary describing the dataset without loading all text.

    Keys
    ----
    archive_path   : resolved path to the ZIP file
    num_categories : number of categories found in the archive
    categories     : list of category names
    counts         : dict mapping category → {train, test, unstructured, total}
    totals         : dict with aggregate train / test / unstructured / total counts
    """
    archive = _resolve_archive(archive_path)
    counts: dict[str, dict] = {}

    with zipfile.ZipFile(archive, "r") as zf:
        for category in CATEGORIES:
            filename = f"{category}.txt"
            if filename not in {e.filename for e in zf.infolist()}:
                continue
            with zf.open(filename) as fh:
                content = fh.read().decode("utf-8", errors="replace")

            train_n = len(re.findall(r"^document_id:", content, re.MULTILINE))
            test_n = len(re.findall(r"^Document_id:", content, re.MULTILINE))
            from_n = len(re.findall(r"^From: ", content, re.MULTILINE))
            unstruct_n = from_n - train_n - test_n

            counts[category] = {
                "train": train_n,
                "test": test_n,
                "unstructured": max(unstruct_n, 0),
                "total": from_n,
            }

    totals = {
        "train": sum(v["train"] for v in counts.values()),
        "test": sum(v["test"] for v in counts.values()),
        "unstructured": sum(v["unstructured"] for v in counts.values()),
        "total": sum(v["total"] for v in counts.values()),
    }

    return {
        "archive_path": str(archive),
        "num_categories": len(counts),
        "categories": list(counts.keys()),
        "counts": counts,
        "totals": totals,
    }
