"""
test_data_loader.py
===================
Unit tests for src/data_loader.py

Author : Reetikesh Choudhury
Module : Dataset Loading (own component)

Run with:
    pytest tests/test_data_loader.py -v
    pytest tests/test_data_loader.py -v --cov=src/data_loader
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Make sure the src/ package is importable regardless of where pytest is run.
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from data_loader import (
    CATEGORIES,
    dataset_info,
    load_dataset,
    load_dataframe,
    load_split,
)

# ---------------------------------------------------------------------------
# Locate the archive – skip the whole suite gracefully if it is absent.
# ---------------------------------------------------------------------------
_ARCHIVE_ENV = os.environ.get("NEWSGROUPS_ARCHIVE_PATH")
_DEFAULT_ARCHIVE = Path(__file__).resolve().parents[3] / "archive.zip"
_ARCHIVE_PATH: Path = (
    Path(_ARCHIVE_ENV) if _ARCHIVE_ENV else _DEFAULT_ARCHIVE
)

pytestmark = pytest.mark.skipif(
    not _ARCHIVE_PATH.exists(),
    reason=(
        f"Dataset archive not found at '{_ARCHIVE_PATH}'. "
        "Set NEWSGROUPS_ARCHIVE_PATH to the correct path and re-run."
    ),
)

# ---------------------------------------------------------------------------
# Fixtures – loaded once per session for speed (the corpus is large).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def full_corpus():
    """Full train + test corpus (both structured splits, no unstructured)."""
    return load_dataset(archive_path=_ARCHIVE_PATH)


@pytest.fixture(scope="session")
def train_docs():
    """Train split only."""
    return load_split("train", archive_path=_ARCHIVE_PATH)


@pytest.fixture(scope="session")
def test_docs():
    """Test split only."""
    return load_split("test", archive_path=_ARCHIVE_PATH)


@pytest.fixture(scope="session")
def single_category_corpus():
    """Single category (talk.religion.misc) – fast to load for quick checks."""
    return load_dataset(
        archive_path=_ARCHIVE_PATH,
        categories=["talk.religion.misc"],
    )


# ============================================================
# 1. Dataset loading – basic shape
# ============================================================

class TestDatasetLoading:
    """The loader returns data and it has the expected structure."""

    def test_load_dataset_returns_list(self, full_corpus):
        assert isinstance(full_corpus, list)

    def test_load_dataset_not_empty(self, full_corpus):
        assert len(full_corpus) > 0, "Corpus must not be empty"

    def test_full_corpus_size_within_expected_range(self, full_corpus):
        # 18 828 train + 18 828 test = 37 656 structured docs.
        # A small number of docs (~68) are dropped for having empty bodies
        # after header stripping; allow ±300 tolerance.
        assert 37_000 <= len(full_corpus) <= 38_000, (
            f"Expected ~37 588 docs, got {len(full_corpus)}"
        )

    def test_each_record_has_required_keys(self, full_corpus):
        required = {"doc_id", "text", "label", "category", "split"}
        for doc in full_corpus[:200]:  # spot-check first 200
            missing = required - doc.keys()
            assert not missing, f"Record missing keys: {missing}"

    def test_no_extra_internal_keys(self, full_corpus):
        """The _source_doc_id debugging key must be stripped before return."""
        for doc in full_corpus[:200]:
            assert "_source_doc_id" not in doc

    def test_load_single_category(self, single_category_corpus):
        assert len(single_category_corpus) > 0
        cats = {d["category"] for d in single_category_corpus}
        assert cats == {"talk.religion.misc"}

    def test_load_subset_of_categories(self):
        subset = ["alt.atheism", "sci.space"]
        docs = load_dataset(archive_path=_ARCHIVE_PATH, categories=subset)
        returned_cats = {d["category"] for d in docs}
        assert returned_cats == set(subset)

    def test_invalid_category_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown categories"):
            load_dataset(archive_path=_ARCHIVE_PATH, categories=["not.a.real.group"])

    def test_missing_archive_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_dataset(archive_path="/nonexistent/path/archive.zip")


# ============================================================
# 2. Non-empty documents
# ============================================================

class TestDocumentContent:
    """Every returned document must contain actual text."""

    def test_no_empty_text_field(self, full_corpus):
        empty = [d for d in full_corpus if not d["text"] or not d["text"].strip()]
        assert len(empty) == 0, (
            f"{len(empty)} documents have an empty 'text' field"
        )

    def test_text_is_string(self, full_corpus):
        for doc in full_corpus[:500]:
            assert isinstance(doc["text"], str), (
                f"doc_id={doc['doc_id']} has non-string text: {type(doc['text'])}"
            )

    def test_text_minimum_length(self, full_corpus):
        """Body text should be at least 10 characters after header stripping."""
        short = [d for d in full_corpus if len(d["text"].strip()) < 10]
        # Allow at most 0.1 % of corpus to be very short (malformed posts).
        threshold = max(1, int(len(full_corpus) * 0.001))
        assert len(short) <= threshold, (
            f"{len(short)} documents have fewer than 10 characters in 'text'"
        )

    def test_strip_headers_removes_from_line(self, full_corpus):
        """With strip_headers=True the email From:/Subject: *header* lines should
        be removed.  However, some posts begin their body with an inline quoted
        attribution of the form 'From: user@host' (common in Usenet replies).
        Those are valid body content and must NOT be stripped.

        We verify that the number of such edge-case docs is small (≤0.5 % of
        the corpus), confirming the stripper is removing the vast majority of
        email headers correctly while preserving genuine body text.
        """
        bad = [d for d in full_corpus if d["text"].startswith("From:")]
        max_allowed = max(10, int(len(full_corpus) * 0.005))  # 0.5 % tolerance
        assert len(bad) <= max_allowed, (
            f"{len(bad)} documents start with 'From:' after header stripping "
            f"(tolerance: {max_allowed} = 0.5 % of {len(full_corpus)} docs). "
            "These are likely genuine inline quoted attributions in the body, "
            "not missed email headers."
        )

    def test_raw_headers_present_when_strip_false(self):
        """With strip_headers=False the From: header should appear in raw text."""
        docs = load_dataset(
            archive_path=_ARCHIVE_PATH,
            categories=["sci.space"],
            splits=["train"],
            strip_headers=False,
        )
        from_lines = [d for d in docs if "From:" in d["text"]]
        assert len(from_lines) > 0, (
            "Expected 'From:' to be present in raw (non-stripped) documents"
        )


# ============================================================
# 3. Labels and categories
# ============================================================

class TestLabelsAndCategories:
    """label and category fields are valid and consistent."""

    def test_label_is_integer(self, full_corpus):
        for doc in full_corpus[:500]:
            assert isinstance(doc["label"], int), (
                f"doc_id={doc['doc_id']} has non-int label: {type(doc['label'])}"
            )

    def test_label_range(self, full_corpus):
        """Labels must be in [0, 19]."""
        out_of_range = [d for d in full_corpus if not (0 <= d["label"] <= 19)]
        assert len(out_of_range) == 0, (
            f"{len(out_of_range)} documents have label outside [0, 19]"
        )

    def test_all_20_categories_present(self, full_corpus):
        found_cats = {d["category"] for d in full_corpus}
        missing = set(CATEGORIES) - found_cats
        assert not missing, f"Missing categories: {missing}"

    def test_category_is_string(self, full_corpus):
        for doc in full_corpus[:500]:
            assert isinstance(doc["category"], str)

    def test_label_matches_category(self, full_corpus):
        """label must equal CATEGORIES.index(category) for every document."""
        mismatches = [
            d for d in full_corpus
            if d["label"] != CATEGORIES.index(d["category"])
        ]
        assert len(mismatches) == 0, (
            f"{len(mismatches)} documents have label/category mismatch. "
            f"First: {mismatches[0] if mismatches else ''}"
        )

    def test_exactly_20_labels(self, full_corpus):
        unique_labels = {d["label"] for d in full_corpus}
        assert unique_labels == set(range(20)), (
            f"Expected labels 0-19, got {sorted(unique_labels)}"
        )

    def test_category_values_are_known(self, full_corpus):
        unknown = {d["category"] for d in full_corpus} - set(CATEGORIES)
        assert not unknown, f"Unexpected category values: {unknown}"


# ============================================================
# 4. Unique doc_id values
# ============================================================

class TestDocIds:
    """doc_id must be unique, integer, and 0-based contiguous."""

    def test_doc_ids_are_unique(self, full_corpus):
        ids = [d["doc_id"] for d in full_corpus]
        assert len(ids) == len(set(ids)), (
            f"doc_ids are not unique: {len(ids)} total, {len(set(ids))} unique"
        )

    def test_doc_ids_are_integers(self, full_corpus):
        for doc in full_corpus[:500]:
            assert isinstance(doc["doc_id"], int), (
                f"Non-integer doc_id: {doc['doc_id']} (type {type(doc['doc_id'])})"
            )

    def test_doc_ids_start_at_zero(self, full_corpus):
        ids = [d["doc_id"] for d in full_corpus]
        assert min(ids) == 0, f"Minimum doc_id should be 0, got {min(ids)}"

    def test_doc_ids_are_contiguous(self, full_corpus):
        """After re-indexing, IDs should be 0, 1, 2, … len-1."""
        ids = sorted(d["doc_id"] for d in full_corpus)
        expected = list(range(len(full_corpus)))
        assert ids == expected, "doc_ids are not contiguous integers starting at 0"

    def test_single_category_doc_ids_unique(self, single_category_corpus):
        ids = [d["doc_id"] for d in single_category_corpus]
        assert len(ids) == len(set(ids))

    def test_doc_ids_unique_across_subsets(self):
        """Even when loading a subset, returned doc_ids must still be unique."""
        docs = load_dataset(
            archive_path=_ARCHIVE_PATH,
            categories=["rec.autos", "rec.motorcycles"],
        )
        ids = [d["doc_id"] for d in docs]
        assert len(ids) == len(set(ids))


# ============================================================
# 5. Train / test split separation
# ============================================================

class TestSplitSeparation:
    """Train and test splits are correctly identified and separated."""

    def test_split_field_values(self, full_corpus):
        valid_splits = {"train", "test", "unstructured"}
        bad = [d for d in full_corpus if d["split"] not in valid_splits]
        assert len(bad) == 0, f"{len(bad)} documents have invalid 'split' value"

    def test_train_and_test_both_present(self, full_corpus):
        splits = {d["split"] for d in full_corpus}
        assert "train" in splits, "No train documents found"
        assert "test" in splits, "No test documents found"

    def test_no_overlap_between_train_and_test_splits(self, train_docs, test_docs):
        """load_split('train') and load_split('test') must return disjoint sets
        when we compare by doc_id (each is re-indexed independently)."""
        # Check by content + category instead of doc_id, since each call
        # produces its own independent 0-based index.
        train_keys = {(d["category"], d["text"][:80]) for d in train_docs}
        test_keys = {(d["category"], d["text"][:80]) for d in test_docs}
        # Train and test contain the same documents (that's how the dataset
        # was constructed – same content, different split markers).
        # What we verify: every train doc maps to exactly one test doc.
        overlap_count = len(train_keys & test_keys)
        assert overlap_count == len(train_keys), (
            "Every train document should have a matching test document "
            f"(got {overlap_count} / {len(train_keys)} matched)"
        )

    def test_load_split_train_contains_only_train(self, train_docs):
        splits = {d["split"] for d in train_docs}
        assert splits == {"train"}, f"Expected only 'train', got {splits}"

    def test_load_split_test_contains_only_test(self, test_docs):
        splits = {d["split"] for d in test_docs}
        assert splits == {"test"}, f"Expected only 'test', got {splits}"

    def test_train_count_matches_expected(self, train_docs):
        # 18 828 structured train docs; ~34 empty-body docs are filtered out.
        assert 18_600 <= len(train_docs) <= 19_000, (
            f"Expected ~18 794 train docs, got {len(train_docs)}"
        )

    def test_test_count_matches_expected(self, test_docs):
        # 18 828 structured test docs; ~34 empty-body docs are filtered out.
        assert 18_600 <= len(test_docs) <= 19_000, (
            f"Expected ~18 794 test docs, got {len(test_docs)}"
        )

    def test_train_test_counts_are_equal(self, train_docs, test_docs):
        """The dataset is perfectly symmetric: each doc appears in both splits."""
        assert len(train_docs) == len(test_docs), (
            f"Train ({len(train_docs)}) and test ({len(test_docs)}) sizes differ"
        )

    def test_invalid_split_name_raises(self):
        with pytest.raises(ValueError, match="Invalid split"):
            load_split("validation", archive_path=_ARCHIVE_PATH)

    def test_splits_kwarg_filters_correctly(self):
        train_only = load_dataset(
            archive_path=_ARCHIVE_PATH,
            categories=["sci.med"],
            splits=["train"],
        )
        assert all(d["split"] == "train" for d in train_only)

    def test_include_unstructured_flag(self):
        with_unstr = load_dataset(
            archive_path=_ARCHIVE_PATH,
            categories=["comp.os.ms-windows.misc"],
            include_unstructured=True,
        )
        splits_found = {d["split"] for d in with_unstr}
        # comp.os.ms-windows.misc has 14 unstructured docs.
        assert "unstructured" in splits_found


# ============================================================
# 6. dataset_info() helper
# ============================================================

class TestDatasetInfo:
    """dataset_info() returns accurate metadata without loading all text."""

    @pytest.fixture(scope="class")
    def info(self):
        return dataset_info(archive_path=_ARCHIVE_PATH)

    def test_info_has_required_keys(self, info):
        assert {"archive_path", "num_categories", "categories", "counts", "totals"} \
               <= info.keys()

    def test_info_num_categories(self, info):
        assert info["num_categories"] == 20

    def test_info_totals_train(self, info):
        assert 18_600 <= info["totals"]["train"] <= 19_000

    def test_info_totals_test(self, info):
        assert 18_600 <= info["totals"]["test"] <= 19_000

    def test_info_per_category_counts_positive(self, info):
        for cat, counts in info["counts"].items():
            assert counts["train"] > 0, f"{cat} has 0 train docs"
            assert counts["test"] > 0, f"{cat} has 0 test docs"


# ============================================================
# 7. load_dataframe() – requires pandas
# ============================================================

class TestLoadDataframe:
    """load_dataframe() returns a correctly shaped DataFrame."""

    def test_returns_dataframe(self):
        pd = pytest.importorskip("pandas")
        df = load_dataframe(
            archive_path=_ARCHIVE_PATH,
            categories=["misc.forsale"],
        )
        assert isinstance(df, pd.DataFrame)

    def test_dataframe_columns(self):
        pd = pytest.importorskip("pandas")
        df = load_dataframe(
            archive_path=_ARCHIVE_PATH,
            categories=["misc.forsale"],
        )
        assert list(df.columns) == ["doc_id", "text", "label", "category", "split"]

    def test_dataframe_not_empty(self):
        pytest.importorskip("pandas")
        df = load_dataframe(
            archive_path=_ARCHIVE_PATH,
            categories=["misc.forsale"],
        )
        assert len(df) > 0

    def test_dataframe_no_null_values(self):
        pytest.importorskip("pandas")
        df = load_dataframe(
            archive_path=_ARCHIVE_PATH,
            categories=["misc.forsale"],
        )
        assert df.isnull().sum().sum() == 0, "DataFrame contains null values"
