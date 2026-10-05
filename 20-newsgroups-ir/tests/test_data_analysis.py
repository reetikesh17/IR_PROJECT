"""
test_data_analysis.py
=====================
Unit tests for src/data_analysis.py

Author : Reetikesh Choudhury
Module : Dataset Analysis (own component)

Run with:
    pytest tests/test_data_analysis.py -v
    pytest tests/test_data_analysis.py -v --cov=src/data_analysis
"""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Make src/ importable regardless of where pytest is invoked.
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from data_analysis import (
    AnalysisResult,
    CategoryStats,
    analyse_dataset,
    print_report,
    save_results,
)

# ---------------------------------------------------------------------------
# Archive location – skip whole suite if archive is absent.
# ---------------------------------------------------------------------------
_ARCHIVE_ENV = os.environ.get("NEWSGROUPS_ARCHIVE_PATH")
_DEFAULT_ARCHIVE = Path(__file__).resolve().parents[3] / "archive.zip"
_ARCHIVE_PATH: Path = Path(_ARCHIVE_ENV) if _ARCHIVE_ENV else _DEFAULT_ARCHIVE

pytestmark = pytest.mark.skipif(
    not _ARCHIVE_PATH.exists(),
    reason=(
        f"Dataset archive not found at '{_ARCHIVE_PATH}'. "
        "Set NEWSGROUPS_ARCHIVE_PATH and re-run."
    ),
)

# ---------------------------------------------------------------------------
# Session-scoped fixture – run the analysis once, reuse across all tests.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def result() -> AnalysisResult:
    """Run the full analysis once per test session."""
    return analyse_dataset(archive_path=_ARCHIVE_PATH)


@pytest.fixture(scope="session")
def result_with_unstructured() -> AnalysisResult:
    """Analysis including unstructured docs (single category for speed)."""
    from data_loader import CATEGORIES
    return analyse_dataset(
        archive_path=_ARCHIVE_PATH,
        include_unstructured=True,
    )


# ============================================================
# 1. Return type and structure
# ============================================================

class TestReturnType:
    """analyse_dataset() must return the correct dataclass."""

    def test_returns_analysis_result(self, result):
        assert isinstance(result, AnalysisResult)

    def test_per_category_list_of_category_stats(self, result):
        assert isinstance(result.per_category, list)
        for cs in result.per_category:
            assert isinstance(cs, CategoryStats)

    def test_duplicate_pairs_is_list(self, result):
        assert isinstance(result.duplicate_pairs, list)

    def test_generated_at_is_iso_string(self, result):
        import re
        assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", result.generated_at)

    def test_analysis_duration_is_positive(self, result):
        assert result.analysis_duration_s > 0


# ============================================================
# 2. Statistic 1 – Total number of documents
# ============================================================

class TestTotalDocuments:
    """Statistic 1: total_documents."""

    def test_total_documents_positive(self, result):
        assert result.total_documents > 0

    def test_total_documents_in_expected_range(self, result):
        # 37 588 verified from actual run; allow ±200 tolerance
        assert 37_000 <= result.total_documents <= 38_000, (
            f"Expected ~37 588, got {result.total_documents}"
        )

    def test_total_equals_train_plus_test(self, result):
        assert result.total_documents == result.train_count + result.test_count + result.unstructured_count

    def test_total_is_integer(self, result):
        assert isinstance(result.total_documents, int)


# ============================================================
# 3. Statistic 2 – Number of categories
# ============================================================

class TestNumCategories:
    """Statistic 2: num_categories."""

    def test_num_categories_is_20(self, result):
        assert result.num_categories == 20

    def test_per_category_length_matches(self, result):
        assert len(result.per_category) == result.num_categories

    def test_num_categories_is_integer(self, result):
        assert isinstance(result.num_categories, int)


# ============================================================
# 4. Statistic 3 – Documents per category
# ============================================================

class TestPerCategoryStats:
    """Statistic 3: per-category breakdown."""

    def test_all_20_categories_present(self, result):
        from data_loader import CATEGORIES
        found = {cs.category for cs in result.per_category}
        assert found == set(CATEGORIES), f"Missing: {set(CATEGORIES) - found}"

    def test_per_category_total_count_positive(self, result):
        for cs in result.per_category:
            assert cs.total_count > 0, f"{cs.category} has 0 total docs"

    def test_per_category_label_range(self, result):
        for cs in result.per_category:
            assert 0 <= cs.label <= 19, f"{cs.category} has label {cs.label}"

    def test_per_category_label_consistent_with_categories_list(self, result):
        from data_loader import CATEGORIES
        for cs in result.per_category:
            assert CATEGORIES[cs.label] == cs.category, (
                f"label {cs.label} → {CATEGORIES[cs.label]} but category={cs.category}"
            )

    def test_per_category_train_test_equal(self, result):
        """Every category should have equal train and test counts."""
        mismatches = [
            cs for cs in result.per_category
            if cs.train_count != cs.test_count
        ]
        assert len(mismatches) == 0, (
            f"Categories with unequal train/test: "
            f"{[(cs.category, cs.train_count, cs.test_count) for cs in mismatches]}"
        )

    def test_per_category_counts_sum_to_total(self, result):
        cat_sum = sum(cs.total_count for cs in result.per_category)
        assert cat_sum == result.total_documents, (
            f"Sum of per-category totals {cat_sum} != total_documents {result.total_documents}"
        )

    def test_per_category_avg_length_positive(self, result):
        for cs in result.per_category:
            assert cs.avg_length > 0, f"{cs.category}: avg_length = {cs.avg_length}"

    def test_per_category_min_le_avg_le_max(self, result):
        for cs in result.per_category:
            assert cs.min_length <= cs.avg_length <= cs.max_length, (
                f"{cs.category}: min={cs.min_length} avg={cs.avg_length} max={cs.max_length}"
            )

    def test_per_category_duplicate_count_non_negative(self, result):
        for cs in result.per_category:
            assert cs.duplicate_count >= 0


# ============================================================
# 5. Statistic 4 – Training document count
# ============================================================

class TestTrainCount:
    """Statistic 4: train_count."""

    def test_train_count_in_expected_range(self, result):
        assert 18_600 <= result.train_count <= 19_000, (
            f"Expected ~18 794, got {result.train_count}"
        )

    def test_train_count_is_integer(self, result):
        assert isinstance(result.train_count, int)

    def test_train_count_matches_per_category_sum(self, result):
        cat_train_sum = sum(cs.train_count for cs in result.per_category)
        assert cat_train_sum == result.train_count


# ============================================================
# 6. Statistic 5 – Testing document count
# ============================================================

class TestTestCount:
    """Statistic 5: test_count."""

    def test_test_count_in_expected_range(self, result):
        assert 18_600 <= result.test_count <= 19_000, (
            f"Expected ~18 794, got {result.test_count}"
        )

    def test_test_count_equals_train_count(self, result):
        """The dataset is perfectly symmetric."""
        assert result.test_count == result.train_count

    def test_test_count_is_integer(self, result):
        assert isinstance(result.test_count, int)

    def test_test_count_matches_per_category_sum(self, result):
        cat_test_sum = sum(cs.test_count for cs in result.per_category)
        assert cat_test_sum == result.test_count


# ============================================================
# 7. Statistic 6 / 7 / 8 – Document lengths
# ============================================================

class TestDocumentLengths:
    """Statistics 6, 7, 8: avg / min / max document length."""

    def test_avg_length_is_float(self, result):
        assert isinstance(result.avg_length, float)

    def test_avg_length_in_expected_range(self, result):
        # Verified: 1712.26 chars
        assert 1_000 <= result.avg_length <= 3_000, (
            f"Expected ~1712, got {result.avg_length}"
        )

    def test_min_length_is_positive(self, result):
        assert result.min_length > 0, "min_length must be > 0 (empty docs are filtered)"

    def test_min_length_is_integer(self, result):
        assert isinstance(result.min_length, int)

    def test_max_length_is_integer(self, result):
        assert isinstance(result.max_length, int)

    def test_max_length_in_expected_range(self, result):
        # Verified: 160 470 chars (comp.os.ms-windows.misc)
        assert 100_000 <= result.max_length <= 200_000, (
            f"Expected ~160 470, got {result.max_length}"
        )

    def test_min_le_avg_le_max(self, result):
        assert result.min_length <= result.avg_length <= result.max_length, (
            f"min={result.min_length} avg={result.avg_length} max={result.max_length}"
        )

    def test_avg_length_consistent_with_per_category(self, result):
        """Weighted mean of per-category averages should be close to global average."""
        weighted_sum = sum(
            cs.avg_length * cs.total_count for cs in result.per_category
        )
        total = sum(cs.total_count for cs in result.per_category)
        weighted_avg = weighted_sum / total
        # Allow 1 % relative tolerance (floating-point rounding across categories)
        rel_diff = abs(weighted_avg - result.avg_length) / result.avg_length
        assert rel_diff < 0.01, (
            f"Weighted avg {weighted_avg:.2f} deviates >1% from global avg {result.avg_length:.2f}"
        )


# ============================================================
# 8. Statistic 9 – Empty / null documents
# ============================================================

class TestEmptyDocuments:
    """Statistic 9: empty_null_count."""

    def test_empty_null_count_is_zero(self, result):
        """load_dataset() filters empty-body docs; count must be 0."""
        assert result.empty_null_count == 0

    def test_empty_null_count_is_integer(self, result):
        assert isinstance(result.empty_null_count, int)


# ============================================================
# 9. Statistic 10 – Duplicates
# ============================================================

class TestDuplicates:
    """Statistic 10: duplicate_count (unexpected duplicates only)."""

    def test_duplicate_count_non_negative(self, result):
        assert result.duplicate_count >= 0

    def test_duplicate_count_is_integer(self, result):
        assert isinstance(result.duplicate_count, int)

    def test_duplicate_count_in_expected_range(self, result):
        # Verified: 184 unexpected duplicates; well under 1 % of corpus
        assert 0 <= result.duplicate_count <= int(result.total_documents * 0.01), (
            f"Unexpected duplicate count {result.duplicate_count} exceeds 1% of corpus"
        )

    def test_expected_twin_count_is_correct(self, result):
        """Every doc should have exactly one twin (train↔test), so
        expected_twin_count == total_documents - unexpected dups."""
        expected = result.total_documents - result.duplicate_count
        assert result.expected_twin_count == expected, (
            f"expected_twin_count={result.expected_twin_count} but "
            f"total-dups={expected}"
        )

    def test_duplicate_pairs_list_capped_at_100(self, result):
        assert len(result.duplicate_pairs) <= 100

    def test_duplicate_pairs_have_required_keys(self, result):
        required = {
            "doc_id_a", "doc_id_b", "category_a", "category_b",
            "split_a", "split_b", "text_length", "note",
        }
        for pair in result.duplicate_pairs:
            missing = required - pair.keys()
            assert not missing, f"Duplicate pair missing keys: {missing}"

    def test_duplicate_pairs_text_length_positive(self, result):
        for pair in result.duplicate_pairs:
            assert pair["text_length"] > 0

    def test_duplicate_note_valid_value(self, result):
        valid_notes = {"same-split duplicate", "cross-category duplicate"}
        for pair in result.duplicate_pairs:
            assert pair["note"] in valid_notes, (
                f"Unexpected note value: {pair['note']!r}"
            )


# ============================================================
# 10. save_results()
# ============================================================

class TestSaveResults:
    """save_results() writes valid JSON and CSV files."""

    @pytest.fixture()
    def tmp_dir(self, tmp_path):
        return tmp_path

    def test_save_returns_two_paths(self, result, tmp_dir):
        json_path, csv_path = save_results(result, tmp_dir)
        assert isinstance(json_path, Path)
        assert isinstance(csv_path, Path)

    def test_json_file_created(self, result, tmp_dir):
        json_path, _ = save_results(result, tmp_dir)
        assert json_path.exists()
        assert json_path.suffix == ".json"

    def test_csv_file_created(self, result, tmp_dir):
        _, csv_path = save_results(result, tmp_dir)
        assert csv_path.exists()
        assert csv_path.suffix == ".csv"

    def test_json_is_valid_and_has_required_keys(self, result, tmp_dir):
        json_path, _ = save_results(result, tmp_dir)
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)
        required = {
            "total_documents", "num_categories", "train_count", "test_count",
            "avg_length", "min_length", "max_length", "empty_null_count",
            "duplicate_count", "per_category",
        }
        missing = required - data.keys()
        assert not missing, f"JSON missing keys: {missing}"

    def test_json_values_match_result(self, result, tmp_dir):
        json_path, _ = save_results(result, tmp_dir)
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["total_documents"] == result.total_documents
        assert data["num_categories"]  == result.num_categories
        assert data["train_count"]     == result.train_count
        assert data["test_count"]      == result.test_count
        assert data["empty_null_count"]== result.empty_null_count
        assert data["duplicate_count"] == result.duplicate_count

    def test_csv_has_20_data_rows(self, result, tmp_dir):
        _, csv_path = save_results(result, tmp_dir)
        with open(csv_path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 20, f"Expected 20 rows, got {len(rows)}"

    def test_csv_has_required_columns(self, result, tmp_dir):
        _, csv_path = save_results(result, tmp_dir)
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            cols = set(reader.fieldnames or [])
        required = {
            "category", "label", "train_count", "test_count",
            "total_count", "avg_length", "min_length", "max_length",
            "duplicate_count",
        }
        assert required <= cols, f"CSV missing columns: {required - cols}"

    def test_csv_totals_consistent_with_json(self, result, tmp_dir):
        json_path, csv_path = save_results(result, tmp_dir)
        with open(json_path, encoding="utf-8") as fh:
            jdata = json.load(fh)
        with open(csv_path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        csv_total = sum(int(r["total_count"]) for r in rows)
        assert csv_total == jdata["total_documents"]

    def test_save_creates_directory_if_missing(self, result, tmp_path):
        new_dir = tmp_path / "brand_new_subdir"
        assert not new_dir.exists()
        save_results(result, new_dir)
        assert new_dir.exists()


# ============================================================
# 11. include_unstructured flag
# ============================================================

class TestIncludeUnstructured:
    """analyse_dataset(include_unstructured=True) inflates counts correctly."""

    def test_unstructured_count_zero_by_default(self, result):
        assert result.unstructured_count == 0

    def test_unstructured_count_positive_when_requested(self, result_with_unstructured):
        assert result_with_unstructured.unstructured_count > 0

    def test_total_larger_with_unstructured(self, result, result_with_unstructured):
        assert result_with_unstructured.total_documents > result.total_documents

    def test_unstructured_in_expected_range(self, result_with_unstructured):
        # ~1 624 unstructured docs across all 20 categories
        assert 1_400 <= result_with_unstructured.unstructured_count <= 2_000, (
            f"Expected ~1 624, got {result_with_unstructured.unstructured_count}"
        )


# ============================================================
# 12. print_report() – smoke test (no crash, produces output)
# ============================================================

class TestPrintReport:
    """print_report() must produce output without raising."""

    def test_print_report_runs_without_error(self, result, capsys):
        print_report(result)
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_print_report_contains_key_values(self, result, capsys):
        print_report(result)
        out = capsys.readouterr().out
        assert str(result.total_documents) in out.replace(",", "")
        assert str(result.num_categories)  in out
        assert str(result.train_count)     in out.replace(",", "")

    def test_print_report_mentions_all_categories(self, result, capsys):
        from data_loader import CATEGORIES
        print_report(result)
        out = capsys.readouterr().out
        for cat in CATEGORIES:
            assert cat in out, f"Category '{cat}' not found in report output"


# ============================================================
# 13. Error handling
# ============================================================

class TestErrorHandling:
    """analyse_dataset() raises appropriate errors for bad inputs."""

    def test_missing_archive_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            analyse_dataset(archive_path="/nonexistent/archive.zip")
