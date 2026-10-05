"""
test_tfidf.py
=============
Unit tests for src/tfidf.py (TF-IDF retrieval system).

Author : Reetikesh Choudhury
Module : TF-IDF Retrieval (own component)

Run with:
    pytest tests/test_tfidf.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Make src/ importable regardless of working directory
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from tfidf import TFIDFSearcher, build_tfidf, search_tfidf


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_documents() -> list[dict]:
    """Sample corpus with distinct doc_ids and realistic text content."""
    return [
        {
            "doc_id": 101,
            "text": "The NASA space shuttle program launched rockets into Earth orbit.",
            "clean_text": "nasa space shuttl program launch rocket earth orbit",
            "label": 14,
            "category": "sci.space",
            "split": "train",
        },
        {
            "doc_id": 202,
            "text": "Computer graphics software renders 3D animation and realistic textures.",
            "clean_text": "comput graphic softwar render 3d anim realist textur",
            "label": 1,
            "category": "comp.graphics",
            "split": "train",
        },
        {
            "doc_id": 303,
            "text": "Public key encryption and PGP cryptography keep email communications secure.",
            "clean_text": "public key encrypt pgp cryptographi keep email commun secur",
            "label": 11,
            "category": "sci.crypt",
            "split": "train",
        },
        {
            "doc_id": 404,
            "text": "Motorcycle helmets and protective gear reduce risks for street riding.",
            "clean_text": "motorcycl helmet protect gear reduc risk street ride",
            "label": 8,
            "category": "rec.motorcycles",
            "split": "train",
        },
        {
            "doc_id": 505,
            "text": "Space exploration missions investigate planets and astronomy theories.",
            "clean_text": "space explor mission investig planet astronom theori",
            "label": 14,
            "category": "sci.space",
            "split": "test",
        },
    ]


@pytest.fixture
def fitted_searcher(sample_documents) -> TFIDFSearcher:
    """Pre-built TFIDFSearcher on sample documents."""
    return build_tfidf(sample_documents)


# ============================================================
# 1. Successful model building
# ============================================================

class TestModelBuilding:
    """Verify TFIDFSearcher initialization and build process."""

    def test_build_tfidf_returns_searcher_instance(self, sample_documents):
        searcher = build_tfidf(sample_documents)
        assert isinstance(searcher, TFIDFSearcher)

    def test_searcher_dimensions(self, fitted_searcher, sample_documents):
        assert fitted_searcher.num_documents == len(sample_documents)
        assert fitted_searcher.vocab_size > 0

    def test_build_tfidf_from_raw_text_without_clean_text(self):
        raw_docs = [
            {"doc_id": 1, "text": "Apollo astronauts landed on the moon in 1969."},
            {"doc_id": 2, "text": "3D graphics rendering software for computers."},
        ]
        searcher = build_tfidf(raw_docs)
        assert searcher.num_documents == 2
        results = searcher.search("moon astronaut")
        assert len(results) > 0
        assert results[0]["doc_id"] == 1

    def test_build_empty_corpus_raises_value_error(self):
        with pytest.raises(ValueError, match="Cannot build TF-IDF index on an empty"):
            build_tfidf([])


# ============================================================
# 2. Basic search functionality
# ============================================================

class TestBasicSearch:
    """Verify relevant documents are retrieved for standard queries."""

    def test_single_term_query(self, fitted_searcher):
        results = fitted_searcher.search("graphics", top_k=5)
        assert len(results) > 0
        assert results[0]["doc_id"] == 202

    def test_multi_term_query(self, fitted_searcher):
        results = fitted_searcher.search("space shuttle orbit", top_k=5)
        assert len(results) > 0
        assert results[0]["doc_id"] == 101

    def test_search_tfidf_convenience_function(self, fitted_searcher):
        results = search_tfidf("cryptography encryption", searcher=fitted_searcher, top_k=5)
        assert len(results) > 0
        assert results[0]["doc_id"] == 303

    def test_search_tfidf_global_state_flow(self, sample_documents):
        """Teammates pattern: build_tfidf(docs) then search_tfidf('query')."""
        build_tfidf(sample_documents)
        results = search_tfidf("space exploration", top_k=5)
        assert len(results) > 0
        assert results[0]["doc_id"] in (101, 505)

    def test_get_document_helper(self, sample_documents):
        """Teammates pattern: get_document(doc_id) returns document dict."""
        build_tfidf(sample_documents)
        from tfidf import get_document
        doc = get_document(101)
        assert doc is not None
        assert doc["doc_id"] == 101
        assert doc["category"] == "sci.space"
        assert "text" in doc
        assert "clean_text" in doc


# ============================================================
# 3. Ranking order
# ============================================================

class TestRankingOrder:
    """Verify documents are sorted strictly in descending order of similarity score."""

    def test_scores_strictly_descending(self, fitted_searcher):
        results = fitted_searcher.search("space shuttle orbit planet rocket", top_k=5)
        assert len(results) >= 2
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), f"Scores not descending: {scores}"

    def test_most_relevant_doc_ranked_first(self, fitted_searcher):
        # Query matching doc 505 ("space explor mission planet astronomy") and doc 101 ("space shuttle ... rocket")
        results = fitted_searcher.search("astronomy planets exploration", top_k=5)
        assert len(results) > 0
        assert results[0]["doc_id"] == 505

    def test_ranks_are_sequential_starting_at_one(self, fitted_searcher):
        results = fitted_searcher.search("space graphics encryption motorcycle", top_k=10)
        ranks = [r["rank"] for r in results]
        expected_ranks = list(range(1, len(results) + 1))
        assert ranks == expected_ranks, f"Ranks expected {expected_ranks}, got {ranks}"


# ============================================================
# 4. Top-K functionality
# ============================================================

class TestTopK:
    """Verify top_k parameter behavior."""

    def test_top_k_limits_result_count(self, fitted_searcher):
        # "space" matches doc 101 and doc 505
        results_top1 = fitted_searcher.search("space", top_k=1)
        assert len(results_top1) == 1

        results_top5 = fitted_searcher.search("space", top_k=5)
        assert len(results_top5) == 2  # total 2 matching docs

    def test_top_k_exceeding_total_docs_returns_all_matching(self, fitted_searcher):
        results = fitted_searcher.search("space", top_k=100)
        assert len(results) == 2

    def test_top_k_zero_returns_empty_list(self, fitted_searcher):
        results = fitted_searcher.search("space", top_k=0)
        assert results == []

    def test_negative_top_k_returns_empty_list(self, fitted_searcher):
        results = fitted_searcher.search("space", top_k=-5)
        assert results == []


# ============================================================
# 5. Unknown query terms & empty query handling
# ============================================================

class TestEdgeCases:
    """Verify graceful handling of degenerate or OOV queries."""

    def test_empty_string_query_returns_empty_list(self, fitted_searcher):
        assert fitted_searcher.search("") == []

    def test_none_query_returns_empty_list(self, fitted_searcher):
        assert fitted_searcher.search(None) == []

    def test_whitespace_query_returns_empty_list(self, fitted_searcher):
        assert fitted_searcher.search("   \n\t  ") == []

    def test_only_stopword_query_returns_empty_list(self, fitted_searcher):
        # "the is are" pre-processes to ""
        assert fitted_searcher.search("the is are") == []

    def test_unknown_out_of_vocab_query_returns_empty_list(self, fitted_searcher):
        # Terms not in corpus vocabulary
        results = fitted_searcher.search("xyzzy nonexistingsuperword 99999")
        assert results == []

    def test_mixed_known_and_unknown_terms(self, fitted_searcher):
        # "graphics" is known, "nonexistentword" is unknown
        results = fitted_searcher.search("graphics nonexistentword")
        assert len(results) > 0
        assert results[0]["doc_id"] == 202


# ============================================================
# 6. Stable doc_id mapping
# ============================================================

class TestStableDocIdMapping:
    """Verify doc_id returned matches original corpus doc_id, NOT matrix row index."""

    def test_doc_ids_match_custom_corpus_ids(self, fitted_searcher):
        results = fitted_searcher.search("motorcycle", top_k=1)
        assert len(results) == 1
        # Row 3 in matrix corresponds to doc_id 404
        assert results[0]["doc_id"] == 404
        assert isinstance(results[0]["doc_id"], int)

    def test_doc_id_mapping_retains_arbitrary_ids(self):
        docs = [
            {"doc_id": 9999, "clean_text": "alpha beta gamma"},
            {"doc_id": 8888, "clean_text": "delta epsilon zeta"},
        ]
        searcher = build_tfidf(docs)
        res = searcher.search("gamma", top_k=1)
        assert res[0]["doc_id"] == 9999

        res2 = searcher.search("zeta", top_k=1)
        assert res2[0]["doc_id"] == 8888


# ============================================================
# 7. Result format specification
# ============================================================

class TestResultFormat:
    """Verify returned result data structure matches requirement exactly."""

    def test_result_structure_keys_and_types(self, fitted_searcher):
        results = fitted_searcher.search("graphics software 3d", top_k=3)
        assert len(results) > 0
        for item in results:
            assert set(item.keys()) == {"doc_id", "score", "rank"}
            assert isinstance(item["doc_id"], int)
            assert isinstance(item["score"], float)
            assert isinstance(item["rank"], int)

    def test_first_rank_is_one(self, fitted_searcher):
        results = fitted_searcher.search("space", top_k=2)
        assert results[0]["rank"] == 1
        assert results[1]["rank"] == 2


# ============================================================
# 8. Persistence (save / load)
# ============================================================

class TestPersistence:
    """Verify saving and loading TFIDFSearcher artifacts."""

    def test_save_and_load(self, fitted_searcher, tmp_path):
        save_path = tmp_path / "tfidf_test.pkl"
        fitted_searcher.save(save_path)
        assert save_path.exists()

        loaded_searcher = TFIDFSearcher.load(save_path)
        assert loaded_searcher.num_documents == fitted_searcher.num_documents

        orig_res = fitted_searcher.search("graphics", top_k=3)
        loaded_res = loaded_searcher.search("graphics", top_k=3)
        assert orig_res == loaded_res
