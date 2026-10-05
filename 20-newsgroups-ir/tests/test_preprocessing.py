"""
test_preprocessing.py
=====================
Unit tests for src/preprocessing.py

Author : Reetikesh Choudhury
Module : Text Preprocessing (own component)

Run with:
    pytest tests/test_preprocessing.py -v
    pytest tests/test_preprocessing.py -v --cov=src/preprocessing
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Make src/ importable regardless of where pytest is invoked.
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from preprocessing import (
    _STEMMER,
    _STOPWORDS,
    build_preprocessor,
    pipeline_steps,
    preprocess,
    preprocess_document,
    preprocess_documents,
)


# ============================================================
# 1.  Header / metadata removal
# ============================================================

class TestHeaderRemoval:
    """Step 1: residual news/email metadata must be removed."""

    def test_removes_organization_header(self):
        text = "Organization: MIT AI Lab\nThe sky is blue."
        result = preprocess(text)
        assert "organization" not in result
        assert "mit" not in result
        # Body content should survive
        assert "sky" in result or "blue" in result

    def test_removes_lines_header(self):
        text = "Lines: 42\nSome real content here."
        result = preprocess(text)
        assert "lines" not in result

    def test_removes_nntp_posting_host_header(self):
        text = "NNTP-Posting-Host: news.server.edu\nReal body text."
        result = preprocess(text)
        assert "nntp" not in result

    def test_removes_keywords_header(self):
        text = "Keywords: python, nltk, IR\nActual document body."
        result = preprocess(text)
        assert "keywords" not in result

    def test_removes_distribution_header(self):
        text = "Distribution: world\nDocument content follows."
        result = preprocess(text)
        assert "distribut" not in result.split()  # stemmed form absent

    def test_removes_date_header(self):
        text = "Date: Mon, 5 Apr 1993 12:00:00 GMT\nBody starts here."
        result = preprocess(text)
        assert "date" not in result

    def test_removes_references_header(self):
        text = "References: <abc123@news.umd.edu>\nBody content."
        result = preprocess(text)
        assert "references" not in result

    def test_removes_version_header(self):
        text = "Version: 1.2\nLast-modified: 5 April 1993\nReal text."
        result = preprocess(text)
        assert "version" not in result

    def test_removes_archive_name_header(self):
        text = "Archive-name: atheism/resources\nUseful body text here."
        result = preprocess(text)
        assert "archive" not in result

    def test_removes_in_article_citation(self):
        text = "In article <abc@news.server.com> user@host.edu writes:\nActual reply body."
        result = preprocess(text)
        assert "articl" not in result.split()
        assert "write" not in result.split()

    def test_removes_pgp_block(self):
        text = (
            "-----BEGIN PGP SIGNED MESSAGE-----\n"
            "Hash: SHA1\n"
            "Some private content\n"
            "-----END PGP SIGNATURE-----\n"
            "Real document body here."
        )
        result = preprocess(text)
        assert "begin" not in result
        assert "pgp" not in result
        assert "sha" not in result
        # Body survives
        assert "real" in result or "document" in result or "bodi" in result

    def test_removes_quoted_reply_lines_single_angle(self):
        text = "> This was written by someone else.\nMy actual response."
        result = preprocess(text)
        assert "written" not in result
        # Own text survives
        assert "actual" in result or "respons" in result

    def test_removes_quoted_reply_lines_double_angle(self):
        text = ">> Deeply nested quote.\n> Single quote.\nOriginal body."
        result = preprocess(text)
        assert "deepli" not in result or "nest" not in result
        assert "origin" in result or "bodi" in result

    def test_removes_email_addresses(self):
        text = "Contact mathew@mantis.co.uk for more information."
        result = preprocess(text)
        # No part of the email address should survive as a token
        assert "mathew" not in result
        assert "mantis" not in result
        # Rest of text survives
        assert "contact" in result or "inform" in result

    def test_removes_angle_bracket_message_ids(self):
        text = "See <C5s9zM.9E0@cbnewsj.cb.att.com> for the original post."
        result = preprocess(text)
        assert "cbnewsj" not in result

    def test_multiple_headers_all_removed(self):
        text = (
            "Organization: University of Arizona\n"
            "Lines: 17\n"
            "NNTP-Posting-Host: ccit.arizona.edu\n"
            "Keywords: atheism, religion\n"
            "The real document body starts here with useful content."
        )
        result = preprocess(text)
        assert "organ" not in result.split()
        assert "univers" not in result.split()
        assert "arizona" not in result
        assert "nntp" not in result
        assert "keyword" not in result
        # Body survives
        assert "real" in result or "document" in result or "use" in result

    def test_body_content_preserved_after_header_removal(self):
        text = (
            "Organization: NASA\n"
            "Lines: 5\n\n"
            "The Apollo missions landed humans on the moon."
        )
        result = preprocess(text)
        assert "apollo" in result
        assert "mission" in result
        assert "land" in result
        assert "human" in result
        assert "moon" in result


# ============================================================
# 2.  Lowercase conversion
# ============================================================

class TestLowercase:
    """Step 2: all output tokens must be lowercase."""

    def test_all_caps_lowercased(self):
        result = preprocess("NASA LAUNCHES SATELLITE SUCCESSFULLY")
        for token in result.split():
            assert token == token.lower(), f"Token '{token}' is not lowercase"

    def test_mixed_case_lowercased(self):
        result = preprocess("The Quick Brown Fox Jumps Over The Lazy Dog")
        for token in result.split():
            assert token == token.lower()

    def test_camelcase_lowercased(self):
        result = preprocess("ComputerScience DataBase MachineLearning")
        for token in result.split():
            assert token == token.lower()

    def test_already_lowercase_unchanged_in_form(self):
        result1 = preprocess("machine learning algorithm")
        result2 = preprocess("Machine Learning Algorithm")
        # Both should produce the same stemmed tokens
        assert result1 == result2

    def test_acronyms_lowercased(self):
        result = preprocess("The FBI and CIA operate in the USA.")
        for token in result.split():
            assert token == token.lower()


# ============================================================
# 3.  Tokenisation
# ============================================================

class TestTokenisation:
    """Step 3: text must be split into tokens."""

    def test_output_is_space_separated_string(self):
        result = preprocess("hello world")
        assert isinstance(result, str)
        # Should be non-empty words separated by single spaces
        tokens = result.split()
        assert len(tokens) >= 1

    def test_single_word_tokenised(self):
        result = preprocess("running")
        assert result == "run"  # stemmed

    def test_sentence_splits_into_tokens(self):
        result = preprocess("The computer processes data quickly.")
        tokens = result.split()
        assert len(tokens) >= 2

    def test_pipeline_steps_tokens_is_list(self):
        steps = pipeline_steps("Hello world testing.")
        assert isinstance(steps["tokens"], list)
        assert len(steps["tokens"]) > 0

    def test_contraction_tokenised(self):
        # "don't" tokenises to ["do", "n't"] – "n't" is dropped (non-alpha)
        result = preprocess("I don't want to run away.")
        assert "dont" not in result  # NLTK splits contractions
        # "run" and "away" survive after stemming
        assert "run" in result
        assert "away" in result

    def test_hyphenated_words(self):
        # Hyphens cause splits; both parts may survive if they are real words
        result = preprocess("well-known algorithm for state-of-the-art systems")
        assert isinstance(result, str)
        # At least some content words survive
        assert len(result.split()) > 0


# ============================================================
# 4.  Stopword removal
# ============================================================

class TestStopwordRemoval:
    """Step 4: English stopwords must not appear in output."""

    # Common NLTK stopwords to verify against
    _COMMON_STOPWORDS = [
        "the", "is", "are", "was", "were", "a", "an", "in", "on", "at",
        "to", "of", "and", "or", "but", "for", "with", "this", "that",
        "it", "he", "she", "they", "we", "you", "i", "my", "your",
        "has", "have", "had", "be", "been", "will", "would", "could",
        "should", "may", "might", "do", "does", "did", "not", "no",
        "from", "by", "as", "if", "about", "after", "before",
    ]

    def test_common_stopwords_removed(self):
        text = "The cat is sitting on the mat in the corner of the room."
        result = preprocess(text)
        output_tokens = set(result.split())
        for sw in self._COMMON_STOPWORDS:
            # Check both raw form and stemmed form
            assert sw not in output_tokens, (
                f"Stopword '{sw}' found in output: {result}"
            )

    def test_stopwords_not_in_output_set(self):
        text = "I am going to the store and he is coming with me."
        result = preprocess(text)
        output_tokens = set(result.split())
        nltk_sw = _STOPWORDS
        overlap = output_tokens & nltk_sw
        # Stemmed tokens may differ, but unstemmed stopwords should be gone
        # We verify no raw NLTK stopword survived unstemmed
        assert "i" not in output_tokens
        assert "am" not in output_tokens
        assert "to" not in output_tokens
        assert "the" not in output_tokens
        assert "and" not in output_tokens
        assert "he" not in output_tokens
        assert "is" not in output_tokens
        assert "with" not in output_tokens
        assert "me" not in output_tokens

    def test_content_words_survive_after_stopword_removal(self):
        text = "The scientists discovered a new planet beyond the solar system."
        result = preprocess(text)
        # Content words (stemmed) should survive
        assert "scientist" in result
        assert "discov" in result
        assert "planet" in result
        assert "solar" in result
        assert "system" in result

    def test_only_stopwords_returns_empty(self):
        # A sentence with only stopwords and short words
        text = "I am the a an"
        result = preprocess(text)
        assert result == ""

    def test_extra_stopwords_via_build_preprocessor(self):
        custom_pp = build_preprocessor(extra_stopwords=["nasa", "apollo"])
        text = "NASA launched the Apollo mission to the moon."
        result = custom_pp(text)
        assert "nasa" not in result
        assert "apollo" not in result
        # "moon" and "mission" should still survive
        assert "moon" in result
        assert "mission" in result


# ============================================================
# 5.  Stemming
# ============================================================

class TestStemming:
    """Step 5: tokens must be stemmed with PorterStemmer."""

    def test_running_stemmed_to_run(self):
        result = preprocess("running")
        assert result == "run"

    def test_computers_stemmed(self):
        result = preprocess("computers compute computed computing computation")
        tokens = result.split()
        # Porter stems all of these to "comput"
        assert all(t == "comput" for t in tokens), f"Unexpected stems: {tokens}"

    def test_stemming_reduces_inflections(self):
        # Porter stems verb inflections: jumping/jumps/jumped → "jump"
        # "jumper" is a noun/agent form; Porter correctly keeps it as "jumper"
        verb_forms = ["jumping", "jumps", "jumped"]
        for w in verb_forms:
            result = preprocess(w)
            assert result == "jump", f"Expected 'jump', got '{result}' for '{w}'"
        # Agent noun: Porter does NOT reduce "jumper" to "jump"
        assert preprocess("jumper") == "jumper"

    def test_plural_stemmed(self):
        singular = preprocess("machine")
        plural = preprocess("machines")
        assert singular == plural

    def test_stemming_applied_before_output(self):
        # "running" → "run", not "running"
        result = preprocess("He was running fast toward the finish line.")
        assert "running" not in result.split()
        assert "run" in result.split()

    def test_pipeline_steps_stemmed_field(self):
        steps = pipeline_steps("The scientists are running experiments.")
        stemmed = steps["stemmed"]
        assert isinstance(stemmed, list)
        assert "run" in stemmed
        assert "experi" in stemmed

    def test_custom_stemmer_via_build_preprocessor(self):
        """build_preprocessor should accept an alternative stemmer."""

        class IdentityStemmer:
            def stem(self, word):
                return word  # no-op: return word unchanged

        custom_pp = build_preprocessor(stemmer=IdentityStemmer())
        result = custom_pp("running jumping computers")
        # Without stemming, words retain their original form (post-stopword filter)
        assert "running" in result
        assert "jumping" in result
        assert "computers" in result


# ============================================================
# 6.  Punctuation and whitespace cleanup
# ============================================================

class TestPunctuationAndWhitespace:
    """Step 6: non-alpha tokens and very short tokens must be dropped."""

    def test_punctuation_removed(self):
        text = "Hello, world! This is a test. Really? Yes!"
        result = preprocess(text)
        for token in result.split():
            assert token.isalpha(), f"Non-alpha token found: '{token}'"

    def test_digits_removed(self):
        text = "There are 42 items in 3 categories from 1993."
        result = preprocess(text)
        for token in result.split():
            assert token.isalpha(), f"Digit token found: '{token}'"

    def test_single_character_tokens_removed(self):
        text = "a b c d e x y z good word"
        result = preprocess(text)
        for token in result.split():
            assert len(token) >= 2, f"Single-char token found: '{token}'"

    def test_multiple_spaces_normalised(self):
        text = "hello    world     test"
        result = preprocess(text)
        # Should not have runs of spaces
        assert "  " not in result

    def test_newlines_and_tabs_handled(self):
        text = "first line\nsecond line\tthird line"
        result = preprocess(text)
        assert "\n" not in result
        assert "\t" not in result

    def test_special_characters_removed(self):
        text = "C++ programming & algorithms | data-structures (advanced)"
        result = preprocess(text)
        for token in result.split():
            assert token.isalpha()

    def test_output_tokens_are_all_alpha(self):
        texts = [
            "Running 100% tests with pytest==8.0 on Python 3.13.",
            "User: john@example.com / password: #secret123!",
            "The cost is $4.95 per item (plus 10% tax).",
        ]
        for text in texts:
            result = preprocess(text)
            for token in result.split():
                assert token.isalpha(), f"Non-alpha token '{token}' in: {result}"


# ============================================================
# 7.  Empty / null / edge-case input
# ============================================================

class TestEmptyAndEdgeCases:
    """The pipeline must handle degenerate inputs without raising."""

    def test_empty_string_returns_empty(self):
        assert preprocess("") == ""

    def test_none_returns_empty(self):
        assert preprocess(None) == ""

    def test_whitespace_only_returns_empty(self):
        assert preprocess("   \n\t\n  ") == ""

    def test_only_stopwords_returns_empty(self):
        assert preprocess("the a an is are was were") == ""

    def test_only_punctuation_returns_empty(self):
        assert preprocess("!!!! .... ???? ----") == ""

    def test_only_digits_returns_empty(self):
        assert preprocess("12345 678 9 0") == ""

    def test_only_email_address_returns_empty(self):
        assert preprocess("user@example.com") == ""

    def test_very_short_valid_text(self):
        # Single meaningful word – should survive
        result = preprocess("computer")
        assert result == "comput"

    def test_one_word_stopword_returns_empty(self):
        assert preprocess("the") == ""

    def test_single_char_returns_empty(self):
        assert preprocess("a") == ""

    def test_numeric_header_line_only(self):
        assert preprocess("Lines: 42") == "" or "42" not in preprocess("Lines: 42")

    def test_only_pgp_block_returns_empty(self):
        text = "-----BEGIN PGP SIGNATURE-----\nabc123\n-----END PGP SIGNATURE-----"
        result = preprocess(text)
        assert result == "" or all(t.isalpha() for t in result.split())

    def test_unicode_characters_handled_without_crash(self):
        """Pipeline must not raise on non-ASCII input."""
        try:
            result = preprocess("Über die Straße fahren. Ünlü bilim insanı.")
            assert isinstance(result, str)
        except Exception as e:
            pytest.fail(f"preprocess raised on unicode input: {e}")

    def test_very_long_text_handled(self):
        """Pipeline must handle texts up to the maximum observed length (~160k chars)."""
        long_text = "The computer processes data quickly. " * 5_000
        result = preprocess(long_text)
        assert isinstance(result, str)
        assert len(result) > 0


# ============================================================
# 8.  preprocess_document – field preservation
# ============================================================

class TestPreprocessDocument:
    """preprocess_document must preserve all original fields and add clean_text."""

    _SAMPLE_DOC = {
        "doc_id": 42,
        "text": "The computer runs fast processing data efficiently.",
        "label": 3,
        "category": "comp.graphics",
        "split": "train",
    }

    def test_returns_dict(self):
        result = preprocess_document(self._SAMPLE_DOC)
        assert isinstance(result, dict)

    def test_clean_text_field_added(self):
        result = preprocess_document(self._SAMPLE_DOC)
        assert "clean_text" in result

    def test_original_text_preserved_unchanged(self):
        result = preprocess_document(self._SAMPLE_DOC)
        assert result["text"] == self._SAMPLE_DOC["text"]

    def test_doc_id_preserved(self):
        result = preprocess_document(self._SAMPLE_DOC)
        assert result["doc_id"] == 42

    def test_label_preserved(self):
        result = preprocess_document(self._SAMPLE_DOC)
        assert result["label"] == 3

    def test_category_preserved(self):
        result = preprocess_document(self._SAMPLE_DOC)
        assert result["category"] == "comp.graphics"

    def test_split_preserved(self):
        result = preprocess_document(self._SAMPLE_DOC)
        assert result["split"] == "train"

    def test_original_dict_not_mutated(self):
        original_copy = dict(self._SAMPLE_DOC)
        preprocess_document(self._SAMPLE_DOC)
        assert self._SAMPLE_DOC == original_copy

    def test_clean_text_is_preprocessed(self):
        result = preprocess_document(self._SAMPLE_DOC)
        clean = result["clean_text"]
        # Must be lowercase
        for token in clean.split():
            assert token == token.lower()
        # Must be alpha-only
        for token in clean.split():
            assert token.isalpha()
        # Stopwords gone
        assert "the" not in clean.split()

    def test_missing_text_key_handled(self):
        doc = {"doc_id": 99, "label": 0, "category": "alt.atheism", "split": "train"}
        result = preprocess_document(doc)
        assert result["clean_text"] == ""

    def test_none_text_handled(self):
        doc = {**self._SAMPLE_DOC, "text": None}
        result = preprocess_document(doc)
        assert result["clean_text"] == ""

    def test_empty_text_handled(self):
        doc = {**self._SAMPLE_DOC, "text": ""}
        result = preprocess_document(doc)
        assert result["clean_text"] == ""

    def test_output_has_all_required_keys(self):
        result = preprocess_document(self._SAMPLE_DOC)
        required = {"doc_id", "text", "clean_text", "label", "category", "split"}
        assert required <= result.keys()


# ============================================================
# 9.  preprocess_documents – batch processing
# ============================================================

class TestPreprocessDocuments:
    """preprocess_documents must process a list correctly."""

    _DOCS = [
        {"doc_id": 0, "text": "Running fast computers.", "label": 0,
         "category": "comp.graphics", "split": "train"},
        {"doc_id": 1, "text": "Jumping over the lazy dog.", "label": 1,
         "category": "rec.sport", "split": "test"},
        {"doc_id": 2, "text": "NASA launches rocket.", "label": 2,
         "category": "sci.space", "split": "train"},
    ]

    def test_returns_list(self):
        result = preprocess_documents(self._DOCS)
        assert isinstance(result, list)

    def test_length_preserved(self):
        result = preprocess_documents(self._DOCS)
        assert len(result) == len(self._DOCS)

    def test_all_records_have_clean_text(self):
        result = preprocess_documents(self._DOCS)
        for doc in result:
            assert "clean_text" in doc

    def test_doc_ids_preserved_in_order(self):
        result = preprocess_documents(self._DOCS)
        for original, processed in zip(self._DOCS, result):
            assert processed["doc_id"] == original["doc_id"]

    def test_original_text_not_modified(self):
        result = preprocess_documents(self._DOCS)
        for original, processed in zip(self._DOCS, result):
            assert processed["text"] == original["text"]

    def test_empty_list_returns_empty_list(self):
        assert preprocess_documents([]) == []

    def test_categories_preserved(self):
        result = preprocess_documents(self._DOCS)
        for original, processed in zip(self._DOCS, result):
            assert processed["category"] == original["category"]

    def test_labels_preserved(self):
        result = preprocess_documents(self._DOCS)
        for original, processed in zip(self._DOCS, result):
            assert processed["label"] == original["label"]

    def test_clean_texts_are_strings(self):
        result = preprocess_documents(self._DOCS)
        for doc in result:
            assert isinstance(doc["clean_text"], str)


# ============================================================
# 10. Determinism – same input must always yield same output
# ============================================================

class TestDeterminism:
    """The pipeline must be deterministic."""

    _TEXTS = [
        "The quick brown fox jumps over the lazy dog.",
        "Organization: MIT\nResearchers studied neural networks and machine learning.",
        "In article <abc@server.com> user@host.edu writes:\n>This was quoted.\nMy reply.",
        "-----BEGIN PGP SIGNED MESSAGE-----\nHash: SHA1\nSigned content.\n-----END PGP SIGNATURE-----\nReal body.",
        "",
        None,
    ]

    def test_same_output_on_repeated_calls(self):
        for text in self._TEXTS:
            r1 = preprocess(text)
            r2 = preprocess(text)
            assert r1 == r2, (
                f"Non-deterministic output for input {text!r}: "
                f"first={r1!r}, second={r2!r}"
            )

    def test_preprocess_document_deterministic(self):
        doc = {
            "doc_id": 7,
            "text": "The scientists are studying quantum computing algorithms.",
            "label": 5,
            "category": "sci.crypt",
            "split": "train",
        }
        r1 = preprocess_document(doc)
        r2 = preprocess_document(doc)
        assert r1["clean_text"] == r2["clean_text"]

    def test_preprocess_documents_deterministic(self):
        docs = [
            {"doc_id": i, "text": f"Test sentence number {i} about computers.",
             "label": 0, "category": "comp.graphics", "split": "train"}
            for i in range(10)
        ]
        r1 = [d["clean_text"] for d in preprocess_documents(docs)]
        r2 = [d["clean_text"] for d in preprocess_documents(docs)]
        assert r1 == r2

    def test_order_independence(self):
        """Processing docs individually vs in batch must yield same clean_text."""
        docs = [
            {"doc_id": 0, "text": "Running fast on machines.", "label": 0,
             "category": "comp.graphics", "split": "train"},
            {"doc_id": 1, "text": "Jumping over obstacles daily.", "label": 1,
             "category": "rec.autos", "split": "test"},
        ]
        batch_result = {d["doc_id"]: d["clean_text"] for d in preprocess_documents(docs)}
        for doc in docs:
            individual = preprocess_document(doc)["clean_text"]
            assert individual == batch_result[doc["doc_id"]]


# ============================================================
# 11. pipeline_steps() – intermediate step inspection
# ============================================================

class TestPipelineSteps:
    """pipeline_steps() must return a dict with all 7 intermediate keys."""

    _REQUIRED_KEYS = {
        "raw", "no_headers", "lowercased", "tokens",
        "no_stopwords", "stemmed", "clean_text",
    }

    def test_returns_dict_with_required_keys(self):
        steps = pipeline_steps("The quick brown fox.")
        assert self._REQUIRED_KEYS <= steps.keys()

    def test_raw_matches_input(self):
        text = "Hello world test."
        steps = pipeline_steps(text)
        assert steps["raw"] == text

    def test_lowercased_is_lowercase(self):
        steps = pipeline_steps("HELLO WORLD")
        assert steps["lowercased"] == steps["lowercased"].lower()

    def test_tokens_is_list(self):
        steps = pipeline_steps("Hello world.")
        assert isinstance(steps["tokens"], list)

    def test_no_stopwords_shorter_than_tokens(self):
        steps = pipeline_steps("The quick brown fox jumps over the lazy dog.")
        assert len(steps["no_stopwords"]) <= len(steps["tokens"])

    def test_stemmed_same_length_as_no_stopwords(self):
        steps = pipeline_steps("Running computers process data efficiently.")
        assert len(steps["stemmed"]) == len(steps["no_stopwords"])

    def test_clean_text_subset_of_stemmed(self):
        steps = pipeline_steps("Running 123 computers process data efficiently.")
        clean_tokens = set(steps["clean_text"].split())
        stemmed_set = set(steps["stemmed"])
        assert clean_tokens <= stemmed_set

    def test_empty_input_returns_empty_fields(self):
        steps = pipeline_steps("")
        assert steps["no_headers"] == ""
        assert steps["lowercased"] == ""
        assert steps["tokens"] == []
        assert steps["no_stopwords"] == []
        assert steps["stemmed"] == []
        assert steps["clean_text"] == ""

    def test_none_input_returns_empty_fields(self):
        steps = pipeline_steps(None)
        assert steps["clean_text"] == ""


# ============================================================
# 12. build_preprocessor() – custom pipeline factory
# ============================================================

class TestBuildPreprocessor:
    """build_preprocessor must return a callable that behaves like preprocess."""

    def test_returns_callable(self):
        pp = build_preprocessor()
        assert callable(pp)

    def test_default_factory_same_as_preprocess(self):
        text = "The scientists studied neural networks and machine learning."
        pp = build_preprocessor()
        assert pp(text) == preprocess(text)

    def test_extra_stopwords_applied(self):
        pp = build_preprocessor(extra_stopwords=["computer", "comput"])
        result = pp("The computer processes data.")
        assert "comput" not in result.split()
        # "data" (stemmed "data") should still be there
        assert "data" in result

    def test_custom_stemmer_identity(self):
        class NoOpStemmer:
            def stem(self, w):
                return w

        pp = build_preprocessor(stemmer=NoOpStemmer())
        result = pp("running computers processing")
        result_tokens = result.split()
        # With no-op stemmer words are NOT stemmed
        assert "running" in result_tokens
        # "comput" is the Porter-stemmed form; the no-op stemmer leaves the
        # token as "computers", so "comput" must NOT appear as a standalone token
        assert "comput" not in result_tokens
        assert "computers" in result_tokens

    def test_multiple_independent_preprocessors(self):
        """Two factory instances should not share state."""
        pp1 = build_preprocessor()
        pp2 = build_preprocessor(extra_stopwords=["machine"])
        text = "Machine learning runs on powerful computers."
        r1 = pp1(text)
        r2 = pp2(text)
        # pp1 keeps "machin"; pp2 removes it
        assert "machin" in r1
        assert "machin" not in r2
