"""
preprocessing.py
================
Text preprocessing pipeline for the 20 Newsgroups IR project.

Author : Reetikesh Choudhury
Module : Text Preprocessing (own component)

Pipeline (applied in this fixed order)
---------------------------------------
  Step 1 – Residual header / metadata removal
             Removes leftover RFC-2822 / news-protocol header lines that the
             data loader did not strip (Archive-name:, Organization:, Lines:,
             Keywords:, Distribution:, NNTP-Posting-Host:, References:, Date:,
             Version:, Last-modified:, etc.).
             Also strips:
               • PGP armour blocks (-----BEGIN / -----END PGP …-----)
               • Quoted reply lines  (lines beginning with > or |>)
               • Usenet citation lines ("In article <…> … writes:")
               • Email addresses    (user@domain.tld)
               • Angle-bracket message-id tokens  (<…@…>)
  Step 2 – Lowercase conversion
  Step 3 – Tokenisation           (NLTK word_tokenize)
  Step 4 – Stopword removal       (NLTK English stopwords)
  Step 5 – Stemming               (NLTK PorterStemmer)
  Step 6 – Punctuation / whitespace cleanup
             Non-alphabetic tokens are dropped; result rejoined with spaces.

Public API
----------
  preprocess(text: str) -> str
      Apply the full pipeline to a single text string.
      Returns a clean, space-separated string of stemmed tokens.
      Returns "" for empty / null input.

  preprocess_document(doc: dict) -> dict
      Apply the pipeline to one corpus record.
      Returns a NEW dict with all original keys PLUS ``clean_text``.
      The original ``text`` field is NEVER modified.

  preprocess_documents(documents: list[dict]) -> list[dict]
      Apply the pipeline to an entire list of records.
      Preserves doc_id ordering.

  build_preprocessor() -> callable
      Returns a configured ``preprocess`` function (useful if callers want
      to pass the function around or swap tokeniser/stemmer later).

Design constraints
------------------
  • Deterministic: identical input always yields identical output.
  • Non-destructive: original ``text`` field is preserved unchanged.
  • Thread-safe: the PorterStemmer and stopword set are module-level
    singletons; no mutable state is modified at call time.
  • Graceful: empty string, None, and whitespace-only inputs return "".
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional

# ---------------------------------------------------------------------------
# Allow running as a script from any working directory.
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# ---------------------------------------------------------------------------
# NLTK imports — download required corpora on first use.
# ---------------------------------------------------------------------------
import nltk
from nltk.corpus import stopwords as _nltk_stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


def _ensure_nltk_data() -> None:
    """Download required NLTK packages if not already present (silent after first run)."""
    packages = [
        ("tokenizers/punkt_tab",   "punkt_tab"),
        ("tokenizers/punkt",       "punkt"),
        ("corpora/stopwords",      "stopwords"),
    ]
    for resource_path, download_id in packages:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(download_id, quiet=True)


_ensure_nltk_data()

# ---------------------------------------------------------------------------
# Module-level singletons (initialised once, reused on every call).
# ---------------------------------------------------------------------------
_STEMMER: PorterStemmer = PorterStemmer()
_STOPWORDS: frozenset[str] = frozenset(_nltk_stopwords.words("english"))

# ---------------------------------------------------------------------------
# Compiled regex patterns (compiled once at import time for speed).
# ---------------------------------------------------------------------------

# Residual RFC-2822 / news header lines that begin with "Key: value".
# Covers the full set observed in the 20-Newsgroups archive.
_RESIDUAL_HEADER_RE = re.compile(
    r"^("
    r"archive-name|alt-\w+-archive-name|last-modified|version|"
    r"in-reply-to|summary|keywords|expires|followup-to|distribution|"
    r"lines|path|message-id|nntp-posting-host|xref|date|references|"
    r"reply-to|organization|sender|cc|to|bcc|content-type|"
    r"content-transfer-encoding|mime-version|x-[\w-]+"
    r"):.*$",
    re.IGNORECASE | re.MULTILINE,
)

# PGP armour blocks.
_PGP_BLOCK_RE = re.compile(
    r"-----BEGIN PGP.*?-----END PGP[^\n]*-----",
    re.DOTALL | re.IGNORECASE,
)

# Quoted reply lines: lines that start with one or more > or | characters
# (Usenet quoting conventions).
_QUOTE_LINE_RE = re.compile(
    r"^[ \t]*[>|]{1,}.*$",
    re.MULTILINE,
)

# "In article <msg-id> user@host writes:" citation preamble.
_IN_ARTICLE_RE = re.compile(
    r"In article\s+<[^>]*>.*?writes:\s*",
    re.IGNORECASE | re.DOTALL,
)

# Email addresses.
_EMAIL_RE = re.compile(
    r"\b[\w.+\-]+@[\w.\-]+\.[a-zA-Z]{2,}\b",
)

# Angle-bracket message-ID tokens such as <abc123@news.server.com>.
_MSGID_RE = re.compile(
    r"<[^>\s]+@[^>\s]+>",
)

# Runs of whitespace (for normalisation).
_WHITESPACE_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Step 1 – Residual header / metadata removal
# ---------------------------------------------------------------------------

def _remove_residual_headers(text: str) -> str:
    """
    Remove leftover news/email metadata lines and noise that persist after
    the data loader's basic header stripping.

    Applied in order:
      1. PGP armour blocks
      2. "In article … writes:" preambles
      3. Residual header lines (key: value)
      4. Quoted reply lines (lines starting with > or |)
      5. Email addresses
      6. Angle-bracket message-ID tokens
    """
    text = _PGP_BLOCK_RE.sub(" ", text)
    text = _IN_ARTICLE_RE.sub(" ", text)
    text = _RESIDUAL_HEADER_RE.sub(" ", text)
    text = _QUOTE_LINE_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _MSGID_RE.sub(" ", text)
    return text


# ---------------------------------------------------------------------------
# Step 2 – Lowercase
# ---------------------------------------------------------------------------

def _to_lowercase(text: str) -> str:
    return text.lower()


# ---------------------------------------------------------------------------
# Step 3 – Tokenisation
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> List[str]:
    """
    Tokenise text using NLTK's ``word_tokenize``.

    Handles contractions ("don't" → ["do", "n't"]) and punctuation
    naturally; non-alphabetic tokens are filtered in step 6.
    """
    return word_tokenize(text)


# ---------------------------------------------------------------------------
# Step 4 – Stopword removal
# ---------------------------------------------------------------------------

def _remove_stopwords(tokens: List[str]) -> List[str]:
    """Remove English stopwords using NLTK's stopword list."""
    return [t for t in tokens if t not in _STOPWORDS]


# ---------------------------------------------------------------------------
# Step 5 – Stemming
# ---------------------------------------------------------------------------

def _stem(tokens: List[str]) -> List[str]:
    """Apply Porter stemming to each token."""
    return [_STEMMER.stem(t) for t in tokens]


# ---------------------------------------------------------------------------
# Step 6 – Punctuation / whitespace cleanup
# ---------------------------------------------------------------------------

def _clean_tokens(tokens: List[str]) -> List[str]:
    """
    Keep only purely alphabetic tokens of length ≥ 2.

    This removes punctuation marks, digits, single-character leftovers,
    and any residual noise that survived earlier steps.
    """
    return [t for t in tokens if t.isalpha() and len(t) >= 2]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess(text: str | None) -> str:
    """
    Apply the full 6-step preprocessing pipeline to a single text string.

    Parameters
    ----------
    text : str or None
        Raw document text (may already have email headers stripped by
        ``data_loader``).  ``None`` and empty / whitespace-only strings
        are handled safely and return ``""``.

    Returns
    -------
    str
        Space-separated string of stemmed, lowercased, stopword-free,
        alpha-only tokens.  Returns ``""`` if no tokens survive the pipeline.

    Pipeline
    --------
    1. Residual header / metadata removal
    2. Lowercase conversion
    3. Tokenisation (NLTK word_tokenize)
    4. Stopword removal (NLTK English stopwords)
    5. Stemming (PorterStemmer)
    6. Punctuation / whitespace cleanup (keep alpha tokens ≥ 2 chars)

    Examples
    --------
    >>> preprocess("Running tests on the server machines.")
    'run test server machin'

    >>> preprocess("")
    ''

    >>> preprocess(None)
    ''

    >>> preprocess("Organization: MIT  \\nThe quick brown fox jumps.")
    'quick brown fox jump'
    """
    # Guard: None / empty / whitespace-only
    if not text or not text.strip():
        return ""

    # Step 1 – residual header removal
    text = _remove_residual_headers(text)

    # Step 2 – lowercase
    text = _to_lowercase(text)

    # Step 3 – tokenise
    tokens = _tokenise(text)

    # Step 4 – stopword removal
    tokens = _remove_stopwords(tokens)

    # Step 5 – stemming
    tokens = _stem(tokens)

    # Step 6 – drop non-alpha / short tokens
    tokens = _clean_tokens(tokens)

    return " ".join(tokens)


def preprocess_document(doc: dict) -> dict:
    """
    Apply the preprocessing pipeline to one corpus record.

    Parameters
    ----------
    doc : dict
        A record as returned by ``data_loader.load_dataset()``.
        Must contain at minimum: ``doc_id``, ``text``, ``label``,
        ``category``, ``split``.

    Returns
    -------
    dict
        A new dict with all original keys preserved PLUS ``clean_text``.
        The original ``text`` field is NEVER modified.
        If the input dict is missing ``text``, ``clean_text`` is set to ``""``.

    Notes
    -----
    ``doc_id``, ``label``, ``category``, and ``split`` are passed through
    unchanged.
    """
    result = dict(doc)                          # shallow copy – original untouched
    raw_text = doc.get("text") or ""
    result["clean_text"] = preprocess(raw_text)
    return result


def preprocess_documents(
    documents: Iterable[dict],
    *,
    verbose: bool = False,
) -> List[dict]:
    """
    Apply the preprocessing pipeline to a list of corpus records.

    Parameters
    ----------
    documents : iterable of dict
        Records as returned by ``data_loader.load_dataset()``.
    verbose : bool, default False
        If True, print a progress update every 5 000 documents.

    Returns
    -------
    list of dict
        Each dict has all original keys plus ``clean_text``.
        Order is preserved.  ``doc_id`` values are unchanged.

    Examples
    --------
    >>> from data_loader import load_dataset
    >>> docs = load_dataset()
    >>> processed = preprocess_documents(docs)
    >>> processed[0].keys()
    dict_keys(['doc_id', 'text', 'label', 'category', 'split', 'clean_text'])
    """
    results: List[dict] = []
    for i, doc in enumerate(documents):
        results.append(preprocess_document(doc))
        if verbose and (i + 1) % 5_000 == 0:
            print(f"  Preprocessed {i + 1:,} documents ...", flush=True)
    return results


def build_preprocessor(
    stemmer=None,
    extra_stopwords: Optional[Iterable[str]] = None,
) -> "callable[[str], str]":
    """
    Return a configured preprocessing function.

    Useful when callers need a variant pipeline (e.g. different stopword
    lists or a different stemmer) without monkey-patching module globals.

    Parameters
    ----------
    stemmer : object with a ``stem(word) -> str`` method, optional
        Defaults to the module-level ``PorterStemmer``.
    extra_stopwords : iterable of str, optional
        Additional words to treat as stopwords (merged with NLTK defaults).

    Returns
    -------
    callable
        A function with the same signature as ``preprocess()``.
    """
    _local_stemmer = stemmer if stemmer is not None else _STEMMER
    _local_stopwords = (
        _STOPWORDS | frozenset(extra_stopwords)
        if extra_stopwords
        else _STOPWORDS
    )

    def _custom_preprocess(text: str | None) -> str:
        if not text or not text.strip():
            return ""
        t = _remove_residual_headers(text)
        t = _to_lowercase(t)
        tokens = _tokenise(t)
        tokens = [tok for tok in tokens if tok not in _local_stopwords]
        tokens = [_local_stemmer.stem(tok) for tok in tokens]
        tokens = _clean_tokens(tokens)
        return " ".join(tokens)

    return _custom_preprocess


# ---------------------------------------------------------------------------
# Convenience: expose the pipeline steps individually for inspection/testing
# ---------------------------------------------------------------------------

def pipeline_steps(text: str | None) -> dict:
    """
    Return intermediate outputs at each pipeline step.

    Useful for debugging and demonstrating the pipeline.

    Returns
    -------
    dict with keys:
        ``raw``         – original input
        ``no_headers``  – after step 1 (residual header removal)
        ``lowercased``  – after step 2
        ``tokens``      – after step 3 (tokenisation)
        ``no_stopwords``– after step 4
        ``stemmed``     – after step 5
        ``clean_text``  – final output (step 6)
    """
    if not text or not text.strip():
        return {
            "raw": text,
            "no_headers": "",
            "lowercased": "",
            "tokens": [],
            "no_stopwords": [],
            "stemmed": [],
            "clean_text": "",
        }

    step1 = _remove_residual_headers(text)
    step2 = _to_lowercase(step1)
    step3 = _tokenise(step2)
    step4 = _remove_stopwords(step3)
    step5 = _stem(step4)
    step6 = _clean_tokens(step5)

    return {
        "raw": text,
        "no_headers": step1,
        "lowercased": step2,
        "tokens": step3,
        "no_stopwords": step4,
        "stemmed": step5,
        "clean_text": " ".join(step6),
    }
