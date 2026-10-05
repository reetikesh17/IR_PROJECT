"""
tfidf.py
========
TF-IDF retrieval system for the 20 Newsgroups IR project.

Author : Reetikesh Choudhury
Module : TF-IDF Retrieval (own component)

Overview
--------
Implements TF-IDF document indexing and cosine-similarity vector search
over the preprocessed 20 Newsgroups corpus.

Pipeline
--------
1. Clean documents (preprocessed via src/preprocessing.py)
2. Fit sklearn TfidfVectorizer on corpus
3. Generate sparse TF-IDF document matrix
4. Preprocess incoming queries using the SAME preprocessing pipeline
5. Transform query to TF-IDF query vector
6. Compute cosine similarity against document matrix
7. Rank and return top-k matching results

Public API
----------
* build_tfidf(documents) -> TFIDFSearcher
* search_tfidf(query, top_k=10, searcher=None) -> list[dict]
* load_processed_dataset() -> pandas.DataFrame
* get_document(doc_id) -> dict or None
* TFIDFSearcher (class with search(), get_document(), save(), load() methods)

Result Format
-------------
[
    {
        "doc_id": 123,
        "score": 0.82,
        "rank": 1
    }
]
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# Allow importing src modules when running standalone
# ---------------------------------------------------------------------------
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from preprocessing import preprocess

# Module-level singleton holding the default searcher instance
_GLOBAL_SEARCHER: Optional[TFIDFSearcher] = None


class TFIDFSearcher:
    """
    TF-IDF index and cosine similarity search engine.

    Holds the fitted TfidfVectorizer, document matrix, and stable doc_id mapping.
    Avoids rebuilding the TF-IDF model on every search.
    """

    def __init__(
        self,
        vectorizer: TfidfVectorizer,
        doc_matrix: sp.csr_matrix,
        doc_ids: Sequence[int],
        documents: Optional[List[dict]] = None,
    ) -> None:
        self.vectorizer = vectorizer
        self.doc_matrix = doc_matrix
        self.doc_ids = np.asarray(doc_ids, dtype=np.int32)
        self.documents = documents
        self._doc_map = (
            {doc["doc_id"]: doc for doc in documents} if documents is not None else {}
        )

    @property
    def num_documents(self) -> int:
        """Return total number of documents in the index."""
        return int(self.doc_matrix.shape[0])

    @property
    def vocab_size(self) -> int:
        """Return total number of terms in the vocabulary."""
        return len(self.vectorizer.vocabulary_)

    def search(self, query: Optional[str], top_k: int = 10) -> List[dict]:
        """
        Search the indexed corpus for a given query string using cosine similarity.

        Parameters
        ----------
        query : str or None
            Raw user query text.
        top_k : int, default 10
            Maximum number of top ranked results to return.

        Returns
        -------
        list of dict
            Each dict has keys: 'doc_id' (int), 'score' (float), 'rank' (int).
            Returns [] if query is empty or contains no matching vocabulary terms.
        """
        if top_k <= 0:
            return []

        # 1. Guard against empty / None query
        if not query or not query.strip():
            return []

        # 2. Preprocess query using the exact same pipeline as documents
        clean_query = preprocess(query)
        if not clean_query or not clean_query.strip():
            return []

        # 3. Transform query to TF-IDF vector
        query_vec = self.vectorizer.transform([clean_query])

        # 4. Handle queries with no known terms in the vocabulary
        if query_vec.nnz == 0:
            return []

        # 5. Calculate cosine similarity (doc_matrix rows are L2-normalized)
        scores = self.doc_matrix.dot(query_vec.T).toarray().ravel()

        # 6. Filter only positive similarity scores
        pos_mask = scores > 0.0
        pos_indices = np.flatnonzero(pos_mask)

        if pos_indices.size == 0:
            return []

        pos_scores = scores[pos_indices]
        pos_doc_ids = self.doc_ids[pos_indices]

        # 7. Sort by score descending, breaking ties by doc_id ascending
        # np.lexsort sorts by primary key last: (secondary, primary)
        sorted_order = np.lexsort((pos_doc_ids, -pos_scores))
        top_indices = pos_indices[sorted_order[:top_k]]

        # 8. Format results with rank starting at 1
        results: List[dict] = []
        for rank, idx in enumerate(top_indices, start=1):
            results.append(
                {
                    "doc_id": int(self.doc_ids[idx]),
                    "score": float(round(float(scores[idx]), 6)),
                    "rank": int(rank),
                }
            )

        return results

    def get_document(self, doc_id: int) -> Optional[dict]:
        """Retrieve stored document dict by doc_id if documents were provided at build time."""
        return self._doc_map.get(doc_id)

    def save(self, filepath: Union[str, Path]) -> Path:
        """
        Save the fitted searcher state (vectorizer, matrix, doc_ids) to disk.

        Parameters
        ----------
        filepath : str or Path
            Target file path (e.g. 'models/tfidf_searcher.pkl').

        Returns
        -------
        Path
            Absolute path to saved pickle artifact.
        """
        path = Path(filepath).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vectorizer": self.vectorizer,
            "doc_matrix": self.doc_matrix,
            "doc_ids": self.doc_ids,
            "documents": self.documents,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "TFIDFSearcher":
        """
        Load a saved TFIDFSearcher instance from disk.

        Parameters
        ----------
        filepath : str or Path
            Path to saved pickle artifact.

        Returns
        -------
        TFIDFSearcher
        """
        path = Path(filepath).resolve()
        if not path.exists():
            raise FileNotFoundError(f"TFIDFSearcher artifact not found at '{path}'")
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        return cls(
            vectorizer=payload["vectorizer"],
            doc_matrix=payload["doc_matrix"],
            doc_ids=payload["doc_ids"],
            documents=payload.get("documents"),
        )


def load_processed_dataset(parquet_path: Optional[Union[str, Path]] = None):
    """
    Convenience helper to load the shared processed dataset artifact.

    Columns: ``doc_id``, ``text``, ``clean_text``, ``label``, ``category``, ``split``.

    Parameters
    ----------
    parquet_path : str or Path, optional
        Defaults to data/processed/processed_documents.parquet.

    Returns
    -------
    pandas.DataFrame
    """
    from data_loader import load_processed_dataset as _loader
    return _loader(parquet_path=parquet_path)


def build_tfidf(
    documents: Optional[Union[Iterable[dict], "pd.DataFrame"]] = None,
    *,
    sublinear_tf: bool = True,
    min_df: Union[int, float] = 1,
    max_df: Union[int, float] = 1.0,
    norm: str = "l2",
    store_documents: bool = True,
) -> TFIDFSearcher:
    """
    Build a TF-IDF retrieval index from preprocessed documents.

    Stores the built searcher instance globally so subsequent search_tfidf()
    calls use it automatically.

    Parameters
    ----------
    documents : list of dict, iterable of dict, or pandas.DataFrame, optional
        Document records containing 'doc_id' and 'clean_text' (or 'text').
        If None, automatically loads the common processed dataset parquet artifact.
    sublinear_tf : bool, default True
        Apply sublinear scaling 1 + log(tf).
    min_df : int or float, default 1
        Minimum document frequency threshold.
    max_df : int or float, default 1.0
        Maximum document frequency threshold.
    norm : str, default 'l2'
        Normalization strategy ('l2' enables fast dot-product cosine similarity).
    store_documents : bool, default True
        Whether to store reference document dicts inside the searcher instance.

    Returns
    -------
    TFIDFSearcher
        Fitted TF-IDF searcher ready for queries.
    """
    global _GLOBAL_SEARCHER

    if documents is None:
        documents = load_processed_dataset()

    # Handle pandas DataFrame input
    if hasattr(documents, "to_dict"):
        records: List[dict] = documents.to_dict("records")
    else:
        records = list(documents)

    if not records:
        raise ValueError("Cannot build TF-IDF index on an empty document collection.")

    corpus_texts: List[str] = []
    doc_ids: List[int] = []

    for idx, doc in enumerate(records):
        doc_id = doc.get("doc_id", idx)
        doc_ids.append(doc_id)

        # Use pre-computed clean_text if available, otherwise preprocess text
        clean = doc.get("clean_text")
        if clean is None:
            raw_text = doc.get("text") or ""
            clean = preprocess(raw_text)
        corpus_texts.append(str(clean))

    # Configure TfidfVectorizer to accept preprocessed whitespace-separated tokens
    vectorizer = TfidfVectorizer(
        token_pattern=r"\S+",
        lowercase=False,
        sublinear_tf=sublinear_tf,
        min_df=min_df,
        max_df=max_df,
        norm=norm,
    )

    doc_matrix = vectorizer.fit_transform(corpus_texts)

    stored_docs = records if store_documents else None
    searcher = TFIDFSearcher(
        vectorizer=vectorizer,
        doc_matrix=doc_matrix,
        doc_ids=doc_ids,
        documents=stored_docs,
    )
    _GLOBAL_SEARCHER = searcher
    return searcher


def search_tfidf(
    query: Optional[str] = None,
    top_k: int = 10,
    searcher: Optional[TFIDFSearcher] = None,
    documents: Optional[Union[Iterable[dict], "pd.DataFrame"]] = None,
) -> List[dict]:
    """
    Search documents using TF-IDF and cosine similarity.

    Can be called with an explicit searcher instance or rely on a previously built index.

    Parameters
    ----------
    query : str or None
        Raw user query text.
    top_k : int, default 10
        Number of top ranked results to return.
    searcher : TFIDFSearcher, optional
        Pre-built searcher instance. If None, uses the globally built searcher
        or auto-builds one from the common processed dataset.
    documents : iterable of dict or DataFrame, optional
        Documents to build searcher from if searcher is None.

    Returns
    -------
    list of dict
        List of result dicts: [{'doc_id': int, 'score': float, 'rank': int}]
    """
    global _GLOBAL_SEARCHER

    target_searcher = searcher or _GLOBAL_SEARCHER

    if target_searcher is None:
        if documents is not None:
            target_searcher = build_tfidf(documents)
        else:
            # Auto-load processed dataset if available
            df = load_processed_dataset()
            target_searcher = build_tfidf(df)

    return target_searcher.search(query, top_k=top_k)


def get_document(doc_id: int, searcher: Optional[TFIDFSearcher] = None) -> Optional[dict]:
    """
    Retrieve document record by doc_id (includes category, text, clean_text, etc.).

    Parameters
    ----------
    doc_id : int
        Unique document identifier.
    searcher : TFIDFSearcher, optional
        Optional searcher instance containing stored documents.

    Returns
    -------
    dict or None
    """
    target_searcher = searcher or _GLOBAL_SEARCHER
    if target_searcher is not None:
        doc = target_searcher.get_document(doc_id)
        if doc is not None:
            return doc

    # Fallback to searching parquet dataset directly
    try:
        df = load_processed_dataset()
        match = df[df["doc_id"] == doc_id]
        if not match.empty:
            return match.iloc[0].to_dict()
    except Exception:
        pass

    return None
