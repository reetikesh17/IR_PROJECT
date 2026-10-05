# Reetikesh Module Integration Guide

**Owner:** Reetikesh Choudhury  
**Components:** Dataset Loader, Preprocessing Pipeline, Common Processed Dataset, TF-IDF Retrieval System

---

## Overview

This directory provides the shared data foundation and the TF-IDF search engine for the 20 Newsgroups IR project.  
Teammates (Srijan – BM25, Vidur – Semantic Search / RRF) can easily consume preprocessed documents, perform TF-IDF queries, or retrieve document metadata.

---

## Required Dependencies

Install project dependencies via `requirements.txt`:

```bash
pip install pandas pyarrow scikit-learn nltk scipy numpy
```

---

## 1. How to Generate the Processed Dataset

Run the reproducible dataset builder from the `20-newsgroups-ir/` directory:

```bash
python src/build_dataset.py
```

This generates:
- `data/processed/processed_documents.parquet` (37,588 rows, full preprocessed corpus)
- `data/processed/metadata.json` (Dataset statistics and schema metadata)

---

## 2. How to Load the Processed Dataset

You can load the processed dataset into a `pandas.DataFrame` using `load_processed_dataset()`:

```python
from src.tfidf import load_processed_dataset

df = load_processed_dataset()

print(df.shape)  # (37588, 6)
print(df.columns.tolist())
# ['doc_id', 'text', 'clean_text', 'label', 'category', 'split']
```

### Accessing Document Fields

Each row / document dict contains:
- `doc_id` (`int32`): Stable 0-based unique identifier
- `text` (`string`): Original raw text (email headers stripped)
- `clean_text` (`string`): Preprocessed space-separated stemmed tokens
- `label` (`int8`): Numeric class label (0–19)
- `category` (`category`): Newsgroup name (e.g. `'sci.space'`)
- `split` (`category`): `'train'` or `'test'`

---

## 3. How to Build TF-IDF

Build the TF-IDF index over the dataset:

```python
from src.tfidf import build_tfidf, load_processed_dataset

# Load documents and build index
documents = load_processed_dataset()
build_tfidf(documents)
```

> **Note:** `build_tfidf(documents)` automatically sets up a global searcher instance so you don't need to pass the searcher around.

---

## 4. How to Perform a Search

Perform TF-IDF search queries using `search_tfidf`:

```python
from src.tfidf import build_tfidf, search_tfidf, get_document

# Build index once
build_tfidf()  # auto-loads dataset if arguments omitted

# Execute search
results = search_tfidf("space exploration", top_k=10)

# Inspect top result document details
top_doc_id = results[0]["doc_id"]
doc = get_document(top_doc_id)

print(f"Doc ID: {doc['doc_id']}")
print(f"Category: {doc['category']}")
print(f"Original Text: {doc['text'][:100]}...")
print(f"Clean Text: {doc['clean_text'][:100]}...")
```

---

## 5. Standardized Output Format

`search_tfidf()` returns a list of dictionaries with the exact standardized schema:

```json
[
    {
        "doc_id": 27983,
        "score": 0.361942,
        "rank": 1
    },
    {
        "doc_id": 28969,
        "score": 0.361942,
        "rank": 2
    },
    ...
]
```

- `doc_id` (`int`): Unique document ID corresponding to the corpus
- `score` (`float`): Cosine similarity score in range `[0.0, 1.0]`
- `rank` (`int`): 1-based rank position (`1`, `2`, `3`, ..., `top_k`)

---

## Quick Integration Example for Teammates

```python
from src.tfidf import build_tfidf, search_tfidf, get_document

# 1. Initialize index
build_tfidf()

# 2. Query
results = search_tfidf("cryptography encryption", top_k=5)

# 3. Process results
for item in results:
    doc = get_document(item["doc_id"])
    print(f"Rank {item['rank']}: ID={item['doc_id']}, Score={item['score']:.4f}, Category={doc['category']}")
```
