# Processed Dataset

This directory contains the shared processed dataset for the
20 Newsgroups IR project.

## Files

| File | Description |
|------|-------------|
| `processed_documents.parquet` | Full preprocessed corpus (Snappy-compressed Parquet) |
| `metadata.json` | Dataset statistics and build parameters |
| `README.md` | This file |

## Dataset Statistics

| Stat | Value |
|------|-------|
| Total documents | 19,961 |
| Training documents | 15,968 (~80%) |
| Testing documents | 3,993 (~20%) |
| Categories | 20 |
| Avg clean tokens | 96.56 |
| Generated at | 2026-10-06T17:24:53.366285+00:00 |

## Split Method

The train/test split is created by `build_dataset.py` using:
```python
from sklearn.model_selection import train_test_split
train_docs, test_docs = train_test_split(
    processed,
    test_size=0.20,
    random_state=42,
    stratify=labels,
)
```

## Loading the Dataset

```python
import sys
sys.path.insert(0, 'src')
from data_loader import load_processed_dataset

df = load_processed_dataset()
train_df = df[df['split'] == 'train']
test_df  = df[df['split'] == 'test']
```

## Columns

| Column | Type | Description |
|--------|------|-------------|
| `doc_id` | int32 | Globally unique 0-based document ID |
| `text` | string | Raw document body (headers stripped) |
| `clean_text` | string | Preprocessed text (stemmed, stopwords removed) |
| `label` | int8 | Integer category label (0–19) |
| `category` | category | Newsgroup name |
| `split` | category | `"train"` or `"test"` |

## Important: No Data Leakage

The TF-IDF vectorizer (and any other data-dependent feature extractor)
**must be fitted only on training documents**.  Always filter by split
before building an index:

```python
from tfidf import build_tfidf
train_df = df[df['split'] == 'train']
searcher = build_tfidf(train_df)  # fit on train only
```
