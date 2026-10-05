# 20 Newsgroups — Processed Dataset

> **This directory is generated automatically.**  Do NOT commit `processed_documents.parquet` to Git.
> Only `metadata.json` and `README.md` are tracked.
> To regenerate, run: `python src/build_dataset.py`

## Files

| File | In Git | Description |
|---|---|---|
| `processed_documents.parquet` | ❌ | Full processed corpus (~37 500 rows) |
| `metadata.json` | ✅ | Dataset description and stats |
| `README.md` | ✅ | This file |

## Quick-start (pandas)

```python
import pandas as pd

df = pd.read_parquet('data/processed/processed_documents.parquet')
print(df.shape)          # (~37500, 6)
print(df.columns.tolist())
# ['doc_id', 'text', 'clean_text', 'label', 'category', 'split']
```

## Quick-start (PyArrow)

```python
import pyarrow.parquet as pq

table = pq.read_table('data/processed/processed_documents.parquet')
# Filter to train split only:
import pyarrow.compute as pc
train = table.filter(pc.equal(table['split'], 'train'))
```

## Schema

| Column | Type | Description |
|---|---|---|
| `doc_id` | int32 | Globally unique 0-based integer |
| `text` | string | Raw document body (email headers stripped) |
| `clean_text` | string | Preprocessed text: lowercase stemmed tokens |
| `label` | int8 | Numeric class label 0–19 |
| `category` | category | Newsgroup name e.g. `alt.atheism` |
| `split` | category | `train` or `test` |

## Dataset statistics

| Statistic | Value |
|---|---|
| Total documents | 37,588 |
| Categories | 20 |
| Train documents | 18,794 |
| Test documents | 18,794 |
| Empty clean_text | 64 |
| Avg clean tokens | 96.57 |
| Preprocessing version | `1.0.0` |
| Generated at | 2026-10-05T16:23:49.104845+00:00 |

## Preprocessing pipeline

Implemented in `src/preprocessing.py`.  Steps applied in order:

1. Residual header / metadata removal
2. Lowercase
3. Tokenisation (NLTK `word_tokenize`)
4. Stopword removal (NLTK English)
5. Stemming (Porter Stemmer)
6. Drop non-alpha / length < 2 tokens

## Reproducing the dataset

```bash
# From the 20-newsgroups-ir/ directory:
python src/build_dataset.py

# With a custom archive path:
python src/build_dataset.py --archive /path/to/archive.zip

# Verify an existing file without rebuilding:
python src/build_dataset.py --verify-only
```

The `NEWSGROUPS_ARCHIVE_PATH` environment variable can also be set.
