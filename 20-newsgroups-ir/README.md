# 20 Newsgroups IR Project

An Information Retrieval system built on the 20 Newsgroups dataset, implementing
TF-IDF, BM25, and semantic search with Reciprocal Rank Fusion (RRF) and a
Streamlit evaluation UI.

---

## Team & Module Ownership

| Module | Owner | Files |
|---|---|---|
| Dataset Loading & Preprocessing & TF-IDF | Reetikesh Choudhury | `src/data_loader.py`, `src/preprocessing.py`, `src/tfidf.py`, `src/utils.py` |
| BM25 Retrieval | *(teammate)* | `src/bm25.py` |
| Semantic Search | *(teammate)* | `src/semantic_search.py` |
| Reciprocal Rank Fusion | *(teammate)* | `src/rrf.py` |
| Evaluation (MAP, nDCG, P@10, Recall) | *(teammate)* | `src/evaluation.py` |
| Streamlit UI | *(teammate)* | `app/streamlit_app.py` |

---

## Repository Structure

```
20-newsgroups-ir/
├── app/
│   └── streamlit_app.py          # Streamlit demo UI
├── notebooks/
│   ├── preprocessing.ipynb       # Dataset inspection & preprocessing demo
│   ├── bm25.ipynb
│   ├── evaluation.ipynb
│   └── semantic.ipynb
├── results/
│   ├── comparison.csv
│   ├── metrics.json
│   └── precision_recall.png
├── src/
│   ├── data_loader.py            # ← Reetikesh: dataset loading (this doc)
│   ├── preprocessing.py          # ← Reetikesh: text preprocessing pipeline
│   ├── tfidf.py                  # ← Reetikesh: TF-IDF retrieval
│   ├── utils.py                  # ← Reetikesh: shared utilities
│   ├── bm25.py
│   ├── semantic_search.py
│   ├── rrf.py
│   └── evaluation.py
├── tests/
│   ├── __init__.py
│   └── test_data_loader.py       # ← Reetikesh: 47 unit tests
├── README.md
└── requirements.txt
```

> **Dataset note:** `archive.zip` lives **outside** the repository at the
> workspace root (`../archive.zip` relative to this folder). It is **not**
> committed to Git and must not be added. See [Dataset Setup](#dataset-setup).

---

## Dataset Setup

The raw dataset is a ZIP archive (`archive.zip`, ~27 MB) stored locally at:

```
d:\Project-ir\archive.zip          # default location
```

It is **not** in Git. Do not commit or push it.

If you need to point the loader to a different path, set the environment variable:

```powershell
$env:NEWSGROUPS_ARCHIVE_PATH = "C:\path\to\archive.zip"
```

or pass `archive_path=` explicitly to any loader function (see API below).

### Archive Contents

| File | Description |
|---|---|
| `alt.atheism.txt` … `talk.religion.misc.txt` | 20 category files, each containing all documents for that newsgroup |
| `list.csv` | Index of `(newsgroup, document_id)` pairs for `talk.religion.misc` |

Each category `.txt` file encodes **two splits** using an RFC-2822-style header
convention:

```
Newsgroup: comp.graphics          ← train document starts here
document_id: 37261                ← lowercase 'd' → split = "train"
From: user@host
Subject: ...

<body text>

Newsgroup: comp.graphics          ← test document starts here
Document_id: 37261                ← uppercase 'D' → split = "test"
From: user@host
Subject: ...

<body text>
```

Each document appears once in the train section and once in the test section
with identical content — the split is signalled solely by the capitalisation
of `document_id:` vs `Document_id:`.

### Corpus Statistics

| Split | Documents |
|---|---|
| Train | 18,794 |
| Test | 18,794 |
| Unstructured (no header) | ~1,600 |
| **Total** | **~37,588** |

20 categories, ~940–1,990 documents per category per split.

---

## Installation

```powershell
pip install -r requirements.txt
```

Python 3.10+ is required.

---

## Data Loader — `src/data_loader.py`

**Author:** Reetikesh Choudhury

### Public API

#### `load_dataset(archive_path=None, *, categories=None, splits=None, strip_headers=True, include_unstructured=False) → list[dict]`

Load the full corpus (or a filtered subset) as a list of record dicts.

```python
from src.data_loader import load_dataset

# Full train + test corpus (default)
docs = load_dataset()
print(len(docs))          # 37588
print(docs[0].keys())     # dict_keys(['doc_id', 'text', 'label', 'category', 'split'])

# Single category
atheism = load_dataset(categories=["alt.atheism"])

# Train split only
train = load_dataset(splits=["train"])

# Keep raw email headers (do not strip From:/Subject: lines)
raw = load_dataset(strip_headers=False)

# Override archive location
docs = load_dataset(archive_path="/data/archive.zip")
```

**Each record contains:**

| Key | Type | Description |
|---|---|---|
| `doc_id` | `int` | Globally unique, 0-based integer stable within a single call |
| `text` | `str` | Document body (email headers stripped by default) |
| `label` | `int` | Integer class label 0–19 (index into `CATEGORIES`) |
| `category` | `str` | Newsgroup name, e.g. `"alt.atheism"` |
| `split` | `str` | `"train"` \| `"test"` \| `"unstructured"` |

---

#### `load_split(split, archive_path=None, **kwargs) → list[dict]`

Convenience wrapper for loading a single split.

```python
from src.data_loader import load_split

train_docs = load_split("train")
test_docs  = load_split("test")
```

---

#### `load_dataframe(archive_path=None, **kwargs) → pandas.DataFrame`

Returns the corpus as a DataFrame with columns
`doc_id, text, label, category, split`.

```python
from src.data_loader import load_dataframe

df = load_dataframe()
print(df.shape)           # (37588, 5)
print(df["category"].value_counts())
```

Requires `pandas`. Install via `requirements.txt`.

---

#### `dataset_info(archive_path=None) → dict`

Returns metadata (doc counts per category/split) without loading all text.
Fast — suitable for inspection without paying the full parse cost.

```python
from src.data_loader import dataset_info

info = dataset_info()
print(info["totals"])
# {'train': 18828, 'test': 18828, 'unstructured': 1624, 'total': 39280}

print(info["counts"]["sci.space"])
# {'train': 987, 'test': 987, 'unstructured': 20, 'total': 1994}
```

---

#### `CATEGORIES` — ordered list of the 20 newsgroup names

```python
from src.data_loader import CATEGORIES
print(CATEGORIES[0])   # 'alt.atheism'
print(CATEGORIES[19])  # 'talk.religion.misc'
```

`label` integers in every record correspond to the index of `category` in this list.

---

### Design Notes

- **No internet access.** The loader reads only from the local `archive.zip`.
- **No side effects.** The archive is never modified or extracted to disk.
- **Configurable path.** Override via `NEWSGROUPS_ARCHIVE_PATH` env var or
  `archive_path=` argument; no hard-coded absolute paths in source.
- **Header stripping.** With `strip_headers=True` (default), RFC-2822-style
  header lines (`From:`, `Subject:`, `Date:`, `Organization:`, etc.) are
  removed from each document, leaving only the prose body.  About 108 posts
  (~0.3 %) begin their body with a quoted attribution line
  (`From: user@host`) — this is valid body content and is preserved.
- **Stable doc_ids.** `doc_id` values are re-indexed to a contiguous 0-based
  range on every call, so `doc_id == list index` always holds. They are stable
  within a single call but may differ across calls with different `categories=`
  or `splits=` arguments.
- **Unstructured docs.** A small set (~1,600) of posts lack `Newsgroup:` /
  `document_id:` headers and are tagged `split="unstructured"`. They are
  excluded by default; enable with `include_unstructured=True`.

---

## Running the Tests

```powershell
# From the project root (20-newsgroups-ir/)
pytest tests/test_data_loader.py -v

# With coverage report
pytest tests/test_data_loader.py -v --cov=src/data_loader --cov-report=term-missing
```

The suite contains **47 tests** across 7 classes:

| Class | What it tests |
|---|---|
| `TestDatasetLoading` | Return type, size, record keys, category filtering, error handling |
| `TestDocumentContent` | Non-empty text, string type, minimum length, header stripping |
| `TestLabelsAndCategories` | Integer labels 0–19, all 20 categories present, label↔category consistency |
| `TestDocIds` | Uniqueness, integer type, 0-based, contiguous |
| `TestSplitSeparation` | Train/test presence, correct isolation, symmetric counts, invalid split error |
| `TestDatasetInfo` | Metadata accuracy without full load |
| `TestLoadDataframe` | DataFrame shape, columns, no nulls |

Expected result: **47 passed** in ~12 seconds.

> The tests skip automatically if `archive.zip` is not found at its default
> location. Set `NEWSGROUPS_ARCHIVE_PATH` to run them in a different
> environment.

---

## .gitignore Recommendation

Add the following to prevent the dataset from being committed:

```gitignore
# Raw dataset – do not commit
archive.zip
*.zip

# Python artifacts
__pycache__/
*.pyc
*.pyo
.pytest_cache/
*.egg-info/
dist/
build/

# Data outputs
data/processed_corpus.pkl
data/*.pkl

# Notebooks
.ipynb_checkpoints/
```
