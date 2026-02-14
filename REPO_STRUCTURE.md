# fppc-opinions-search-lab — Repo Structure

## Current State (what you have)

```
fppc-opinions-search-lab/
├── src/
│   ├── __init__.py
│   ├── interface.py          # SearchEngine ABC
│   └── scorer.py             # Scoring harness
├── eval/
│   └── dataset.json          # 65 queries, 877 relevance judgments
├── data/
│   └── extracted/            # ~14,100 opinion JSONs by year
├── tests/
│   ├── __init__.py
│   └── test_scorer.py
└── SEARCH_ENGINE_TESTING_GUIDE.md
```

## What to Add

```
fppc-opinions-search-lab/
├── src/
│   ├── __init__.py
│   ├── interface.py
│   ├── scorer.py
│   └── engines/
│       ├── __init__.py       # empty
│       ├── bm25_baseline.py  # your first engine
│       └── ...               # one file per engine, never overwrite old ones
├── eval/
│   └── dataset.json
├── data/
│   └── extracted/            # gitignored
├── indexes/                  # pre-built indexes live here (gitignored)
│   └── .gitkeep
├── results/                  # scorer JSON output (git-tracked)
│   └── .gitkeep
├── experiments/              # your lab notebook
│   ├── _TEMPLATE.md          # copy this to start a new experiment
│   ├── 001-bm25-baseline.md
│   └── ...
├── tests/
│   ├── __init__.py
│   └── test_scorer.py
├── .gitignore
├── SEARCH_ENGINE_TESTING_GUIDE.md
└── README.md
```

### New directories explained

**`src/engines/`** — One Python file per search engine implementation. The naming
convention is descriptive, lowercase, underscores: `bm25_baseline.py`,
`bm25_statute_boost.py`, `semantic_qat.py`, `hybrid_v1.py`. Never modify or
delete an old engine file when iterating — create a new file instead. Disk is
free and you'll want to re-run old engines against new ideas.

**`results/`** — Scorer JSON output. Name files to match their experiment:
`001-bm25-baseline.json`, `002-bm25-statute-boost.json`. Track these in git.
They're small and having the history of your scores in version control is
valuable.

**`experiments/`** — One markdown file per experiment. This is your lab notebook.
Not formal, not a template you have to fill out — just a running log of what you
tried, what the numbers were, and what you learned. More on format below.

**`indexes/`** — Pre-built search indexes (BM25 indexes, embedding caches,
serialized models). Gitignored because they're large and reproducible. Your
engine's `__init__` should build the index if it's missing, load from here if it
exists.

---

## Experiment Workflow

### Starting a new experiment

1. Copy `experiments/_TEMPLATE.md` → `experiments/NNN-short-name.md`
2. Create or identify the engine file in `src/engines/`
3. Run the scorer:
   ```bash
   python src/scorer.py \
     --search-module src.engines.bm25_baseline \
     --dataset eval/dataset.json \
     --output results/001-bm25-baseline.json
   ```
4. Paste the headline numbers into your experiment notes
5. Write down what you observed and what to try next

### Experiment notes format

Keep it simple. Here's the template:

```markdown
# 001 — BM25 Baseline

**Engine:** `src/engines/bm25_baseline.py`
**Results:** `results/001-bm25-baseline.json`
**Date:** 2026-02-13

## What I tried
Plain BM25 over embedding.qa_text field. No statute boosting, no
metadata, no semantic component. Just want a baseline number.

## Key numbers
| Metric   | Overall | Keyword | NatLang | FactPat |
|----------|---------|---------|---------|---------|
| MRR      |         |         |         |         |
| nDCG@5   |         |         |         |         |
| nDCG@10  |         |         |         |         |

## Observations
(What worked, what didn't, queries that scored 0, patterns you noticed)

## Next
(What to try based on what you learned)
```

You don't have to use this format exactly. The point is that future-you can
open any experiment file and immediately understand what happened without
having to re-run anything or reverse-engineer your thinking.

---

## Engine Portability

Every engine should be designed to work in both the lab and your eventual app
with minimal changes. Two conventions make this easy:

### 1. Configurable corpus path with a lab-friendly default

```python
class BM25Baseline(SearchEngine):
    def __init__(self, corpus_path: str = "data/extracted"):
        self.corpus_path = Path(corpus_path)
        # ... load corpus, build index
```

The scorer calls `__init__()` with no arguments, so the default works here.
When you drop this file into your app repo, you pass the real path:
`BM25Baseline(corpus_path="/app/data/opinions")`.

### 2. Build-or-load index initialization

```python
def __init__(self, corpus_path: str = "data/extracted",
             index_dir: str = "indexes"):
    self.corpus_path = Path(corpus_path)
    self.index_dir = Path(index_dir)
    self.index_path = self.index_dir / f"{self.name()}_index.pkl"

    if self.index_path.exists():
        self._load_index()
    else:
        self._build_index()
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._save_index()
```

First run builds the index (slow). Every run after that loads from disk (fast).
In your app, you'd pre-build the index at deploy time and ship it.

### What this means in practice

When your search engine is ready for the app, you:
1. Copy the engine `.py` file into the app repo
2. Copy `src/interface.py` (the ABC) into the app repo
3. Instantiate with the app's corpus path and index directory
4. Call `engine.search(query, top_k)` — same method, same return type

No adapter layer, no refactoring. The engine file *is* the portable unit.

---

## .gitignore additions

```
# Large / reproducible data
data/extracted/
indexes/

# Python
__pycache__/
*.pyc
.venv/
```

Results and experiment notes should be tracked — they're your record of what
happened.
