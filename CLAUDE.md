# CLAUDE.md — fppc-opinions-search-lab

## What this repo is

An experimental workbench for building and evaluating search engines against a
corpus of ~14,100 FPPC (Fair Political Practices Commission) advisory opinions.
The corpus spans 1975 to present and covers conflicts of interest, campaign
finance, gifts, lobbying, and related topics.

The goal is to find the best search engine implementation, then port it to a
separate app repo for a web-based search tool.

## Repo structure

```
src/interface.py          # SearchEngine ABC — do not modify
src/scorer.py             # Scoring harness — do not modify
src/engines/              # One file per search engine implementation
eval/dataset.json         # 65 queries, 877 relevance judgments — do not modify
data/extracted/           # ~14,100 opinion JSON files organized by year
indexes/                  # Pre-built indexes (gitignored, reproducible)
results/                  # Scorer JSON output (git-tracked)
experiments/              # Lab notebook — one markdown file per experiment
tests/test_scorer.py      # Scorer tests
```

## Rules

### Never modify eval infrastructure
`src/interface.py`, `src/scorer.py`, `eval/dataset.json`, and `tests/test_scorer.py`
are the fixed evaluation framework. Do not modify them. If there's a bug, flag it
rather than fixing it silently.

### One engine per file, never overwrite
Every search engine implementation goes in its own file under `src/engines/`.
When iterating on an approach, create a new file rather than modifying an old one.
Example: `bm25_baseline.py` → `bm25_statute_boost.py` → `bm25_hybrid_v1.py`.

### Engine portability pattern
Every engine must be portable to an app repo with zero refactoring. This means:

1. **Configurable corpus path with a default that works in this repo:**
   ```python
   def __init__(self, corpus_path: str = "data/extracted",
                index_dir: str = "indexes"):
   ```

2. **Build-or-load index initialization:**
   ```python
   if self.index_path.exists():
       self._load_index()
   else:
       self._build_index()
       self._save_index()
   ```

3. **No hardcoded paths** anywhere except the default arguments above.

4. **The scorer calls `__init__()` with no arguments**, so defaults must work
   for this repo's layout.

### Experiment tracking
Every scorer run should have a corresponding experiment note in `experiments/`.
Naming convention: results and experiments share the same prefix number.
- Engine: `src/engines/bm25_baseline.py`
- Results: `results/001-bm25-baseline.json`
- Notes: `experiments/001-bm25-baseline.md`

Copy `experiments/_TEMPLATE.md` to start a new experiment.

### Running the scorer
```bash
python src/scorer.py \
  --search-module src.engines.bm25_baseline \
  --dataset eval/dataset.json \
  --output results/001-bm25-baseline.json
```
Run from repo root. The `--search-module` is a dotted import path.

## Opinion data reference

Opinion JSON files are in `data/extracted/{year}/{id}.json`. Prefer these fields:

| Field | Coverage | Use for |
|-------|----------|---------|
| `embedding.qa_text` | ~97% | Primary indexing — combined question + conclusion |
| `sections.question` + `sections.conclusion` | 66% | Structured Q&A when available |
| `embedding.summary` | varies | Brief summaries |
| `content.full_text` | 100% | Fallback — noisy for older OCR'd opinions |
| `classification.topic_primary` | — | Topic filtering |
| `citations.government_code` | — | Statute sections cited |
| `citations.prior_opinions` | — | Cross-references to other opinions |
| `parsed.date` | — | Opinion date |

## Opinion ID formats
| Format | Example | Era |
|--------|---------|-----|
| 5-digit | `76188` | 1975–1983 |
| Letter-number | `82A155` | 1982–1985 |
| Year-dash | `90-162` | 1984–2003 |
| A-year | `A-24-003` | 2004–present |
| I-year | `I-19-145` | Informal advice |
| Special suffix | `16-073-1090` | Companion opinions |

## Context for the developer

The developer (Nick) is a California attorney who specializes in public agency
law. He has practical experience researching FPPC opinions and understands the
domain deeply, but is a Python beginner. Explain code decisions clearly. When
suggesting approaches, favor readability over cleverness.
