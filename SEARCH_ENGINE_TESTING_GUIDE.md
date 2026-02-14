# Search Engine Testing Setup Guide

How to set up a new repo to build and evaluate search engines against the FPPC Opinions eval suite.

## 1. Files You Need

Copy these from `fppc-opinions-eval` into your new repo:

```
# Eval infrastructure (required)
src/__init__.py              # Package init
src/interface.py             # SearchEngine ABC
src/scorer.py                # Scoring harness

# Ground truth (required)
eval/dataset.json            # 65 queries with 877 relevance judgments

# Tests (recommended)
tests/__init__.py
tests/test_scorer.py         # 28 tests for scorer correctness

# Opinion corpus (required by your engines, not by the scorer)
data/extracted/              # ~14,100 opinion JSON files organized by year
```

The scorer and interface are stdlib-only (no pip install needed). Your engines will have their own dependencies.

## 2. Suggested Repo Structure

```
fppc-search/
├── src/
│   ├── __init__.py
│   ├── interface.py              # SearchEngine ABC (copied from eval suite)
│   ├── scorer.py                 # Scoring harness (copied from eval suite)
│   └── engines/
│       ├── __init__.py
│       ├── bm25_engine.py        # Example First Engine
├── eval/
│   └── dataset.json              # Ground truth (copied from eval suite)
├── results/                      # Scorer JSON output goes here
├── data/
│   └── extracted/                # Opinion corpus (gitignored)
│       ├── 1975/
│       │   ├── 75001.json
│       │   └── ...
│       ├── 1976/
│       └── ...
└── tests/
    ├── __init__.py
    └── test_scorer.py            # Scorer tests (copied from eval suite)
```

Put each engine approach in its own file under `src/engines/`. This keeps experiments isolated and makes A/B comparison straightforward.

## 3. Implementing a Search Engine

Every engine implements one method. The scorer calls `obj()` with no arguments to instantiate your engine, so `__init__` must work with defaults or no parameters.

```python
from src.interface import SearchEngine

class BM25Engine(SearchEngine):
    def __init__(self):
        # Load your index, model, embeddings, etc.
        # This runs once before all 65 queries.
        pass

    def search(self, query: str, top_k: int = 20) -> list[str]:
        # Your retrieval logic here.
        # Return opinion IDs as strings, ordered by relevance (best first).
        # Examples: ["A-24-003", "90-162", "76188", "82A155"]
        return results

    def name(self) -> str:
        # Optional: human-readable name for the scorecard header.
        return "BM25"
```

**Key requirements:**

- Return opinion IDs as strings matching the filenames in `data/extracted/{year}/` (minus `.json`)
- Return at most `top_k` results (default 20)
- Order results by relevance, best first — ranking order directly affects nDCG and MRR
- The scorer deduplicates results automatically, but avoid returning duplicates if possible
- Opinion IDs not in the ground truth judgments are treated as score 0 (not relevant)

**Opinion ID formats** you'll encounter in the corpus:
| Format | Example | Era |
|--------|---------|-----|
| 5-digit number | `76188` | 1975-1983 |
| Letter-number | `82A155` | 1982-1985 |
| Year-dash-number | `90-162` | 1984-2003 |
| A-year-number | `A-24-003` | 2004-present |
| I-year-number | `I-19-145` | Informal advice |
| Special suffix | `16-073-1090` | Companion opinions |

## 4. Running the Scorer

```bash
# Scorecard printed to stdout
python src/scorer.py --search-module src.engines.bm25_engine --dataset eval/dataset.json

# Scorecard + detailed JSON output
python src/scorer.py --search-module src.engines.bm25_engine --dataset eval/dataset.json \
    --output results/bm25.json
```

The `--search-module` argument is a dotted Python import path. The scorer imports the module, finds the first `SearchEngine` subclass, and calls its constructor with no arguments.

Run the scorer from the repo root so that `src.engines.bm25_engine` resolves correctly.

## 5. Understanding the Output

### Scorecard (stdout)

The scorer prints an 80-character-wide table:

```
================================================================================
  FPPC Opinions Search Evaluation — BM25
  65 queries evaluated
================================================================================

                          MRR   nDCG@5  nDCG@10      P@5     P@10     R@10     R@20
--------------------------------------------------------------------------------
             Overall    0.652    0.483    0.401    0.388    0.302    0.225    0.318
--------------------------------------------------------------------------------
       By Query Type
        fact_pattern    0.580    0.421    0.355    0.341    0.271    0.198    0.287
             keyword    0.731    0.552    0.448    0.438    0.338    0.256    0.351
    natural_language    0.623    0.460    0.390    0.373    0.291    0.217    0.310
--------------------------------------------------------------------------------
            By Topic
    campaign_finance    0.610    0.445    0.372    ...
  conflicts_of_inter    0.688    0.510    0.425    ...
     gifts_honoraria    0.571    0.401    0.340    ...
            lobbying    0.540    0.380    0.315    ...
               other    0.620    0.455    0.385    ...
--------------------------------------------------------------------------------
```

### What the metrics mean

| Metric      | What it measures                                     | What to look for                                                                    |
| ----------- | ---------------------------------------------------- | ----------------------------------------------------------------------------------- |
| **MRR**     | Is a highly relevant (score=2) opinion near the top? | Higher = best answer shows up sooner. 1.0 means a score-2 opinion is always rank 1. |
| **nDCG@5**  | Ranking quality of top 5 results                     | The primary quality metric. Rewards putting score-2 above score-1 above score-0.    |
| **nDCG@10** | Ranking quality of top 10 results                    | Same as nDCG@5 but over a larger window.                                            |
| **P@5**     | Fraction of top 5 that are relevant (score >= 1)     | Higher = fewer irrelevant results cluttering the top.                               |
| **P@10**    | Fraction of top 10 that are relevant (score >= 1)    | Same over 10 results.                                                               |
| **R@10**    | How many known relevant opinions appear in top 10    | Higher = better at finding relevant opinions.                                       |
| **R@20**    | How many known relevant opinions appear in top 20    | Higher = better at comprehensive retrieval.                                         |

**Metric semantics to be aware of:**

- MRR only counts score=2 (highly relevant), not score=1
- nDCG computes the ideal ranking from ALL judged documents, not just returned ones — this penalizes engines that miss relevant docs
- Precision divides by k even if your engine returns fewer than k results
- Recall counts score >= 1 as relevant (both 1 and 2)

### JSON output (--output)

The JSON file contains everything needed for detailed analysis:

```json
{
  "timestamp": "2026-02-12T...",
  "engine": "BM25",
  "overall": {
    "mrr": 0.652,
    "ndcg@5": 0.483,
    ...
  },
  "by_type": {
    "keyword": { "mrr": 0.731, ... },
    "natural_language": { "mrr": 0.623, ... },
    "fact_pattern": { "mrr": 0.580, ... }
  },
  "by_topic": {
    "conflicts_of_interest": { "mrr": 0.688, ... },
    ...
  },
  "per_query": [
    {
      "query_id": "q001",
      "query_text": "Section 87103(a) disqualification business entity...",
      "query_type": "keyword",
      "query_topic": "conflicts_of_interest",
      "num_results": 20,
      "results": ["82A155", "00-014", "10-007", ...],
      "metrics": {
        "mrr": 1.0,
        "ndcg@5": 0.85,
        ...
      }
    },
    ...
  ]
}
```

The `per_query` array is the most useful for debugging. It shows exactly which opinions your engine returned for each query and how each query scored individually.

## 6. Comparing Engines

Save each engine's output to a separate JSON file:

```bash
python src/scorer.py --search-module src.engines.bm25_engine --dataset eval/dataset.json --output results/bm25.json
python src/scorer.py --search-module src.engines.semantic_engine --dataset eval/dataset.json --output results/semantic.json
```

Then compare per-query results to understand where approaches differ:

```python
import json

with open("results/bm25.json") as f:
    bm25 = json.load(f)
with open("results/semantic.json") as f:
    semantic = json.load(f)

for bq, sq in zip(bm25["per_query"], semantic["per_query"]):
    diff = sq["metrics"]["ndcg@10"] - bq["metrics"]["ndcg@10"]
    if abs(diff) > 0.1:
        winner = "semantic" if diff > 0 else "bm25"
        print(f"{bq['query_id']}: {winner} wins by {abs(diff):.3f}")
        print(f"  query: {bq['query_text'][:80]}")
        print(f"  bm25 nDCG@10={bq['metrics']['ndcg@10']:.3f}, semantic={sq['metrics']['ndcg@10']:.3f}")
```

**What patterns to look for:**

- Keyword queries favoring one engine over another (BM25 often wins on keyword, semantic on natural language)
- Specific topics where an engine struggles (thin topics like lobbying may need different handling)
- Queries where an engine scores 0.0 on MRR (it failed to put any score-2 opinion in the results at all)

## 7. Opinion Data Reference

Your search engines will read opinions from `data/extracted/{year}/{id}.json`. Prefer these fields in order:

| Field                                       | Coverage | Best for                                               |
| ------------------------------------------- | -------- | ------------------------------------------------------ |
| `embedding.qa_text`                         | ~97%     | Single-field indexing — combined question + conclusion |
| `sections.question` + `sections.conclusion` | 66%      | Structured Q&A pairs                                   |
| `embedding.summary`                         | varies   | Brief summaries                                        |
| `content.full_text`                         | 100%     | Last resort — often noisy for older OCR'd opinions     |

Useful metadata for search:

- `classification.topic_primary` — topic label (conflicts_of_interest, campaign_finance, etc.)
- `citations.government_code` — statute sections cited (e.g., ["87103(a)", "87100"])
- `citations.prior_opinions` — other FPPC opinions cited
- `parsed.date` — opinion date

## 8. Query Distribution

The 65 test queries are distributed across:

| Topic                 | Queries | % of Corpus |
| --------------------- | ------- | ----------- |
| Conflicts of interest | 29      | 48%         |
| Campaign finance      | 14      | 17%         |
| Gifts/honoraria       | 7       | 5%          |
| Lobbying              | 5       | 1%          |
| Other                 | 10      | 8%          |

| Query Type       | Count |
| ---------------- | ----- |
| keyword          | 26    |
| natural_language | 22    |
| fact_pattern     | 17    |

Each query has 10-25 judged opinions (avg 13.5). Scores: 39% highly relevant (2), 57% relevant (1), 4% not relevant (0).

## 9. Practical Tips

**Start simple.** A basic keyword/BM25 engine over `embedding.qa_text` is a good first baseline. It will beat random (which scores ~0.000) and give you real numbers to improve on.

**Watch for ID format mismatches.** Your engine must return opinion IDs that exactly match the filenames in `data/extracted/`. If your index uses a different format (e.g., adding `.json` or normalizing dashes), the scorer will treat them as unjudged and score them 0.

**The scorer is fast.** It evaluates 65 queries in seconds. The bottleneck is your engine's `search()` method. If you're iterating quickly, keep initialization (index loading, model loading) in `__init__` so it only runs once.

**Don't optimize for the test set.** The 877 judgments cover a tiny fraction of the ~14,100 opinions. An engine that memorizes which opinion IDs appear in the judgments would score well but be useless in practice. Build engines that work on the full corpus.

**Use `--output` for every run.** JSON results are cheap to store and invaluable for comparing experiments later. Consider naming files with timestamps or experiment IDs: `results/bm25_v2_2026-02-12.json`.
