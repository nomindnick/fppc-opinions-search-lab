# 002 — Semantic Baseline

**Engine:** `src/engines/semantic_baseline.py`
**Results:** `results/002-semantic-baseline.json`
**Date:** 2026-02-13

## What I tried

Pure embedding search using OpenAI `text-embedding-3-small` over the `embedding.qa_text` field (falling back to `content.full_text` for opinions without qa_text). All 14,096 corpus documents are embedded at index time, L2-normalized, and stored in a pickle. At search time, the query is embedded via the same model and scored against all documents using a single dot product (equivalent to cosine similarity on normalized vectors).

No BM25, no hybrid scoring, no field boosting — this is a clean semantic-only baseline to contrast with the BM25 results from 001/001b.

Embedding index built in 28 batches of 512 texts each. Texts exceeding the model's 8,191-token limit are truncated using tiktoken (cl100k_base encoding). The index pickle is ~82 MB and loads in under a second on subsequent runs.

## Key numbers

| Metric   | Overall | Keyword | NatLang | FactPat |
|----------|---------|---------|---------|---------|
| MRR      |   0.485 |   0.367 |   0.565 |   0.560 |
| nDCG@5   |   0.262 |   0.198 |   0.316 |   0.291 |
| nDCG@10  |   0.237 |   0.191 |   0.268 |   0.267 |
| R@20     |   0.194 |   0.184 |   0.210 |   0.188 |

Delta from 001b (BM25 full-text):

| Metric   | Overall | Keyword | NatLang | FactPat |
|----------|---------|---------|---------|---------|
| MRR      |  -0.135 |  -0.060 |  -0.157 |  -0.223 |
| nDCG@5   |  -0.096 |  -0.085 |  -0.091 |  -0.119 |
| nDCG@10  |  -0.084 |  -0.071 |  -0.089 |  -0.098 |
| R@20     |  -0.058 |  -0.067 |  -0.055 |  -0.047 |

By topic:

| Metric   | Conflicts | Campaign | Gifts | Lobbying | Other |
|----------|-----------|----------|-------|----------|-------|
| MRR      |     0.259 |    0.522 | 0.905 |    1.000 | 0.535 |
| nDCG@5   |     0.129 |    0.266 | 0.626 |    0.558 | 0.243 |
| R@20     |     0.076 |    0.253 | 0.372 |    0.411 | 0.220 |

Topic deltas from 001b:

| Metric   | Conflicts | Campaign | Gifts | Lobbying | Other |
|----------|-----------|----------|-------|----------|-------|
| MRR      |    -0.064 |   -0.341 |-0.024 |   +0.100 |-0.248 |
| nDCG@5   |    -0.027 |   -0.260 |+0.085 |   -0.049 |-0.218 |
| R@20     |    -0.007 |   -0.161 |-0.064 |   -0.105 |-0.032 |

## Observations

**Semantic search underperforms BM25 overall.** MRR drops from 0.620 to 0.485 (-22%). Every query type and 4 of 5 topics see MRR declines. This is not surprising in hindsight — FPPC opinions are a statute-heavy domain where precise section numbers and legal terms carry enormous signal, and BM25 excels at exact matching.

**Semantic search has 16 zero-MRR queries vs 13 for BM25.** It rescued 6 of BM25's 13 zeros but created 9 new zeros — a net loss. The new failures are mostly queries where BM25 scored 1.0 (perfect top-1) thanks to exact term matching on statute numbers or distinctive vocabulary.

**The two approaches are strongly complementary.** Across 65 queries: 13 semantic wins, 22 semantic losses, 30 ties. The wins and losses are on almost entirely different queries. This is the strongest argument for a hybrid approach.

### Where semantic search helped (rescued from BM25 zero-MRR)

| Query | BM25 | Semantic | What happened |
|-------|------|----------|---------------|
| q018  | 0.00 | **1.00** | "revolving door" / post-employment — semantic understood the concept even though opinions don't use "revolving door" literally |
| q027  | 0.00 | **0.33** | Spouse community property income — semantic bridged "community property" to related opinions |
| q001  | 0.00 | **0.25** | 87103(a) disqualification — semantic found semantically related opinions even without exact section match |
| q026  | 0.00 | **0.20** | Spouse conflict of interest — similar vocabulary gap bridging |
| q015  | 0.00 | **0.13** | Gov Code 1090 self-dealing — semantic found related contract interest opinions |
| q004  | 0.00 | **0.07** | Real property proximity recusal — modest improvement |

### Where semantic search collapsed (BM25 scored 1.0, semantic scored 0 or near 0)

| Query | BM25 | Semantic | What happened |
|-------|------|----------|---------------|
| q009  | 1.00 | **0.00** | Fact pattern about source of income — too many conceptually similar opinions diluted ranking |
| q013  | 1.00 | **0.00** | Development permit conflict — semantic couldn't distinguish this specific scenario |
| q007  | 1.00 | **0.10** | 87103(c) source of income — statute number matching lost |
| q036  | 1.00 | **0.13** | 84308 subcontractor contribution — precise statute matching failed |
| q040  | 1.00 | **0.13** | Party committee member communications — complex campaign finance fact pattern |
| q041  | 1.00 | **0.17** | Behested payments 84224 — statute-specific keyword query degraded |
| q038  | 1.00 | **0.20** | Transfer ban 85304 — statute number matching lost |
| q065  | 1.00 | **0.10** | Lobbyist board service — specific fact pattern swamped by semantic noise |
| q060  | 1.00 | **0.25** | JPA code reviewing body — specific procedural question |

**Campaign finance suffered the most** (MRR 0.864 → 0.522, -0.341). This topic relies heavily on precise statute numbers (84308, 85304, 85800) and regulatory terminology that BM25 matches exactly. Semantic search dilutes these signals among conceptually similar but wrong opinions.

**Lobbying was the one bright spot** (MRR 0.900 → 1.000, +0.100). The lobbying sub-corpus is small and topically distinct — semantic similarity works well when the relevant cluster is tight and well-separated from other topics.

**Gifts held steady** (MRR 0.929 → 0.905, -0.024). The gift/honoraria opinions have distinctive enough content that both approaches find them.

**Conflicts didn't improve much** (MRR 0.323 → 0.259, -0.064). Despite the hypothesis that semantic search would help with vocabulary mismatch in conflicts queries, the improvement was marginal. The 6 rescues were offset by new failures on fact patterns and natural language queries where BM25 was already performing well.

## Next

- **Hybrid BM25+semantic** — the complementary win/loss pattern is the clearest signal from this experiment. A weighted combination (e.g., `0.7*BM25_score + 0.3*cosine_sim`) should rescue the vocabulary-mismatch queries while preserving BM25's precise matching. Need to normalize scores to the same scale before combining.
- **BM25 on qa_text + semantic** — 001b showed full_text BM25 is better than qa_text BM25, but the semantic embeddings were built on qa_text. A hybrid might benefit from BM25 on full_text (for statute numbers) plus semantic on qa_text (for meaning).
- **Reciprocal rank fusion** — rather than score-level combination, RRF combines ranked lists. This avoids the score normalization problem and is simple to implement.
- **Re-ranker** — use a cross-encoder to re-rank BM25's top results, which would be more precise than bi-encoder similarity while avoiding the recall problem of pure semantic search.
