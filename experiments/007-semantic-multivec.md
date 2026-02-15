# 007 — Multi-Vector Semantic Search

**Engine:** `src/engines/semantic_multivec.py`
**Results:** `results/007-semantic-multivec.json`
**Date:** 2026-02-15

## What I tried

Extended the semantic baseline (002) by embedding three different text representations per opinion — `embedding.qa_text`, `sections.facts`, and `sections.analysis` + `sections.conclusion` — and taking MAX similarity across all three at search time. The hypothesis: different query types will match different aspects of an opinion (the question, the facts, or the legal analysis), improving recall over a single-vector approach.

Three embedding indexes, all using `text-embedding-3-small` with L2-normalized vectors:

| Index | Source field | Coverage |
|-------|------------|----------|
| qa_text | `embedding.qa_text` | 14,096 (100%) — reused existing index |
| facts | `sections.facts` | 7,440 (52.8%) |
| analysis | `sections.analysis` + `sections.conclusion` (concat) | 10,825 (76.8%) |

At search time: embed query once, dot-product against all three matrices, scatter sparse scores into full-size arrays via pre-computed alignment mappings, take element-wise MAX, return top-20.

OpenAI API batch size had to be reduced from 512 to 64 — the analysis+conclusion concatenations are long enough (avg ~2400 tokens) that 128-text batches exceeded the 300k tokens-per-request limit.

## Key numbers

| Metric   | Overall | Keyword | NatLang | FactPat |
|----------|---------|---------|---------|---------|
| MRR      |   0.465 |   0.302 |   0.513 |   0.654 |
| nDCG@5   |   0.240 |   0.164 |   0.258 |   0.334 |
| nDCG@10  |   0.238 |   0.174 |   0.249 |   0.324 |
| R@20     |   0.213 |   0.203 |   0.213 |   0.227 |

Delta from 002 (semantic baseline, qa_text only):

| Metric   | Overall | Keyword | NatLang | FactPat |
|----------|---------|---------|---------|---------|
| MRR      |  -0.019 |  -0.065 |  -0.053 |  +0.094 |
| nDCG@5   |  -0.022 |  -0.034 |  -0.058 |  +0.043 |
| nDCG@10  |  +0.002 |  -0.017 |  -0.019 |  +0.057 |
| R@20     |  +0.019 |  +0.019 |  +0.002 |  +0.039 |

By topic:

| Metric   | Conflicts | Campaign | Gifts | Lobbying | Other |
|----------|-----------|----------|-------|----------|-------|
| MRR      |     0.190 |    0.586 | 0.857 |    1.000 | 0.553 |
| nDCG@5   |     0.073 |    0.316 | 0.510 |    0.643 | 0.232 |
| R@20     |     0.081 |    0.299 | 0.387 |    0.436 | 0.240 |

Topic deltas from 002:

| Metric   | Conflicts | Campaign | Gifts | Lobbying | Other |
|----------|-----------|----------|-------|----------|-------|
| MRR      |    -0.069 |   +0.064 |-0.048 |    0.000 |+0.018 |
| nDCG@5   |    -0.056 |   +0.050 |-0.116 |   +0.085 |-0.012 |
| R@20     |    +0.005 |   +0.046 |+0.015 |   +0.025 |+0.020 |

## Observations

**Overall MRR regressed slightly** (0.485 → 0.465, -0.019). 10 wins, 18 losses, 37 ties. The losses cancel out the wins at the top-rank level.

**Fact pattern queries improved** (+0.094 MRR). This was the hypothesis — fact-rich queries match better against the facts/analysis vectors. The best wins:
- q065 (lobbyist board service): 0.10 → 1.00 — analysis text captured the legal reasoning
- q050 (contractor campaign funds): 0.33 → 1.00 — facts section matched the scenario
- q006 (rental property planning): 0.50 → 1.00 — facts alignment
- q035 (subcontractor contribution): 0.50 → 1.00 — analysis reasoning match

**Keyword and natural language queries regressed** (-0.065 and -0.053 MRR). The MAX operation lets facts/analysis vectors override what was a correct qa_text match by promoting unrelated opinions whose sections happen to be generically similar to the query.

**Five queries lost rank-1 results** — the same dilution pattern seen in hybrid experiments:
- q018 (revolving door): 1.00 → 0.143 — analysis vectors from other post-employment opinions flooded the top
- q025 (3% entity stake): 1.00 → 0.167 — facts from similar-but-wrong conflict opinions diluted
- q047 (charity golf gift): 1.00 → 0.500 — modest dilution
- q048 (travel payment exemption): 1.00 → 0.500 — analysis text similarity too broad
- q063 (electronic filing): 1.00 → 0.500

**Conflicts of interest worst affected** (MRR -0.069). This topic has the largest corpus (~6,600 opinions) with highly overlapping facts/analysis language, so the MAX operation amplifies noise.

**Depth metrics improved.** R@20 up +0.019 overall, with gains across every topic. The additional vectors do surface relevant opinions that qa_text misses — they just don't promote them to rank 1.

**Rescued 3 zero-MRR queries** (q019, q032, q058) but **dropped 2** (q004, q015). Net reduction from 16 to 15 zeros.

### Root cause: MAX is too aggressive

The core problem is that MAX(qa, facts, analysis) lets any single vector override the others. For opinions with verbose analysis sections, the analysis vector can produce high generic similarity to many queries, promoting irrelevant opinions above the correct one. The qa_text vector is more focused (question + conclusion) and better discriminates, but MAX lets the noisier vectors dilute it.

## Next

- **Weighted MAX or mean fusion** — instead of pure MAX, try `0.5*qa + 0.25*max(facts, analysis)` to keep qa_text as the primary signal while using facts/analysis as a boost. This preserves qa_text's discriminative power while still capturing the multi-vector benefit.
- **Use multi-vector as a re-ranker** — let qa_text determine the candidate set (top 50–100), then re-rank using the max-of-three score. This prevents facts/analysis from promoting garbage into the pool.
- **Selective MAX** — only use facts/analysis vectors for opinions that have high qa_text similarity (e.g., top 200), avoiding the noise from the full corpus.
- **The fact_pattern win is real** — +0.094 MRR on fact patterns suggests these vectors should be part of a hybrid system, but gated rather than applied unconditionally.
