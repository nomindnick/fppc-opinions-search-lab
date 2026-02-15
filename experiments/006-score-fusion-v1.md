# 006 — Score-Based Fusion with BM25 Circuit Breaker

**Engine:** `src/engines/score_fusion_v1.py`
**Results:** `results/006-score-fusion-v1.json`
**Date:** 2026-02-14

## What I tried

Starting from the hybrid fusion concept in 003 but replacing rank-based RRF
with score-based fusion. Two key changes:

1. **Min-max normalized score fusion (50/50)** — instead of RRF's uniform rank
   increments, use actual BM25 and cosine similarity scores. Normalize each
   pool to [0, 1] via min-max, then combine 0.5 * norm_bm25 + 0.5 * norm_sem.
   This preserves BM25's confidence signal: a dominant top score stays dominant
   after normalization.

2. **Circuit breaker** — when BM25's top score is ≥1.3× the second score, skip
   fusion entirely and return BM25 results. This protects high-confidence BM25
   queries from any semantic dilution. The semantic embedding API call is also
   skipped, saving latency/cost.

The circuit breaker fired on 14/65 queries (22%).

## Key numbers

| Metric   | Overall | Keyword | NatLang | FactPat |
|----------|---------|---------|---------|---------|
| MRR      | 0.632   | 0.482   | 0.681   | 0.796   |
| nDCG@5   | 0.372   | 0.305   | 0.404   | 0.434   |
| nDCG@10  | 0.334   | 0.290   | 0.357   | 0.373   |
| R@20     | 0.268   | 0.280   | 0.270   | 0.245   |

Comparison to baselines:

| Metric | 001b BM25 | 003 RRF | 005 Cit-Hybrid | **006 Score Fusion** |
|--------|-----------|---------|----------------|----------------------|
| MRR    | 0.620     | 0.620   | **0.665**      | 0.632                |
| nDCG@5 | 0.345     | 0.373   | 0.394          | 0.372                |
| R@20   | 0.252     | 0.277   | 0.282          | 0.268                |
| Zeros  | 13        | 12      | 10             | 9                    |

## Observations

**Dilution problem partially fixed:**
- q036: 1.0 in BM25, dropped to 0.5 in 005 → **back to 1.0** (circuit breaker did NOT fire — score fusion preserved BM25 dominance naturally)
- q038: 1.0 in BM25, dropped to 0.25 in 005 → **0.5** (improved but not fully fixed — circuit breaker did NOT fire, semantic still pulled it down)

**Fewest zero-MRR queries (9)** but composition shifted:
- New rescues vs 001b: q001 (0→0.2), q015 (0→0.059), q018 (0→0.143), q026 (0→0.067), q027 (0→0.167)
- Lost 005's strong rescues: q001 (1.0→0.2), q012 (1.0→0.0) — these relied on 005's citation-filtered pooling, not available here
- New zero: q029 (0.053→0.0)

**Notable new regressions from BM25:**
- q009: 1.0→0.333, q013: 1.0→0.5, q060: 1.0→0.333 — queries where BM25 had clear rank-1 hits but score fusion with semantic pulled them down. The circuit breaker didn't fire because BM25's score gap was below 1.3x.

**Big semantic win:** q006 jumped from 0.05 (BM25) to 1.0 — a genuine rescue where semantic found the right opinion that BM25 buried.

**vs 001b BM25:** 13 wins, 10 losses, 42 ties
**vs 005 Cit-Hybrid:** 10 wins, 12 losses, 43 ties — net negative

**Circuit breaker tuning:** The 1.3x threshold may be too aggressive. q009
and q013 have BM25 score ratios just under 1.3 but still had confident rank-1
results that got diluted. A lower threshold (e.g., 1.15 or 1.2) might catch
these but could over-fire on queries that benefit from fusion.

## Next

The core insight: **score-based fusion reduces dilution better than RRF** (q036
fully fixed, q038 partially fixed) but **can't replicate 005's citation-filtered
pooling wins** (q001, q012 rescues lost). The best next step combines both:

1. **Experiment 007: Citation-filtered score fusion** — use 005's citation
   pooling strategy but replace RRF with min-max score fusion + circuit
   breaker. Should get 005's rescue capability with less dilution.

2. **CB threshold tuning** — try 1.15 or 1.2 to protect q009/q013/q060 from
   dilution. Could also try asymmetric weighting (0.6 BM25 / 0.4 semantic)
   for non-CB queries.

3. **Conditional fusion** — only fuse when semantic's top score is high enough
   to indicate it found something meaningful (e.g., cosine > 0.5). When
   semantic has nothing confident, just use BM25.
