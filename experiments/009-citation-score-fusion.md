# 009 — Citation-Filtered Score Fusion

**Engine:** `src/engines/citation_score_fusion.py`
**Results:** `results/009-citation-score-fusion.json`
**Date:** 2026-02-15

## What I tried

Combined the best ideas from experiments 005 and 006:

- **From 005:** Citation pooling for citation queries — build candidate pool from
  citation matches ∪ BM25 top-100, use semantic similarity to discriminate
  within statute-specific pools
- **From 006:** Min-max score fusion (0.4 BM25 / 0.6 semantic) instead of RRF,
  plus circuit breaker (top1/top2 ≥ 1.3) to protect confident BM25 results

Two-path architecture:
- **Non-citation queries (29):** Pure BM25 on full_text — no API call, no
  dilution risk
- **Citation queries (36):** Citation-pooled score fusion with circuit breaker

## Key numbers

| Metric   | Overall | Keyword | NatLang | FactPat |
|----------|---------|---------|---------|---------|
| MRR      | 0.684   | 0.588   | 0.722   | 0.783   |
| nDCG@5   | 0.387   | 0.349   | 0.409   | 0.418   |
| nDCG@10  | 0.336   | 0.307   | 0.340   | 0.377   |
| R@20     | 0.270   | 0.291   | 0.258   | 0.254   |

### Comparison with parent experiments

| Metric   | 005 (cite+RRF) | 006 (score fuse) | **009** | 001b (BM25) |
|----------|----------------|------------------|---------|-------------|
| MRR      | 0.665          | 0.632            | **0.684** | 0.620     |
| nDCG@5   | 0.371          | 0.372            | **0.387** | 0.358     |
| nDCG@10  | 0.322          | 0.334            | **0.336** | 0.321     |
| R@20     | 0.252          | 0.268            | **0.270** | 0.252     |
| Zeros    | 10             | 9                | **10**    | 13        |

## Observations

### 005's rescues: preserved

| Query | BM25 | 005  | 006  | **009** |
|-------|------|------|------|---------|
| q001  | 0    | 1.0  | 0.2  | **1.0** |
| q012  | 0    | 1.0  | 0    | **0.5** |
| q018  | 0    | 0.17 | 0.14 | **1.0** |

q001 and q018 fully rescued (MRR=1.0). q012 partially rescued (0.5 vs 005's 1.0)
— score fusion re-ordered within the citation pool differently than RRF.

### 005's dilution losses: fixed

| Query | BM25 | 005  | 006  | **009** |
|-------|------|------|------|---------|
| q036  | 1.0  | 0.5  | 1.0  | **1.0** |
| q038  | 1.0  | 0.25 | 0.5  | **0.5** |

q036 fully fixed. q038 improved to 0.5 (same as 006), not fully fixed — the
circuit breaker didn't fire for this query, so score fusion still dilutes
slightly.

### 006's non-citation regressions: eliminated

| Query | BM25 | 005  | 006   | **009** |
|-------|------|------|-------|---------|
| q009  | 1.0  | 1.0  | 0.333 | **1.0** |
| q013  | 1.0  | 1.0  | 0.5   | **1.0** |
| q060  | 1.0  | 1.0  | 0.333 | **1.0** |

All three fully recovered because non-citation queries use pure BM25.

### Circuit breaker behavior

CB fired on 6 of 36 citation queries (17%): q034, q040, q049, q054, q055, q063.
Lower fire rate than 006's 14/65 (22%) because non-citation queries are excluded
from the citation path entirely (and many of 006's CB fires were on non-citation
queries that 009 handles as pure BM25).

### Zero-MRR queries (10)

q004, q010, q015, q017, q020, q022, q023, q026, q027, q028

Same count as 005 (10), one more than 006 (9). The zero set is mostly the same
hard conflicts_of_interest queries that all engines struggle with. q015 (1090
self-dealing) went from 006's 0.059 back to 0 — its citation pool is too large
(~4200 opinions cite "1090" base) for semantic to find the needle.

### Per-type improvements

- **Keyword:** 0.588 vs 005's 0.541, 006's 0.482 — big gains from both
  citation pooling and CB protecting BM25 results
- **Natural language:** 0.722 vs 005's 0.745, 006's 0.681 — slight regression
  from 005 (score fusion less favorable than RRF for some NL queries)
- **Fact pattern:** 0.783 vs 005's 0.753, 006's 0.796 — middle ground; pure
  BM25 path eliminates 006's regressions

## Next

1. **q038 investigation** — CB didn't fire despite this being a confident BM25
   result. The citation pool dilutes it. Could try lowering CB threshold to 1.2
   for citation queries, or add a secondary CB check before fusion.

2. **q012 regression from 005** — score fusion ranked it at position 2 vs RRF's
   position 1. The BM25 score gap is small enough that semantic doesn't preserve
   the rank. Could try adaptive weighting: more BM25 weight when BM25 scores are
   closer together.

3. **Persistent zeros** — q004 (18702.2 proximity), q015 (1090 self-dealing),
   q020 (public generally exception) remain hard. These need either:
   - Larger-than-qa_text embeddings (full_text or analysis)
   - Cross-encoder re-ranking
   - Query expansion / reformulation

4. **Deploy candidate** — At MRR 0.684, this is the best engine so far. Could
   port to the app repo with the pure BM25 fallback as a fast path and citation
   fusion as the enhanced path for statute-referencing queries.
