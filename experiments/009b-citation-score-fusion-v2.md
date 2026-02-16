# 009b — Citation Score Fusion v2 (Lower CB Threshold)

**Engine:** `src/engines/citation_score_fusion_v2.py`
**Results:** `results/009b-citation-score-fusion-v2.json`
**Date:** 2026-02-15

## What I tried

Starting from 009, lowered the circuit breaker threshold from 1.3 to 1.2. The
hypothesis was that a more aggressive CB would protect more queries from
dilution by returning BM25-only when BM25 has a moderately clear winner, while
preserving citation pooling's rescue ability for truly ambiguous queries.

Everything else identical to 009: two-path (non-citation → pure BM25, citation →
pooled score fusion), 0.4 BM25 / 0.6 semantic weighting, min-max normalization.

## Key numbers

| Metric   | Overall | Keyword | NatLang | FactPat |
|----------|---------|---------|---------|---------|
| MRR      | 0.684   | 0.588   | 0.722   | 0.783   |
| nDCG@5   | 0.387   | 0.349   | 0.409   | 0.418   |
| nDCG@10  | 0.335   | 0.307   | 0.340   | 0.371   |
| R@20     | 0.270   | 0.291   | 0.258   | 0.254   |

Comparison to 009 (CB threshold 1.3):

| Metric      |   009 |  009b | Delta  |
|-------------|-------|-------|--------|
| MRR         | 0.684 | 0.684 | +0.000 |
| nDCG@5      | 0.387 | 0.387 | +0.000 |
| nDCG@10     | 0.336 | 0.335 | -0.001 |
| R@20        | 0.270 | 0.270 | +0.000 |
| Zeros       |    10 |    10 |      0 |

## Observations

**Almost no effect.** Lowering the CB threshold from 1.3 to 1.2 captured exactly
one additional query: q031 (ratio 1.29). All other citation queries either had
ratios already above 1.3 (already caught by 009) or below 1.2 (still go to
fusion in both).

The one new CB hit (q031) was a slight regression: nDCG@10 dropped from 0.375
to 0.281 because BM25-only lost one relevant doc from the top-10 that fusion
had surfaced. MRR was unaffected (same rank-1 result).

CB firing stats: 7/36 citation queries (19%) vs 009's 6/36 (17%).

**Why the threshold doesn't matter much:** The BM25 top1/top2 ratio distribution
is bimodal — most queries have ratios either well above 1.3 or well below 1.2.
Very few queries fall in the 1.2–1.3 gap, so shifting the threshold within this
range has minimal impact.

## Next

The CB threshold is not a productive tuning knob — the gap between 1.2 and 1.3
contains almost no queries. More promising directions:

- **Cross-encoder re-ranking:** Use a cross-encoder to re-rank the top-N
  candidates from citation-pooled fusion. This could fix deep ranking issues
  without the dilution/rescue tradeoff.
- **Adaptive BM25/semantic weighting:** Instead of fixed 0.4/0.6, weight based
  on query characteristics (e.g., more semantic weight for natural language
  queries, more BM25 weight for keyword queries).
- **Query expansion:** Use LLM to expand terse keyword queries before semantic
  search, addressing the gap between keyword query style and opinion text style.
