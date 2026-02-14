# 005 — Citation-Filtered Hybrid (BM25 + Semantic)

**Engine:** `src/engines/citation_semantic_hybrid.py`
**Results:** `results/005-citation-semantic-hybrid.json`
**Date:** 2026-02-14

## What I tried

Building on the key insight from experiment 004 — that BM25 full_text cannot
rank accurately within citation pools — this engine uses a two-path approach:

**Citation path** (~30 queries with statute/regulation references):
1. Parse citations from the query using `parse_query_citations()`
2. Build a candidate pool: all opinions matching those citations (via gc_exact,
   gc_base, reg_exact indexes) **union** BM25 top-100 (safety net)
3. Rank candidates with two arms: BM25 scores and semantic (cosine similarity
   of `text-embedding-3-small` embeddings on qa_text)
4. Fuse with semantic-favoring weighted RRF: `0.4/(60+bm25_rank) + 0.6/(60+sem_rank)`

**No-citation path** (~35 queries without statute references):
Pure BM25 on full_text (identical to experiment 001b).

The semantic-favoring 0.4/0.6 weighting reflects experiment 004's finding that
within citation-filtered pools, semantic similarity is more discriminative than
BM25 full_text. The RRF formulation still provides natural protection for BM25's
strong rank-1 results.

## Key numbers

| Metric   | Overall | Keyword | NatLang | FactPat |
|----------|---------|---------|---------|---------|
| MRR      | 0.665   | 0.541   | 0.745   | 0.753   |
| nDCG@5   | 0.371   | 0.320   | 0.409   | 0.400   |
| nDCG@10  | 0.322   | 0.287   | 0.340   | 0.354   |
| R@20     | 0.252   | 0.256   | 0.254   | 0.245   |

**vs 001b (BM25 full_text baseline):**

| Metric   | 001b  | 005   | Delta  |
|----------|-------|-------|--------|
| MRR      | 0.620 | 0.665 | +0.046 |
| nDCG@5   | 0.358 | 0.371 | +0.013 |
| nDCG@10  | 0.321 | 0.322 | +0.002 |
| R@20     | 0.252 | 0.252 | +0.000 |
| Zeros    | 13    | 10    | -3     |

## Observations

### Wins (8 queries improved)

| Query | 001b MRR | 005 MRR | Delta  | Notes |
|-------|----------|---------|--------|-------|
| q001  | 0.000    | 1.000   | +1.000 | Full rescue. 87103(a) pool + semantic |
| q012  | 0.000    | 1.000   | +1.000 | Surprise rescue (87103(e)). Not in target list |
| q061  | 0.250    | 1.000   | +0.750 | 82041 definition — semantic lifted correct opinion |
| q043  | 0.500    | 1.000   | +0.500 | 85102(d) broad-based committee |
| q044  | 0.500    | 1.000   | +0.500 | 89501/89502 honorarium ban |
| q030  | 0.091    | 0.333   | +0.242 | 82015 contribution definition |
| q018  | 0.000    | 0.167   | +0.167 | Partial rescue. 87400 revolving door |
| q024  | 0.167    | 0.333   | +0.167 | 1091/1091.5 remote interest |

### Losses (3 queries regressed)

| Query | 001b MRR | 005 MRR | Delta  | Notes |
|-------|----------|---------|--------|-------|
| q038  | 1.000    | 0.250   | -0.750 | 85304 transfer ban. Semantic diluted BM25 rank-1 |
| q036  | 1.000    | 0.500   | -0.500 | 84308 subcontractor. Same dilution pattern |
| q056  | 0.250    | 0.143   | -0.107 | 89001 mass mailing. Minor regression |

### Rescue target scorecard

Of the 5 queries expected to be rescued based on semantic search (002) performance:

| Query | Semantic MRR | 005 MRR | Rescued? |
|-------|-------------|---------|----------|
| q001  | 0.250       | 1.000   | Yes — better than semantic alone |
| q004  | 0.067       | 0.000   | No |
| q015  | 0.125       | 0.000   | No |
| q018  | 1.000       | 0.167   | Partial — semantic's rank-1 was diluted by pool size |
| q026  | 0.200       | 0.000   | No |

q004, q015, q026: The citation pools are very large (87103 base → ~4,200
opinions), drowning the semantic signal. The relevant opinions may rank too low
even in semantic space within such large pools.

q018: Semantic had MRR=1.0 alone, but within the 87400 citation pool + BM25
top-100, the relevant opinion dropped to semantic rank ~6 in a pool where BM25
rankings also pushed it down.

### Regression analysis

q036 and q038 are the same pattern seen in experiment 003 — RRF fusion dilutes
BM25's confident rank-1 placement. Both are campaign finance keyword queries
where BM25 alone had the right answer at rank 1, but semantic rankings were weak
(rank 8-10 in semantic space). With 0.4 BM25 / 0.6 semantic weighting, even a
moderate semantic ranking can displace BM25's rank-1 result.

### Zero-MRR queries (10 remaining)

q004, q010, q015, q017, q020, q022, q023, q026, q027, q028

These cluster heavily in conflicts_of_interest keyword queries and represent
fundamental limitations: the relevant opinions either don't match the expected
citation patterns, or rank too low in both BM25 and semantic space within
large citation pools.

## Next

1. **Protect BM25 rank-1 results:** The q036/q038 regressions suggest a
   conditional fusion approach: if BM25 has a strong rank-1 result (high score
   gap between rank 1 and rank 2), trust BM25 and skip fusion. This would
   protect confident BM25 results while still allowing semantic rescue for
   queries where BM25 is uncertain.

2. **Smaller citation pools:** The q004/q015/q026 failures suggest that
   base-only citation matching (e.g., all of "87103") creates pools too large
   for semantic to discriminate. Try using only exact citation matches
   (subsection-level) and falling back to base only when exact matches are
   sparse.

3. **Full-text semantic embeddings:** The qa_text embeddings miss opinions
   without qa_text (~3% of corpus). Embedding full_text (with truncation)
   could capture more opinions and provide better semantic coverage for the
   hard cases.

4. **Try 0.5/0.5 weighting as variant 005b:** More conservative weighting
   might protect q036/q038 while preserving most of the gains.
