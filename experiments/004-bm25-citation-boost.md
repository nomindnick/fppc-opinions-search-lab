# 004 — BM25 + Citation Boost (IDF-weighted)

**Engine:** `src/engines/bm25_citation_boost.py`
**Results:** `results/004-bm25-citation-boost.json`
**Date:** 2026-02-14

## What I tried

Starting from 001b (BM25 on full_text, MRR 0.620), added IDF-weighted additive
boosts for opinions whose structured `citations.government_code` and
`citations.regulations` fields match statute/regulation references extracted from
the query. Also added a small topic boost when the query's topic could be
confidently inferred from citations and keywords.

**How it works:**
1. **Query parser** extracts statute references (e.g., "Section 87103(a)" → base
   "87103", subsection "(a)") and regulation references (e.g., "Regulation 18702.2")
   using regex patterns.
2. **Citation index** maps each statute/regulation string to the set of opinions
   that cite it (built from structured citation fields, ~80% coverage).
3. **IDF-weighted boost** for each citation match: `boost = max_bm25 × 0.30 × Σ(match_score × IDF)`.
   Exact subsection match scores 1.0, base-only match scores 0.2. IDF = log(N/df)
   so rare citations get larger boosts.
4. **Topic boost**: `max_bm25 × 0.03` for opinions matching inferred topic.
5. **Combined score**: BM25 + citation boost + topic boost, ranked over all 14K opinions.

**Hypothesis:** The structured citation fields precisely record which statutes
each opinion *analyzes*, unlike BM25 which can't distinguish analysis from
boilerplate mentions. Citation boosting should help the 9 statute-heavy zero-MRR
queries (q001, q004, q010, q012, q015, q018, q020, q022, q026).

## Key numbers

| Metric   | Overall | Keyword | NatLang | FactPat |
|----------|---------|---------|---------|---------|
| MRR      | 0.615   | 0.422   | 0.715   | 0.783   |
| nDCG@5   | 0.367   | 0.286   | 0.411   | 0.436   |
| nDCG@10  | 0.330   | 0.277   | 0.357   | 0.378   |
| R@20     | 0.277   | 0.272   | 0.272   | 0.290   |

Comparison to baseline:

| Metric    | BM25 (001b) | Citation Boost (004) | Delta   |
|-----------|-------------|----------------------|---------|
| MRR       | 0.620       | 0.615                | -0.004  |
| nDCG@5    | 0.358       | 0.367                | +0.009  |
| nDCG@10   | 0.321       | 0.330                | +0.010  |
| R@20      | 0.252       | 0.277                | +0.025  |
| Zero-MRR  | 13          | 13                   | 0       |

Per-query: 1 win, 2 losses, 62 ties. No MRR=1.0 regressions.

## Observations

**The 9 target zero-MRR queries all remained at zero.** Citation boosting
completely failed to rescue any of them.

**Root cause analysis — three compounding problems:**

1. **Citation saturation in BM25 top-20.** For most target queries, the BM25
   top-20 already overwhelmingly cite the same statute. For q001 (87103(a)),
   18/20 top BM25 results cite "87103(a)" exactly. For q012 (87103(e)), 19/20
   match. A uniform boost applied to both top-20 and the relevant opinion changes
   nothing — it's adding a constant.

2. **Relevant opinions rank poorly even within citation-matching subsets.** I
   tested using citations as a hard filter (BM25 only over opinions citing the
   target statute). Results: q001's best relevant opinion ranks 32nd among 1,463
   opinions citing "87103(a)". q018's best relevant ranks 27th among 217 citing
   "87400". The relevant opinions don't have high BM25 full_text scores even
   within their citation pool — the problem is BM25 scoring, not filtering.

3. **Inconsistent subsection extraction.** For q010 (87103(d)), all three
   score-2 relevant opinions have `cites_87103(d)=False` in the citation index —
   they only have base "87103". The citation extractor didn't capture subsection-
   level detail for these opinions, so exact-match boosting can't help.

**Minor depth metric improvements.** Despite failing on MRR, the citation boost
improved all other metrics — especially R@20 (+0.025, +10% relative). The boost
shuffles some relevant documents higher in the list (just not to rank 1). The
R@20 gain was largest for fact_pattern queries (+0.055).

**Small regressions.** q043 dropped 0.500→0.333, q058 dropped 0.333→0.200. Both
are keyword queries where the citation boost for common statutes shifted rankings
enough to demote a previously good result.

## Next

Citation boosting on BM25 full_text is a dead end for the zero-MRR queries.
The fundamental problem: **BM25 full_text can't rank accurately within large
statute-specific pools** because the same keywords (disqualification, financial
interest, conflict, etc.) appear in thousands of opinions about the same statute.

More promising directions:

1. **Semantic re-ranking within citation pools.** Use citation matching to filter
   to ~200-2000 candidates, then re-rank by embedding similarity. This combines
   the precision of citation matching (narrows the pool) with the semantic
   understanding that BM25 lacks (ranks within the pool).

2. **Hybrid RRF parameter tuning (experiment 003 follow-up).** The semantic arm
   in experiment 002 actually found relevant results for several zero-MRR queries
   (q001, q004, q015, q018, q026, q027). The 70/30 BM25-heavy weighting just
   prevented them from surfacing. A 50/50 split or score-based fusion might
   rescue those queries while keeping BM25's strong results.

3. **Field-specific BM25.** Score BM25 on qa_text (more focused) instead of
   full_text (noisy), combined with citation filtering. The higher precision of
   qa_text might rank better within citation-matching subsets.
