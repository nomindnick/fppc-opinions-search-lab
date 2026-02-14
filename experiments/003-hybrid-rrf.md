# 003 — Hybrid Weighted RRF (BM25 + Semantic)

**Engine:** `src/engines/hybrid_rrf_v1.py`
**Results:** `results/003-hybrid-rrf.json`
**Date:** 2026-02-14

## What I tried

Weighted Reciprocal Rank Fusion combining BM25 on full_text (001b) with semantic
search on qa_text (002). Formula:

    score(d) = 0.7/(60 + rank_bm25) + 0.3/(60 + rank_sem)

Top 100 candidates from each arm, unioned for fusion. Documents appearing in
only one pool get contribution from that engine only. Standard RRF constant k=60.

Weights chosen at 70/30 BM25-favoring because BM25 has the stronger individual
MRR (0.620 vs 0.485).

## Key numbers

| Metric   | Overall | Keyword | NatLang | FactPat |
|----------|---------|---------|---------|---------|
| MRR      | 0.620   | 0.458   | 0.720   | 0.737   |
| nDCG@5   | 0.373   | 0.308   | 0.416   | 0.416   |
| nDCG@10  | 0.326   | 0.273   | 0.358   | 0.367   |
| R@20     | 0.277   | 0.276   | 0.301   | 0.248   |

Comparison to component engines:

| Metric    | BM25 (001b) | Semantic (002) | Hybrid (003) |
|-----------|-------------|----------------|--------------|
| MRR       | 0.620       | 0.485          | 0.620        |
| nDCG@5    | 0.358       | 0.262          | 0.373        |
| nDCG@10   | 0.321       | 0.237          | 0.326        |
| R@20      | 0.252       | 0.194          | 0.277        |
| Zero-MRR  | 13          | 16             | 12           |

## Observations

**MRR is flat at 0.620** — same as BM25 alone. The per-query wins (12) and losses
(9) cancel out, with 44 ties.

**Ranking quality improved slightly.** nDCG@5 gained +0.015, meaning the top-5
results got modestly better even though the first relevant result didn't move
(MRR). Recall@20 gained +0.025, meaning fusion surfaces more relevant documents
across the full results list.

**Keyword queries improved** from 0.427→0.458 MRR, the only type to gain on MRR.

**Big wins** where semantic boosted BM25:
- q061 (definition of local government agency): 0.250→1.000
- q043 (committee qualification requirements): 0.500→1.000
- q044 (honorarium ban): 0.500→1.000
- q053 (lobbying firm gift): 0.500→1.000
- q011 (redevelopment recusal): 0.067→0.500

**Big losses** where semantic diluted BM25's correct rankings:
- q038 (transfer ban): 1.000→0.125 — BM25 had it at rank 1, fusion pushed it down
- q009 (technology company contract): 1.000→0.200 — same pattern
- q013 (adjacent property vote): 1.000→0.500
- q060 (JPA code reviewing body): 1.000→0.500

**70/30 weighting kills semantic-only rescues.** Seven queries where semantic found
the answer but BM25 didn't (q001, q004, q015, q018, q026, q027) still show zero
MRR in the hybrid. A semantic-only rank-1 score (0.3/61 = 0.0049) is dominated
by any BM25-only rank-1 score (0.7/61 = 0.0115), so purely semantic hits can't
reach the top 20. This is the biggest weakness of the current weighting.

## Next

The 70/30 weighting is too aggressive for BM25. It prevents semantic from
rescuing queries that BM25 completely misses, while also diluting BM25's
confident rank-1 results. Several directions to explore:

1. **Equal weighting (50/50) or semantic-favoring for RRF** — test whether
   letting semantic compete equally improves the rescue rate without degrading
   BM25's strong queries too much.

2. **Larger candidate pool** — currently top 100. Expanding to 200+ might help
   semantic contributions survive into the final top 20.

3. **Score-based fusion instead of rank-based** — normalize BM25 and cosine
   scores to [0,1] and do weighted sum. This preserves score magnitude, so a
   very strong BM25 match stays at rank 1 even with semantic contribution, while
   a strong semantic-only match can still break through.

4. **Conditional fusion** — use BM25 alone when BM25 confidence is high
   (e.g., top score >> 2nd score), add semantic only when BM25 is uncertain.

5. **Semantic on full_text instead of qa_text** — the BM25 improvement from
   qa_text→full_text was substantial (0.541→0.620). Embedding full_text might
   similarly improve the semantic arm, though token truncation is a concern.
