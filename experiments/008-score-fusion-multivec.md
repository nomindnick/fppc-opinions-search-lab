# 008 — Score Fusion + Multi-Vector Semantic

**Engine:** `src/engines/score_fusion_multivec.py`
**Results:** `results/008-score-fusion-multivec.json`
**Date:** 2026-02-15

## What I tried

Combined experiment 006's score-based fusion framework (min-max normalization,
50/50 BM25/semantic weighting, circuit breaker at ≥1.3× ratio) with experiment
007's multi-vector semantic approach (MAX over qa_text, facts, and analysis
embeddings).

The hypothesis: the circuit breaker protects confident BM25 results from
dilution (007's main failure), while the richer multi-vector signal improves the
semantic arm when fusion fires — particularly for fact-pattern queries where 007
showed a +0.094 MRR gain.

Circuit breaker fired on the same 14/65 queries (22%) as 006, so differences
only appear on the 51 fusion queries.

## Key numbers

| Metric   | Overall | Keyword | NatLang | FactPat |
|----------|---------|---------|---------|---------|
| MRR      |   0.620 |   0.446 |   0.710 |   0.771 |
| nDCG@5   |   0.366 |   0.294 |   0.405 |   0.425 |
| nDCG@10  |   0.323 |   0.265 |   0.349 |   0.379 |
| R@20     |   0.255 |   0.246 |   0.263 |   0.259 |

Comparison to prior experiments:

| Metric | 001b BM25 | 005 Cit-Hybrid | 006 Score Fusion | **008 SF+MultiVec** |
|--------|-----------|----------------|------------------|---------------------|
| MRR    | 0.620     | **0.665**      | 0.632            | 0.620               |
| nDCG@5 | 0.345     | 0.394          | 0.372            | 0.366               |
| R@20   | 0.252     | 0.282          | 0.268            | 0.255               |
| Zeros  | 13        | 10             | **9**            | 10                  |

## Observations

**Overall MRR regressed** (0.632 → 0.620, -0.012). 9 wins, 14 losses, 42 ties
vs 006. The multi-vector semantic arm is noisier than single-vector qa_text,
hurting more queries than it helps.

**Fact pattern queries did NOT improve** — 0.771 vs 006's 0.796 (-0.025). This
is the opposite of what 007 suggested. The +0.094 fact-pattern gain in 007 came
from the multi-vector approach promoting different opinions into the top-20.
But within score fusion, those same opinions compete against BM25 candidates.
The MAX operation makes the semantic scores noisier, so more irrelevant opinions
enter the semantic pool, diluting the fusion.

**Natural language queries improved** (+0.029, 0.681 → 0.710). Wins on q002
(+0.108), q019 (+0.133), q043 (+0.500), q060 (+0.167). The richer analysis
vectors helped with reasoning-style queries.

**Keyword queries regressed** (-0.037, 0.482 → 0.446). Losses on q001 (-0.200),
q024 (-0.133), q038 (-0.167), q044 (-0.500), q061 (-0.500).

**10 zero-MRR queries** (vs 006's 9):
q001, q004, q010, q012, q015, q017, q018, q020, q023, q028.
Lost 006's partial rescues of q001 (0.2→0), q015 (0.059→0), q018 (0.143→0) —
the multi-vector noise pushed correct opinions out of the semantic top-100 pool.

**Biggest losses from 006:**
- q006: 1.0 → 0.5 — 006's big semantic win got diluted by noisy facts/analysis
- q044: 1.0 → 0.5 — analysis vectors from similar gift opinions crowded out
- q061: 1.0 → 0.5 — same dilution pattern

**Biggest wins over 006:**
- q058: 0.5 → 1.0 — analysis vectors rescued a campaign funds query
- q043: 0.5 → 1.0 — committee qualification matched analysis reasoning
- q032: 0.2 → 0.5 — modest improvement

### Root cause: multi-vector MAX hurts the semantic arm's precision

The circuit breaker protects BM25-confident queries (those 14 queries are
identical). But for the 51 fusion queries, the semantic arm's quality
*decreased*. MAX over 3 vectors introduces ~7,400 facts embeddings and ~10,800
analysis embeddings that compete with qa_text's focused signal. Within score
fusion, a noisier semantic pool means more irrelevant opinions enter the union,
and min-max normalization can't distinguish "confidently right" from "generically
similar."

This is a different failure mode than 007's: there, MAX hurt the top rank
globally. Here, MAX specifically hurts the semantic *pool quality*, which
cascades through fusion. The circuit breaker only prevents BM25 dilution — it
can't fix a weak semantic arm.

## Next

1. **Weighted multi-vector** — instead of MAX(qa, facts, analysis), try
   `0.5*qa + 0.25*max(facts, analysis)` as the semantic score within score
   fusion. This keeps qa_text dominant while using facts/analysis as a tie-
   breaker rather than an override.

2. **Multi-vector as re-ranker** — use single-vector qa_text for the semantic
   pool (preserving 006's precision), then re-rank the fused top-50 using the
   MAX-of-three score. This separates candidate generation (where precision
   matters) from ranking (where signal diversity helps).

3. **Combine 005 + 006** — citation-filtered pooling with score fusion. This
   is the most promising unexplored combination: 005's citation filtering was
   the only approach to rescue q001 and q012 to MRR 1.0, and score fusion
   reduces dilution better than RRF. Multi-vector is a distraction from this
   more fundamental architectural improvement.
