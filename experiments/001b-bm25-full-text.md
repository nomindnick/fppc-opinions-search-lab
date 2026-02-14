# 001b — BM25 Full-Text Baseline

**Engine:** `src/engines/bm25_full_text.py`
**Results:** `results/001b-bm25-full-text.json`
**Date:** 2026-02-13

## What I tried

Same BM25Okapi setup as 001 (k1=1.5, b=0.75, same tokenizer, same stopwords) but indexing `content.full_text` exclusively instead of `embedding.qa_text`. The hypothesis was that full_text contains statute numbers, regulation citations, and analysis details that keyword queries target but qa_text omits.

## Key numbers

| Metric   | Overall | Keyword | NatLang | FactPat |
|----------|---------|---------|---------|---------|
| MRR      |   0.620 |   0.427 |   0.722 |   0.783 |
| nDCG@5   |   0.358 |   0.284 |   0.407 |   0.410 |
| nDCG@10  |   0.321 |   0.261 |   0.357 |   0.365 |
| R@20     |   0.252 |   0.251 |   0.265 |   0.235 |

Delta from 001 (qa_text baseline):

| Metric   | Overall | Keyword | NatLang | FactPat |
|----------|---------|---------|---------|---------|
| MRR      |  +0.079 |  +0.038 |  +0.103 |  +0.110 |
| nDCG@5   |  +0.057 |  +0.064 |  +0.062 |  +0.043 |
| nDCG@10  |  +0.053 |  +0.052 |  +0.048 |  +0.059 |
| R@20     |  +0.049 |  +0.045 |  +0.045 |  +0.059 |

By topic:

| Metric   | Conflicts | Campaign | Gifts | Lobbying | Other |
|----------|-----------|----------|-------|----------|-------|
| MRR      |     0.323 |    0.864 | 0.929 |    0.900 | 0.783 |
| nDCG@5   |     0.155 |    0.527 | 0.541 |    0.607 | 0.461 |
| R@20     |     0.083 |    0.414 | 0.436 |    0.516 | 0.252 |

Topic deltas from 001:

| Metric   | Conflicts | Campaign | Gifts | Lobbying | Other |
|----------|-----------|----------|-------|----------|-------|
| MRR      |    +0.060 |   +0.082 |+0.143 |    0.000 |+0.125 |
| nDCG@5   |    +0.042 |   +0.102 |+0.004 |   -0.065 |+0.142 |
| R@20     |    -0.006 |   +0.145 |+0.083 |   +0.107 |+0.017 |

## Observations

**Full-text indexing improves every aggregate metric.** MRR jumps from 0.541 to 0.620 (+14.6%). The improvement is broad — every query type and 4 of 5 topics see MRR gains. R@20 also improves from 0.203 to 0.252, meaning BM25 surfaces more relevant documents when given the full opinion text.

**Fact patterns and natural language benefit more than keyword queries.** MRR gains: fact_pattern +0.110, natural_language +0.103, keyword only +0.038. The longer, richer text of full opinions provides more vocabulary overlap with descriptive queries. Keyword queries (dense statute-number strings) get a smaller boost — the extra text helps surface some statute references, but also introduces noise from the much larger document length.

**12 queries still score MRR=0** (down from 14 in 001). The zero-MRR queries remain heavily concentrated in conflicts_of_interest keyword queries: q001, q004, q010, q012, q015, q018, q020, q022, q026, q028 are all keyword/conflicts. Plus q023 (natural_language/conflicts) and q027 (natural_language/conflicts). The conflicts topic is still by far the hardest (MRR 0.323), and keyword queries into that topic are essentially broken for BM25.

**Gifts/honoraria improved the most** (MRR 0.786 → 0.929, +0.143). The full opinion text for gift opinions likely contains the specific dollar thresholds and exemption language that queries target.

**Lobbying MRR unchanged at 0.900**, though nDCG@5 dropped slightly (-0.065). The lobbying sub-corpus has distinctive terminology that works well with either field; full_text adds length without adding discriminating vocabulary, slightly diluting ranking precision.

**Conflicts R@20 actually decreased slightly** (0.089 → 0.083). The longer documents in full_text make BM25's length normalization work harder, and the overlapping vocabulary across the large conflicts corpus means more noise documents score above zero.

**Campaign finance saw notable nDCG@5 improvement** (+0.102), suggesting full_text helps rank relevant campaign opinions higher, not just find them.

## Next

- **Multi-field BM25** — combine qa_text and full_text scores with tunable weights. qa_text provides focused, high-precision signal; full_text provides broader recall. A weighted combination could beat both individual field approaches.
- **Stemming** — still untested. The conflicts_of_interest keyword failures suggest vocabulary mismatch is a core issue. Stemming could help "disqualification"/"disqualified"/"disqualifying" all match.
- **Hybrid BM25+semantic** — the 12 MRR=0 queries are almost all conflicts_of_interest. These likely need embedding-based retrieval to handle the vocabulary overlap problem. BM25 alone may have hit its ceiling on this topic.
- **Query analysis** — the keyword queries that score 0 tend to be abstract concept searches (e.g., "public generally exception", "legally required participation rule of necessity") where the relevant opinions may not use those exact phrases in the analysis text. Query expansion or synonym injection could help.
