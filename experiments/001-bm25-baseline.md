# 001 — BM25 Baseline

**Engine:** `src/engines/bm25_baseline.py`
**Results:** `results/001-bm25-baseline.json`
**Date:** 2026-02-13

## What I tried

Plain BM25Okapi (k1=1.5, b=0.75) over the `embedding.qa_text` field, which concatenates each opinion's question and conclusion/short answer. Fallback to `content.full_text` for the ~59 opinions with degenerate qa_text (<20 chars).

Tokenization: lowercase, merge parenthetical statute subsections (e.g. `87103(a)` → `87103a`), strip non-alphanumeric chars (preserving hyphens), remove ~130 English stopwords (keeping "not"/"no" for legal significance). No stemming.

This is the simplest reasonable keyword baseline — no field boosting, no query expansion, no stemming.

## Key numbers

| Metric   | Overall | Keyword | NatLang | FactPat |
|----------|---------|---------|---------|---------|
| MRR      |   0.541 |   0.389 |   0.619 |   0.673 |
| nDCG@5   |   0.301 |   0.220 |   0.345 |   0.367 |
| nDCG@10  |   0.268 |   0.209 |   0.309 |   0.306 |
| R@20     |   0.203 |   0.206 |   0.220 |   0.176 |

By topic:

| Metric   | Conflicts | Campaign | Gifts | Lobbying | Other |
|----------|-----------|----------|-------|----------|-------|
| MRR      |     0.263 |    0.782 |  0.786|    0.900 |  0.658|
| nDCG@5   |     0.113 |    0.425 |  0.537|    0.672 |  0.319|
| R@20     |     0.089 |    0.269 |  0.353|    0.409 |  0.235|

## Observations

**29 of 65 queries (45%) achieved MRR=1.0** — BM25 ranks the best answer first nearly half the time, mostly on narrower topics (lobbying, gifts) where distinctive vocabulary helps.

**14 queries (22%) scored MRR=0** — the engine completely missed the best answer in the top 20. These cluster in conflicts_of_interest and queries referencing specific regulations by number (e.g. "Regulation 18702.2", "Section 87103(d)/(e)"), the rule of necessity, the public generally exception, and common law conflicts. The qa_text field may not always contain the regulation numbers that keyword queries target.

**Conflicts of interest is the hardest topic by far** (MRR 0.263 vs next-worst 0.658). This is the largest topic in the corpus — more opinions means more noise for BM25 to sift through, and the vocabulary across conflict-of-interest opinions is highly overlapping (the same statutes and phrases appear in many opinions).

**Lobbying is the easiest** (MRR 0.900). Small, specialized sub-corpus with distinctive terminology.

**Keyword queries underperform fact patterns and natural language** on MRR (0.389 vs 0.673/0.619). Many keyword queries are dense statute-number strings that may partially match many documents without clearly distinguishing the target. Fact pattern queries, despite being longer, provide richer discriminating vocabulary.

**Recall@20 is uniformly low (~20%)** across all query types. 10 queries had zero recall — BM25 couldn't surface a single relevant document in the top 20. This is the ceiling for a re-ranker; improving recall requires better retrieval.

## Next

- **Index full_text instead of (or in addition to) qa_text** — the analysis sections contain statute numbers and regulatory citations that keyword queries target but qa_text may omit.
- **Try stemming** — legal text has many inflected forms ("disqualified"/"disqualification"/"disqualifying") that stemming would collapse.
- **Multi-field BM25** — score qa_text and full_text separately, then combine with tunable weights.
- **Semantic search** — the 14 MRR=0 queries and the conflicts_of_interest topic likely need vector/embedding-based retrieval to handle vocabulary mismatch. A hybrid BM25+embedding approach could preserve BM25's strength on keyword queries while rescuing the ones it misses entirely.
- **Query expansion** — for keyword queries with statute numbers, automatically expand to include common co-occurring terms.
