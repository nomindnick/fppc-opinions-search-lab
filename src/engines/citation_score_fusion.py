"""
Citation-Filtered Score Fusion for FPPC opinions (experiment 009).

Combines the best of experiments 005 and 006:
- From 005: citation pooling to rescue statute-specific queries BM25 misses
- From 006: min-max score fusion + circuit breaker to prevent dilution

Two-path search:
- Non-citation queries: pure BM25 on full_text (no API call, no dilution risk)
- Citation queries: build candidate pool from citation matches ∪ BM25 top-100,
  then fuse with 0.4 BM25 / 0.6 semantic using min-max normalized scores.
  Circuit breaker fires when BM25 top1/top2 ratio >= 1.3, returning BM25 only.

Key difference from 005: score fusion instead of RRF preserves BM25 confidence
gaps, so strong BM25 results don't get diluted into uniform rank increments.
Key difference from 006: citation pooling constrains candidates before fusion,
letting semantic search discriminate within statute-specific pools.
"""

import os
import pickle
import sys

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

from src.interface import SearchEngine
from src.engines.bm25_full_text import tokenize
from src.engines.bm25_citation_boost import parse_query_citations

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_BM25_INDEX = os.path.join(_PROJECT_ROOT, "indexes", "BM25FullText_index.pkl")
_SEM_INDEX = os.path.join(
    _PROJECT_ROOT, "indexes", "embeddings_text-embedding-3-small_qa_text.pkl"
)
_CITATION_INDEX = os.path.join(
    _PROJECT_ROOT, "indexes", "BM25CitationBoost_citation_index.pkl"
)
_ENV_PATH = os.path.join(_PROJECT_ROOT, ".env")

_MODEL = "text-embedding-3-small"
_BM25_POOL = 100  # BM25 top-N to union into candidate pool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _min_max_normalize(pool: dict[str, float]) -> dict[str, float]:
    """Normalize scores to [0, 1] range within a pool."""
    if not pool:
        return pool
    vals = list(pool.values())
    lo, hi = min(vals), max(vals)
    rng = hi - lo
    if rng == 0:
        return {k: 1.0 for k in pool}
    return {k: (v - lo) / rng for k, v in pool.items()}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class CitationScoreFusion(SearchEngine):
    """Citation-filtered score fusion: BM25 + semantic within citation pools."""

    def __init__(self, cb_threshold: float = 1.3, w_bm25: float = 0.4,
                 w_sem: float = 0.6):
        self._cb_threshold = cb_threshold
        self._w_bm25 = w_bm25
        self._w_sem = w_sem

        # OpenAI client for query embedding
        load_dotenv(_ENV_PATH, override=True)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found. Set it in .env or as an env variable."
            )
        self._client = OpenAI(api_key=api_key)

        # Load BM25 index
        print(f"Loading BM25 index from {_BM25_INDEX}...")
        with open(_BM25_INDEX, "rb") as f:
            bm25_data = pickle.load(f)
        self._bm25_ids = bm25_data["opinion_ids"]
        self._bm25 = bm25_data["bm25"]
        self._bm25_id_to_idx = {oid: i for i, oid in enumerate(self._bm25_ids)}
        print(f"  BM25: {len(self._bm25_ids)} opinions")

        # Load semantic index
        print(f"Loading semantic index from {_SEM_INDEX}...")
        with open(_SEM_INDEX, "rb") as f:
            sem_data = pickle.load(f)
        self._sem_ids = sem_data["opinion_ids"]
        self._embeddings = sem_data["embeddings"]
        self._sem_id_to_idx = {oid: i for i, oid in enumerate(self._sem_ids)}
        print(f"  Semantic: {len(self._sem_ids)} opinions")

        # Load citation index
        print(f"Loading citation index from {_CITATION_INDEX}...")
        with open(_CITATION_INDEX, "rb") as f:
            self._cite_index = pickle.load(f)
        gc_exact = self._cite_index["gc_exact"]
        gc_base = self._cite_index["gc_base"]
        reg_exact = self._cite_index["reg_exact"]
        print(f"  gc_exact entries: {len(gc_exact)}, "
              f"gc_base entries: {len(gc_base)}, "
              f"reg_exact entries: {len(reg_exact)}")

    def search(self, query: str, top_k: int = 20) -> list[str]:
        # --- BM25 scoring (always needed) ---
        tokens = tokenize(query)
        if not tokens:
            return []
        bm25_scores = self._bm25.get_scores(tokens)

        # --- Check for citations in query ---
        parsed = parse_query_citations(query)
        has_citations = bool(parsed["gov_code"] or parsed["regulations"])

        if not has_citations:
            # Path B: pure BM25, no API call
            top_indices = bm25_scores.argsort()[::-1][:top_k]
            return [self._bm25_ids[i] for i in top_indices if bm25_scores[i] > 0]

        # --- Path A: citation-pooled score fusion ---

        # Step 1: Build candidate pool from citation matches
        pool = set()
        gc_exact = self._cite_index["gc_exact"]
        gc_base = self._cite_index["gc_base"]
        reg_exact = self._cite_index["reg_exact"]

        for cite in parsed["gov_code"]:
            pool |= gc_exact.get(cite["raw"], set())
            pool |= gc_base.get(cite["base"], set())

        for cite in parsed["regulations"]:
            pool |= reg_exact.get(cite["raw"], set())
            if cite["subsection"]:
                pool |= reg_exact.get(cite["base"], set())

        # Union with BM25 top-100 (safety net)
        bm25_top100 = {
            self._bm25_ids[i]
            for i in bm25_scores.argsort()[::-1][:_BM25_POOL]
            if bm25_scores[i] > 0
        }
        candidate_pool = pool | bm25_top100

        if not candidate_pool:
            return []

        # Step 2: Extract raw BM25 scores for pool members
        bm25_pool = {}
        for oid in candidate_pool:
            idx = self._bm25_id_to_idx.get(oid)
            if idx is not None:
                bm25_pool[oid] = float(bm25_scores[idx])
            else:
                bm25_pool[oid] = 0.0

        # Step 3: Circuit breaker on pool-scoped BM25 scores
        sorted_scores = sorted(bm25_pool.values(), reverse=True)
        if len(sorted_scores) <= 1:
            ratio = float("inf")
        else:
            ratio = (sorted_scores[0] / sorted_scores[1]
                     if sorted_scores[1] > 0 else float("inf"))

        if ratio >= self._cb_threshold:
            print(
                f"  [CB] ratio={ratio:.2f} >= {self._cb_threshold} — "
                f"returning BM25-only for: {query[:80]}",
                file=sys.stderr,
            )
            return sorted(bm25_pool, key=bm25_pool.get, reverse=True)[:top_k]

        # Step 4: Embed query, compute cosine similarities for pool members
        resp = self._client.embeddings.create(model=_MODEL, input=[query])
        query_vec = np.array(resp.data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec /= norm

        cos_scores = self._embeddings @ query_vec

        sem_pool = {}
        for oid in candidate_pool:
            idx = self._sem_id_to_idx.get(oid)
            if idx is not None:
                sem_pool[oid] = float(cos_scores[idx])
            else:
                sem_pool[oid] = 0.0

        # Step 5: Min-max normalize both score sets within the pool
        norm_bm25 = _min_max_normalize(bm25_pool)
        norm_sem = _min_max_normalize(sem_pool)

        # Step 6: Combine with weighted scores
        combined = {}
        for oid in candidate_pool:
            b = norm_bm25.get(oid, 0.0)
            s = norm_sem.get(oid, 0.0)
            combined[oid] = self._w_bm25 * b + self._w_sem * s

        # Step 7: Return top-k by combined score
        return sorted(combined, key=combined.get, reverse=True)[:top_k]

    def name(self) -> str:
        return "CitationScoreFusion"
