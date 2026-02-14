"""
Citation-Filtered Hybrid search engine for FPPC opinions.

For queries with citation references (statute/regulation numbers), builds a
candidate pool from citation-matching opinions + BM25 top-100, then fuses BM25
and semantic rankings with semantic-favoring weights (0.4/0.6) using RRF.
For queries without citations, falls back to pure BM25.

Key insight from experiment 004: BM25 full_text cannot rank accurately within
citation pools (relevant opinion ranks 32nd among ~1,463 citing "87103(a)").
Semantic similarity is more discriminative within these statute-specific pools,
so we weight it higher when citations narrow the candidate set.

RRF formula: score(d) = 0.4/(60 + bm25_rank) + 0.6/(60 + sem_rank)
"""

import os
import pickle

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

from src.interface import SearchEngine
from src.engines.bm25_full_text import tokenize
from src.engines.bm25_citation_boost import parse_query_citations

# Project root (two levels up from this file)
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

# RRF parameters
_K = 60       # standard RRF constant
_W_BM25 = 0.4  # BM25 weight (lower — BM25 is noisy within citation pools)
_W_SEM = 0.6   # semantic weight (higher — more discriminative in pools)
_BM25_POOL = 100  # BM25 top-N to union into candidate pool


class CitationSemanticHybrid(SearchEngine):
    """Citation-filtered hybrid: RRF of BM25 + semantic within citation pools."""

    def __init__(self):
        # OpenAI client for query embedding
        load_dotenv(_ENV_PATH, override=True)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found. Set it in .env or as an environment variable."
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
        # BM25 scores for all documents (cheap numpy operation)
        tokens = tokenize(query)
        if not tokens:
            return []
        bm25_scores = self._bm25.get_scores(tokens)

        # Parse citations from query
        parsed = parse_query_citations(query)
        has_citations = bool(parsed["gov_code"] or parsed["regulations"])

        if not has_citations:
            # No citations — pure BM25 fallback
            top_indices = bm25_scores.argsort()[::-1][:top_k]
            return [self._bm25_ids[i] for i in top_indices if bm25_scores[i] > 0]

        # --- Citation path: build candidate pool ---
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

        # --- BM25 arm: rank candidates by BM25 score ---
        bm25_pool_scores = {}
        for oid in candidate_pool:
            idx = self._bm25_id_to_idx.get(oid)
            if idx is not None:
                bm25_pool_scores[oid] = bm25_scores[idx]
            else:
                bm25_pool_scores[oid] = 0.0

        bm25_ranked = sorted(bm25_pool_scores, key=bm25_pool_scores.get, reverse=True)
        bm25_ranks = {oid: rank for rank, oid in enumerate(bm25_ranked, start=1)}

        # --- Semantic arm: rank candidates by cosine similarity ---
        resp = self._client.embeddings.create(model=_MODEL, input=[query])
        query_vec = np.array(resp.data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec /= norm

        # Full matrix multiply then extract pool members
        cos_scores = self._embeddings @ query_vec

        sem_pool_scores = {}
        for oid in candidate_pool:
            idx = self._sem_id_to_idx.get(oid)
            if idx is not None:
                sem_pool_scores[oid] = cos_scores[idx]
            else:
                sem_pool_scores[oid] = 0.0

        sem_ranked = sorted(sem_pool_scores, key=sem_pool_scores.get, reverse=True)
        sem_ranks = {oid: rank for rank, oid in enumerate(sem_ranked, start=1)}

        # --- RRF fusion ---
        rrf = {}
        for oid in candidate_pool:
            rrf[oid] = (
                _W_BM25 / (_K + bm25_ranks[oid])
                + _W_SEM / (_K + sem_ranks[oid])
            )

        return sorted(rrf, key=rrf.get, reverse=True)[:top_k]

    def name(self) -> str:
        return "CitationSemanticHybrid"
