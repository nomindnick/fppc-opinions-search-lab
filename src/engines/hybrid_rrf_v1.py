"""
Hybrid Weighted RRF search engine for FPPC opinions.

Combines BM25 on full_text (experiment 001b) with semantic search on qa_text
(experiment 002) using weighted Reciprocal Rank Fusion. BM25 is weighted at
0.7 and semantic at 0.3, reflecting BM25's stronger overall performance while
preserving semantic's ability to rescue queries that BM25 misses entirely.

RRF formula: score(d) = w_bm25/(k + rank_bm25) + w_sem/(k + rank_sem)
where k=60 (standard RRF constant).
"""

import os
import pickle

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

from src.interface import SearchEngine
from src.engines.bm25_full_text import tokenize

# Project root (two levels up from this file)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BM25_INDEX = os.path.join(_PROJECT_ROOT, "indexes", "BM25FullText_index.pkl")
_SEM_INDEX = os.path.join(_PROJECT_ROOT, "indexes", "embeddings_text-embedding-3-small_qa_text.pkl")
_ENV_PATH = os.path.join(_PROJECT_ROOT, ".env")

_MODEL = "text-embedding-3-small"

# RRF parameters
_K = 60        # standard RRF constant
_W_BM25 = 0.7  # BM25 weight
_W_SEM = 0.3   # semantic weight
_POOL = 100     # top-N candidates from each arm


class HybridRRFv1(SearchEngine):
    """Weighted Reciprocal Rank Fusion of BM25 (full_text) + Semantic (qa_text)."""

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
        print(f"  BM25: {len(self._bm25_ids)} opinions")

        # Load semantic index
        print(f"Loading semantic index from {_SEM_INDEX}...")
        with open(_SEM_INDEX, "rb") as f:
            sem_data = pickle.load(f)
        self._sem_ids = sem_data["opinion_ids"]
        self._embeddings = sem_data["embeddings"]
        print(f"  Semantic: {len(self._sem_ids)} opinions")

    def search(self, query: str, top_k: int = 20) -> list[str]:
        # --- BM25 arm ---
        bm25_ranks = {}
        tokens = tokenize(query)
        if tokens:
            scores = self._bm25.get_scores(tokens)
            top_idx = scores.argsort()[::-1][:_POOL]
            for rank, idx in enumerate(top_idx, start=1):
                if scores[idx] > 0:
                    bm25_ranks[self._bm25_ids[idx]] = rank

        # --- Semantic arm ---
        resp = self._client.embeddings.create(model=_MODEL, input=[query])
        query_vec = np.array(resp.data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec /= norm
        cos_scores = self._embeddings @ query_vec
        top_idx = cos_scores.argsort()[::-1][:_POOL]
        sem_ranks = {}
        for rank, idx in enumerate(top_idx, start=1):
            sem_ranks[self._sem_ids[idx]] = rank

        # --- Weighted RRF ---
        all_ids = set(bm25_ranks) | set(sem_ranks)
        rrf = {}
        for oid in all_ids:
            s = 0.0
            if oid in bm25_ranks:
                s += _W_BM25 / (_K + bm25_ranks[oid])
            if oid in sem_ranks:
                s += _W_SEM / (_K + sem_ranks[oid])
            rrf[oid] = s

        return sorted(rrf, key=rrf.get, reverse=True)[:top_k]

    def name(self) -> str:
        return "HybridRRFv1"
