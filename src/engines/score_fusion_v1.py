"""
Score-based fusion with BM25 circuit breaker for FPPC opinions.

Combines BM25 on full_text with semantic search on qa_text using min-max
normalized score fusion (50/50 weighting). A circuit breaker protects
confident BM25 results from dilution: when BM25's top score is much higher
than the second score (ratio >= threshold), fusion is skipped and BM25
results are returned as-is.

This addresses the dilution problem seen in experiments 003 and 005, where
rank-based fusion (RRF) hurt queries like q038 (MRR 1.0 → 0.125) by
flattening BM25's score signal into uniform rank increments.
"""

import os
import pickle
import re
import sys

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

from src.interface import SearchEngine

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BM25_INDEX = os.path.join(_PROJECT_ROOT, "indexes", "BM25FullText_index.pkl")
_SEM_INDEX = os.path.join(_PROJECT_ROOT, "indexes", "embeddings_text-embedding-3-small_qa_text.pkl")
_ENV_PATH = os.path.join(_PROJECT_ROOT, ".env")

_MODEL = "text-embedding-3-small"
_POOL = 100  # top-N candidates from each arm

# ---------------------------------------------------------------------------
# Tokenizer (copied from bm25_full_text.py for self-containment)
# ---------------------------------------------------------------------------
STOPWORDS = frozenset([
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "could", "did", "do", "does", "doing", "done", "down", "during",
    "each", "few", "for", "from", "further", "get", "got", "had", "has",
    "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its",
    "itself", "just", "let", "may", "me", "might", "more", "most", "much",
    "must", "my", "myself", "nor", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shall", "she", "should", "so", "some", "such", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "upon", "us", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "would", "yet",
    "you", "your", "yours", "yourself", "yourselves",
    "about", "above", "after", "again", "against", "all", "am", "any",
    "because", "before", "below", "between", "both", "also",
])

_PAREN_SUB = re.compile(r"(\d+)\(([a-zA-Z0-9])\)")
_NON_ALNUM = re.compile(r"[^a-z0-9\-]+")


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 indexing/querying."""
    text = text.lower()
    text = _PAREN_SUB.sub(r"\1\2", text)
    text = _NON_ALNUM.sub(" ", text)
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class ScoreFusionV1(SearchEngine):
    """Score-based fusion of BM25 + Semantic with BM25 circuit breaker."""

    def __init__(self, cb_threshold: float = 1.3):
        self._cb_threshold = cb_threshold

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

        # Lookup: opinion_id → index in semantic arrays
        self._sem_id_to_idx = {oid: i for i, oid in enumerate(self._sem_ids)}

    def search(self, query: str, top_k: int = 20) -> list[str]:
        # --- Step 1: BM25 scoring ---
        tokens = tokenize(query)
        if not tokens:
            return []

        bm25_scores = self._bm25.get_scores(tokens)

        # --- Step 2: Extract top-100 with score > 0 ---
        top_idx = bm25_scores.argsort()[::-1][:_POOL]
        bm25_pool = {}
        for idx in top_idx:
            if bm25_scores[idx] > 0:
                bm25_pool[self._bm25_ids[idx]] = float(bm25_scores[idx])

        if not bm25_pool:
            return []

        # --- Step 3: Circuit breaker ---
        sorted_scores = sorted(bm25_pool.values(), reverse=True)
        if len(sorted_scores) == 1:
            ratio = float("inf")
        else:
            ratio = sorted_scores[0] / sorted_scores[1] if sorted_scores[1] > 0 else float("inf")

        if ratio >= self._cb_threshold:
            print(
                f"  [CB] ratio={ratio:.2f} >= {self._cb_threshold} — "
                f"returning BM25-only for: {query[:80]}",
                file=sys.stderr,
            )
            return sorted(bm25_pool, key=bm25_pool.get, reverse=True)[:top_k]

        # --- Step 4: Semantic scoring (only if circuit breaker didn't fire) ---
        resp = self._client.embeddings.create(model=_MODEL, input=[query])
        query_vec = np.array(resp.data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec /= norm
        cos_scores = self._embeddings @ query_vec

        # --- Step 5: Extract top-100 semantic candidates ---
        sem_top_idx = cos_scores.argsort()[::-1][:_POOL]
        sem_pool = {}
        for idx in sem_top_idx:
            sem_pool[self._sem_ids[idx]] = float(cos_scores[idx])

        # --- Step 6: Union candidates ---
        all_ids = set(bm25_pool) | set(sem_pool)

        # --- Step 7: Min-max normalize each pool to [0, 1] ---
        def min_max_normalize(pool: dict[str, float]) -> dict[str, float]:
            if not pool:
                return pool
            vals = list(pool.values())
            lo, hi = min(vals), max(vals)
            rng = hi - lo
            if rng == 0:
                return {k: 1.0 for k in pool}
            return {k: (v - lo) / rng for k, v in pool.items()}

        norm_bm25 = min_max_normalize(bm25_pool)
        norm_sem = min_max_normalize(sem_pool)

        # --- Step 8: Combined score (0.5 / 0.5) ---
        combined = {}
        for oid in all_ids:
            b = norm_bm25.get(oid, 0.0)
            s = norm_sem.get(oid, 0.0)
            combined[oid] = 0.5 * b + 0.5 * s

        # --- Step 9: Return top-k ---
        return sorted(combined, key=combined.get, reverse=True)[:top_k]

    def name(self) -> str:
        return "ScoreFusionV1"
