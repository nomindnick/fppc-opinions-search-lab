"""
Score-based fusion with BM25 circuit breaker and multi-vector semantic arm.

Combines experiment 006's score fusion framework (min-max normalization, 50/50
weighting, circuit breaker at ratio >= 1.3) with experiment 007's multi-vector
semantic scoring (MAX over qa_text, facts, and analysis embeddings).

The hypothesis: multi-vector MAX improves the semantic arm for fact-pattern
queries (007's strength), while the circuit breaker prevents keyword dilution
(007's main failure mode). Score-based fusion avoids the rank-flattening
problem that hurt RRF in experiments 003 and 005.
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
_QA_INDEX = os.path.join(_PROJECT_ROOT, "indexes", "embeddings_text-embedding-3-small_qa_text.pkl")
_FACTS_INDEX = os.path.join(_PROJECT_ROOT, "indexes", "embeddings_text-embedding-3-small_facts.pkl")
_ANALYSIS_INDEX = os.path.join(_PROJECT_ROOT, "indexes", "embeddings_text-embedding-3-small_analysis.pkl")
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
class ScoreFusionMultiVec(SearchEngine):
    """Score fusion of BM25 + multi-vector semantic with BM25 circuit breaker."""

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

        # --- Load BM25 index ---
        print(f"Loading BM25 index from {_BM25_INDEX}...")
        with open(_BM25_INDEX, "rb") as f:
            bm25_data = pickle.load(f)
        self._bm25_ids = bm25_data["opinion_ids"]
        self._bm25 = bm25_data["bm25"]
        print(f"  BM25: {len(self._bm25_ids)} opinions")

        # --- Load semantic indexes (all three) ---
        print(f"Loading qa_text index from {_QA_INDEX}...")
        with open(_QA_INDEX, "rb") as f:
            qa_data = pickle.load(f)
        self._master_ids = qa_data["opinion_ids"]
        self._qa_embeddings = qa_data["embeddings"]
        n = len(self._master_ids)
        print(f"  QA: {n} opinions")

        master_id_to_idx = {oid: i for i, oid in enumerate(self._master_ids)}

        print(f"Loading facts index from {_FACTS_INDEX}...")
        with open(_FACTS_INDEX, "rb") as f:
            facts_data = pickle.load(f)
        self._facts_embeddings = facts_data["embeddings"]
        self._facts_mapping = np.array(
            [master_id_to_idx[oid] for oid in facts_data["opinion_ids"]],
            dtype=np.int32,
        )
        print(f"  Facts: {len(self._facts_mapping)} opinions")

        print(f"Loading analysis index from {_ANALYSIS_INDEX}...")
        with open(_ANALYSIS_INDEX, "rb") as f:
            analysis_data = pickle.load(f)
        self._analysis_embeddings = analysis_data["embeddings"]
        self._analysis_mapping = np.array(
            [master_id_to_idx[oid] for oid in analysis_data["opinion_ids"]],
            dtype=np.int32,
        )
        print(f"  Analysis: {len(self._analysis_mapping)} opinions")

        print(
            f"ScoreFusionMultiVec ready: {len(self._bm25_ids)} BM25, "
            f"{n} QA, {len(self._facts_mapping)} facts, "
            f"{len(self._analysis_mapping)} analysis"
        )

    def search(self, query: str, top_k: int = 20) -> list[str]:
        # --- Step 1: BM25 scoring ---
        tokens = tokenize(query)
        if not tokens:
            return []

        bm25_scores = self._bm25.get_scores(tokens)

        # --- Step 2: Extract top-100 BM25 candidates with score > 0 ---
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

        # --- Step 4: Multi-vector semantic scoring ---
        resp = self._client.embeddings.create(model=_MODEL, input=[query])
        query_vec = np.array(resp.data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec /= norm

        n = len(self._master_ids)

        # QA scores (full coverage)
        sem_scores = self._qa_embeddings @ query_vec

        # Facts scores (sparse — scatter into full array, then MAX)
        facts_scores = self._facts_embeddings @ query_vec
        facts_full = np.zeros(n, dtype=np.float32)
        facts_full[self._facts_mapping] = facts_scores
        np.maximum(sem_scores, facts_full, out=sem_scores)

        # Analysis scores (sparse — scatter into full array, then MAX)
        analysis_scores = self._analysis_embeddings @ query_vec
        analysis_full = np.zeros(n, dtype=np.float32)
        analysis_full[self._analysis_mapping] = analysis_scores
        np.maximum(sem_scores, analysis_full, out=sem_scores)

        # --- Step 5: Extract top-100 semantic candidates ---
        sem_top_idx = sem_scores.argsort()[::-1][:_POOL]
        sem_pool = {}
        for idx in sem_top_idx:
            sem_pool[self._master_ids[idx]] = float(sem_scores[idx])

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
        return "ScoreFusionMultiVec"
