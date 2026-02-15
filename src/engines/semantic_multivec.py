"""
Multi-Vector Semantic search engine for FPPC opinions.

Extends the semantic baseline (002) by embedding three different text
representations per opinion — qa_text, facts, and analysis — and taking
the MAX similarity across all three at search time. The hypothesis is
that different query types will match different aspects of an opinion,
improving recall over a single-vector approach.

Uses OpenAI text-embedding-3-small with L2-normalized vectors and cosine
similarity via dot product, same as the semantic baseline.
"""

import json
import os
import pickle
import sys

import numpy as np
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv

from src.interface import SearchEngine

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "extracted")
_INDEX_DIR = os.path.join(_PROJECT_ROOT, "indexes")
_ENV_PATH = os.path.join(_PROJECT_ROOT, ".env")

_QA_INDEX_PATH = os.path.join(_INDEX_DIR, "embeddings_text-embedding-3-small_qa_text.pkl")
_FACTS_INDEX_PATH = os.path.join(_INDEX_DIR, "embeddings_text-embedding-3-small_facts.pkl")
_ANALYSIS_INDEX_PATH = os.path.join(_INDEX_DIR, "embeddings_text-embedding-3-small_analysis.pkl")

_MODEL = "text-embedding-3-small"
_BATCH_SIZE = 64
_MAX_TOKENS = 8191
_MIN_TEXT_LEN = 20  # skip fields shorter than this

_enc = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(text: str) -> str:
    """Truncate text to fit within the model's token limit."""
    tokens = _enc.encode(text)
    if len(tokens) <= _MAX_TOKENS:
        return text
    return _enc.decode(tokens[:_MAX_TOKENS])


def _get_facts_text(opinion: dict) -> str:
    """Extract the facts section text from an opinion, or empty string."""
    text = opinion.get("sections", {}).get("facts", "") or ""
    return text if len(text) >= _MIN_TEXT_LEN else ""


def _get_analysis_text(opinion: dict) -> str:
    """Extract analysis + conclusion text, concatenating if both exist."""
    sections = opinion.get("sections", {})
    analysis = sections.get("analysis", "") or ""
    conclusion = sections.get("conclusion", "") or ""
    # Concat if both are present; use whichever exists otherwise
    if len(analysis) >= _MIN_TEXT_LEN and len(conclusion) >= _MIN_TEXT_LEN:
        return analysis + "\n\n" + conclusion
    if len(analysis) >= _MIN_TEXT_LEN:
        return analysis
    if len(conclusion) >= _MIN_TEXT_LEN:
        return conclusion
    return ""


def _load_opinions_full() -> list[tuple[str, dict]]:
    """Walk data/extracted/ and return list of (opinion_id, full_dict) tuples."""
    opinions = []
    for year_dir in sorted(os.listdir(_DATA_DIR)):
        year_path = os.path.join(_DATA_DIR, year_dir)
        if not os.path.isdir(year_path):
            continue
        for filename in sorted(os.listdir(year_path)):
            if not filename.endswith(".json"):
                continue
            filepath = os.path.join(year_path, filename)
            with open(filepath, "r") as f:
                opinion = json.load(f)
            oid = opinion.get("id", filename.replace(".json", ""))
            opinions.append((oid, opinion))
    return opinions


def _embed_texts(client: OpenAI, texts: list[str], label: str = "") -> np.ndarray:
    """Embed texts in batches, returning L2-normalized float32 array."""
    all_embeddings = []
    total = len(texts)
    for i in range(0, total, _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        batch = [_truncate(t) if t.strip() else " " for t in batch]
        print(
            f"  [{label}] Embedding batch {i // _BATCH_SIZE + 1}/"
            f"{(total + _BATCH_SIZE - 1) // _BATCH_SIZE}"
            f" ({len(batch)} texts)...",
            file=sys.stderr,
        )
        response = client.embeddings.create(model=_MODEL, input=batch)
        batch_vecs = [item.embedding for item in response.data]
        all_embeddings.extend(batch_vecs)

    matrix = np.array(all_embeddings, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    matrix /= norms
    return matrix


def _build_sparse_index(
    client: OpenAI,
    opinions: list[tuple[str, dict]],
    extract_fn,
    index_path: str,
    label: str,
) -> dict:
    """Build an embedding index for a subset of opinions where extract_fn returns text."""
    opinion_ids = []
    texts = []
    for oid, opinion in opinions:
        text = extract_fn(opinion)
        if text:
            opinion_ids.append(oid)
            texts.append(text)

    print(
        f"  [{label}] {len(texts)} of {len(opinions)} opinions have text "
        f"({100 * len(texts) / len(opinions):.1f}% coverage)",
        file=sys.stderr,
    )

    embeddings = _embed_texts(client, texts, label=label)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    data = {"opinion_ids": opinion_ids, "embeddings": embeddings}
    with open(index_path, "wb") as f:
        pickle.dump(data, f)
    print(f"  [{label}] Index saved to {index_path}", file=sys.stderr)
    return data


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SemanticMultiVec(SearchEngine):
    """Multi-vector semantic search over qa_text, facts, and analysis fields."""

    def __init__(self):
        load_dotenv(_ENV_PATH, override=True)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found. Set it in .env or as an environment variable."
            )
        self._client = OpenAI(api_key=api_key)

        # --- Load or build all three indexes ---
        need_facts = not os.path.exists(_FACTS_INDEX_PATH)
        need_analysis = not os.path.exists(_ANALYSIS_INDEX_PATH)
        need_qa = not os.path.exists(_QA_INDEX_PATH)

        # Load corpus once if we need to build any index
        opinions = None
        if need_facts or need_analysis or need_qa:
            print("Loading corpus for index building...", file=sys.stderr)
            opinions = _load_opinions_full()
            print(f"Loaded {len(opinions)} opinions.", file=sys.stderr)

        # QA index (reuse existing or build)
        if need_qa:
            print("Building qa_text index...", file=sys.stderr)
            qa_ids = []
            qa_texts = []
            for oid, op in opinions:
                text = op.get("embedding", {}).get("qa_text", "") or ""
                if len(text) < _MIN_TEXT_LEN:
                    text = op.get("content", {}).get("full_text", "") or ""
                qa_ids.append(oid)
                qa_texts.append(text)
            qa_embeddings = _embed_texts(self._client, qa_texts, label="qa_text")
            qa_data = {"opinion_ids": qa_ids, "embeddings": qa_embeddings}
            os.makedirs(_INDEX_DIR, exist_ok=True)
            with open(_QA_INDEX_PATH, "wb") as f:
                pickle.dump(qa_data, f)
            print(f"qa_text index saved to {_QA_INDEX_PATH}", file=sys.stderr)
        else:
            print(f"Loading qa_text index from {_QA_INDEX_PATH}...", file=sys.stderr)
            with open(_QA_INDEX_PATH, "rb") as f:
                qa_data = pickle.load(f)

        # Facts index
        if need_facts:
            print("Building facts index...", file=sys.stderr)
            facts_data = _build_sparse_index(
                self._client, opinions, _get_facts_text,
                _FACTS_INDEX_PATH, "facts",
            )
        else:
            print(f"Loading facts index from {_FACTS_INDEX_PATH}...", file=sys.stderr)
            with open(_FACTS_INDEX_PATH, "rb") as f:
                facts_data = pickle.load(f)

        # Analysis index
        if need_analysis:
            print("Building analysis index...", file=sys.stderr)
            analysis_data = _build_sparse_index(
                self._client, opinions, _get_analysis_text,
                _ANALYSIS_INDEX_PATH, "analysis",
            )
        else:
            print(f"Loading analysis index from {_ANALYSIS_INDEX_PATH}...", file=sys.stderr)
            with open(_ANALYSIS_INDEX_PATH, "rb") as f:
                analysis_data = pickle.load(f)

        # --- Store master index (qa_text covers all opinions) ---
        self._master_ids = qa_data["opinion_ids"]
        self._qa_embeddings = qa_data["embeddings"]
        n = len(self._master_ids)

        # --- Build alignment mappings for sparse indexes ---
        master_id_to_idx = {oid: i for i, oid in enumerate(self._master_ids)}

        # Facts alignment
        facts_ids = facts_data["opinion_ids"]
        self._facts_embeddings = facts_data["embeddings"]
        self._facts_mapping = np.array(
            [master_id_to_idx[oid] for oid in facts_ids], dtype=np.int32
        )

        # Analysis alignment
        analysis_ids = analysis_data["opinion_ids"]
        self._analysis_embeddings = analysis_data["embeddings"]
        self._analysis_mapping = np.array(
            [master_id_to_idx[oid] for oid in analysis_ids], dtype=np.int32
        )

        print(
            f"SemanticMultiVec ready: {n} opinions, "
            f"{len(self._facts_mapping)} with facts, "
            f"{len(self._analysis_mapping)} with analysis",
            file=sys.stderr,
        )

    def search(self, query: str, top_k: int = 20) -> list[str]:
        # Embed query
        response = self._client.embeddings.create(model=_MODEL, input=[query])
        query_vec = np.array(response.data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec /= norm

        n = len(self._master_ids)

        # QA scores (full coverage)
        scores = self._qa_embeddings @ query_vec

        # Facts scores (sparse — scatter into full array, then max)
        facts_scores = self._facts_embeddings @ query_vec
        facts_full = np.zeros(n, dtype=np.float32)
        facts_full[self._facts_mapping] = facts_scores
        np.maximum(scores, facts_full, out=scores)

        # Analysis scores (sparse — scatter into full array, then max)
        analysis_scores = self._analysis_embeddings @ query_vec
        analysis_full = np.zeros(n, dtype=np.float32)
        analysis_full[self._analysis_mapping] = analysis_scores
        np.maximum(scores, analysis_full, out=scores)

        top_indices = scores.argsort()[::-1][:top_k]
        return [self._master_ids[idx] for idx in top_indices]

    def name(self) -> str:
        return "SemanticMultiVec"
