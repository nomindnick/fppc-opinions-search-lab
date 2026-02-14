"""
Semantic Baseline search engine for FPPC opinions.

Uses OpenAI text-embedding-3-small embeddings over the qa_text field
(question + conclusion) with cosine similarity via dot product on
L2-normalized vectors. No BM25, no hybrid — pure embedding baseline
to see where semantic search helps and where it falls short.
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

# Project root (two levels up from this file)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "extracted")
_INDEX_PATH = os.path.join(_PROJECT_ROOT, "indexes", "embeddings_text-embedding-3-small_qa_text.pkl")
_ENV_PATH = os.path.join(_PROJECT_ROOT, ".env")

_MODEL = "text-embedding-3-small"
_BATCH_SIZE = 512
_MAX_TOKENS = 8191  # text-embedding-3-small limit

# Tokenizer for truncation — cl100k_base is what text-embedding-3-small uses
_enc = tiktoken.get_encoding("cl100k_base")


def _truncate(text: str) -> str:
    """Truncate text to fit within the model's token limit."""
    tokens = _enc.encode(text)
    if len(tokens) <= _MAX_TOKENS:
        return text
    return _enc.decode(tokens[:_MAX_TOKENS])


def _get_text(opinion: dict) -> str:
    """Extract the best text field for indexing an opinion."""
    qa = opinion.get("embedding", {}).get("qa_text", "") or ""
    if len(qa) >= 20:
        return qa
    return opinion.get("content", {}).get("full_text", "") or ""


def _load_opinions() -> tuple[list[str], list[str]]:
    """Walk data/extracted/ and return (opinion_ids, raw_texts)."""
    opinion_ids = []
    raw_texts = []
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
            text = _get_text(opinion)
            opinion_ids.append(oid)
            raw_texts.append(text)
    return opinion_ids, raw_texts


def _embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    """Embed texts in batches, returning L2-normalized float32 array of shape (N, 1536)."""
    all_embeddings = []
    total = len(texts)
    for i in range(0, total, _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        # Guard against empty/whitespace strings and truncate to token limit
        batch = [_truncate(t) if t.strip() else " " for t in batch]
        print(
            f"  Embedding batch {i // _BATCH_SIZE + 1}/"
            f"{(total + _BATCH_SIZE - 1) // _BATCH_SIZE}"
            f" ({len(batch)} texts)...",
            file=sys.stderr,
        )
        response = client.embeddings.create(model=_MODEL, input=batch)
        batch_vecs = [item.embedding for item in response.data]
        all_embeddings.extend(batch_vecs)

    matrix = np.array(all_embeddings, dtype=np.float32)
    # L2-normalize so dot product = cosine similarity
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero
    matrix /= norms
    return matrix


class SemanticBaseline(SearchEngine):
    """Pure embedding search over qa_text using text-embedding-3-small."""

    def __init__(self):
        load_dotenv(_ENV_PATH, override=True)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found. Set it in .env or as an environment variable."
            )
        self._client = OpenAI(api_key=api_key)

        if os.path.exists(_INDEX_PATH):
            print(f"Loading embedding index from {_INDEX_PATH}...")
            with open(_INDEX_PATH, "rb") as f:
                data = pickle.load(f)
            self._opinion_ids = data["opinion_ids"]
            self._embeddings = data["embeddings"]
            print(f"Loaded index with {len(self._opinion_ids)} opinions.")
        else:
            print("Building embedding index from scratch...")
            opinion_ids, raw_texts = _load_opinions()
            print(f"Loaded {len(opinion_ids)} opinions. Embedding with {_MODEL}...")
            embeddings = _embed_texts(self._client, raw_texts)
            self._opinion_ids = opinion_ids
            self._embeddings = embeddings
            # Persist
            os.makedirs(os.path.dirname(_INDEX_PATH), exist_ok=True)
            with open(_INDEX_PATH, "wb") as f:
                pickle.dump({"opinion_ids": opinion_ids, "embeddings": embeddings}, f)
            print(f"Index saved to {_INDEX_PATH}")

    def search(self, query: str, top_k: int = 20) -> list[str]:
        response = self._client.embeddings.create(model=_MODEL, input=[query])
        query_vec = np.array(response.data[0].embedding, dtype=np.float32)
        # Normalize query vector
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec /= norm
        # Dot product against pre-normalized corpus = cosine similarity
        scores = self._embeddings @ query_vec
        top_indices = scores.argsort()[::-1][:top_k]
        return [self._opinion_ids[idx] for idx in top_indices]

    def name(self) -> str:
        return "SemanticBaseline"
