"""
BM25 Full-Text search engine for FPPC opinions.

Uses BM25Okapi over the content.full_text field instead of embedding.qa_text.
Same tokenization pipeline and BM25 parameters as bm25_baseline.py — the only
difference is which text field is indexed. This tests whether the fuller
analysis text (which contains statute numbers and regulatory citations)
improves retrieval for keyword/statute queries.
"""

import json
import os
import pickle
import re

from rank_bm25 import BM25Okapi

from src.interface import SearchEngine

# Project root (two levels up from this file)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "extracted")
_INDEX_PATH = os.path.join(_PROJECT_ROOT, "indexes", "BM25FullText_index.pkl")

# Stopwords — standard English set minus "not" and "no" which carry legal meaning
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

# Regex to merge parenthetical statute subsections: 87103(a) -> 87103a
_PAREN_SUB = re.compile(r"(\d+)\(([a-zA-Z0-9])\)")
# Regex to replace non-alphanumeric (except hyphens) with space
_NON_ALNUM = re.compile(r"[^a-z0-9\-]+")


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 indexing/querying."""
    text = text.lower()
    text = _PAREN_SUB.sub(r"\1\2", text)
    text = _NON_ALNUM.sub(" ", text)
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS]


def _get_text(opinion: dict) -> str:
    """Extract full_text field for indexing an opinion."""
    return opinion.get("content", {}).get("full_text", "") or ""


def _load_opinions() -> tuple[list[str], list[list[str]]]:
    """Walk data/extracted/ and return (opinion_ids, tokenized_docs)."""
    opinion_ids = []
    tokenized_docs = []
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
            tokenized_docs.append(tokenize(text))
    return opinion_ids, tokenized_docs


class BM25FullText(SearchEngine):
    """BM25Okapi over content.full_text (complete opinion text)."""

    def __init__(self):
        if os.path.exists(_INDEX_PATH):
            print(f"Loading BM25 index from {_INDEX_PATH}...")
            with open(_INDEX_PATH, "rb") as f:
                data = pickle.load(f)
            self._opinion_ids = data["opinion_ids"]
            self._bm25 = data["bm25"]
            print(f"Loaded index with {len(self._opinion_ids)} opinions.")
        else:
            print("Building BM25 index from scratch...")
            opinion_ids, tokenized_docs = _load_opinions()
            print(f"Tokenized {len(opinion_ids)} opinions. Fitting BM25...")
            bm25 = BM25Okapi(tokenized_docs, k1=1.5, b=0.75)
            self._opinion_ids = opinion_ids
            self._bm25 = bm25
            # Persist
            os.makedirs(os.path.dirname(_INDEX_PATH), exist_ok=True)
            with open(_INDEX_PATH, "wb") as f:
                pickle.dump({"opinion_ids": opinion_ids, "bm25": bm25}, f)
            print(f"Index saved to {_INDEX_PATH}")

    def search(self, query: str, top_k: int = 20) -> list[str]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        # Get top_k indices by descending score, filtering score > 0
        top_indices = scores.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(self._opinion_ids[idx])
        return results

    def name(self) -> str:
        return "BM25FullText"
