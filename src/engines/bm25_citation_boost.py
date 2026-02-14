"""
BM25 + Citation Boost search engine for FPPC opinions.

Builds on BM25 full_text (experiment 001b) by adding IDF-weighted additive
boosts for opinions whose structured citation fields match statutes/regulations
extracted from the query. Also applies a small topic boost when the query's
topic can be confidently inferred from citations and keywords.

The key insight: BM25 treats statute numbers like any other token, so common
statutes (e.g. "87103" in ~3,500 opinions) are non-discriminative. The
structured citations.government_code and citations.regulations fields precisely
record which statutes each opinion *analyzes*, making them much more useful for
re-ranking — but only when weighted by IDF so rare citations count more.
"""

import json
import math
import os
import pickle
import re

import numpy as np

from src.interface import SearchEngine
from src.engines.bm25_full_text import tokenize

# Project root (two levels up from this file)
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "extracted")
_BM25_INDEX = os.path.join(_PROJECT_ROOT, "indexes", "BM25FullText_index.pkl")
_CITATION_INDEX = os.path.join(
    _PROJECT_ROOT, "indexes", "BM25CitationBoost_citation_index.pkl"
)

# Boost parameters
_CITE_BOOST = 0.30  # base multiplier for citation boost (scaled by IDF)
_TOPIC_BOOST = 0.03  # topic match boost (fraction of max BM25)
_N_TOTAL = 14096  # total opinions (for IDF calculation)

# ---------------------------------------------------------------------------
# Query citation parser
# ---------------------------------------------------------------------------

# Prefixed statute: "Section 87103(a)", "Government Code 1090", "Gov. Code 87100"
_RE_PREFIXED_STATUTE = re.compile(
    r"(?:Section|Gov(?:ernment)?\.?\s*Code)\s+"
    r"(\d{3,5})(\([a-zA-Z0-9]\))?",
    re.IGNORECASE,
)

# Prefixed regulation: "Regulation 18702.2", "Reg. 18703", "FPPC Reg 18700"
_RE_PREFIXED_REG = re.compile(
    r"(?:Reg(?:ulation)?\.?)\s+"
    r"(\d{4,5}(?:\.\d+)?)",
    re.IGNORECASE,
)

# Bare statute — known FPPC ranges only (to avoid false positives)
# 81000-91014 (Political Reform Act), 1090-1097 (Section 1090)
_RE_BARE_STATUTE = re.compile(r"\b(8[1-9]\d{3}|90\d{3}|91014|109[0-7])(?:\(([a-zA-Z0-9])\))?\b")

# Bare regulation — 18000-18999 range (Title 2, Division 6 regulations)
_RE_BARE_REG = re.compile(r"\b(18\d{3}(?:\.\d+)?)\b")


def parse_query_citations(query: str) -> dict:
    """Extract statute and regulation references from a query string.

    Returns:
        {
            "gov_code": [{"raw": "87103(a)", "base": "87103", "subsection": "(a)"}, ...],
            "regulations": [{"raw": "18702.2", "base": "18702", "subsection": ".2"}, ...]
        }
    """
    gov_code = []
    regulations = []
    seen_gc = set()
    seen_reg = set()

    # Prefixed statutes
    for m in _RE_PREFIXED_STATUTE.finditer(query):
        base = m.group(1)
        sub = m.group(2) or ""
        raw = base + sub
        if raw not in seen_gc:
            seen_gc.add(raw)
            gov_code.append({"raw": raw, "base": base, "subsection": sub})

    # Prefixed regulations
    for m in _RE_PREFIXED_REG.finditer(query):
        full = m.group(1)
        base = full.split(".")[0]
        sub = "." + full.split(".", 1)[1] if "." in full else ""
        if full not in seen_reg:
            seen_reg.add(full)
            regulations.append({"raw": full, "base": base, "subsection": sub})

    # Bare statutes (only if not already captured by prefixed)
    for m in _RE_BARE_STATUTE.finditer(query):
        base = m.group(1)
        sub_letter = m.group(2) or ""
        sub = f"({sub_letter})" if sub_letter else ""
        raw = base + sub
        if raw not in seen_gc:
            seen_gc.add(raw)
            gov_code.append({"raw": raw, "base": base, "subsection": sub})

    # Bare regulations (only if not already captured by prefixed)
    for m in _RE_BARE_REG.finditer(query):
        full = m.group(1)
        base = full.split(".")[0]
        sub = "." + full.split(".", 1)[1] if "." in full else ""
        if full not in seen_reg:
            seen_reg.add(full)
            regulations.append({"raw": full, "base": base, "subsection": sub})

    return {"gov_code": gov_code, "regulations": regulations}


# ---------------------------------------------------------------------------
# Topic inference
# ---------------------------------------------------------------------------

# Statute ranges → topic
_STATUTE_TOPIC = {
    "conflicts_of_interest": {
        "87100", "87101", "87102", "87103", "87104", "87105",
        "87200", "87201", "87202", "87203", "87206", "87207",
        "87300", "87301", "87302", "87302.3", "87302.6",
        "87400", "87450",
        "1090", "1091", "1092", "1093", "1094", "1095", "1096", "1097",
    },
    "campaign_finance": {
        "82015", "84200", "84201", "84202", "84203", "84204",
        "84211", "84300", "84301", "84302",
        "85100", "85101", "85200", "85201", "85300", "85301",
        "85302", "85303", "85304", "85305", "85306",
        "85500", "85601", "85700", "85800",
    },
    "gifts_honoraria": {
        "89501", "89502", "89503", "89506",
        "86201", "86202", "86203", "86204", "86205",
    },
    "lobbying": {
        "86100", "86101", "86102", "86103", "86104", "86105",
        "86110", "86112", "86113", "86114", "86115", "86116",
    },
}

# Keyword stems → topic
_KEYWORD_TOPIC = {
    "conflicts_of_interest": [
        "disqualif", "recus", "conflict", "1090", "self-deal",
        "financial interest", "abstain",
    ],
    "campaign_finance": [
        "contribution", "campaign", "expenditure", "donor",
        "committee", "election",
    ],
    "gifts_honoraria": [
        "gift", "honorari", "travel payment", "behest",
    ],
    "lobbying": [
        "lobbying", "lobbyist", "lobbied",
    ],
}


def infer_topic(query: str, parsed_citations: dict) -> str | None:
    """Infer topic from parsed citations and query keywords.

    Returns a topic string only if confidence >= 2 signal points.
    """
    scores = {}
    query_lower = query.lower()

    # Score from statutes
    for cite in parsed_citations["gov_code"]:
        base = cite["base"]
        for topic, statutes in _STATUTE_TOPIC.items():
            if base in statutes:
                scores[topic] = scores.get(topic, 0) + 1

    # Score from regulation numbers (18700-18707 → conflicts)
    for cite in parsed_citations["regulations"]:
        base = cite["base"]
        base_int = int(base) if base.isdigit() else 0
        if 18700 <= base_int <= 18707:
            scores["conflicts_of_interest"] = scores.get("conflicts_of_interest", 0) + 1
        elif 18215 <= base_int <= 18225:
            scores["campaign_finance"] = scores.get("campaign_finance", 0) + 1
        elif 18730 <= base_int <= 18735:
            scores["gifts_honoraria"] = scores.get("gifts_honoraria", 0) + 1
        elif 18610 <= base_int <= 18618:
            scores["lobbying"] = scores.get("lobbying", 0) + 1

    # Score from keywords
    for topic, keywords in _KEYWORD_TOPIC.items():
        for kw in keywords:
            if kw in query_lower:
                scores[topic] = scores.get(topic, 0) + 1

    if not scores:
        return None

    best_topic = max(scores, key=scores.get)
    if scores[best_topic] >= 2:
        return best_topic
    return None


# ---------------------------------------------------------------------------
# Citation index builder
# ---------------------------------------------------------------------------

def _build_citation_index() -> dict:
    """Walk data/extracted/ and build inverted maps from citations to opinion IDs."""
    gc_exact = {}   # "87103(a)" → {set of opinion IDs}
    gc_base = {}    # "87103" → {set of opinion IDs}
    reg_exact = {}  # "18702.2" → {set of opinion IDs}
    topic_map = {}  # "conflicts_of_interest" → {set of opinion IDs}

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

            # Government code citations
            citations = opinion.get("citations", {})
            for gc in citations.get("government_code", []):
                gc_str = str(gc)
                gc_exact.setdefault(gc_str, set()).add(oid)
                # Extract base number (strip subsections like "(a)")
                base_match = re.match(r"(\d+)", gc_str)
                if base_match:
                    gc_base.setdefault(base_match.group(1), set()).add(oid)

            # Regulation citations
            for reg in citations.get("regulations", []):
                reg_str = str(reg)
                reg_exact.setdefault(reg_str, set()).add(oid)

            # Topic classification
            topic = opinion.get("classification", {}).get("topic_primary")
            if topic and topic not in ("None", "other"):
                topic_map.setdefault(topic, set()).add(oid)

    return {
        "gc_exact": gc_exact,
        "gc_base": gc_base,
        "reg_exact": reg_exact,
        "topic": topic_map,
    }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BM25CitationBoost(SearchEngine):
    """BM25 full-text with additive citation and topic boosts."""

    def __init__(self):
        # Load BM25 index (reuse from bm25_full_text)
        print(f"Loading BM25 index from {_BM25_INDEX}...")
        with open(_BM25_INDEX, "rb") as f:
            bm25_data = pickle.load(f)
        self._opinion_ids = bm25_data["opinion_ids"]
        self._bm25 = bm25_data["bm25"]
        self._id_to_idx = {oid: i for i, oid in enumerate(self._opinion_ids)}
        print(f"  BM25: {len(self._opinion_ids)} opinions")

        # Load or build citation index
        if os.path.exists(_CITATION_INDEX):
            print(f"Loading citation index from {_CITATION_INDEX}...")
            with open(_CITATION_INDEX, "rb") as f:
                self._cite_index = pickle.load(f)
        else:
            print("Building citation index from scratch...")
            self._cite_index = _build_citation_index()
            os.makedirs(os.path.dirname(_CITATION_INDEX), exist_ok=True)
            with open(_CITATION_INDEX, "wb") as f:
                pickle.dump(self._cite_index, f)
            print(f"  Citation index saved to {_CITATION_INDEX}")

        gc_exact = self._cite_index["gc_exact"]
        gc_base = self._cite_index["gc_base"]
        reg_exact = self._cite_index["reg_exact"]
        topic_map = self._cite_index["topic"]
        print(f"  gc_exact entries: {len(gc_exact)}, "
              f"gc_base entries: {len(gc_base)}, "
              f"reg_exact entries: {len(reg_exact)}, "
              f"topics: {len(topic_map)}")

    def search(self, query: str, top_k: int = 20) -> list[str]:
        # BM25 scores for all documents
        tokens = tokenize(query)
        if not tokens:
            return []
        bm25_scores = self._bm25.get_scores(tokens)
        max_bm25 = bm25_scores.max()
        if max_bm25 <= 0:
            return []

        # Parse citations from query
        parsed = parse_query_citations(query)
        has_citations = bool(parsed["gov_code"] or parsed["regulations"])

        # Citation boost: IDF-weighted, summed across matching citations.
        # Each citation contributes: match_score * idf(citation)
        # where idf = log(N / df) and df = number of opinions citing it.
        # Exact subsection match → match_score = 1.0
        # Base-only match → match_score = 0.2
        cite_boost = np.zeros(len(self._opinion_ids), dtype=np.float32)

        if has_citations:
            gc_exact = self._cite_index["gc_exact"]
            gc_base = self._cite_index["gc_base"]
            reg_exact = self._cite_index["reg_exact"]

            for cite in parsed["gov_code"]:
                raw = cite["raw"]
                base = cite["base"]
                has_subsection = bool(cite["subsection"])

                # Exact match: idf-weighted 1.0
                exact_ids = gc_exact.get(raw, set())
                if exact_ids:
                    idf = math.log(_N_TOTAL / len(exact_ids))
                    for oid in exact_ids:
                        idx = self._id_to_idx.get(oid)
                        if idx is not None:
                            cite_boost[idx] += 1.0 * idf

                # Base match: idf-weighted 0.2 (only opinions NOT exact-matched)
                base_ids = gc_base.get(base, set())
                if base_ids:
                    base_idf = math.log(_N_TOTAL / len(base_ids))
                    for oid in base_ids:
                        if oid not in exact_ids:
                            idx = self._id_to_idx.get(oid)
                            if idx is not None:
                                cite_boost[idx] += 0.2 * base_idf

                # If query has bare base (no subsection), give exact match
                # credit to opinions citing that exact base number
                if not has_subsection and base not in (raw,):
                    base_exact_ids = gc_exact.get(base, set())
                    if base_exact_ids:
                        base_exact_idf = math.log(_N_TOTAL / len(base_exact_ids))
                        for oid in base_exact_ids:
                            if oid not in exact_ids:
                                idx = self._id_to_idx.get(oid)
                                if idx is not None:
                                    cite_boost[idx] += 1.0 * base_exact_idf

            for cite in parsed["regulations"]:
                raw = cite["raw"]
                exact_ids = reg_exact.get(raw, set())
                if exact_ids:
                    idf = math.log(_N_TOTAL / len(exact_ids))
                    for oid in exact_ids:
                        idx = self._id_to_idx.get(oid)
                        if idx is not None:
                            cite_boost[idx] += 1.0 * idf

                # For regulations with subsection, base-match fallback
                if cite["subsection"]:
                    base = cite["base"]
                    base_ids = reg_exact.get(base, set())
                    if base_ids:
                        base_idf = math.log(_N_TOTAL / len(base_ids))
                        for oid in base_ids:
                            if oid not in exact_ids:
                                idx = self._id_to_idx.get(oid)
                                if idx is not None:
                                    cite_boost[idx] += 0.2 * base_idf

        # Topic boost
        topic_boost = np.zeros(len(self._opinion_ids), dtype=np.float32)
        topic = infer_topic(query, parsed)
        if topic:
            topic_ids = self._cite_index["topic"].get(topic, set())
            for oid in topic_ids:
                idx = self._id_to_idx.get(oid)
                if idx is not None:
                    topic_boost[idx] = 1.0

        # Combine: BM25 + adaptive citation boost + topic boost
        combined = (
            bm25_scores
            + max_bm25 * _CITE_BOOST * cite_boost
            + max_bm25 * _TOPIC_BOOST * topic_boost
        )

        # Top-k by descending combined score
        top_indices = combined.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            if combined[idx] > 0:
                results.append(self._opinion_ids[idx])
        return results

    def name(self) -> str:
        return "BM25CitationBoost"
