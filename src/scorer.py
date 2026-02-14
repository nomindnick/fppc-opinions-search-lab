"""
Scoring harness for the FPPC Opinions Search Evaluation Suite.

Computes IR metrics (MRR, nDCG, Precision, Recall) for search engines
implementing the SearchEngine ABC.

Usage:
    python src/scorer.py --search-module <dotted.path> --dataset eval/dataset.json [--output results.json]
"""

import argparse
import importlib
import inspect
import json
import math
import os
import sys
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Metric functions (pure, no side effects)
# ---------------------------------------------------------------------------

def compute_mrr(results: list[str], judgments: dict[str, int]) -> float:
    """Compute Mean Reciprocal Rank.

    Returns 1/rank of the first score=2 result, or 0.0 if none found.
    Only score=2 counts (not score=1).
    """
    for rank, doc_id in enumerate(results, start=1):
        if judgments.get(doc_id, 0) == 2:
            return 1.0 / rank
    return 0.0


def compute_ndcg(results: list[str], judgments: dict[str, int], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at k.

    Uses graded relevance with log2 discounting. The ideal ranking (IDCG) is
    computed from ALL judged documents sorted by score descending, not just
    the documents returned by the engine.

    Returns 0.0 if IDCG is 0 (no relevant documents in judgments).
    """
    def dcg(scores: list[int], n: int) -> float:
        total = 0.0
        for i in range(min(n, len(scores))):
            total += scores[i] / math.log2(i + 2)  # i+2 because rank starts at 1
        return total

    # DCG from actual results
    actual_scores = [judgments.get(doc_id, 0) for doc_id in results[:k]]
    dcg_val = dcg(actual_scores, k)

    # IDCG from all judged docs sorted descending
    ideal_scores = sorted(judgments.values(), reverse=True)
    idcg_val = dcg(ideal_scores, k)

    if idcg_val == 0.0:
        return 0.0
    return dcg_val / idcg_val


def compute_precision(results: list[str], judgments: dict[str, int], k: int) -> float:
    """Compute Precision at k.

    Counts results with score >= 1 in the top-k positions.
    Divides by k even if fewer results are returned.
    """
    relevant_count = 0
    for doc_id in results[:k]:
        if judgments.get(doc_id, 0) >= 1:
            relevant_count += 1
    return relevant_count / k


def compute_recall(results: list[str], judgments: dict[str, int], k: int) -> float:
    """Compute Recall at k.

    Fraction of all relevant opinions (score >= 1) found in top-k results.
    Returns 0.0 if there are no relevant documents in judgments.
    """
    total_relevant = sum(1 for score in judgments.values() if score >= 1)
    if total_relevant == 0:
        return 0.0
    found = 0
    for doc_id in results[:k]:
        if judgments.get(doc_id, 0) >= 1:
            found += 1
    return found / total_relevant


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def evaluate_query(query: dict, engine) -> dict:
    """Run a single query through the engine and compute all metrics.

    Args:
        query: A query dict from the dataset (must have 'text' and 'relevance_judgments').
        engine: A SearchEngine instance.

    Returns:
        A dict with the query metadata and all 7 computed metrics.
    """
    results = engine.search(query["text"], top_k=20)

    # Deduplicate results, preserving order
    seen = set()
    deduped = []
    for doc_id in results:
        if doc_id in seen:
            print(f"  Warning: duplicate result '{doc_id}' in query '{query['id']}' — keeping first occurrence")
        else:
            seen.add(doc_id)
            deduped.append(doc_id)
    results = deduped

    # Build judgments dict: opinion_id -> score
    judgments = {j["opinion_id"]: j["score"] for j in query["relevance_judgments"]}

    metrics = {
        "mrr": compute_mrr(results, judgments),
        "ndcg@5": compute_ndcg(results, judgments, 5),
        "ndcg@10": compute_ndcg(results, judgments, 10),
        "precision@5": compute_precision(results, judgments, 5),
        "precision@10": compute_precision(results, judgments, 10),
        "recall@10": compute_recall(results, judgments, 10),
        "recall@20": compute_recall(results, judgments, 20),
    }

    return {
        "query_id": query["id"],
        "query_text": query["text"],
        "query_type": query.get("type", "unknown"),
        "query_topic": query.get("topic", "unknown"),
        "num_results": len(results),
        "results": results,
        "metrics": metrics,
    }


def aggregate_metrics(query_results: list[dict]) -> dict:
    """Compute mean of each metric across a list of query results."""
    if not query_results:
        return {}
    metric_keys = list(query_results[0]["metrics"].keys())
    aggregated = {}
    for key in metric_keys:
        values = [qr["metrics"][key] for qr in query_results]
        aggregated[key] = sum(values) / len(values)
    return aggregated


def load_dataset(path: str) -> dict:
    """Load the eval dataset from a JSON file.

    Skips queries with empty relevance_judgments (with a warning).
    """
    with open(path, "r") as f:
        dataset = json.load(f)

    filtered_queries = []
    for query in dataset.get("queries", []):
        if not query.get("relevance_judgments"):
            print(f"Warning: skipping query '{query.get('id', '?')}' — empty relevance_judgments")
        else:
            filtered_queries.append(query)

    dataset["queries"] = filtered_queries
    return dataset


def load_engine(module_path: str):
    """Import a module and find/instantiate a SearchEngine subclass.

    Args:
        module_path: Dotted module path (e.g., 'src.baselines.random_baseline').

    Returns:
        An instantiated SearchEngine subclass.

    Raises:
        ImportError: If the module cannot be imported.
        RuntimeError: If no SearchEngine subclass is found in the module.
    """
    from src.interface import SearchEngine

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Could not import module '{module_path}': {e}") from e

    # Scan module for SearchEngine subclasses
    for _name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, SearchEngine) and obj is not SearchEngine:
            return obj()

    raise RuntimeError(
        f"No SearchEngine subclass found in module '{module_path}'. "
        f"Ensure the module defines a class that inherits from src.interface.SearchEngine."
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_scorecard(
    engine_name: str,
    overall: dict,
    by_type: dict[str, dict],
    by_topic: dict[str, dict],
    num_queries: int,
):
    """Print a formatted scorecard to stdout.

    80-char-wide table with metrics to 3 decimal places.
    """
    metric_keys = ["mrr", "ndcg@5", "ndcg@10", "precision@5", "precision@10", "recall@10", "recall@20"]
    short_labels = ["MRR", "nDCG@5", "nDCG@10", "P@5", "P@10", "R@10", "R@20"]

    sep = "=" * 80
    thin_sep = "-" * 80

    print(sep)
    print(f"  FPPC Opinions Search Evaluation — {engine_name}")
    print(f"  {num_queries} queries evaluated")
    print(sep)
    print()

    # Header row
    header = f"{'':>20s}"
    for label in short_labels:
        header += f"  {label:>7s}"
    print(header)
    print(thin_sep)

    # Overall row
    row = f"{'Overall':>20s}"
    for key in metric_keys:
        row += f"  {overall.get(key, 0.0):>7.3f}"
    print(row)
    print(thin_sep)

    # By query type
    if by_type:
        print(f"{'By Query Type':>20s}")
        for type_name in sorted(by_type.keys()):
            metrics = by_type[type_name]
            row = f"{'  ' + type_name:>20s}"
            for key in metric_keys:
                row += f"  {metrics.get(key, 0.0):>7.3f}"
            print(row)
        print(thin_sep)

    # By topic
    if by_topic:
        print(f"{'By Topic':>20s}")
        for topic_name in sorted(by_topic.keys()):
            metrics = by_topic[topic_name]
            label = topic_name[:18]
            row = f"{'  ' + label:>20s}"
            for key in metric_keys:
                row += f"  {metrics.get(key, 0.0):>7.3f}"
            print(row)
        print(thin_sep)

    print()


def write_results(
    path: str,
    engine_name: str,
    overall: dict,
    by_type: dict[str, dict],
    by_topic: dict[str, dict],
    per_query: list[dict],
):
    """Write detailed evaluation results to a JSON file."""
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "engine": engine_name,
        "overall": overall,
        "by_type": by_type,
        "by_topic": by_topic,
        "per_query": per_query,
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results written to {path}")


# ---------------------------------------------------------------------------
# Main / CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FPPC Opinions Search Evaluation Scoring Harness"
    )
    parser.add_argument(
        "--search-module",
        required=True,
        help="Dotted module path to a SearchEngine implementation (e.g., src.baselines.random_baseline)",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the eval dataset JSON file (e.g., eval/dataset.json)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write detailed JSON results",
    )
    args = parser.parse_args()

    # Add project root to sys.path so module imports work
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    dataset = load_dataset(args.dataset)
    queries = dataset["queries"]

    if not queries:
        print("No queries with relevance judgments found. Nothing to evaluate.")
        sys.exit(0)

    # Load engine
    print(f"Loading search engine from {args.search_module}...")
    engine = load_engine(args.search_module)
    print(f"Engine: {engine.name()}")
    print()

    # Evaluate each query
    print(f"Evaluating {len(queries)} queries...")
    per_query = []
    for i, query in enumerate(queries, start=1):
        print(f"  [{i}/{len(queries)}] {query['id']}: {query['text'][:60]}...")
        result = evaluate_query(query, engine)
        per_query.append(result)

    # Aggregate overall
    overall = aggregate_metrics(per_query)

    # Aggregate by query type
    by_type = {}
    for qr in per_query:
        qtype = qr["query_type"]
        if qtype not in by_type:
            by_type[qtype] = []
        by_type[qtype].append(qr)
    by_type = {k: aggregate_metrics(v) for k, v in by_type.items()}

    # Aggregate by topic
    by_topic = {}
    for qr in per_query:
        topic = qr["query_topic"]
        if topic not in by_topic:
            by_topic[topic] = []
        by_topic[topic].append(qr)
    by_topic = {k: aggregate_metrics(v) for k, v in by_topic.items()}

    # Print scorecard
    print()
    print_scorecard(engine.name(), overall, by_type, by_topic, len(queries))

    # Write results if requested
    if args.output:
        write_results(args.output, engine.name(), overall, by_type, by_topic, per_query)


if __name__ == "__main__":
    main()
