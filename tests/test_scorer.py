"""Unit tests for the scoring harness metric functions."""

import math
import unittest

from src.scorer import (
    aggregate_metrics,
    compute_mrr,
    compute_ndcg,
    compute_precision,
    compute_recall,
)


class TestComputeMRR(unittest.TestCase):
    """Tests for compute_mrr."""

    def test_score2_at_rank1(self):
        results = ["a", "b", "c"]
        judgments = {"a": 2, "b": 1, "c": 0}
        self.assertAlmostEqual(compute_mrr(results, judgments), 1.0, places=4)

    def test_score2_at_rank2(self):
        results = ["a", "b", "c"]
        judgments = {"a": 1, "b": 2, "c": 0}
        self.assertAlmostEqual(compute_mrr(results, judgments), 0.5, places=4)

    def test_score2_at_rank5(self):
        results = ["a", "b", "c", "d", "e"]
        judgments = {"e": 2}
        self.assertAlmostEqual(compute_mrr(results, judgments), 0.2, places=4)

    def test_no_score2(self):
        results = ["a", "b", "c"]
        judgments = {"a": 1, "b": 1, "c": 0}
        self.assertAlmostEqual(compute_mrr(results, judgments), 0.0, places=4)

    def test_only_score1(self):
        results = ["a", "b"]
        judgments = {"a": 1, "b": 1}
        self.assertAlmostEqual(compute_mrr(results, judgments), 0.0, places=4)

    def test_empty_results(self):
        results = []
        judgments = {"a": 2, "b": 1}
        self.assertAlmostEqual(compute_mrr(results, judgments), 0.0, places=4)


class TestComputeNDCG(unittest.TestCase):
    """Tests for compute_ndcg."""

    def test_perfect_ranking(self):
        # Results in perfect order: 2, 1, 0
        results = ["a", "b", "c"]
        judgments = {"a": 2, "b": 1, "c": 0}
        self.assertAlmostEqual(compute_ndcg(results, judgments, 3), 1.0, places=4)

    def test_inverse_ranking(self):
        # Results in worst order: 0, 1, 2
        results = ["c", "b", "a"]
        judgments = {"a": 2, "b": 1, "c": 0}
        ndcg = compute_ndcg(results, judgments, 3)
        self.assertLess(ndcg, 1.0)
        self.assertGreater(ndcg, 0.0)

    def test_graded_relevance_hand_computed(self):
        # Hand-computed example:
        # Results: [2, 1, 0, 1, 2] at k=5
        # DCG = 2/log2(2) + 1/log2(3) + 0/log2(4) + 1/log2(5) + 2/log2(6)
        #     = 2/1 + 1/1.585 + 0/2 + 1/2.322 + 2/2.585
        #     = 2.0 + 0.6309 + 0.0 + 0.4307 + 0.7737
        #     = 3.8353
        # Ideal (all judged sorted): [2, 2, 1, 1, 0]
        # IDCG = 2/log2(2) + 2/log2(3) + 1/log2(4) + 1/log2(5) + 0/log2(6)
        #      = 2/1 + 2/1.585 + 1/2 + 1/2.322 + 0/2.585
        #      = 2.0 + 1.2619 + 0.5 + 0.4307 + 0.0
        #      = 4.1926
        # nDCG = 3.8353 / 4.1926 = 0.9147
        results = ["a", "b", "c", "d", "e"]
        judgments = {"a": 2, "b": 1, "c": 0, "d": 1, "e": 2}
        expected = 3.8353 / 4.1926
        self.assertAlmostEqual(compute_ndcg(results, judgments, 5), expected, places=3)

    def test_no_relevant_docs(self):
        results = ["a", "b", "c"]
        judgments = {"a": 0, "b": 0, "c": 0}
        self.assertAlmostEqual(compute_ndcg(results, judgments, 3), 0.0, places=4)

    def test_empty_results(self):
        results = []
        judgments = {"a": 2, "b": 1}
        self.assertAlmostEqual(compute_ndcg(results, judgments, 5), 0.0, places=4)

    def test_fewer_results_than_k(self):
        # Only 2 results but k=5; should still compute correctly
        results = ["a", "b"]
        judgments = {"a": 2, "b": 1}
        ndcg = compute_ndcg(results, judgments, 5)
        self.assertGreater(ndcg, 0.0)
        self.assertLessEqual(ndcg, 1.0)

    def test_ideal_uses_all_judged_docs(self):
        # Only 1 result returned, but judgments have more docs
        # The IDCG should be based on all judged docs, not just returned ones
        results = ["a"]
        judgments = {"a": 1, "b": 2, "c": 2}
        # DCG = 1/log2(2) = 1.0
        # IDCG at k=3: ideal = [2, 2, 1]
        # IDCG = 2/log2(2) + 2/log2(3) + 1/log2(4) = 2.0 + 1.2619 + 0.5 = 3.7619
        # nDCG = 1.0 / 3.7619 = 0.2658
        expected = 1.0 / (2.0 / math.log2(2) + 2.0 / math.log2(3) + 1.0 / math.log2(4))
        self.assertAlmostEqual(compute_ndcg(results, judgments, 3), expected, places=4)


class TestComputePrecision(unittest.TestCase):
    """Tests for compute_precision."""

    def test_all_relevant(self):
        results = ["a", "b", "c", "d", "e"]
        judgments = {"a": 2, "b": 1, "c": 2, "d": 1, "e": 1}
        self.assertAlmostEqual(compute_precision(results, judgments, 5), 1.0, places=4)

    def test_none_relevant(self):
        results = ["a", "b", "c", "d", "e"]
        judgments = {"a": 0, "b": 0, "c": 0, "d": 0, "e": 0}
        self.assertAlmostEqual(compute_precision(results, judgments, 5), 0.0, places=4)

    def test_mixed_3_of_5(self):
        results = ["a", "b", "c", "d", "e"]
        judgments = {"a": 2, "b": 0, "c": 1, "d": 0, "e": 1}
        self.assertAlmostEqual(compute_precision(results, judgments, 5), 0.6, places=4)

    def test_fewer_results_than_k_penalized(self):
        # Only 3 results, all relevant, but k=5 -> 3/5 = 0.6
        results = ["a", "b", "c"]
        judgments = {"a": 2, "b": 1, "c": 1}
        self.assertAlmostEqual(compute_precision(results, judgments, 5), 0.6, places=4)

    def test_empty_results(self):
        results = []
        judgments = {"a": 2, "b": 1}
        self.assertAlmostEqual(compute_precision(results, judgments, 5), 0.0, places=4)


class TestComputeRecall(unittest.TestCase):
    """Tests for compute_recall."""

    def test_all_found(self):
        results = ["a", "b", "c"]
        judgments = {"a": 2, "b": 1, "c": 1}
        self.assertAlmostEqual(compute_recall(results, judgments, 5), 1.0, places=4)

    def test_none_found(self):
        results = ["x", "y", "z"]
        judgments = {"a": 2, "b": 1, "c": 1}
        self.assertAlmostEqual(compute_recall(results, judgments, 5), 0.0, places=4)

    def test_partial_2_of_4(self):
        results = ["a", "b", "x", "y"]
        judgments = {"a": 2, "b": 1, "c": 1, "d": 2}
        # 2 relevant found out of 4 total relevant
        self.assertAlmostEqual(compute_recall(results, judgments, 5), 0.5, places=4)

    def test_no_relevant_docs_in_judgments(self):
        results = ["a", "b"]
        judgments = {"a": 0, "b": 0}
        self.assertAlmostEqual(compute_recall(results, judgments, 5), 0.0, places=4)

    def test_empty_results(self):
        results = []
        judgments = {"a": 2, "b": 1}
        self.assertAlmostEqual(compute_recall(results, judgments, 5), 0.0, places=4)


class TestAggregateMetrics(unittest.TestCase):
    """Tests for aggregate_metrics."""

    def test_single_query(self):
        query_results = [
            {"metrics": {"mrr": 1.0, "ndcg@10": 0.8, "precision@5": 0.6}}
        ]
        agg = aggregate_metrics(query_results)
        self.assertAlmostEqual(agg["mrr"], 1.0, places=4)
        self.assertAlmostEqual(agg["ndcg@10"], 0.8, places=4)
        self.assertAlmostEqual(agg["precision@5"], 0.6, places=4)

    def test_multi_query_means(self):
        query_results = [
            {"metrics": {"mrr": 1.0, "ndcg@10": 0.8, "precision@5": 0.6}},
            {"metrics": {"mrr": 0.5, "ndcg@10": 0.4, "precision@5": 0.2}},
            {"metrics": {"mrr": 0.0, "ndcg@10": 0.6, "precision@5": 0.8}},
        ]
        agg = aggregate_metrics(query_results)
        # Mean of [1.0, 0.5, 0.0] = 0.5
        self.assertAlmostEqual(agg["mrr"], 0.5, places=4)
        # Mean of [0.8, 0.4, 0.6] = 0.6
        self.assertAlmostEqual(agg["ndcg@10"], 0.6, places=4)
        # Mean of [0.6, 0.2, 0.8] = 0.5333...
        self.assertAlmostEqual(agg["precision@5"], 0.5333, places=4)

    def test_empty_list(self):
        self.assertEqual(aggregate_metrics([]), {})


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases."""

    def test_unjudged_results_treated_as_score_0(self):
        # Results contain docs not in judgments -> score 0
        results = ["a", "unknown1", "unknown2", "b"]
        judgments = {"a": 2, "b": 1}
        # MRR: first score=2 is "a" at rank 1 -> 1.0
        self.assertAlmostEqual(compute_mrr(results, judgments), 1.0, places=4)
        # Precision@4: 2 relevant out of k=4 -> 0.5
        self.assertAlmostEqual(compute_precision(results, judgments, 4), 0.5, places=4)

    def test_duplicate_handling_in_metrics(self):
        # If somehow duplicate doc IDs end up in results, metrics should
        # still compute based on what's passed. Deduplication happens in
        # evaluate_query, but metric functions themselves are pure.
        results = ["a", "a", "b"]
        judgments = {"a": 2, "b": 1}
        # MRR: first score=2 at rank 1 -> 1.0
        self.assertAlmostEqual(compute_mrr(results, judgments), 1.0, places=4)
        # Precision@3: all 3 have score >= 1 -> 3/3 = 1.0
        self.assertAlmostEqual(compute_precision(results, judgments, 3), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
