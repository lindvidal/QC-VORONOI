import unittest
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from src.qc.validation import (
    bootstrap_ci,
    bootstrap_all_metrics,
    mcnemar_test,
    permutation_test,
    _f1_wrapper,
    _precision_wrapper,
    _recall_wrapper,
    _accuracy_wrapper,
)
from src.qc.spatial import compute_confusion_matrix


class TestBootstrapCI(unittest.TestCase):

    def test_perfect_prediction(self):
        y_true = np.array(["HQ"] * 20 + ["LQ"] * 20)
        y_pred = np.array(["HQ"] * 20 + ["LQ"] * 20)
        result = bootstrap_ci(y_true, y_pred, _f1_wrapper, n_iter=100, ci=0.95)
        self.assertAlmostEqual(result["mean"], 1.0, places=3)
        self.assertEqual(result["ci"], 0.95)

    def test_all_wrong_prediction(self):
        y_true = np.array(["HQ"] * 20 + ["LQ"] * 20)
        y_pred = np.array(["LQ"] * 20 + ["HQ"] * 20)
        result = bootstrap_ci(y_true, y_pred, _f1_wrapper, n_iter=100, ci=0.95)
        self.assertEqual(result["mean"], 0.0)

    def test_small_sample(self):
        y_true = np.array(["HQ", "LQ"])
        y_pred = np.array(["HQ", "LQ"])
        result = bootstrap_ci(y_true, y_pred, _f1_wrapper, n_iter=50, ci=0.95)
        self.assertIn("lower", result)
        self.assertIn("upper", result)
        self.assertIn("mean", result)
        self.assertIn("std", result)


class TestBootstrapAllMetrics(unittest.TestCase):

    def test_all_metrics_returned(self):
        y_true = np.array(["HQ"] * 20 + ["LQ"] * 20)
        y_pred = np.array(["HQ"] * 20 + ["LQ"] * 20)
        result = bootstrap_all_metrics(y_true, y_pred, n_iter=100)
        for metric in ["f1", "precision", "recall", "accuracy"]:
            self.assertIn(metric, result)
            self.assertIn("mean", result[metric])
            self.assertIn("lower", result[metric])
            self.assertIn("upper", result[metric])


class TestMcNemarTest(unittest.TestCase):

    def test_no_difference(self):
        y_true = np.array(["HQ"] * 10 + ["LQ"] * 10)
        y_pred_a = np.array(["HQ"] * 10 + ["LQ"] * 10)
        y_pred_b = np.array(["HQ"] * 10 + ["LQ"] * 10)
        result = mcnemar_test(y_true, y_pred_a, y_pred_b)
        self.assertFalse(result["significant"])
        self.assertGreater(result["p_value"], 0.05)

    def test_significant_difference(self):
        y_true = np.array(["HQ"] * 10 + ["LQ"] * 10)
        y_pred_a = np.array(["HQ"] * 10 + ["LQ"] * 10)
        y_pred_b = np.array(["LQ"] * 10 + ["HQ"] * 10)
        result = mcnemar_test(y_true, y_pred_a, y_pred_b)
        self.assertTrue(result["significant"])
        self.assertLess(result["p_value"], 0.05)

    def test_no_discordant_pairs(self):
        y_true = np.array(["HQ", "LQ"])
        y_pred_a = np.array(["HQ", "LQ"])
        y_pred_b = np.array(["HQ", "LQ"])
        result = mcnemar_test(y_true, y_pred_a, y_pred_b)
        self.assertEqual(result["n_discordant"], 0)
        self.assertEqual(result["p_value"], 1.0)


class TestPermutationTest(unittest.TestCase):

    def test_perfect_model(self):
        np.random.seed(42)
        y_true = np.array(["HQ"] * 10 + ["LQ"] * 10)
        y_pred = np.array(["HQ"] * 10 + ["LQ"] * 10)
        result = permutation_test(y_true, y_pred, n_perm=100)
        self.assertLess(result["p_value"], 0.05)
        self.assertTrue(result["significant"])

    def test_random_model(self):
        np.random.seed(42)
        y_true = np.array(["HQ"] * 10 + ["LQ"] * 10)
        rng = np.random.default_rng(42)
        y_pred = np.array(["HQ"] * 10 + ["LQ"] * 10)
        rng.shuffle(y_pred)
        result = permutation_test(y_true, y_pred, n_perm=100)
        self.assertIn("real_score", result)
        self.assertIn("p_value", result)


class TestConfusionMatrixFunctions(unittest.TestCase):

    def test_compute_confusion_matrix(self):
        pred = ["HQ", "HQ", "LQ", "LQ"]
        true = ["HQ", "LQ", "HQ", "LQ"]
        cm = compute_confusion_matrix(pred, true)
        self.assertEqual(cm.tp, 1)
        self.assertEqual(cm.fp, 1)
        self.assertEqual(cm.fn, 1)
        self.assertEqual(cm.tn, 1)


class TestMetricWrappers(unittest.TestCase):

    def setUp(self):
        self.y_true = np.array(["HQ", "HQ", "LQ", "LQ"])
        self.y_pred = np.array(["HQ", "LQ", "HQ", "LQ"])

    def test_f1_wrapper(self):
        score = _f1_wrapper(self.y_true, self.y_pred)
        self.assertGreater(score, 0)

    def test_precision_wrapper(self):
        score = _precision_wrapper(self.y_true, self.y_pred)
        self.assertAlmostEqual(score, 0.5)

    def test_recall_wrapper(self):
        score = _recall_wrapper(self.y_true, self.y_pred)
        self.assertAlmostEqual(score, 0.5)

    def test_accuracy_wrapper(self):
        score = _accuracy_wrapper(self.y_true, self.y_pred)
        self.assertAlmostEqual(score, 0.5)


if __name__ == "__main__":
    unittest.main()
