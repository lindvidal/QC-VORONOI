import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from src.qc.spatial import ConfusionMatrix, compute_confusion_matrix


class TestConfusionMatrix(unittest.TestCase):

    def test_empty_matrix(self):
        cm = ConfusionMatrix()
        self.assertEqual(cm.precision, 0.0)
        self.assertEqual(cm.recall, 0.0)
        self.assertEqual(cm.accuracy, 0.0)
        self.assertEqual(cm.f1, 0.0)

    def test_perfect_classification(self):
        cm = ConfusionMatrix(tp=10, fp=0, tn=10, fn=0)
        self.assertEqual(cm.precision, 1.0)
        self.assertEqual(cm.recall, 1.0)
        self.assertEqual(cm.accuracy, 1.0)
        self.assertEqual(cm.f1, 1.0)

    def test_all_wrong(self):
        cm = ConfusionMatrix(tp=0, fp=10, tn=0, fn=10)
        self.assertEqual(cm.precision, 0.0)
        self.assertEqual(cm.recall, 0.0)
        self.assertEqual(cm.accuracy, 0.0)
        self.assertEqual(cm.f1, 0.0)

    def test_imbalanced(self):
        cm = ConfusionMatrix(tp=8, fp=2, tn=5, fn=3)
        self.assertAlmostEqual(cm.precision, 0.8)
        self.assertAlmostEqual(cm.recall, 8 / 11)
        self.assertAlmostEqual(cm.accuracy, 13 / 18)
        expected_f1 = 2 * 0.8 * (8 / 11) / (0.8 + 8 / 11)
        self.assertAlmostEqual(cm.f1, expected_f1)

    def test_compute_confusion_matrix_basic(self):
        pred = ["HQ", "HQ", "LQ", "LQ"]
        true = ["HQ", "LQ", "HQ", "LQ"]
        cm = compute_confusion_matrix(pred, true)
        self.assertEqual(cm.tp, 1)
        self.assertEqual(cm.fp, 1)
        self.assertEqual(cm.fn, 1)
        self.assertEqual(cm.tn, 1)

    def test_compute_confusion_matrix_empty(self):
        cm = compute_confusion_matrix([], [])
        self.assertEqual(cm.tp, 0)
        self.assertEqual(cm.fp, 0)
        self.assertEqual(cm.tn, 0)
        self.assertEqual(cm.fn, 0)

    def test_compute_confusion_matrix_custom_class(self):
        pred = ["A", "A", "B", "B"]
        true = ["A", "B", "A", "B"]
        cm = compute_confusion_matrix(pred, true, positive_class="A")
        self.assertEqual(cm.tp, 1)
        self.assertEqual(cm.fp, 1)
        self.assertEqual(cm.fn, 1)
        self.assertEqual(cm.tn, 1)


if __name__ == "__main__":
    unittest.main()
