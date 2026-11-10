import sys
import unittest
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from src.compression import prune_random_forest, prune_gradient_boosting


class TestRFPruning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        X = np.random.rand(200, 5)
        y = (X[:, 0] > 0.5).astype(int)
        cls.rf = RandomForestClassifier(n_estimators=100, random_state=42)
        cls.rf.fit(X, y)
        cls.X = X

    def test_no_compression(self):
        pruned = prune_random_forest(self.rf, level=0.0)
        self.assertEqual(len(pruned.estimators_), 100)

    def test_50_percent(self):
        pruned = prune_random_forest(self.rf, level=0.5)
        self.assertEqual(len(pruned.estimators_), 50)

    def test_90_percent(self):
        pruned = prune_random_forest(self.rf, level=0.9)
        self.assertEqual(len(pruned.estimators_), 10)

    def test_keeps_at_least_one(self):
        pruned = prune_random_forest(self.rf, level=0.99)
        self.assertGreaterEqual(len(pruned.estimators_), 1)

    def test_doesnt_mutate_original(self):
        _ = prune_random_forest(self.rf, level=0.5)
        self.assertEqual(len(self.rf.estimators_), 100)

    def test_predictions_valid(self):
        pruned = prune_random_forest(self.rf, level=0.5)
        probs = pruned.predict_proba(self.X)
        self.assertEqual(probs.shape[0], len(self.X))
        self.assertTrue(np.all(probs >= 0) and np.all(probs <= 1))


class TestGBPruning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        X = np.random.rand(200, 5)
        y = (X[:, 0] > 0.5).astype(int)
        cls.gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        cls.gb.fit(X, y)
        cls.X = X

    def test_no_compression(self):
        pruned = prune_gradient_boosting(self.gb, level=0.0)
        self.assertEqual(len(pruned.estimators_), 100)

    def test_50_percent(self):
        pruned = prune_gradient_boosting(self.gb, level=0.5)
        self.assertEqual(len(pruned.estimators_), 50)

    def test_doesnt_mutate_original(self):
        _ = prune_gradient_boosting(self.gb, level=0.5)
        self.assertEqual(len(self.gb.estimators_), 100)

    def test_predictions_valid(self):
        pruned = prune_gradient_boosting(self.gb, level=0.5)
        probs = pruned.predict_proba(self.X)
        self.assertEqual(probs.shape[0], len(self.X))
        self.assertTrue(np.all(probs >= 0) and np.all(probs <= 1))


if __name__ == '__main__':
    unittest.main()
