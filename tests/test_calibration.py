import sys
import unittest
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration_metrics import (
    expected_calibration_error,
    maximum_calibration_error,
    brier_score,
    compute_overall_metrics,
    compute_groupwise_metrics,
    calibration_gap,
)


class TestECE(unittest.TestCase):
    def test_perfect_calibration(self):
        """Perfect predictions should give ECE=0."""
        y = np.array([0, 0, 1, 1])
        p = np.array([0.0, 0.0, 1.0, 1.0])
        ece = expected_calibration_error(y, p, n_bins=2)
        self.assertAlmostEqual(ece, 0.0, places=5)

    def test_worst_calibration(self):
        """Completely wrong predictions should give high ECE."""
        y = np.array([0, 0, 1, 1])
        p = np.array([1.0, 1.0, 0.0, 0.0])
        ece = expected_calibration_error(y, p, n_bins=2)
        self.assertGreater(ece, 0.5)

    def test_empty_input(self):
        ece = expected_calibration_error([], [], n_bins=10)
        self.assertEqual(ece, 0.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            expected_calibration_error([0, 1], [0.5], n_bins=10)

    def test_ece_range(self):
        np.random.seed(42)
        y = np.random.randint(0, 2, 1000)
        p = np.random.rand(1000)
        ece = expected_calibration_error(y, p)
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)


class TestBrierScore(unittest.TestCase):
    def test_perfect(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([0.0, 1.0, 0.0, 1.0])
        self.assertAlmostEqual(brier_score(y, p), 0.0)

    def test_worst(self):
        y = np.array([0, 1])
        p = np.array([1.0, 0.0])
        self.assertAlmostEqual(brier_score(y, p), 1.0)


class TestGroupwise(unittest.TestCase):
    def test_correct_groups(self):
        y = np.array([0, 0, 1, 1, 0, 1])
        p = np.array([0.2, 0.3, 0.8, 0.7, 0.1, 0.9])
        groups = np.array(['A', 'A', 'A', 'B', 'B', 'B'])
        df = compute_groupwise_metrics(y, p, groups)
        self.assertEqual(len(df), 2)
        self.assertIn('A', df['group'].values)
        self.assertIn('B', df['group'].values)

    def test_calibration_gap(self):
        y = np.array([0, 0, 1, 1, 0, 1])
        p = np.array([0.2, 0.3, 0.8, 0.7, 0.1, 0.9])
        groups = np.array(['A', 'A', 'A', 'B', 'B', 'B'])
        df = compute_groupwise_metrics(y, p, groups)
        gap = calibration_gap(df)
        self.assertIn('max_ece_gap', gap)
        self.assertGreaterEqual(gap['max_ece_gap'], 0.0)


class TestOverallMetrics(unittest.TestCase):
    def test_keys_present(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([0.3, 0.7, 0.2, 0.8])
        metrics = compute_overall_metrics(y, p)
        for key in ['ece', 'mce', 'brier', 'accuracy']:
            self.assertIn(key, metrics)


if __name__ == '__main__':
    unittest.main()
