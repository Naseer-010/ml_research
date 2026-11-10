import sys
import unittest
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.neural_network_model import NeuralNetworkModelWrapper, TabularMLP
from src.compression import (
    prune_neural_network,
    quantize_neural_network,
    compress_neural_network,
)
from src.explainability import compute_shap_values, _fallback_permutation_importance


class TestNeuralNetworkCompression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        torch.manual_seed(42)
        cls.X = np.random.randn(100, 10).astype(np.float32)
        cls.y = (cls.X[:, 0] > 0).astype(int)
        cls.wrapper = NeuralNetworkModelWrapper(seed=42)
        cls.wrapper.train(cls.X, cls.y)

    def test_prune_neural_network(self):
        pruned = prune_neural_network(self.wrapper.model, level=0.3)
        self.assertIsNotNone(pruned)
        dummy_x = torch.from_numpy(self.X[:5])
        with torch.no_grad():
            preds = pruned(dummy_x)
        self.assertEqual(preds.shape[0], 5)

    def test_quantize_neural_network_cpu(self):
        quant = quantize_neural_network(self.wrapper.model)
        self.assertIsNotNone(quant)
        dummy_x = torch.from_numpy(self.X[:5]).cpu()
        with torch.no_grad():
            preds = quant(dummy_x)
        self.assertEqual(preds.shape[0], 5)

    def test_compress_wrapper_prune_quantize(self):
        compressed_wrapper = self.wrapper.compress(level=0.3, mode="prune+quantize")
        probs = compressed_wrapper.predict_proba(self.X[:10])
        self.assertEqual(len(probs), 10)
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))

    def test_shap_computation_on_nn(self):
        feature_names = [f"feat_{i}" for i in range(10)]
        # Test on base wrapper
        res_base = compute_shap_values(
            self.wrapper, self.X[:10], feature_names, model_type="neural_network"
        )
        self.assertEqual(res_base.values.shape, (10, 10))

        # Test on compressed wrapper
        compressed_wrapper = self.wrapper.compress(level=0.3, mode="prune+quantize")
        res_comp = compute_shap_values(
            compressed_wrapper, self.X[:10], feature_names, model_type="neural_network"
        )
        self.assertEqual(res_comp.values.shape, (10, 10))

    def test_fallback_permutation_importance_on_pytorch(self):
        feature_names = [f"feat_{i}" for i in range(10)]
        res = _fallback_permutation_importance(
            self.wrapper.model, self.X[:20], feature_names
        )
        self.assertEqual(res.values.shape, (20, 10))
        self.assertFalse(np.isnan(res.values).any())


if __name__ == "__main__":
    unittest.main()

