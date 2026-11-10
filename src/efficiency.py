import io
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config

logger = logging.getLogger(__name__)


def count_parameters(model: Any, model_type: str) -> Dict[str, int]:
    """Count total and non-zero parameters for a model.

    For sklearn tree models: count nodes across all trees.
    For sklearn GB: count nodes across all stages.
    For PyTorch: count weight parameters.

    Returns dict with 'total_params' and 'nonzero_params'.
    """
    m_type = str(model_type).strip().lower()

    # PyTorch NN: count weight parameters, handle quantized models
    if m_type in ("nn", "neuralnetwork", "pytorch", "torch"):
        total_params = 0
        nonzero_params = 0
        try:
            import torch
            if isinstance(model, torch.nn.Module):
                for p in model.parameters():
                    total_params += p.numel()
                    nonzero_params += int((p != 0).sum().item())

                # If total_params == 0 (e.g., dynamically quantized PyTorch model)
                if total_params == 0:
                    for m in model.modules():
                        if hasattr(m, "_weight_bias") and not type(m).__name__.endswith("PackedParams"):
                            w, b = m._weight_bias()
                            total_params += w.numel()
                            try:
                                nonzero_params += int((w.dequantize() != 0).sum().item())
                            except Exception:
                                nonzero_params += int((w != 0).sum().item())
                            if b is not None:
                                total_params += b.numel()
                                nonzero_params += int((b != 0).sum().item())
        except Exception as exc:
            logger.warning("Error counting PyTorch parameters: %s", exc)

        return {"total_params": int(total_params), "nonzero_params": int(nonzero_params)}

    # Sklearn Gradient Boosting: iterate model.estimators_ (2D array), sum tree_.node_count
    elif m_type in ("gb", "gradientboosting", "ensemble"):
        total_nodes = 0
        if hasattr(model, "estimators_"):
            for stage in model.estimators_:
                if hasattr(stage, "__iter__"):
                    for tree in stage:
                        if hasattr(tree, "tree_"):
                            total_nodes += int(tree.tree_.node_count)
                        elif hasattr(tree, "node_count"):
                            total_nodes += int(tree.node_count)
                else:
                    if hasattr(stage, "tree_"):
                        total_nodes += int(stage.tree_.node_count)
                    elif hasattr(stage, "node_count"):
                        total_nodes += int(stage.node_count)
        return {"total_params": int(total_nodes), "nonzero_params": int(total_nodes)}

    # Sklearn Random Forest: iterate model.estimators_, sum tree_.node_count
    elif m_type in ("rf", "randomforest", "tree"):
        total_nodes = 0
        if hasattr(model, "estimators_"):
            for tree in model.estimators_:
                if hasattr(tree, "tree_"):
                    total_nodes += int(tree.tree_.node_count)
                elif hasattr(tree, "node_count"):
                    total_nodes += int(tree.node_count)
        return {"total_params": int(total_nodes), "nonzero_params": int(total_nodes)}

    # Fallback heuristic based on model instance attributes
    else:
        try:
            import torch
            if isinstance(model, torch.nn.Module):
                total = sum(p.numel() for p in model.parameters())
                nonzero = sum((p != 0).sum().item() for p in model.parameters())
                if total == 0:
                    for m in model.modules():
                        if hasattr(m, "_weight_bias") and not type(m).__name__.endswith("PackedParams"):
                            w, b = m._weight_bias()
                            total += w.numel()
                            try:
                                nonzero += int((w.dequantize() != 0).sum().item())
                            except Exception:
                                nonzero += int((w != 0).sum().item())
                            if b is not None:
                                total += b.numel()
                                nonzero += int((b != 0).sum().item())
                return {"total_params": int(total), "nonzero_params": int(nonzero)}
        except ImportError:
            pass

        if hasattr(model, "estimators_"):
            total_nodes = 0
            for item in np.ravel(model.estimators_):
                if hasattr(item, "tree_"):
                    total_nodes += int(item.tree_.node_count)
            return {"total_params": int(total_nodes), "nonzero_params": int(total_nodes)}

        return {"total_params": 0, "nonzero_params": 0}


def measure_model_size_bytes(model: Any, model_type: str) -> int:
    """Estimate model size in bytes using pickle serialization."""
    buffer = io.BytesIO()
    try:
        pickle.dump(model, buffer)
        size = buffer.tell()
    except Exception as exc:
        logger.warning("Pickle dump failed (%s), attempting fallback serialization", exc)
        try:
            import torch
            buffer = io.BytesIO()
            torch.save(model, buffer)
            size = buffer.tell()
        except Exception:
            size = 0
    return size


def measure_inference_latency(
    model_wrapper: Any,
    X: np.ndarray,
    n_repeats: int = 5,
) -> Dict[str, float]:
    """Measure inference latency (mean and std over n_repeats).

    Returns dict with 'latency_mean_ms' and 'latency_std_ms'.
    """
    # Warm-up run
    warmup_n = min(100, len(X)) if hasattr(X, "__len__") else 0
    if warmup_n > 0:
        _ = model_wrapper.predict_proba(X[:warmup_n])

    times = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        _ = model_wrapper.predict_proba(X)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    return {
        "latency_mean_ms": float(np.mean(times)),
        "latency_std_ms": float(np.std(times)),
    }


def compute_efficiency_metrics(
    model_wrapper: Any,
    X_test: np.ndarray,
    model_type: str,
) -> Dict[str, Any]:
    """Compute all efficiency metrics for a model.

    Returns dict with:
    - total_params, nonzero_params, sparsity
    - model_size_bytes
    - latency_mean_ms, latency_std_ms
    """
    model = model_wrapper.get_model()

    params = count_parameters(model, model_type)
    size = measure_model_size_bytes(model, model_type)
    latency = measure_inference_latency(model_wrapper, X_test)

    sparsity = 1.0 - (params["nonzero_params"] / max(params["total_params"], 1))

    return {
        **params,
        "sparsity": float(sparsity),
        "model_size_bytes": size,
        **latency,
    }
