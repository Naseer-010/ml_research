"""Compression utility functions for tabular machine learning models.

Provides pruning and quantization utilities for Random Forests, Gradient Boosted
Decision Trees, and PyTorch Neural Networks. All functions produce new model copies
and avoid modifying input models in-place.
"""

import copy
import logging
from typing import Any, List, Optional
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

logger = logging.getLogger(__name__)


def _trim_decision_tree_depth(tree_estimator: Any, max_depth: int) -> None:
    """Trim a fitted scikit-learn DecisionTree to a target maximum depth.

    Recursively converts internal nodes at or beyond max_depth into leaf nodes.
    Modifications are made on the copied tree structure if writeable.

    Args:
        tree_estimator: A fitted scikit-learn tree estimator having a `tree_` attribute.
        max_depth: Maximum tree depth allowed.
    """
    if not hasattr(tree_estimator, "tree_"):
        return

    tree_ = tree_estimator.tree_

    try:
        # Check if arrays are writeable or can be made writeable
        for arr in [tree_.children_left, tree_.children_right, tree_.feature, tree_.threshold]:
            if hasattr(arr, "flags") and not arr.flags.writeable:
                try:
                    arr.flags.writeable = True
                except ValueError:
                    pass

        def _trim(node_id: int, current_depth: int) -> None:
            if node_id == -1:
                return
            if current_depth >= max_depth:
                tree_.children_left[node_id] = -1
                tree_.children_right[node_id] = -1
                tree_.feature[node_id] = -2
                tree_.threshold[node_id] = -2.0
            else:
                _trim(tree_.children_left[node_id], current_depth + 1)
                _trim(tree_.children_right[node_id], current_depth + 1)

        _trim(0, 0)
        if hasattr(tree_, "max_depth"):
            try:
                tree_.max_depth = min(tree_.max_depth, max_depth)
            except (AttributeError, TypeError):
                pass
    except Exception as exc:
        logger.debug("Tree internal depth trimming skipped or unsupported: %s", exc)


def prune_random_forest(
    rf_model: RandomForestClassifier, level: float
) -> RandomForestClassifier:
    """Prune a trained RandomForestClassifier by reducing estimators and max depth.

    Removes a fraction of estimators equal to `level` (keeping `1.0 - level` fraction).
    Estimators are selected sequentially from the ensemble. The tree depth is also
    proportionally reduced.

    Args:
        rf_model: Trained RandomForestClassifier instance.
        level: Compression level between 0.0 and 1.0 (fraction of estimators removed).

    Returns:
        A new RandomForestClassifier instance with pruned estimators and depth.

    Raises:
        ValueError: If level is outside the valid range [0.0, 1.0) or model is unfitted.
    """
    if not hasattr(rf_model, "estimators_") or len(rf_model.estimators_) == 0:
        raise ValueError("The provided RandomForestClassifier has not been fitted yet.")

    if not (0.0 <= level < 1.0):
        raise ValueError(f"Compression level must be in [0.0, 1.0), got {level}")

    if level == 0.0:
        return copy.deepcopy(rf_model)

    pruned_rf = copy.deepcopy(rf_model)
    n_total = len(pruned_rf.estimators_)
    n_keep = max(1, int(np.round(n_total * (1.0 - level))))

    pruned_rf.estimators_ = pruned_rf.estimators_[:n_keep]
    pruned_rf.n_estimators = n_keep
    if hasattr(pruned_rf, "estimators_samples_") and pruned_rf.estimators_samples_ is not None:
        try:
            pruned_rf.estimators_samples_ = pruned_rf.estimators_samples_[:n_keep]
        except (AttributeError, TypeError):
            pass

    # Proportionally reduce max_depth
    observed_depths: List[int] = [
        getattr(est.tree_, "max_depth", None)
        for est in pruned_rf.estimators_
        if hasattr(est, "tree_")
    ]
    valid_depths = [d for d in observed_depths if d is not None]
    base_depth: Optional[int] = pruned_rf.max_depth or (max(valid_depths) if valid_depths else None)

    if base_depth is not None and base_depth > 1:
        new_max_depth = max(1, int(np.round(base_depth * (1.0 - level))))
        pruned_rf.max_depth = new_max_depth
        logger.info(
            "Reduced RandomForest estimators: %d -> %d, max_depth: %s -> %d (level=%.2f)",
            n_total, n_keep, base_depth, new_max_depth, level
        )
        for est in pruned_rf.estimators_:
            est.max_depth = new_max_depth
            _trim_decision_tree_depth(est, new_max_depth)
    else:
        logger.info(
            "Reduced RandomForest estimators: %d -> %d (level=%.2f)",
            n_total, n_keep, level
        )

    return pruned_rf


def prune_gradient_boosting(
    gb_model: GradientBoostingClassifier, level: float
) -> GradientBoostingClassifier:
    """Prune a trained GradientBoostingClassifier by truncating boosting rounds.

    Keeps the first (1.0 - level) fraction of boosting rounds by slicing the
    `estimators_` array. Also reduces max_depth proportionally.

    Args:
        gb_model: Trained GradientBoostingClassifier instance.
        level: Compression level between 0.0 and 1.0 (fraction of rounds removed).

    Returns:
        A new GradientBoostingClassifier instance with truncated boosting stages.

    Raises:
        ValueError: If level is outside the valid range [0.0, 1.0) or model is unfitted.
    """
    if not hasattr(gb_model, "estimators_") or len(gb_model.estimators_) == 0:
        raise ValueError("The provided GradientBoostingClassifier has not been fitted yet.")

    if not (0.0 <= level < 1.0):
        raise ValueError(f"Compression level must be in [0.0, 1.0), got {level}")

    if level == 0.0:
        return copy.deepcopy(gb_model)

    pruned_gb = copy.deepcopy(gb_model)
    n_total = len(pruned_gb.estimators_)
    n_keep = max(1, int(np.round(n_total * (1.0 - level))))

    pruned_gb.estimators_ = pruned_gb.estimators_[:n_keep]
    pruned_gb.n_estimators = n_keep
    if hasattr(pruned_gb, "n_estimators_"):
        pruned_gb.n_estimators_ = n_keep
    if hasattr(pruned_gb, "train_score_") and pruned_gb.train_score_ is not None:
        pruned_gb.train_score_ = pruned_gb.train_score_[:n_keep]
    if hasattr(pruned_gb, "oob_improvement_") and pruned_gb.oob_improvement_ is not None:
        pruned_gb.oob_improvement_ = pruned_gb.oob_improvement_[:n_keep]

    # Proportionally reduce max_depth
    observed_depths: List[int] = [
        getattr(tree[0].tree_, "max_depth", None)
        for tree in pruned_gb.estimators_
        if len(tree) > 0 and hasattr(tree[0], "tree_")
    ]
    valid_depths = [d for d in observed_depths if d is not None]
    base_depth: Optional[int] = pruned_gb.max_depth or (max(valid_depths) if valid_depths else None)

    if base_depth is not None and base_depth > 1:
        new_max_depth = max(1, int(np.round(base_depth * (1.0 - level))))
        pruned_gb.max_depth = new_max_depth
        logger.info(
            "Reduced GradientBoosting stages: %d -> %d, max_depth: %s -> %d (level=%.2f)",
            n_total, n_keep, base_depth, new_max_depth, level
        )
        for stage in pruned_gb.estimators_:
            for tree in stage:
                tree.max_depth = new_max_depth
                _trim_decision_tree_depth(tree, new_max_depth)
    else:
        logger.info(
            "Reduced GradientBoosting stages: %d -> %d (level=%.2f)",
            n_total, n_keep, level
        )

    return pruned_gb


def prune_neural_network(model: torch.nn.Module, level: float) -> torch.nn.Module:
    """Apply L1 unstructured pruning to all Linear layers at the given sparsity level.

    Uses `torch.nn.utils.prune.l1_unstructured` and makes pruning permanent by
    removing parameterization hooks via `torch.nn.utils.prune.remove`.

    Args:
        model: Trained PyTorch neural network module.
        level: Sparsity level between 0.0 and 1.0 (fraction of weights pruned).

    Returns:
        A new pruned copy of the PyTorch module.

    Raises:
        ValueError: If level is outside the valid range [0.0, 1.0).
    """
    if not (0.0 <= level < 1.0):
        raise ValueError(f"Compression level must be in [0.0, 1.0), got {level}")

    model_copy = copy.deepcopy(model)
    if level == 0.0:
        return model_copy

    pruned_count = 0
    for module in model_copy.modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=level)
            prune.remove(module, "weight")
            pruned_count += 1

    logger.info(
        "Applied L1 unstructured pruning (sparsity=%.2f) to %d Linear layers",
        level, pruned_count
    )
    return model_copy


def _quantize_weights_simulated(model: torch.nn.Module) -> torch.nn.Module:
    """Simulate INT8 weight quantization by scaling and clamping float weights.

    Provides a fail-proof fallback that runs identically on CPU and GPU without
    relying on platform-specific C++ quantization kernels.
    """
    model_copy = copy.deepcopy(model)
    model_copy.eval()
    with torch.no_grad():
        for module in model_copy.modules():
            if isinstance(module, nn.Linear):
                w = module.weight.data
                max_val = torch.max(torch.abs(w)).item()
                if max_val > 0:
                    scale = max_val / 127.0
                    w_q = torch.clamp(torch.round(w / scale), -128, 127)
                    module.weight.data = w_q * scale
    logger.info("Applied simulated INT8 weight quantization to Linear layers")
    return model_copy


def quantize_neural_network(model: torch.nn.Module) -> torch.nn.Module:
    """Apply dynamic INT8 quantization to Linear layers using torch.ao.quantization.

    Converts 32-bit floating point weights of Linear layers into 8-bit integers
    for reduced memory footprint and accelerated CPU inference.
    Dynamic quantization in PyTorch is CPU-only, so the model is transferred to CPU first.

    Args:
        model: PyTorch neural network module.

    Returns:
        A new dynamically quantized PyTorch module.
    """
    # PyTorch dynamic quantization is strictly a CPU operator
    model_copy = copy.deepcopy(model).cpu()
    model_copy.eval()

    # Configure quantization engine if available
    try:
        if hasattr(torch.backends, "quantized"):
            supported = getattr(torch.backends.quantized, "supported_engines", [])
            if "fbgemm" in supported:
                torch.backends.quantized.engine = "fbgemm"
            elif "qnnpack" in supported:
                torch.backends.quantized.engine = "qnnpack"
    except Exception as eng_exc:
        logger.debug("Failed setting quantized engine: %s", eng_exc)

    try:
        from torch.ao.quantization import quantize_dynamic
    except ImportError:
        from torch.quantization import quantize_dynamic  # type: ignore

    try:
        quantized_model = quantize_dynamic(
            model_copy,
            {nn.Linear},
            dtype=torch.qint8,
        )
        # Verify 1-sample test forward pass on CPU to confirm packed parameters are valid
        first_linear = next((m for m in quantized_model.modules() if hasattr(m, "in_features")), None)
        in_dim = getattr(first_linear, "in_features", 14)
        dummy_x = torch.zeros((1, in_dim), dtype=torch.float32)
        with torch.no_grad():
            _ = quantized_model(dummy_x)

        logger.info("Applied dynamic INT8 quantization to Linear layers (CPU)")
        return quantized_model
    except Exception as exc:
        logger.warning(
            "Dynamic INT8 quantization failed (%s); using simulated INT8 weight quantization fallback.",
            exc,
        )
        return _quantize_weights_simulated(model_copy)


def fp16_neural_network(model: torch.nn.Module) -> torch.nn.Module:
    """Convert a neural network to FP16 half-precision.

    Args:
        model: PyTorch neural network module.

    Returns:
        A new FP16 copy of the PyTorch module.
    """
    model_copy = copy.deepcopy(model)
    model_copy.eval()
    model_copy = model_copy.half()
    logger.info("Applied FP16 half-precision conversion")
    return model_copy


# Valid NN compression modes
NN_COMPRESSION_MODES = ["prune_only", "quantize_only", "prune+quantize", "fp16"]


def compress_neural_network(
    model: torch.nn.Module,
    level: float,
    mode: str = "prune+quantize",
) -> torch.nn.Module:
    """Compress a neural network using the specified mode.

    Supported modes:
    - 'prune_only': L1 unstructured weight pruning only.
    - 'quantize_only': Dynamic INT8 quantization only (level ignored).
    - 'prune+quantize': Pruning followed by INT8 quantization (default).
    - 'fp16': FP16 half-precision conversion (level ignored).

    Args:
        model: PyTorch neural network module.
        level: Compression level between 0.0 and 1.0 (sparsity for pruning).
        mode: Compression mode string.

    Returns:
        A new compressed PyTorch module.

    Raises:
        ValueError: If level is outside [0.0, 1.0) or mode is unrecognized.
    """
    if mode not in NN_COMPRESSION_MODES:
        raise ValueError(
            f"Unknown NN compression mode '{mode}'. "
            f"Supported modes: {NN_COMPRESSION_MODES}"
        )

    if not (0.0 <= level < 1.0):
        raise ValueError(f"Compression level must be in [0.0, 1.0), got {level}")

    if level <= 0.0 and mode not in ("quantize_only", "fp16"):
        return copy.deepcopy(model)

    if mode == "prune_only":
        logger.info("Compressing neural network at level=%.2f (pruning only)", level)
        return prune_neural_network(model, level=level)

    elif mode == "quantize_only":
        logger.info("Compressing neural network (INT8 quantization only)")
        return quantize_neural_network(model)

    elif mode == "prune+quantize":
        logger.info("Compressing neural network at level=%.2f (pruning + INT8 quantization)", level)
        pruned_model = prune_neural_network(model, level=level)
        return quantize_neural_network(pruned_model)

    elif mode == "fp16":
        logger.info("Compressing neural network (FP16 half-precision)")
        return fp16_neural_network(model)

    raise ValueError(f"Unhandled mode: {mode}")
