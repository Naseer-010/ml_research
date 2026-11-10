"""
Explainability and Feature Attribution Utilities for Tabular Models.

This module provides tools for:
1. SHAP value computation using TreeExplainer (trees) or KernelExplainer (neural networks),
   with automated fallback to scikit-learn permutation importance if SHAP is unavailable.
2. Visualizations for SHAP beeswarm summaries and side-by-side compression comparisons.
3. Feature importance drift quantification between uncompressed and compressed models.
4. Reliability diagrams (calibration curves) partitioned by demographic groups.
5. Grouped feature importance comparison bar charts.
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for headless servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config

# Calibration metrics imports with graceful resolution
try:
    from src.calibration_metrics import expected_calibration_error, reliability_diagram_data
except ImportError:
    from calibration_metrics import expected_calibration_error, reliability_diagram_data

# SHAP optional import
try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    shap = None
    _SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.getLogger("shap").setLevel(logging.WARNING)
logging.getLogger("numba").setLevel(logging.WARNING)


@dataclass
class FallbackExplanation:
    """
    Stand-in representation of feature attributions when SHAP is not installed.

    Attributes
    ----------
    values : np.ndarray
        Feature attribution values of shape (n_samples, n_features) or (n_features,).
    data : np.ndarray
        Input feature matrix evaluated.
    feature_names : List[str]
        Names of the evaluated features.
    """
    values: np.ndarray
    data: np.ndarray
    feature_names: List[str] = field(default_factory=list)


def compute_shap_values(
    model: Any,
    X: Union[np.ndarray, pd.DataFrame],
    feature_names: Optional[List[str]] = None,
    model_type: str = "tree",
) -> Any:
    """
    Compute SHAP explanation values for a trained tabular model.

    Uses:
    - TreeExplainer for decision tree ensembles (RandomForest, GradientBoosting).
    - KernelExplainer with a 100-sample background reference for Neural Networks.
    - Gracefully falls back to scikit-learn permutation importance if SHAP is
      not installed.

    Parameters
    ----------
    model : Any
        Trained model instance (scikit-learn estimator or PyTorch Module).
    X : Union[np.ndarray, pd.DataFrame]
        Evaluation samples to explain.
    feature_names : Optional[List[str]], default=None
        List of feature names matching X's columns. Inferred from X if pd.DataFrame.
    model_type : str, default='tree'
        Model architecture indicator: 'tree' for tree ensembles, 'nn' or 'kernel'
        for neural networks.

    Returns
    -------
    shap.Explanation or FallbackExplanation
        Explanation object containing attribution values, input data, and feature names.
    """
    # Extract feature names if available
    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    X_mat = np.asarray(X)

    if not _SHAP_AVAILABLE:
        logger.warning(
            "SHAP library is not installed. Falling back to sklearn permutation_importance."
        )
        return _fallback_permutation_importance(model, X_mat, feature_names)

    model_type_clean = model_type.strip().lower()

    if model_type_clean == "tree":
        logger.info("Computing attributions using shap.TreeExplainer.")
        try:
            explainer = shap.TreeExplainer(model)
            shap_obj = explainer(X)

            # For binary classification, TreeExplainer may produce (samples, features, classes)
            if hasattr(shap_obj, "values") and shap_obj.values.ndim == 3:
                # Select positive class slice (class 1)
                shap_obj = shap_obj[:, :, 1]

            return shap_obj
        except Exception as exc:
            logger.warning(
                "TreeExplainer failed (%s). Attempting KernelExplainer fallback.", exc
            )

    logger.info("Computing attributions using shap.KernelExplainer (100 background samples).")
    try:
        # Build 100-sample background
        n_samples = len(X_mat)
        bg_size = min(100, n_samples)
        rng = np.random.RandomState(config.RANDOM_SEEDS[0])
        bg_idx = rng.choice(n_samples, size=bg_size, replace=False)
        background = X_mat[bg_idx]

        predict_fn = _get_prediction_function(model)
        explainer = shap.KernelExplainer(predict_fn, background)
        # Limit evaluation samples if X is exceptionally large to keep kernel shap tractable
        eval_X = X_mat[:min(len(X_mat), 500)]
        shap_values = explainer.shap_values(eval_X, silent=True)

        if isinstance(shap_values, list):
            # Binary classification returns list of [neg_class, pos_class]
            vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            vals = shap_values

        return shap.Explanation(
            values=vals,
            data=eval_X,
            feature_names=feature_names,
        )
    except Exception as exc:
        logger.warning(
            "KernelExplainer failed (%s). Falling back to sklearn permutation_importance.", exc
        )
        return _fallback_permutation_importance(model, X_mat, feature_names)


def _get_prediction_function(model: Any) -> Callable[[np.ndarray], np.ndarray]:
    """Resolve a unified probability prediction function for SHAP KernelExplainer."""
    if hasattr(model, "predict_proba"):
        def predict_proba_fn(data: np.ndarray) -> np.ndarray:
            p = model.predict_proba(data)
            return p[:, 1] if (p.ndim == 2 and p.shape[1] > 1) else p.ravel()
        return predict_proba_fn

    if callable(model) and hasattr(model, "forward"):
        # PyTorch Module
        def torch_predict_fn(data: np.ndarray) -> np.ndarray:
            import torch
            is_quant = any(
                "quantized" in m.__class__.__module__.lower() or "quantized" in m.__class__.__name__.lower()
                for m in model.modules()
            )
            if is_quant:
                dev = torch.device("cpu")
                model.cpu()
            else:
                params = list(model.parameters())
                dev = params[0].device if params else torch.device("cpu")

            inp = torch.as_tensor(data, dtype=torch.float32, device=dev)
            model.eval()
            with torch.no_grad():
                try:
                    out = model(inp)
                except Exception:
                    # In case of device mismatch or quantized operator issues, fall back to CPU
                    model.cpu()
                    inp_cpu = torch.as_tensor(data, dtype=torch.float32, device="cpu")
                    out = model(inp_cpu)

                if out.ndim > 1 and out.shape[1] > 1:
                    out = torch.softmax(out, dim=1)[:, 1]
                elif out.ndim > 1 and out.shape[1] == 1:
                    # Check if sigmoid is needed (e.g. outputs raw logits outside [0, 1])
                    if (out < 0).any() or (out > 1).any():
                        out = torch.sigmoid(out)
                return out.cpu().numpy().ravel()
        return torch_predict_fn

    if hasattr(model, "get_model"):
        return _get_prediction_function(model.get_model())

    if hasattr(model, "predict"):
        return lambda data: np.asarray(model.predict(data), dtype=float).ravel()

    raise ValueError(f"Unable to resolve prediction function for model: {type(model)}")


def _fallback_permutation_importance(
    model: Any,
    X: np.ndarray,
    feature_names: List[str],
) -> FallbackExplanation:
    """Compute permutation importance as a fallback when SHAP cannot be used.

    Supports scikit-learn estimators, PyTorch modules, and model wrappers.
    Directly computes feature score variations via feature shuffling.
    """
    logger.info("Computing feature importance via permutation importance.")
    try:
        # Check if model natively supports sklearn permutation_importance (has fit method)
        if hasattr(model, "fit"):
            if hasattr(model, "predict"):
                y_pseudo = model.predict(X)
            elif hasattr(model, "predict_proba"):
                y_pseudo = (model.predict_proba(X)[:, 1] >= 0.5).astype(int)
            else:
                y_pseudo = np.zeros(len(X), dtype=int)

            res = permutation_importance(
                model,
                X,
                y_pseudo,
                n_repeats=5,
                random_state=config.RANDOM_SEEDS[0],
                n_jobs=-1,
            )
            mean_imp = res.importances_mean
        else:
            # Model is a PyTorch Module or Wrapper without fit() - compute permutation importance directly
            predict_fn = _get_prediction_function(model)
            base_preds = predict_fn(X)
            n_features = X.shape[1]
            mean_imp = np.zeros(n_features, dtype=float)
            rng = np.random.RandomState(config.RANDOM_SEEDS[0])
            n_eval = min(len(X), 300)
            X_eval = X[:n_eval]
            base_eval = base_preds[:n_eval]

            for j in range(n_features):
                col_diffs = []
                for _ in range(3):
                    X_perm = X_eval.copy()
                    X_perm[:, j] = rng.permutation(X_perm[:, j])
                    perm_preds = predict_fn(X_perm)
                    col_diffs.append(np.mean(np.abs(perm_preds - base_eval)))
                mean_imp[j] = float(np.mean(col_diffs))

            total_imp = np.sum(mean_imp)
            if total_imp > 0:
                mean_imp = mean_imp / total_imp
            else:
                mean_imp = np.ones(n_features, dtype=float) / n_features

    except Exception as exc:
        logger.warning("Permutation importance failed (%s); using uniform attribution.", exc)
        mean_imp = np.ones(X.shape[1], dtype=float) / X.shape[1]

    # Tile mean importance across samples to mimic sample-level attributions
    values = np.tile(mean_imp, (len(X), 1))
    return FallbackExplanation(values=values, data=X, feature_names=feature_names)


def plot_shap_summary(
    shap_values: Any,
    X: Union[np.ndarray, pd.DataFrame],
    feature_names: Optional[List[str]] = None,
    title: str = "SHAP Summary Plot",
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot SHAP beeswarm summary or fallback attribution bar chart.

    Parameters
    ----------
    shap_values : Any
        SHAP Explanation object or FallbackExplanation.
    X : Union[np.ndarray, pd.DataFrame]
        Underlying feature values.
    feature_names : Optional[List[str]], default=None
        Names of the features.
    title : str, default='SHAP Summary Plot'
        Plot title.
    save_path : Optional[Union[str, Path]], default=None
        Path to save the generated plot. If parent directories do not exist,
        they are automatically created.
    """
    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
        elif hasattr(shap_values, "feature_names") and shap_values.feature_names:
            feature_names = list(shap_values.feature_names)

    fig = plt.figure(figsize=(10, 6))

    if _SHAP_AVAILABLE and hasattr(shap_values, "values") and not isinstance(shap_values, FallbackExplanation):
        try:
            shap.plots.beeswarm(shap_values, max_display=15, show=False)
            plt.title(title, fontsize=12, pad=15)
        except Exception as exc:
            logger.warning("shap.plots.beeswarm failed (%s); drawing bar plot instead.", exc)
            _plot_mean_abs_bar(shap_values.values, feature_names, title)
    else:
        vals = shap_values.values if hasattr(shap_values, "values") else np.asarray(shap_values)
        _plot_mean_abs_bar(vals, feature_names, title)

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, bbox_inches="tight", dpi=300)
        logger.info("Saved SHAP summary plot to: %s", p)

    plt.close("all")


def _plot_mean_abs_bar(
    values: np.ndarray,
    feature_names: Optional[List[str]],
    title: str,
) -> None:
    """Helper to render mean absolute feature attribution horizontal bar plot."""
    vals = np.asarray(values)
    if vals.ndim == 1:
        mean_abs = np.abs(vals)
    else:
        mean_abs = np.mean(np.abs(vals), axis=0)

    n_feats = len(mean_abs)
    names = feature_names if feature_names and len(feature_names) == n_feats else [f"f_{i}" for i in range(n_feats)]

    top_k = min(15, n_feats)
    top_indices = np.argsort(mean_abs)[::-1][:top_k][::-1]

    plt.barh(range(top_k), mean_abs[top_indices], color="#1f77b4", align="center")
    plt.yticks(range(top_k), [names[i] for i in top_indices])
    plt.xlabel("Mean |Attribution|")
    plt.title(title, fontsize=12, pad=12)
    plt.grid(True, linestyle=":", alpha=0.5, axis="x")


def plot_shap_comparison(
    shap_before: Any,
    shap_after: Any,
    feature_names: List[str],
    title: str = "SHAP Feature Importance: Baseline vs Compressed",
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Render side-by-side SHAP importance comparison before and after model compression.

    Parameters
    ----------
    shap_before : Any
        SHAP values / Explanation for the baseline uncompressed model.
    shap_after : Any
        SHAP values / Explanation for the compressed model.
    feature_names : List[str]
        Names of the evaluated features.
    title : str, default='SHAP Feature Importance: Baseline vs Compressed'
        Super-title for the figure.
    save_path : Optional[Union[str, Path]], default=None
        Path to save the generated plot.
    """
    vals_b = shap_before.values if hasattr(shap_before, "values") else np.asarray(shap_before)
    vals_a = shap_after.values if hasattr(shap_after, "values") else np.asarray(shap_after)

    imp_b = np.mean(np.abs(vals_b), axis=0) if vals_b.ndim > 1 else np.abs(vals_b)
    imp_a = np.mean(np.abs(vals_a), axis=0) if vals_a.ndim > 1 else np.abs(vals_a)

    # Order by baseline importance
    top_k = min(15, len(feature_names))
    order = np.argsort(imp_b)[::-1][:top_k][::-1]

    ordered_names = [feature_names[i] for i in order]
    b_ordered = imp_b[order]
    a_ordered = imp_a[order]

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, top_k * 0.35)), sharey=True)

    # Baseline panel
    axes[0].barh(range(top_k), b_ordered, color="#2ca02c", align="center")
    axes[0].set_yticks(range(top_k))
    axes[0].set_yticklabels(ordered_names)
    axes[0].set_xlabel("Mean |SHAP Value|")
    axes[0].set_title("Baseline (Uncompressed)", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.5, axis="x")

    # Compressed panel
    axes[1].barh(range(top_k), a_ordered, color="#d62728", align="center")
    axes[1].set_xlabel("Mean |SHAP Value|")
    axes[1].set_title("Compressed Model", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.5, axis="x")

    fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, bbox_inches="tight", dpi=300)
        logger.info("Saved SHAP comparison plot to: %s", p)

    plt.close(fig)


def compute_feature_importance_drift(
    model_before: Any,
    model_after: Any,
    X: Union[np.ndarray, pd.DataFrame],
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Quantify feature importance drift between uncompressed and compressed models.

    Computes:
    - Absolute importance values before and after compression.
    - Absolute and relative drift per feature.
    - Rank shifts (positive = feature gained relative importance, negative = dropped).

    Parameters
    ----------
    model_before : Any
        Baseline (uncompressed) model.
    model_after : Any
        Compressed model.
    X : Union[np.ndarray, pd.DataFrame]
        Representative feature dataset for attribution estimation.
    feature_names : List[str]
        Names of the evaluated features.

    Returns
    -------
    pd.DataFrame
        DataFrame sorted by descending absolute drift containing columns:
        [
            'feature', 'importance_before', 'importance_after',
            'absolute_drift', 'relative_drift',
            'rank_before', 'rank_after', 'rank_change'
        ]
    """
    imp_before = _extract_model_importance(model_before, X, feature_names)
    imp_after = _extract_model_importance(model_after, X, feature_names)

    # Normalize importances so sum is 1.0 (relative importance distribution)
    sum_b = np.sum(imp_before)
    sum_a = np.sum(imp_after)
    norm_before = imp_before / sum_b if sum_b > 0 else imp_before
    norm_after = imp_after / sum_a if sum_a > 0 else imp_after

    abs_drift = np.abs(norm_after - norm_before)
    rel_drift = np.where(
        norm_before > 0,
        (norm_after - norm_before) / (norm_before + 1e-9),
        0.0,
    )

    rank_before = pd.Series(norm_before).rank(ascending=False, method="min").astype(int).values
    rank_after = pd.Series(norm_after).rank(ascending=False, method="min").astype(int).values
    # Negative rank_change means rank number decreased -> feature moved higher in priority
    rank_change = rank_before - rank_after

    drift_df = pd.DataFrame({
        "feature": feature_names,
        "importance_before": norm_before,
        "importance_after": norm_after,
        "absolute_drift": abs_drift,
        "relative_drift": rel_drift,
        "rank_before": rank_before,
        "rank_after": rank_after,
        "rank_change": rank_change,
    })

    return drift_df.sort_values(by="absolute_drift", ascending=False).reset_index(drop=True)


def _extract_model_importance(
    model: Any,
    X: Union[np.ndarray, pd.DataFrame],
    feature_names: List[str],
) -> np.ndarray:
    """Internal helper to retrieve native or SHAP-based feature importance from a model."""
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)

    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        return np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)

    # Compute via SHAP / permutation fallback
    explanation = compute_shap_values(model, X, feature_names=feature_names, model_type="tree")
    vals = explanation.values if hasattr(explanation, "values") else np.asarray(explanation)
    if vals.ndim > 1:
        return np.mean(np.abs(vals), axis=0)
    return np.abs(vals)


def plot_calibration_curves_by_group(
    y_true: Union[np.ndarray, Sequence[int], Sequence[float]],
    y_prob: Union[np.ndarray, Sequence[float]],
    group_labels: Union[np.ndarray, Sequence[Any], pd.Series],
    group_name: str,
    title: str = "Calibration Curves by Demographic Group",
    save_path: Optional[Union[str, Path]] = None,
    n_bins: int = config.ECE_NUM_BINS,
) -> None:
    """
    Plot reliability diagram (calibration curve) with a distinct curve per demographic group.

    Parameters
    ----------
    y_true : Union[np.ndarray, Sequence[int], Sequence[float]]
        Binary ground-truth labels in {0, 1}.
    y_prob : Union[np.ndarray, Sequence[float]]
        Predicted probabilities for the positive class.
    group_labels : Union[np.ndarray, Sequence[Any], pd.Series]
        Demographic group indicators for each sample.
    group_name : str
        Name of the sensitive demographic attribute (e.g. 'Sex', 'Race', 'Age Group').
    title : str, default='Calibration Curves by Demographic Group'
        Plot title.
    save_path : Optional[Union[str, Path]], default=None
        Path to save the resulting figure.
    n_bins : int, default=config.ECE_NUM_BINS (15)
        Number of equal-width bins for reliability estimation.
    """
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.clip(np.asarray(y_prob, dtype=float).ravel(), 0.0, 1.0)
    groups = np.asarray(group_labels).ravel()

    fig, ax = plt.subplots(figsize=(8, 7))

    # Diagonal reference line: perfect calibration
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration", alpha=0.7)

    unique_groups = pd.unique(groups)
    palette = sns.color_palette("tab10", n_colors=max(len(unique_groups), 1))

    for idx, grp in enumerate(unique_groups):
        mask = groups == grp
        sub_size = int(np.sum(mask))
        if sub_size == 0:
            continue

        mean_pred, frac_pos, _ = reliability_diagram_data(y_t[mask], y_p[mask], n_bins=n_bins)
        grp_ece = expected_calibration_error(y_t[mask], y_p[mask], n_bins=n_bins)

        color = palette[idx % len(palette)]
        ax.plot(
            mean_pred,
            frac_pos,
            marker="o",
            linewidth=2,
            markersize=6,
            color=color,
            label=f"{grp} (N={sub_size}, ECE={grp_ece:.3f})",
        )

    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Fraction of Positives", fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title, fontsize=12, pad=12)
    ax.legend(title=group_name, loc="best", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, bbox_inches="tight", dpi=300)
        logger.info("Saved group-wise calibration plot to: %s", p)

    plt.close(fig)


def plot_feature_importance_comparison(
    importance_before: Union[np.ndarray, Sequence[float]],
    importance_after: Union[np.ndarray, Sequence[float]],
    feature_names: List[str],
    title: str = "Feature Importance Comparison: Baseline vs Compressed",
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Render grouped bar chart comparing feature importances before and after compression.

    Parameters
    ----------
    importance_before : Union[np.ndarray, Sequence[float]]
        Feature importance vector for baseline model.
    importance_after : Union[np.ndarray, Sequence[float]]
        Feature importance vector for compressed model.
    feature_names : List[str]
        Names of the evaluated features.
    title : str, default='Feature Importance Comparison: Baseline vs Compressed'
        Plot title.
    save_path : Optional[Union[str, Path]], default=None
        Path to save the generated plot.
    """
    imp_b = np.asarray(importance_before, dtype=float).ravel()
    imp_a = np.asarray(importance_after, dtype=float).ravel()

    if len(imp_b) != len(feature_names) or len(imp_a) != len(feature_names):
        raise ValueError(
            f"Length mismatch: len(importance_before)={len(imp_b)}, "
            f"len(importance_after)={len(imp_a)}, len(feature_names)={len(feature_names)}"
        )

    # Sort by baseline importance and display top 15 features for clarity
    top_k = min(15, len(feature_names))
    order = np.argsort(imp_b)[::-1][:top_k][::-1]

    sub_names = [feature_names[i] for i in order]
    sub_b = imp_b[order]
    sub_a = imp_a[order]

    fig, ax = plt.subplots(figsize=(10, max(5, top_k * 0.4)))

    y = np.arange(top_k)
    bar_height = 0.38

    ax.barh(y - bar_height / 2, sub_b, height=bar_height, color="#1f77b4", label="Baseline (0.0)")
    ax.barh(y + bar_height / 2, sub_a, height=bar_height, color="#ff7f0e", label="Compressed")

    ax.set_yticks(y)
    ax.set_yticklabels(sub_names)
    ax.set_xlabel("Relative Feature Importance", fontsize=11)
    ax.set_title(title, fontsize=12, pad=12)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.5, axis="x")

    plt.tight_layout()

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, bbox_inches="tight", dpi=300)
        logger.info("Saved feature importance comparison plot to: %s", p)

    plt.close(fig)
