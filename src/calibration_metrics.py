"""
Calibration Quality Metrics for Tabular Models.

This module provides functions to compute calibration error metrics (Expected
Calibration Error, Maximum Calibration Error, Brier Score), generate data for
reliability diagrams, evaluate metrics overall and across demographic groups,
and compute calibration gaps between demographic groups.

All functions handle edge cases such as empty bins, small sample groups,
and extreme predicted probabilities gracefully.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# Ensure project root is in sys.path for config import
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config

logger = logging.getLogger(__name__)


def expected_calibration_error(
    y_true: Union[np.ndarray, Sequence[int], Sequence[float]],
    y_prob: Union[np.ndarray, Sequence[float]],
    n_bins: int = config.ECE_NUM_BINS,
) -> float:
    """
    Compute Expected Calibration Error (ECE) using equal-width bins.

    ECE partitions predicted probabilities into M equal-width bins on [0, 1]
    and computes the weighted average absolute difference between empirical
    accuracy and mean predicted confidence across all bins:
        ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|

    Parameters
    ----------
    y_true : Union[np.ndarray, Sequence[int], Sequence[float]]
        Binary ground-truth labels in {0, 1}.
    y_prob : Union[np.ndarray, Sequence[float]]
        Predicted probabilities for the positive class in [0, 1].
    n_bins : int, default=config.ECE_NUM_BINS (15)
        Number of equal-width bins.

    Returns
    -------
    float
        The Expected Calibration Error in [0.0, 1.0]. Returns 0.0 if inputs
        are empty.

    Raises
    ------
    ValueError
        If lengths of `y_true` and `y_prob` do not match, or `n_bins` < 1.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be a positive integer, got {n_bins}")

    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_prob, dtype=float).ravel()

    if len(y_t) != len(y_p):
        raise ValueError(
            f"Length mismatch: len(y_true)={len(y_t)} vs len(y_prob)={len(y_p)}"
        )

    n_samples = len(y_t)
    if n_samples == 0:
        logger.warning("Empty arrays provided to expected_calibration_error; returning 0.0.")
        return 0.0

    # Clip probabilities to valid unit interval
    y_p = np.clip(y_p, 0.0, 1.0)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Right-inclusive on the last bin to include 1.0
        if i == n_bins - 1:
            in_bin = (y_p >= bin_lower) & (y_p <= bin_upper)
        else:
            in_bin = (y_p >= bin_lower) & (y_p < bin_upper)

        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            bin_acc = float(np.mean(y_t[in_bin]))
            bin_conf = float(np.mean(y_p[in_bin]))
            ece += (bin_count / n_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def maximum_calibration_error(
    y_true: Union[np.ndarray, Sequence[int], Sequence[float]],
    y_prob: Union[np.ndarray, Sequence[float]],
    n_bins: int = config.ECE_NUM_BINS,
) -> float:
    """
    Compute Maximum Calibration Error (MCE) across equal-width bins.

    MCE identifies the worst calibration error across all populated bins:
        MCE = max_{m: |B_m| > 0} |acc(B_m) - conf(B_m)|

    Parameters
    ----------
    y_true : Union[np.ndarray, Sequence[int], Sequence[float]]
        Binary ground-truth labels in {0, 1}.
    y_prob : Union[np.ndarray, Sequence[float]]
        Predicted probabilities for the positive class in [0, 1].
    n_bins : int, default=config.ECE_NUM_BINS (15)
        Number of equal-width bins.

    Returns
    -------
    float
        The Maximum Calibration Error in [0.0, 1.0]. Returns 0.0 if inputs
        are empty or no bins are populated.

    Raises
    ------
    ValueError
        If lengths of `y_true` and `y_prob` do not match, or `n_bins` < 1.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be a positive integer, got {n_bins}")

    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_prob, dtype=float).ravel()

    if len(y_t) != len(y_p):
        raise ValueError(
            f"Length mismatch: len(y_true)={len(y_t)} vs len(y_prob)={len(y_p)}"
        )

    if len(y_t) == 0:
        logger.warning("Empty arrays provided to maximum_calibration_error; returning 0.0.")
        return 0.0

    y_p = np.clip(y_p, 0.0, 1.0)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    max_error = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (y_p >= bin_lower) & (y_p <= bin_upper)
        else:
            in_bin = (y_p >= bin_lower) & (y_p < bin_upper)

        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            bin_acc = float(np.mean(y_t[in_bin]))
            bin_conf = float(np.mean(y_p[in_bin]))
            error = abs(bin_acc - bin_conf)
            if error > max_error:
                max_error = error

    return float(max_error)


def brier_score(
    y_true: Union[np.ndarray, Sequence[int], Sequence[float]],
    y_prob: Union[np.ndarray, Sequence[float]],
) -> float:
    """
    Compute the Brier Score for binary classification.

    The Brier score measures the mean squared error of probability forecasts:
        Brier = (1 / N) * sum_{i=1}^N (p_i - y_i)^2

    Parameters
    ----------
    y_true : Union[np.ndarray, Sequence[int], Sequence[float]]
        Binary ground-truth labels in {0, 1}.
    y_prob : Union[np.ndarray, Sequence[float]]
        Predicted probabilities for the positive class in [0, 1].

    Returns
    -------
    float
        The Brier score in [0.0, 1.0]. Returns 0.0 if inputs are empty.

    Raises
    ------
    ValueError
        If lengths of `y_true` and `y_prob` do not match.
    """
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_prob, dtype=float).ravel()

    if len(y_t) != len(y_p):
        raise ValueError(
            f"Length mismatch: len(y_true)={len(y_t)} vs len(y_prob)={len(y_p)}"
        )

    if len(y_t) == 0:
        logger.warning("Empty arrays provided to brier_score; returning 0.0.")
        return 0.0

    y_p = np.clip(y_p, 0.0, 1.0)
    return float(np.mean((y_p - y_t) ** 2))


def log_loss(
    y_true: Union[np.ndarray, Sequence[int], Sequence[float]],
    y_prob: Union[np.ndarray, Sequence[float]],
    eps: float = 1e-15,
) -> float:
    """
    Compute binary cross-entropy (log loss).

    Formula:
        -(1/N) * sum(y * log(p + eps) + (1 - y) * log(1 - p + eps))

    Parameters
    ----------
    y_true : Union[np.ndarray, Sequence[int], Sequence[float]]
        Binary ground-truth labels in {0, 1}.
    y_prob : Union[np.ndarray, Sequence[float]]
        Predicted probabilities for the positive class in [0, 1].
    eps : float, default=1e-15
        Small epsilon value to avoid undefined log(0).

    Returns
    -------
    float
        The binary cross-entropy loss. Returns 0.0 if inputs are empty.

    Raises
    ------
    ValueError
        If lengths of `y_true` and `y_prob` do not match.
    """
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_prob, dtype=float).ravel()

    if len(y_t) != len(y_p):
        raise ValueError(
            f"Length mismatch: len(y_true)={len(y_t)} vs len(y_prob)={len(y_p)}"
        )

    if len(y_t) == 0:
        logger.warning("Empty arrays provided to log_loss; returning 0.0.")
        return 0.0

    y_p = np.clip(y_p, 0.0, 1.0)
    loss = -float(np.mean(y_t * np.log(y_p + eps) + (1.0 - y_t) * np.log(1.0 - y_p + eps)))
    return float(loss)


def reliability_diagram_data(
    y_true: Union[np.ndarray, Sequence[int], Sequence[float]],
    y_prob: Union[np.ndarray, Sequence[float]],
    n_bins: int = config.ECE_NUM_BINS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute calibration curve points and bin sample counts for reliability diagrams.

    Only populated bins (bin_count > 0) are returned, preventing undefined
    mean confidence / empirical accuracy artifacts when plotting.

    Parameters
    ----------
    y_true : Union[np.ndarray, Sequence[int], Sequence[float]]
        Binary ground-truth labels in {0, 1}.
    y_prob : Union[np.ndarray, Sequence[float]]
        Predicted probabilities for the positive class in [0, 1].
    n_bins : int, default=config.ECE_NUM_BINS (15)
        Number of equal-width bins.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (mean_predicted, fraction_positives, bin_counts)
        - mean_predicted: Mean predicted probability for each populated bin.
        - fraction_positives: Proportion of positive labels for each populated bin.
        - bin_counts: Number of samples falling into each populated bin.

    Raises
    ------
    ValueError
        If lengths of `y_true` and `y_prob` do not match, or `n_bins` < 1.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be a positive integer, got {n_bins}")

    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_prob, dtype=float).ravel()

    if len(y_t) != len(y_p):
        raise ValueError(
            f"Length mismatch: len(y_true)={len(y_t)} vs len(y_prob)={len(y_p)}"
        )

    if len(y_t) == 0:
        logger.warning("Empty arrays provided to reliability_diagram_data; returning empty arrays.")
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=int),
        )

    y_p = np.clip(y_p, 0.0, 1.0)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)

    mean_predicted: list[float] = []
    fraction_positives: list[float] = []
    bin_counts: list[int] = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (y_p >= bin_lower) & (y_p <= bin_upper)
        else:
            in_bin = (y_p >= bin_lower) & (y_p < bin_upper)

        count = int(np.sum(in_bin))
        if count > 0:
            mean_predicted.append(float(np.mean(y_p[in_bin])))
            fraction_positives.append(float(np.mean(y_t[in_bin])))
            bin_counts.append(count)

    return (
        np.array(mean_predicted, dtype=float),
        np.array(fraction_positives, dtype=float),
        np.array(bin_counts, dtype=int),
    )


def compute_overall_metrics(
    y_true: Union[np.ndarray, Sequence[int], Sequence[float]],
    y_prob: Union[np.ndarray, Sequence[float]],
    n_bins: int = config.ECE_NUM_BINS,
) -> Dict[str, float]:
    """
    Compute aggregate calibration and classification performance metrics.

    Metrics computed:
    - Expected Calibration Error (ECE)
    - Maximum Calibration Error (MCE)
    - Brier score
    - Classification accuracy (using a 0.5 decision threshold)

    Parameters
    ----------
    y_true : Union[np.ndarray, Sequence[int], Sequence[float]]
        Binary ground-truth labels in {0, 1}.
    y_prob : Union[np.ndarray, Sequence[float]]
        Predicted probabilities for the positive class in [0, 1].
    n_bins : int, default=config.ECE_NUM_BINS (15)
        Number of equal-width bins for ECE and MCE.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys: {'ece', 'mce', 'brier', 'log_loss', 'accuracy'}.
    """
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_prob, dtype=float).ravel()

    if len(y_t) != len(y_p):
        raise ValueError(
            f"Length mismatch: len(y_true)={len(y_t)} vs len(y_prob)={len(y_p)}"
        )

    if len(y_t) == 0:
        logger.warning("Empty arrays provided to compute_overall_metrics; returning zeros.")
        return {"ece": 0.0, "mce": 0.0, "brier": 0.0, "log_loss": 0.0, "accuracy": 0.0}

    y_p = np.clip(y_p, 0.0, 1.0)
    ece = expected_calibration_error(y_t, y_p, n_bins=n_bins)
    mce = maximum_calibration_error(y_t, y_p, n_bins=n_bins)
    brier = brier_score(y_t, y_p)

    y_pred_binary = (y_p >= 0.5).astype(float)
    accuracy = float(np.mean(y_pred_binary == (y_t >= 0.5).astype(float)))

    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier": float(brier),
        "log_loss": log_loss(y_t, y_p),
        "accuracy": float(accuracy),
    }


def compute_groupwise_metrics(
    y_true: Union[np.ndarray, Sequence[int], Sequence[float]],
    y_prob: Union[np.ndarray, Sequence[float]],
    group_labels: Union[np.ndarray, Sequence[Any], pd.Series],
    n_bins: int = config.ECE_NUM_BINS,
) -> pd.DataFrame:
    """
    Compute calibration and accuracy metrics partitioned by demographic group.

    Handles edge cases gracefully:
    - Empty inputs return an empty DataFrame with expected column schema.
    - Groups with fewer than 10 samples are logged with a warning, but metrics
       are computed on the available samples.
    - Groups with 0 samples receive NaN for metric values.

    Parameters
    ----------
    y_true : Union[np.ndarray, Sequence[int], Sequence[float]]
        Binary ground-truth labels in {0, 1}.
    y_prob : Union[np.ndarray, Sequence[float]]
        Predicted probabilities for the positive class in [0, 1].
    group_labels : Union[np.ndarray, Sequence[Any], pd.Series]
        Categorical group membership identifier for each observation.
    n_bins : int, default=config.ECE_NUM_BINS (15)
        Number of equal-width bins for calibration metrics.

    Returns
    -------
    pd.DataFrame
        DataFrame with rows corresponding to demographic groups and columns:
        ['group', 'group_size', 'ece', 'mce', 'brier', 'log_loss', 'accuracy'].
    """
    columns = ["group", "group_size", "ece", "mce", "brier", "log_loss", "accuracy"]

    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_prob, dtype=float).ravel()
    groups = np.asarray(group_labels).ravel()

    if len(y_t) != len(y_p) or len(y_t) != len(groups):
        raise ValueError(
            f"Length mismatch: len(y_true)={len(y_t)}, len(y_prob)={len(y_p)}, "
            f"len(group_labels)={len(groups)}"
        )

    if len(y_t) == 0:
        logger.warning("Empty data provided to compute_groupwise_metrics; returning empty DataFrame.")
        return pd.DataFrame(columns=columns)

    unique_groups = pd.unique(groups)
    records = []

    for grp in unique_groups:
        mask = groups == grp
        grp_size = int(np.sum(mask))

        if grp_size == 0:
            logger.warning("Group '%s' has 0 samples; metrics will be NaN.", grp)
            records.append({
                "group": grp,
                "group_size": 0,
                "ece": np.nan,
                "mce": np.nan,
                "brier": np.nan,
                "log_loss": np.nan,
                "accuracy": np.nan,
            })
            continue

        if grp_size < 10:
            logger.warning(
                "Group '%s' has only %d samples (< 10). Metric estimates may be unreliable.",
                grp,
                grp_size,
            )

        grp_y_true = y_t[mask]
        grp_y_prob = y_p[mask]
        metrics = compute_overall_metrics(grp_y_true, grp_y_prob, n_bins=n_bins)

        records.append({
            "group": grp,
            "group_size": grp_size,
            "ece": metrics["ece"],
            "mce": metrics["mce"],
            "brier": metrics["brier"],
            "log_loss": metrics["log_loss"],
            "accuracy": metrics["accuracy"],
        })

    df = pd.DataFrame(records, columns=columns)
    return df.sort_values(by="group").reset_index(drop=True)


def calibration_gap(groupwise_metrics_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate maximum calibration disparities across demographic groups.

    Computes:
    - max_ece_gap: Difference between highest and lowest group ECE.
    - max_brier_gap: Difference between highest and lowest group Brier score.
    - worst_group: Demographic group exhibiting highest ECE (poorest calibration).
    - best_group: Demographic group exhibiting lowest ECE (best calibration).

    Parameters
    ----------
    groupwise_metrics_df : pd.DataFrame
        DataFrame produced by `compute_groupwise_metrics`, containing columns:
        ['group', 'group_size', 'ece', 'mce', 'brier', 'accuracy'].

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:
        {'max_ece_gap': float, 'max_brier_gap': float, 'worst_group': Any, 'best_group': Any}.
        Returns zeros and None if the DataFrame is empty or contains no valid rows.
    """
    default_result: Dict[str, Any] = {
        "max_ece_gap": 0.0,
        "max_brier_gap": 0.0,
        "worst_group": None,
        "best_group": None,
    }

    if groupwise_metrics_df is None or groupwise_metrics_df.empty:
        logger.warning("Empty DataFrame passed to calibration_gap; returning default null gap.")
        return default_result

    # Drop rows with NaN in calibration metrics
    valid_df = groupwise_metrics_df.dropna(subset=["ece", "brier"])
    if valid_df.empty:
        logger.warning("No rows with valid ECE/Brier scores in calibration_gap.")
        return default_result

    ece_series = valid_df["ece"].astype(float)
    brier_series = valid_df["brier"].astype(float)

    max_ece_gap = float(ece_series.max() - ece_series.min())
    max_brier_gap = float(brier_series.max() - brier_series.min())

    worst_idx = ece_series.idxmax()
    best_idx = ece_series.idxmin()

    worst_group = valid_df.loc[worst_idx, "group"]
    best_group = valid_df.loc[best_idx, "group"]

    return {
        "max_ece_gap": max_ece_gap,
        "max_brier_gap": max_brier_gap,
        "worst_group": worst_group,
        "best_group": best_group,
    }
