"""
Post-Hoc Recalibration Methods for Tabular Models.

This module provides calibration methods using a held-out calibration set:
1. Platt Scaling: Logistic regression fitted on model predicted probabilities.
2. Isotonic Regression: Non-parametric, piecewise monotonic calibration.
3. Unified recalibration interfaces for seamless benchmarking across methods.

All outputs are guaranteed to be clipped within [0.0, 1.0] and edge cases
(empty datasets, single-class calibration sets) are handled cleanly.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# Ensure project root is in sys.path for config import
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config

logger = logging.getLogger(__name__)


def platt_scaling_fit(
    y_cal: Union[np.ndarray, Sequence[int], Sequence[float]],
    prob_cal: Union[np.ndarray, Sequence[float]],
) -> LogisticRegression:
    """
    Fit Platt scaling (logistic regression) on a held-out calibration set.

    Fits a univariate LogisticRegression estimator:
        sigma(w * p + b)
    mapping raw model predicted probabilities to empirical class posterior
    probabilities.

    Parameters
    ----------
    y_cal : Union[np.ndarray, Sequence[int], Sequence[float]]
        True binary class labels in {0, 1} from the calibration set.
    prob_cal : Union[np.ndarray, Sequence[float]]
        Model-predicted probabilities for the positive class on the calibration set.

    Returns
    -------
    LogisticRegression
        Trained logistic regression calibrator.

    Raises
    ------
    ValueError
        If inputs have mismatched lengths or are empty.
    """
    y_c = np.asarray(y_cal, dtype=int).ravel()
    p_c = np.clip(np.asarray(prob_cal, dtype=float).ravel(), 0.0, 1.0)

    if len(y_c) != len(p_c):
        raise ValueError(
            f"Length mismatch: len(y_cal)={len(y_c)} vs len(prob_cal)={len(p_c)}"
        )

    if len(y_c) == 0:
        raise ValueError("Cannot fit Platt scaling on an empty calibration set.")

    unique_classes = np.unique(y_c)

    # Edge case: calibration set contains only one class
    if len(unique_classes) < 2:
        logger.warning(
            "Calibration set contains only one class (%s). "
            "Platt scaling cannot learn class separation; creating fallback model.",
            unique_classes,
        )
        model = LogisticRegression(random_state=config.RANDOM_SEEDS[0])
        model.classes_ = np.array([0, 1])
        model.coef_ = np.zeros((1, 1), dtype=float)
        single_cls = unique_classes[0]
        # Extreme intercept: large positive for class 1, large negative for class 0
        model.intercept_ = np.array([15.0 if single_cls == 1 else -15.0], dtype=float)
        model.n_features_in_ = 1
        return model

    # Use high C for unregularized post-hoc calibration on 1D probabilities
    model = LogisticRegression(
        C=1e5,
        solver="lbfgs",
        random_state=config.RANDOM_SEEDS[0],
    )
    model.fit(p_c.reshape(-1, 1), y_c)
    return model


def platt_scaling_transform(
    model: LogisticRegression,
    prob_test: Union[np.ndarray, Sequence[float]],
) -> np.ndarray:
    """
    Apply a fitted Platt scaling calibrator to test probabilities.

    Parameters
    ----------
    model : LogisticRegression
        Fitted Platt scaling model.
    prob_test : Union[np.ndarray, Sequence[float]]
        Model-predicted probabilities to recalibrate.

    Returns
    -------
    np.ndarray
        Recalibrated probabilities in [0.0, 1.0], 1D array.
    """
    p_t = np.clip(np.asarray(prob_test, dtype=float).ravel(), 0.0, 1.0)
    if len(p_t) == 0:
        return np.array([], dtype=float)

    proba = model.predict_proba(p_t.reshape(-1, 1))

    # Identify positive class index
    if hasattr(model, "classes_") and 1 in model.classes_:
        pos_idx = int(np.where(model.classes_ == 1)[0][0])
    else:
        pos_idx = proba.shape[1] - 1

    calibrated_prob = proba[:, pos_idx]
    return np.clip(calibrated_prob, 0.0, 1.0).ravel()


def isotonic_fit(
    y_cal: Union[np.ndarray, Sequence[int], Sequence[float]],
    prob_cal: Union[np.ndarray, Sequence[float]],
) -> IsotonicRegression:
    """
    Fit an Isotonic Regression calibrator on a held-out calibration set.

    Fits a non-decreasing step function mapping raw predicted probabilities to
    empirical targets, clipping predictions out of bounds.

    Parameters
    ----------
    y_cal : Union[np.ndarray, Sequence[int], Sequence[float]]
        True binary class labels in {0, 1} from the calibration set.
    prob_cal : Union[np.ndarray, Sequence[float]]
        Model-predicted probabilities for the positive class on the calibration set.

    Returns
    -------
    IsotonicRegression
        Trained isotonic regression calibrator.

    Raises
    ------
    ValueError
        If inputs have mismatched lengths or are empty.
    """
    y_c = np.asarray(y_cal, dtype=float).ravel()
    p_c = np.clip(np.asarray(prob_cal, dtype=float).ravel(), 0.0, 1.0)

    if len(y_c) != len(p_c):
        raise ValueError(
            f"Length mismatch: len(y_cal)={len(y_c)} vs len(prob_cal)={len(p_c)}"
        )

    if len(y_c) == 0:
        raise ValueError("Cannot fit Isotonic Regression on an empty calibration set.")

    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(p_c, y_c)
    return model


def isotonic_transform(
    model: IsotonicRegression,
    prob_test: Union[np.ndarray, Sequence[float]],
) -> np.ndarray:
    """
    Apply a fitted Isotonic Regression calibrator to test probabilities.

    Parameters
    ----------
    model : IsotonicRegression
        Fitted isotonic regression model.
    prob_test : Union[np.ndarray, Sequence[float]]
        Model-predicted probabilities to recalibrate.

    Returns
    -------
    np.ndarray
        Recalibrated probabilities in [0.0, 1.0], 1D array.
    """
    p_t = np.clip(np.asarray(prob_test, dtype=float).ravel(), 0.0, 1.0)
    if len(p_t) == 0:
        return np.array([], dtype=float)

    calibrated_prob = model.predict(p_t)
    return np.clip(calibrated_prob, 0.0, 1.0).ravel()


def recalibrate(
    y_cal: Union[np.ndarray, Sequence[int], Sequence[float]],
    prob_cal: Union[np.ndarray, Sequence[float]],
    prob_test: Union[np.ndarray, Sequence[float]],
    method: str = "platt",
) -> np.ndarray:
    """
    Unified interface to apply post-hoc calibration to test probabilities.

    Supported methods:
    - 'none': Identity mapping (clips inputs to [0.0, 1.0]).
    - 'platt': Fits and applies logistic calibration.
    - 'isotonic': Fits and applies isotonic regression calibration.

    Parameters
    ----------
    y_cal : Union[np.ndarray, Sequence[int], Sequence[float]]
        Ground-truth labels from the held-out calibration set.
    prob_cal : Union[np.ndarray, Sequence[float]]
        Model-predicted probabilities on the held-out calibration set.
    prob_test : Union[np.ndarray, Sequence[float]]
        Model-predicted probabilities on the test set.
    method : str, default='platt'
        Recalibration method to use. Must be in config.RECALIBRATION_METHODS.

    Returns
    -------
    np.ndarray
        Recalibrated test set probabilities in [0.0, 1.0].

    Raises
    ------
    ValueError
        If `method` is unrecognized or calibration inputs are invalid.
    """
    method_key = method.strip().lower()

    if method_key not in config.RECALIBRATION_METHODS:
        raise ValueError(
            f"Unsupported calibration method '{method}'. "
            f"Supported methods are: {config.RECALIBRATION_METHODS}"
        )

    p_test = np.clip(np.asarray(prob_test, dtype=float).ravel(), 0.0, 1.0)

    if method_key == "none":
        return p_test

    if method_key == "platt":
        model = platt_scaling_fit(y_cal, prob_cal)
        return platt_scaling_transform(model, p_test)

    if method_key == "isotonic":
        model = isotonic_fit(y_cal, prob_cal)
        return isotonic_transform(model, p_test)

    raise ValueError(f"Unhandled method: {method_key}")


def recalibrate_all_methods(
    y_cal: Union[np.ndarray, Sequence[int], Sequence[float]],
    prob_cal: Union[np.ndarray, Sequence[float]],
    prob_test: Union[np.ndarray, Sequence[float]],
) -> Dict[str, np.ndarray]:
    """
    Recalibrate test probabilities using all configured methods.

    Returns predictions for 'none', 'platt', and 'isotonic'.

    Parameters
    ----------
    y_cal : Union[np.ndarray, Sequence[int], Sequence[float]]
        Ground-truth labels from the held-out calibration set.
    prob_cal : Union[np.ndarray, Sequence[float]]
        Model-predicted probabilities on the calibration set.
    prob_test : Union[np.ndarray, Sequence[float]]
        Model-predicted probabilities on the test set.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping method name to recalibrated test probabilities.
        Keys match config.RECALIBRATION_METHODS: {'none', 'platt', 'isotonic'}.
    """
    results: Dict[str, np.ndarray] = {}

    for method in config.RECALIBRATION_METHODS:
        try:
            results[method] = recalibrate(
                y_cal=y_cal,
                prob_cal=prob_cal,
                prob_test=prob_test,
                method=method,
            )
        except Exception as exc:
            logger.error("Failed recalibration with method '%s': %s", method, exc)
            # Fall back to uncalibrated probabilities in case of unexpected errors
            results[method] = np.clip(np.asarray(prob_test, dtype=float).ravel(), 0.0, 1.0)

    return results


def recalibrate_groupwise(
    y_cal: Union[np.ndarray, Sequence[int], Sequence[float]],
    prob_cal: Union[np.ndarray, Sequence[float]],
    prob_test: Union[np.ndarray, Sequence[float]],
    groups_cal: Union[np.ndarray, Sequence[Any]],
    groups_test: Union[np.ndarray, Sequence[Any]],
    method: str = "platt",
) -> np.ndarray:
    """
    Apply group-specific post-hoc calibration to test probabilities.

    Fits an independent calibrator per demographic group using calibration data
    belonging to that group. Falls back to a global calibrator for groups with
    insufficient sample size (<10) or single-class labels in the calibration set.

    Parameters
    ----------
    y_cal : Union[np.ndarray, Sequence[int], Sequence[float]]
        Ground-truth labels from the held-out calibration set.
    prob_cal : Union[np.ndarray, Sequence[float]]
        Model-predicted probabilities on the held-out calibration set.
    prob_test : Union[np.ndarray, Sequence[float]]
        Model-predicted probabilities on the test set.
    groups_cal : Union[np.ndarray, Sequence[Any]]
        Demographic group indicators for the calibration set.
    groups_test : Union[np.ndarray, Sequence[Any]]
        Demographic group indicators for the test set.
    method : str, default='platt'
        Recalibration method to use ('none', 'platt', 'isotonic').

    Returns
    -------
    np.ndarray
        Recalibrated test set probabilities.
    """
    if method == "none":
        return np.clip(np.asarray(prob_test, dtype=float).ravel(), 0.0, 1.0)

    y_c = np.asarray(y_cal, dtype=int).ravel()
    p_c = np.clip(np.asarray(prob_cal, dtype=float).ravel(), 0.0, 1.0)
    p_t = np.clip(np.asarray(prob_test, dtype=float).ravel(), 0.0, 1.0)
    g_c = np.asarray(groups_cal).ravel()
    g_t = np.asarray(groups_test).ravel()

    # Pre-fit global fallback calibrator
    global_calibrated = recalibrate(y_c, p_c, p_t, method=method)
    output_probs = global_calibrated.copy()

    unique_test_groups = pd.unique(g_t)
    for grp in unique_test_groups:
        mask_t = (g_t == grp)
        mask_c = (g_c == grp)

        n_cal_samples = int(np.sum(mask_c))
        # If group has fewer than 10 calibration samples or single class, fallback to global
        if n_cal_samples < 10 or len(np.unique(y_c[mask_c])) < 2:
            logger.info(
                "Group '%s' has %d calibration samples; using global calibrator fallback.",
                grp,
                n_cal_samples,
            )
            continue

        try:
            grp_calibrated = recalibrate(
                y_cal=y_c[mask_c],
                prob_cal=p_c[mask_c],
                prob_test=p_t[mask_t],
                method=method,
            )
            output_probs[mask_t] = grp_calibrated
        except Exception as exc:
            logger.warning(
                "Group-specific recalibration failed for group '%s': %s. Retaining global fallback.",
                grp,
                exc,
            )

    return output_probs
