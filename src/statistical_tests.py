"""
Statistical Significance Testing for Calibration Degradation in Compressed Models.

This module evaluates whether model compression significantly degrades calibration
performance (overall or for specific demographic groups) compared to the uncompressed
baseline (compression_level = 0.0).

Methods included:
- Paired Student's t-test across random seeds.
- Wilcoxon signed-rank test (non-parametric paired test).
- Bonferroni multiple comparison correction.
- Automated pipeline to test all (model, compression_level, group) configurations.
"""

import logging
import sys
from pathlib import Path
from typing import Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

# Ensure project root is in sys.path for config import
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config

logger = logging.getLogger(__name__)


def paired_ttest(
    baseline_scores: Union[np.ndarray, Sequence[float]],
    compressed_scores: Union[np.ndarray, Sequence[float]],
    alpha: float = config.SIGNIFICANCE_LEVEL,
) -> Tuple[float, float, bool]:
    """
    Perform a two-sided paired Student's t-test between baseline and compressed scores.

    Tests whether the mean difference between paired scores across seeds is
    significantly different from zero.

    Parameters
    ----------
    baseline_scores : Union[np.ndarray, Sequence[float]]
        Metric scores from the baseline (uncompressed) model across seeds.
    compressed_scores : Union[np.ndarray, Sequence[float]]
        Metric scores from the compressed model across matching seeds.
    alpha : float, default=config.SIGNIFICANCE_LEVEL (0.05)
        Significance threshold.

    Returns
    -------
    Tuple[float, float, bool]
        (t_stat, p_value, is_significant)
        - t_stat: Computed t-statistic (positive indicates higher compressed metric).
        - p_value: Two-tailed p-value in [0.0, 1.0]. Returns 1.0 if all values are identical.
        - is_significant: True if p_value < alpha.

    Raises
    ------
    ValueError
        If lengths of `baseline_scores` and `compressed_scores` do not match.
    """
    b = np.asarray(baseline_scores, dtype=float).ravel()
    c = np.asarray(compressed_scores, dtype=float).ravel()

    if len(b) != len(c):
        raise ValueError(
            f"Length mismatch: len(baseline_scores)={len(b)} vs len(compressed_scores)={len(c)}"
        )

    # Edge cases: fewer than 2 pairs
    if len(b) < 2:
        logger.warning("Fewer than 2 pairs provided for paired_ttest; returning non-significant.")
        return 0.0, 1.0, False

    diff = c - b

    # Edge case: all values are identical or zero variance in differences
    if np.allclose(b, c) or float(np.var(diff)) < 1e-14:
        logger.debug("Identical scores or zero variance across seeds; returning p=1.0.")
        return 0.0, 1.0, False

    res = stats.ttest_rel(c, b)
    t_stat = float(res.statistic)
    p_val = float(res.pvalue)

    if np.isnan(t_stat) or np.isnan(p_val):
        logger.warning("NaN encountered during paired t-test; returning p=1.0.")
        return 0.0, 1.0, False

    is_significant = bool(p_val < alpha)
    return t_stat, p_val, is_significant


def wilcoxon_test(
    baseline_scores: Union[np.ndarray, Sequence[float]],
    compressed_scores: Union[np.ndarray, Sequence[float]],
    alpha: float = config.SIGNIFICANCE_LEVEL,
) -> Tuple[float, float, bool]:
    """
    Perform a two-sided Wilcoxon signed-rank test between baseline and compressed scores.

    Non-parametric alternative to the paired t-test for evaluating paired differences
    without assuming normality.

    Parameters
    ----------
    baseline_scores : Union[np.ndarray, Sequence[float]]
        Metric scores from the baseline (uncompressed) model across seeds.
    compressed_scores : Union[np.ndarray, Sequence[float]]
        Metric scores from the compressed model across matching seeds.
    alpha : float, default=config.SIGNIFICANCE_LEVEL (0.05)
        Significance threshold.

    Returns
    -------
    Tuple[float, float, bool]
        (stat, p_value, is_significant)
        - stat: Wilcoxon test statistic.
        - p_value: Two-tailed p-value. Returns 1.0 if all values are identical.
        - is_significant: True if p_value < alpha.

    Raises
    ------
    ValueError
        If lengths of `baseline_scores` and `compressed_scores` do not match.
    """
    b = np.asarray(baseline_scores, dtype=float).ravel()
    c = np.asarray(compressed_scores, dtype=float).ravel()

    if len(b) != len(c):
        raise ValueError(
            f"Length mismatch: len(baseline_scores)={len(b)} vs len(compressed_scores)={len(c)}"
        )

    if len(b) < 2:
        logger.warning("Fewer than 2 pairs provided for wilcoxon_test; returning non-significant.")
        return 0.0, 1.0, False

    diff = c - b

    # Edge case: all differences are zero or scores identical
    if np.allclose(b, c) or np.all(diff == 0.0):
        logger.debug("Identical scores across seeds in Wilcoxon test; returning p=1.0.")
        return 0.0, 1.0, False

    try:
        res = stats.wilcoxon(c, b, zero_method="pratt")
        stat = float(res.statistic)
        p_val = float(res.pvalue)
    except Exception as exc:
        logger.warning("Wilcoxon test failed with error: %s; returning p=1.0.", exc)
        return 0.0, 1.0, False

    if np.isnan(stat) or np.isnan(p_val):
        return 0.0, 1.0, False

    is_significant = bool(p_val < alpha)
    return stat, p_val, is_significant


def apply_bonferroni_correction(
    p_values: Union[np.ndarray, Sequence[float]],
    alpha: float = config.SIGNIFICANCE_LEVEL,
) -> Tuple[np.ndarray, float]:
    """
    Apply Bonferroni multiple testing correction to a collection of p-values.

    Adjusts significance threshold:
        alpha_adj = alpha / M

    Parameters
    ----------
    p_values : Union[np.ndarray, Sequence[float]]
        Collection of unadjusted p-values.
    alpha : float, default=config.SIGNIFICANCE_LEVEL (0.05)
        Family-wise error rate threshold.

    Returns
    -------
    Tuple[np.ndarray, float]
        (is_significant, adjusted_alpha)
        - is_significant: Boolean numpy array indicating which hypotheses reject H0.
        - adjusted_alpha: The Bonferroni-adjusted alpha threshold.
    """
    p_arr = np.asarray(p_values, dtype=float)
    n_tests = len(p_arr)

    if n_tests == 0:
        return np.array([], dtype=bool), float(alpha)

    adjusted_alpha = float(alpha / n_tests)
    is_significant = (p_arr <= adjusted_alpha) & ~np.isnan(p_arr)
    return is_significant, adjusted_alpha


def run_significance_tests(
    results_df: pd.DataFrame,
    metric: str = "ece",
    alpha: float = config.SIGNIFICANCE_LEVEL,
) -> pd.DataFrame:
    """
    Execute paired significance tests against baseline for each model, compression, and group.

    Matches seeds between baseline (compression_level=0.0) and compressed
    conditions to perform paired Student's t-test and Wilcoxon signed-rank test.
    Applies Bonferroni correction across all evaluated comparisons.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results DataFrame containing experiment runs. Expected columns:
        ['seed', 'model', 'compression_level', 'group', metric].
    metric : str, default='ece'
        The evaluation metric column to test (e.g. 'ece', 'brier', 'accuracy', 'mce').
    alpha : float, default=config.SIGNIFICANCE_LEVEL (0.05)
        Nominal significance level before multiple-testing correction.

    Returns
    -------
    pd.DataFrame
        DataFrame with rows corresponding to each (model, compression_level, group) test:
        [
            'model', 'compression_level', 'group', 'metric', 'n_seeds',
            'mean_baseline', 'mean_compressed', 'mean_diff',
            'ttest_stat', 'ttest_p_value', 'ttest_significant',
            'wilcoxon_stat', 'wilcoxon_p_value', 'wilcoxon_significant',
            'bonferroni_significant', 'adjusted_alpha'
        ]

    Raises
    ------
    ValueError
        If required columns are missing from `results_df`.
    """
    output_cols = [
        "model",
        "compression_level",
        "group",
        "metric",
        "n_seeds",
        "mean_baseline",
        "mean_compressed",
        "mean_diff",
        "ttest_stat",
        "ttest_p_value",
        "ttest_significant",
        "wilcoxon_stat",
        "wilcoxon_p_value",
        "wilcoxon_significant",
        "bonferroni_significant",
        "adjusted_alpha",
    ]

    if results_df is None or results_df.empty:
        logger.warning("Empty results_df supplied to run_significance_tests.")
        base_cols = list(output_cols)
        if results_df is not None and "dataset" in results_df.columns:
            base_cols.insert(0, "dataset")
        return pd.DataFrame(columns=base_cols)

    # Standardize column naming if 'random_seed' is used instead of 'seed'
    df = results_df.copy()
    if "seed" not in df.columns and "random_seed" in df.columns:
        df["seed"] = df["random_seed"]
    if "group" not in df.columns and "group_value" in df.columns:
        df["group"] = df["group_value"]

    required_cols = {"model", "compression_level", "group", "seed", metric}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"results_df is missing required columns: {missing}")

    has_dataset = "dataset" in df.columns
    recal_col = None
    if "recalibration" in df.columns:
        recal_col = "recalibration"
    elif "recalibration_method" in df.columns:
        recal_col = "recalibration_method"

    output_cols = []
    if has_dataset:
        output_cols.append("dataset")
    output_cols.extend([
        "model",
        "compression_level",
        "group",
    ])
    if recal_col:
        output_cols.append(recal_col)
    output_cols.extend([
        "metric",
        "n_seeds",
        "mean_baseline",
        "mean_compressed",
        "mean_diff",
        "ttest_stat",
        "ttest_p_value",
        "ttest_significant",
        "wilcoxon_stat",
        "wilcoxon_p_value",
        "wilcoxon_significant",
        "bonferroni_significant",
        "adjusted_alpha",
    ])

    # Baseline has compression_level == 0.0
    baseline_df = df[np.isclose(df["compression_level"].astype(float), 0.0)]
    if baseline_df.empty:
        logger.warning("No baseline observations with compression_level=0.0 found.")
        return pd.DataFrame(columns=output_cols)

    datasets = df["dataset"].unique() if has_dataset else [None]

    records = []

    for dataset in datasets:
        ds_subset = df[df["dataset"] == dataset] if has_dataset else df
        ds_baseline = baseline_df[baseline_df["dataset"] == dataset] if has_dataset else baseline_df

        models = ds_subset["model"].unique()
        for mdl in models:
            m_subset = ds_subset[ds_subset["model"] == mdl]
            m_baseline = ds_baseline[ds_baseline["model"] == mdl]

            groups = m_subset["group"].unique()
            for grp in groups:
                g_subset = m_subset[m_subset["group"] == grp]
                g_baseline = m_baseline[m_baseline["group"] == grp]

                recal_methods = (
                    g_subset[recal_col].unique() if recal_col else [None]
                )

                for recal in recal_methods:
                    if recal_col:
                        b_subset = g_baseline[g_baseline[recal_col] == recal]
                        r_subset = g_subset[g_subset[recal_col] == recal]
                    else:
                        b_subset = g_baseline
                        r_subset = g_subset

                    if b_subset.empty:
                        continue

                    # Map seed -> baseline score
                    b_map = dict(zip(b_subset["seed"], b_subset[metric].astype(float)))

                    all_comp_levels = np.sort(r_subset["compression_level"].astype(float).unique())
                    compressed_levels = [lvl for lvl in all_comp_levels if not np.isclose(lvl, 0.0)]

                    for comp_lvl in compressed_levels:
                        c_subset = r_subset[np.isclose(r_subset["compression_level"].astype(float), comp_lvl)]
                        if c_subset.empty:
                            continue

                        c_map = dict(zip(c_subset["seed"], c_subset[metric].astype(float)))

                        # Intersect seeds to guarantee strictly paired observations
                        common_seeds = sorted(set(b_map.keys()).intersection(set(c_map.keys())))
                        if len(common_seeds) == 0:
                            logger.warning(
                                "No matching seeds for dataset=%s, model=%s, group=%s, comp_level=%.2f",
                                dataset,
                                mdl,
                                grp,
                                comp_lvl,
                            )
                            continue

                        b_scores = [b_map[s] for s in common_seeds]
                        c_scores = [c_map[s] for s in common_seeds]

                        t_stat, t_pval, t_sig = paired_ttest(b_scores, c_scores, alpha=alpha)
                        w_stat, w_pval, w_sig = wilcoxon_test(b_scores, c_scores, alpha=alpha)

                        mean_b = float(np.mean(b_scores))
                        mean_c = float(np.mean(c_scores))
                        mean_d = float(mean_c - mean_b)

                        record = {}
                        if has_dataset:
                            record["dataset"] = dataset
                        record["model"] = mdl
                        record["compression_level"] = comp_lvl
                        record["group"] = grp
                        if recal_col:
                            record[recal_col] = recal
                        record.update({
                            "metric": metric,
                            "n_seeds": len(common_seeds),
                            "mean_baseline": mean_b,
                            "mean_compressed": mean_c,
                            "mean_diff": mean_d,
                            "ttest_stat": t_stat,
                            "ttest_p_value": t_pval,
                            "ttest_significant": t_sig,
                            "wilcoxon_stat": w_stat,
                            "wilcoxon_p_value": w_pval,
                            "wilcoxon_significant": w_sig,
                        })

                        records.append(record)

    if not records:
        logger.warning("No testable comparisons could be constructed from results_df.")
        return pd.DataFrame(columns=output_cols)

    test_results_df = pd.DataFrame(records)

    # Apply Bonferroni correction over all performed paired t-tests
    bonf_sig, adj_alpha = apply_bonferroni_correction(
        test_results_df["ttest_p_value"].values,
        alpha=alpha,
    )
    test_results_df["bonferroni_significant"] = bonf_sig
    test_results_df["adjusted_alpha"] = adj_alpha

    return test_results_df[output_cols]
