#!/usr/bin/env python3
"""
results_analysis.py — Generate all tables, plots, and statistical tests.

Reads the CSV results produced by run_experiment.py and generates:

Tables:
  1. Overall accuracy & ECE per model × compression level
  2. Group-wise ECE breakdown per dataset
  3. Recalibration effectiveness (ECE before vs after)
  4. Statistical significance table

Plots:
  1. Calibration degradation curves (ECE vs compression level)
  2. Group-wise calibration gap bar charts
  3. Reliability diagrams per group
  4. Recalibration effectiveness paired bar charts
  5. Heatmap: Model × Compression × Group ECE
  6. SHAP comparison plots (if available)
  7. Statistical significance matrix

Usage:
    python results_analysis.py
    python results_analysis.py --results-file results/results_all.csv
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.statistical_tests import run_significance_tests

# ──────────────────────────────────────────────────────────────────────
# Logging & style setup
# ──────────────────────────────────────────────────────────────────────
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# Matplotlib style
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "figure.figsize": (10, 6),
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "font.family": "sans-serif",
})
sns.set_palette("colorblind")


# ══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════
def load_results(results_path: Path) -> pd.DataFrame:
    """Load and validate the results CSV."""
    df = pd.read_csv(results_path)
    required = [
        "dataset", "model", "seed", "compression_level",
        "recalibration", "group_attribute", "group_value",
        "ece", "mce", "brier", "accuracy",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in results CSV: {missing}")

    logger.info(f"Loaded {len(df)} rows from {results_path}")
    logger.info(f"  Datasets: {df['dataset'].unique().tolist()}")
    logger.info(f"  Models:   {df['model'].unique().tolist()}")
    logger.info(f"  Seeds:    {sorted(df['seed'].unique().tolist())}")
    return df


# ══════════════════════════════════════════════════════════════════════
# TABLE 1: Overall Performance Summary
# ══════════════════════════════════════════════════════════════════════
def table_overall_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Average accuracy & ECE per (dataset, model, compression_level).

    Only uses 'overall' group and 'none' recalibration.
    """
    mask = (df["group_attribute"] == "overall") & (df["recalibration"] == "none")
    subset = df[mask].copy()

    agg_dict = {
        "accuracy_mean": ("accuracy", "mean"),
        "accuracy_std": ("accuracy", "std"),
        "ece_mean": ("ece", "mean"),
        "ece_std": ("ece", "std"),
        "brier_mean": ("brier", "mean"),
        "brier_std": ("brier", "std"),
    }
    if "log_loss" in subset.columns:
        agg_dict["log_loss_mean"] = ("log_loss", "mean")
        agg_dict["log_loss_std"] = ("log_loss", "std")

    summary = (
        subset
        .groupby(["dataset", "model", "compression_level"])
        .agg(**agg_dict)
        .reset_index()
    )

    # Format for display
    summary["Accuracy"] = summary.apply(
        lambda r: f"{r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}", axis=1
    )
    summary["ECE"] = summary.apply(
        lambda r: f"{r['ece_mean']:.4f} ± {r['ece_std']:.4f}", axis=1
    )
    summary["Brier"] = summary.apply(
        lambda r: f"{r['brier_mean']:.4f} ± {r['brier_std']:.4f}", axis=1
    )
    if "log_loss_mean" in summary.columns:
        summary["LogLoss"] = summary.apply(
            lambda r: f"{r['log_loss_mean']:.4f} ± {r['log_loss_std']:.4f}", axis=1
        )

    save_path = config.RESULTS_DIR / "table_overall_performance.csv"
    summary.to_csv(save_path, index=False)
    logger.info(f"Table 1 saved: {save_path}")

    return summary


# ══════════════════════════════════════════════════════════════════════
# TABLE 2: Group-Wise ECE Breakdown
# ══════════════════════════════════════════════════════════════════════
def table_groupwise_ece(df: pd.DataFrame) -> pd.DataFrame:
    """Group-wise ECE for each (dataset, model, compression_level, group).

    Uses 'none' recalibration.
    """
    mask = (df["group_attribute"] != "overall") & (df["recalibration"] == "none")
    subset = df[mask].copy()

    summary = (
        subset
        .groupby(["dataset", "model", "compression_level", "group_attribute", "group_value"])
        .agg(
            ece_mean=("ece", "mean"),
            ece_std=("ece", "std"),
            group_size_mean=("group_size", "mean"),
        )
        .reset_index()
    )

    save_path = config.RESULTS_DIR / "table_groupwise_ece.csv"
    summary.to_csv(save_path, index=False)
    logger.info(f"Table 2 saved: {save_path}")

    return summary


# ══════════════════════════════════════════════════════════════════════
# TABLE 3: Recalibration Effectiveness
# ══════════════════════════════════════════════════════════════════════
def table_recalibration_effectiveness(df: pd.DataFrame) -> pd.DataFrame:
    """Compare ECE before and after recalibration."""
    mask = df["group_attribute"] == "overall"
    subset = df[mask].copy()

    summary = (
        subset
        .groupby(["dataset", "model", "compression_level", "recalibration"])
        .agg(ece_mean=("ece", "mean"), ece_std=("ece", "std"))
        .reset_index()
    )

    # Pivot to show none/platt/isotonic side by side
    pivot = summary.pivot_table(
        index=["dataset", "model", "compression_level"],
        columns="recalibration",
        values="ece_mean",
    ).reset_index()

    save_path = config.RESULTS_DIR / "table_recalibration_effectiveness.csv"
    pivot.to_csv(save_path, index=False)
    logger.info(f"Table 3 saved: {save_path}")

    return pivot


# ══════════════════════════════════════════════════════════════════════
# TABLE 4: Statistical Significance
# ══════════════════════════════════════════════════════════════════════
def table_statistical_significance(df: pd.DataFrame) -> pd.DataFrame:
    """Run paired t-tests and Wilcoxon tests for compression vs baseline."""
    mask = df["recalibration"] == "none"
    subset = df[mask].copy()

    # Create combined group column expected by run_significance_tests
    subset["group"] = subset["group_attribute"] + ":" + subset["group_value"].astype(str)

    sig_results = run_significance_tests(subset, metric="ece")

    save_path = config.RESULTS_DIR / "table_statistical_significance.csv"
    sig_results.to_csv(save_path, index=False)
    logger.info(f"Table 4 saved: {save_path}")

    return sig_results


# ══════════════════════════════════════════════════════════════════════
# PLOT 1: Calibration Degradation Curves
# ══════════════════════════════════════════════════════════════════════
def plot_calibration_degradation(df: pd.DataFrame) -> None:
    """ECE vs compression level, one line per model, faceted by dataset."""
    mask = (df["group_attribute"] == "overall") & (df["recalibration"] == "none")
    subset = df[mask].copy()

    summary = (
        subset
        .groupby(["dataset", "model", "compression_level"])
        .agg(ece_mean=("ece", "mean"), ece_std=("ece", "std"))
        .reset_index()
    )

    datasets = summary["dataset"].unique()
    n_cols = 2
    n_rows = (len(datasets) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows), squeeze=False)

    for idx, dataset in enumerate(datasets):
        ax = axes[idx // n_cols][idx % n_cols]
        data = summary[summary["dataset"] == dataset]

        for model_name in data["model"].unique():
            model_data = data[data["model"] == model_name]
            ax.errorbar(
                model_data["compression_level"] * 100,
                model_data["ece_mean"],
                yerr=model_data["ece_std"],
                marker="o", capsize=4, linewidth=2,
                label=model_name,
            )

        ax.set_title(config.DATASETS[dataset].name, fontweight="bold")
        ax.set_xlabel("Compression Level (%)")
        ax.set_ylabel("Expected Calibration Error (ECE)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))

    # Hide unused subplots
    for idx in range(len(datasets), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle("Calibration Degradation Under Compression", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = config.PLOTS_DIR / "plot1_calibration_degradation.png"
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Plot 1 saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════
# PLOT 2: Group-Wise Calibration Gap
# ══════════════════════════════════════════════════════════════════════
def plot_groupwise_calibration_gap(df: pd.DataFrame) -> None:
    """Bar chart of ECE per group, before/after compression, faceted by model."""
    mask = (df["group_attribute"] != "overall") & (df["recalibration"] == "none")
    subset = df[mask].copy()

    # Compare baseline (0.0) vs max compression
    max_comp = max(config.COMPRESSION_LEVELS)
    compare_mask = subset["compression_level"].isin([0.0, max_comp])
    compare_data = subset[compare_mask].copy()

    compare_data["Compression"] = compare_data["compression_level"].map(
        {0.0: "Baseline", max_comp: f"{int(max_comp*100)}% Compressed"}
    )

    for dataset in compare_data["dataset"].unique():
        ds_data = compare_data[compare_data["dataset"] == dataset]
        ds_config = config.DATASETS[dataset]

        for attr in ds_data["group_attribute"].unique():
            attr_data = ds_data[ds_data["group_attribute"] == attr]

            summary = (
                attr_data
                .groupby(["model", "group_value", "Compression"])
                .agg(ece_mean=("ece", "mean"), ece_std=("ece", "std"))
                .reset_index()
            )

            fig, axes = plt.subplots(
                1, len(summary["model"].unique()),
                figsize=(6 * len(summary["model"].unique()), 5),
                squeeze=False,
            )

            for i, model_name in enumerate(sorted(summary["model"].unique())):
                ax = axes[0][i]
                model_data = summary[summary["model"] == model_name]

                groups = sorted(model_data["group_value"].unique())
                x = np.arange(len(groups))
                width = 0.35

                for j, comp_label in enumerate(["Baseline", f"{int(max_comp*100)}% Compressed"]):
                    comp_data = model_data[model_data["Compression"] == comp_label]
                    comp_data = comp_data.set_index("group_value").reindex(groups)
                    bars = ax.bar(
                        x + j * width, comp_data["ece_mean"], width,
                        yerr=comp_data["ece_std"], capsize=3,
                        label=comp_label, alpha=0.85,
                    )

                ax.set_title(model_name, fontweight="bold")
                ax.set_xlabel(f"Group ({attr})")
                ax.set_ylabel("ECE")
                ax.set_xticks(x + width / 2)
                ax.set_xticklabels(groups, rotation=45, ha="right")
                ax.legend()
                ax.grid(True, alpha=0.3, axis="y")

            fig.suptitle(
                f"Group-Wise ECE — {ds_config.name} ({attr})",
                fontsize=14, fontweight="bold",
            )
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            save_path = config.PLOTS_DIR / f"plot2_groupwise_gap_{dataset}_{attr}.png"
            fig.savefig(save_path)
            plt.close(fig)
            logger.info(f"Plot 2 saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════
# PLOT 3: Reliability Diagrams
# ══════════════════════════════════════════════════════════════════════
def plot_reliability_diagrams(df: pd.DataFrame) -> None:
    """Reliability diagrams are generated during run_experiment.

    Here we create a summary version showing baseline vs compressed
    calibration curves using the raw probabilities saved by the experiment.

    Since we don't have raw probabilities in the CSV (only summary metrics),
    we create a visual summary of ECE components instead.
    """
    # This plot uses the bin-level data which would need raw predictions.
    # Instead, create a visual summary of how calibration metrics shift.
    mask = (df["group_attribute"] == "overall") & (df["recalibration"] == "none")
    subset = df[mask].copy()

    summary = (
        subset
        .groupby(["dataset", "model", "compression_level"])
        .agg(
            ece_mean=("ece", "mean"),
            mce_mean=("mce", "mean"),
            brier_mean=("brier", "mean"),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [("ece_mean", "ECE"), ("mce_mean", "MCE"), ("brier_mean", "Brier Score")]

    for ax, (col, label) in zip(axes, metrics):
        for model_name in summary["model"].unique():
            model_data = summary[summary["model"] == model_name]
            # Average across datasets
            avg = model_data.groupby("compression_level")[col].mean()
            ax.plot(
                avg.index * 100, avg.values,
                marker="o", linewidth=2, label=model_name,
            )

        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Compression Level (%)")
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Calibration Metrics Under Compression (Averaged Across Datasets)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    save_path = config.PLOTS_DIR / "plot3_calibration_metrics_summary.png"
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Plot 3 saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════
# PLOT 4: Recalibration Effectiveness
# ══════════════════════════════════════════════════════════════════════
def plot_recalibration_effectiveness(df: pd.DataFrame) -> None:
    """Paired bar chart: ECE before/after Platt & Isotonic recalibration."""
    mask = df["group_attribute"] == "overall"
    subset = df[mask].copy()

    summary = (
        subset
        .groupby(["dataset", "model", "compression_level", "recalibration"])
        .agg(ece_mean=("ece", "mean"))
        .reset_index()
    )

    # Focus on max compression level
    max_comp = max(config.COMPRESSION_LEVELS)
    comp_data = summary[summary["compression_level"] == max_comp].copy()

    fig, axes = plt.subplots(
        1, len(comp_data["dataset"].unique()),
        figsize=(5 * len(comp_data["dataset"].unique()), 5),
        squeeze=False,
    )

    for i, dataset in enumerate(sorted(comp_data["dataset"].unique())):
        ax = axes[0][i]
        ds_data = comp_data[comp_data["dataset"] == dataset]
        ds_config = config.DATASETS[dataset]

        models = sorted(ds_data["model"].unique())
        x = np.arange(len(models))
        width = 0.25

        for j, method in enumerate(["none", "platt", "isotonic"]):
            method_data = ds_data[ds_data["recalibration"] == method]
            method_data = method_data.set_index("model").reindex(models)
            label = {"none": "No Recalibration", "platt": "Platt Scaling", "isotonic": "Isotonic"}[method]
            ax.bar(
                x + j * width, method_data["ece_mean"], width,
                label=label, alpha=0.85,
            )

        ax.set_title(f"{ds_config.name}\n(Compression: {int(max_comp*100)}%)", fontweight="bold")
        ax.set_xlabel("Model")
        ax.set_ylabel("ECE")
        ax.set_xticks(x + width)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Recalibration Effectiveness at Maximum Compression",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    save_path = config.PLOTS_DIR / "plot4_recalibration_effectiveness.png"
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Plot 4 saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════
# PLOT 5: Heatmap — Model × Compression × Group ECE
# ══════════════════════════════════════════════════════════════════════
def plot_ece_heatmap(df: pd.DataFrame) -> None:
    """Heatmap of ECE per (Model+Compression) × Group for each dataset."""
    mask = (df["group_attribute"] != "overall") & (df["recalibration"] == "none")
    subset = df[mask].copy()

    for dataset in subset["dataset"].unique():
        ds_data = subset[subset["dataset"] == dataset]
        ds_config = config.DATASETS[dataset]

        for attr in ds_data["group_attribute"].unique():
            attr_data = ds_data[ds_data["group_attribute"] == attr]

            summary = (
                attr_data
                .groupby(["model", "compression_level", "group_value"])
                .agg(ece_mean=("ece", "mean"))
                .reset_index()
            )

            # Create combined model+compression label
            summary["Model_Compression"] = summary.apply(
                lambda r: f"{r['model']}\n({int(r['compression_level']*100)}%)",
                axis=1,
            )

            pivot = summary.pivot_table(
                index="Model_Compression",
                columns="group_value",
                values="ece_mean",
            )

            # Sort index logically
            sort_order = []
            for model in sorted(summary["model"].unique()):
                for level in sorted(summary["compression_level"].unique()):
                    label = f"{model}\n({int(level*100)}%)"
                    if label in pivot.index:
                        sort_order.append(label)
            pivot = pivot.reindex(sort_order)

            fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 2), max(6, len(pivot) * 0.6)))
            sns.heatmap(
                pivot, annot=True, fmt=".4f", cmap="YlOrRd",
                linewidths=0.5, ax=ax,
                cbar_kws={"label": "ECE"},
            )
            ax.set_title(
                f"Group-Wise ECE Heatmap — {ds_config.name} ({attr})",
                fontweight="bold",
            )
            ax.set_xlabel("Demographic Group")
            ax.set_ylabel("Model (Compression %)")
            plt.tight_layout()

            save_path = config.PLOTS_DIR / f"plot5_heatmap_{dataset}_{attr}.png"
            fig.savefig(save_path)
            plt.close(fig)
            logger.info(f"Plot 5 saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════
# PLOT 6: SHAP Comparison
# ══════════════════════════════════════════════════════════════════════
def plot_shap_comparison() -> None:
    """Load saved SHAP values and create comparison plots."""
    shap_dir = config.RESULTS_DIR / "shap"
    if not shap_dir.exists():
        logger.info("No SHAP directory found — skipping SHAP plots.")
        return

    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed — skipping SHAP comparison plots.")
        return

    shap_files = list(shap_dir.glob("*.npz"))
    if not shap_files:
        logger.info("No SHAP files found — skipping SHAP plots.")
        return

    # Group by (dataset, model) and compare comp0 vs comp90
    import re
    from collections import defaultdict
    grouped = defaultdict(dict)
    for f in shap_files:
        # shap_{dataset}_{model}_comp{level}_seed{seed}
        match = re.match(
            r"shap_(.+?)_(RandomForest|GradientBoosting|NeuralNetwork)_(comp\d+)_(seed\d+)",
            f.stem,
        )
        if not match:
            logger.warning(f"Could not parse SHAP filename: {f.name}")
            continue
        dataset = match.group(1)
        model = match.group(2)
        comp_level = match.group(3)
        seed = match.group(4)
        key = (dataset, model, seed)
        grouped[key][comp_level] = f

    for (dataset, model, seed), files in grouped.items():
        if "comp0" in files and f"comp{int(max(config.COMPRESSION_LEVELS)*100)}" in files:
            baseline_data = np.load(files["comp0"], allow_pickle=True)
            compressed_data = np.load(files[f"comp{int(max(config.COMPRESSION_LEVELS)*100)}"], allow_pickle=True)

            feature_names = list(baseline_data["feature_names"])

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # Baseline SHAP
            ax = axes[0]
            shap_vals_base = baseline_data["shap_values"]
            mean_abs_shap = np.abs(shap_vals_base).mean(axis=0)
            sorted_idx = np.argsort(mean_abs_shap)[-15:]  # Top 15 features
            ax.barh(
                [feature_names[i] for i in sorted_idx],
                mean_abs_shap[sorted_idx],
                color=sns.color_palette()[0],
            )
            ax.set_title("Baseline (No Compression)", fontweight="bold")
            ax.set_xlabel("Mean |SHAP value|")

            # Compressed SHAP
            ax = axes[1]
            shap_vals_comp = compressed_data["shap_values"]
            mean_abs_shap = np.abs(shap_vals_comp).mean(axis=0)
            sorted_idx = np.argsort(mean_abs_shap)[-15:]
            ax.barh(
                [feature_names[i] for i in sorted_idx],
                mean_abs_shap[sorted_idx],
                color=sns.color_palette()[1],
            )
            max_comp_pct = int(max(config.COMPRESSION_LEVELS) * 100)
            ax.set_title(f"Compressed ({max_comp_pct}%)", fontweight="bold")
            ax.set_xlabel("Mean |SHAP value|")

            fig.suptitle(
                f"Feature Importance (SHAP) — {model} on {dataset} (seed={seed})",
                fontsize=13, fontweight="bold",
            )
            plt.tight_layout(rect=[0, 0, 1, 0.94])

            save_path = config.PLOTS_DIR / f"plot6_shap_{dataset}_{model}_{seed}.png"
            fig.savefig(save_path)
            plt.close(fig)
            logger.info(f"Plot 6 saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════
# PLOT 7: Statistical Significance Matrix
# ══════════════════════════════════════════════════════════════════════
def plot_significance_matrix(sig_df: pd.DataFrame) -> None:
    """Visualize p-values from statistical tests as a heatmap."""
    if sig_df.empty:
        logger.info("No significance results — skipping significance plot.")
        return

    for dataset in sig_df["dataset"].unique():
        ds_data = sig_df[sig_df["dataset"] == dataset]
        ds_config = config.DATASETS[dataset]

        # Create combined model+group label
        ds_data = ds_data.copy()
        ds_data["Model_Group"] = ds_data["model"] + " / " + ds_data["group"].astype(str)

        for test_type in ["ttest_p_value", "wilcoxon_p_value"]:
            if test_type not in ds_data.columns:
                continue

            pivot = ds_data.pivot_table(
                index="Model_Group",
                columns="compression_level",
                values=test_type,
            )

            # Remove baseline column (trivially 1.0)
            if 0.0 in pivot.columns:
                pivot = pivot.drop(columns=[0.0])

            if pivot.empty:
                continue

            fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 2.5), max(6, len(pivot) * 0.5)))

            # Custom colormap: green (not significant) to red (significant)
            mask_sig = pivot < config.SIGNIFICANCE_LEVEL

            sns.heatmap(
                pivot, annot=True, fmt=".3f",
                cmap="RdYlGn_r",
                vmin=0, vmax=0.1,
                linewidths=0.5, ax=ax,
                cbar_kws={"label": "p-value"},
            )

            # Add asterisks for significant cells
            for i in range(len(pivot)):
                for j in range(len(pivot.columns)):
                    val = pivot.iloc[i, j]
                    if not np.isnan(val) and val < config.SIGNIFICANCE_LEVEL:
                        ax.text(
                            j + 0.5, i + 0.15, "*",
                            ha="center", va="center",
                            fontsize=16, fontweight="bold", color="white",
                        )

            test_label = "Paired t-test" if "ttest" in test_type else "Wilcoxon"
            ax.set_title(
                f"Statistical Significance — {ds_config.name}\n({test_label}, α={config.SIGNIFICANCE_LEVEL})",
                fontweight="bold",
            )
            ax.set_xlabel("Compression Level")
            ax.set_ylabel("Model / Group")
            plt.tight_layout()

            test_suffix = "ttest" if "ttest" in test_type else "wilcoxon"
            save_path = config.PLOTS_DIR / f"plot7_significance_{dataset}_{test_suffix}.png"
            fig.savefig(save_path)
            plt.close(fig)
            logger.info(f"Plot 7 saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════
# PLOT 8: Accuracy vs ECE Trade-off (Bonus)
# ══════════════════════════════════════════════════════════════════════
def plot_accuracy_vs_ece(df: pd.DataFrame) -> None:
    """Scatter plot of accuracy vs ECE, colored by compression level."""
    mask = (df["group_attribute"] == "overall") & (df["recalibration"] == "none")
    subset = df[mask].copy()

    summary = (
        subset
        .groupby(["dataset", "model", "compression_level"])
        .agg(accuracy_mean=("accuracy", "mean"), ece_mean=("ece", "mean"))
        .reset_index()
    )

    fig, axes = plt.subplots(
        1, len(summary["dataset"].unique()),
        figsize=(5 * len(summary["dataset"].unique()), 5),
        squeeze=False,
    )

    for i, dataset in enumerate(sorted(summary["dataset"].unique())):
        ax = axes[0][i]
        ds_data = summary[summary["dataset"] == dataset]
        ds_config = config.DATASETS[dataset]

        for model_name in sorted(ds_data["model"].unique()):
            model_data = ds_data[ds_data["model"] == model_name]
            scatter = ax.scatter(
                model_data["ece_mean"],
                model_data["accuracy_mean"],
                c=model_data["compression_level"] * 100,
                cmap="coolwarm",
                s=80, edgecolors="black", linewidths=0.5,
                label=model_name,
                vmin=0, vmax=100,
            )

            # Connect points with arrows to show compression progression
            sorted_data = model_data.sort_values("compression_level")
            ax.plot(
                sorted_data["ece_mean"],
                sorted_data["accuracy_mean"],
                "--", alpha=0.4, linewidth=1,
            )

        ax.set_title(ds_config.name, fontweight="bold")
        ax.set_xlabel("ECE (lower is better →)")
        ax.set_ylabel("Accuracy (higher is better →)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=axes[0][-1], label="Compression %")

    fig.suptitle(
        "Accuracy vs Calibration Trade-off Under Compression",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    save_path = config.PLOTS_DIR / "plot8_accuracy_vs_ece.png"
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Plot 8 saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════
# PLOT 9: Efficiency vs Calibration Trade-off
# ══════════════════════════════════════════════════════════════════════
def plot_efficiency_vs_calibration(df: pd.DataFrame) -> None:
    """Plot Model Size / Latency vs ECE to reveal compression trade-offs."""
    if "model_size_bytes" not in df.columns and "latency_mean_ms" not in df.columns:
        logger.info("No efficiency metrics found in results — skipping Plot 9.")
        return

    mask = (df["group_attribute"] == "overall") & (df["recalibration"] == "none")
    subset = df[mask].copy()

    datasets = sorted(subset["dataset"].unique())
    fig, axes = plt.subplots(len(datasets), 2, figsize=(14, 5 * len(datasets)), squeeze=False)

    for i, dataset in enumerate(datasets):
        ds_data = subset[subset["dataset"] == dataset]
        ds_config = config.DATASETS[dataset]

        ax_size = axes[i][0]
        ax_lat = axes[i][1]

        for model_name in sorted(ds_data["model"].unique()):
            m_data = (
                ds_data[ds_data["model"] == model_name]
                .groupby("compression_level")
                .mean(numeric_only=True)
                .reset_index()
                .sort_values("compression_level")
            )

            if "model_size_bytes" in m_data.columns:
                size_kb = m_data["model_size_bytes"] / 1024.0
                ax_size.plot(
                    size_kb, m_data["ece"],
                    marker="o", linewidth=2, label=model_name
                )
                for _, row in m_data.iterrows():
                    ax_size.annotate(
                        f"{int(row['compression_level']*100)}%",
                        (row["model_size_bytes"] / 1024.0, row["ece"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8
                    )

            if "latency_mean_ms" in m_data.columns:
                ax_lat.plot(
                    m_data["latency_mean_ms"], m_data["ece"],
                    marker="s", linewidth=2, label=model_name
                )
                for _, row in m_data.iterrows():
                    ax_lat.annotate(
                        f"{int(row['compression_level']*100)}%",
                        (row["latency_mean_ms"], row["ece"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8
                    )

        ax_size.set_title(f"{ds_config.name} — Size vs ECE", fontweight="bold")
        ax_size.set_xlabel("Model Size (KB, log scale)")
        ax_size.set_xscale("log")
        ax_size.set_ylabel("Expected Calibration Error (ECE)")
        ax_size.legend()
        ax_size.grid(True, alpha=0.3)

        ax_lat.set_title(f"{ds_config.name} — Latency vs ECE", fontweight="bold")
        ax_lat.set_xlabel("Inference Latency (ms)")
        ax_lat.set_ylabel("Expected Calibration Error (ECE)")
        ax_lat.legend()
        ax_lat.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = config.PLOTS_DIR / "plot9_efficiency_vs_calibration.png"
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Plot 9 saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════
# SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════
def print_summary_report(df: pd.DataFrame, sig_df: pd.DataFrame) -> None:
    """Print a clean text summary of key findings."""
    separator = "=" * 70
    print(f"\n{separator}")
    print("  GROUP-WISE CALIBRATION IN COMPRESSED TABULAR MODELS")
    print(f"  RESULTS SUMMARY")
    print(f"{separator}\n")

    # Overall performance at baseline vs max compression
    mask = (df["group_attribute"] == "overall") & (df["recalibration"] == "none")
    overall = df[mask].copy()

    max_comp = max(config.COMPRESSION_LEVELS)

    for dataset in sorted(overall["dataset"].unique()):
        ds_config = config.DATASETS[dataset]
        print(f"  📊 {ds_config.name}")
        print(f"  {'─' * 60}")

        ds_data = overall[overall["dataset"] == dataset]

        for model in sorted(ds_data["model"].unique()):
            model_data = ds_data[ds_data["model"] == model]

            baseline = model_data[model_data["compression_level"] == 0.0]
            compressed = model_data[model_data["compression_level"] == max_comp]

            base_acc = baseline["accuracy"].mean()
            base_ece = baseline["ece"].mean()
            comp_acc = compressed["accuracy"].mean()
            comp_ece = compressed["ece"].mean()

            acc_drop = base_acc - comp_acc
            ece_increase = comp_ece - base_ece

            print(f"    {model}:")
            print(f"      Baseline     → Acc: {base_acc:.4f}, ECE: {base_ece:.4f}")
            print(f"      {int(max_comp*100)}% Compressed → Acc: {comp_acc:.4f}, ECE: {comp_ece:.4f}")
            print(f"      Δ Accuracy: {acc_drop:+.4f}, Δ ECE: {ece_increase:+.4f}")

        print()

    # Group-wise calibration gaps
    print(f"\n  📐 Group-Wise Calibration Gaps (at {int(max_comp*100)}% compression)")
    print(f"  {'─' * 60}")

    mask_group = (
        (df["group_attribute"] != "overall")
        & (df["recalibration"] == "none")
        & (df["compression_level"] == max_comp)
    )
    group_data = df[mask_group].copy()

    gap_summary = (
        group_data
        .groupby(["dataset", "model", "group_attribute"])
        .agg(max_ece=("ece", "max"), min_ece=("ece", "min"))
    )
    gap_summary["ece_gap"] = gap_summary["max_ece"] - gap_summary["min_ece"]
    gap_summary = gap_summary.reset_index().sort_values("ece_gap", ascending=False)

    for _, row in gap_summary.head(10).iterrows():
        print(
            f"    {row['dataset']}/{row['model']}/{row['group_attribute']}: "
            f"ECE gap = {row['ece_gap']:.4f}"
        )

    # Statistical significance summary
    if not sig_df.empty and "ttest_significant" in sig_df.columns:
        n_sig = sig_df["ttest_significant"].sum()
        n_total = len(sig_df)
        print(f"\n  🔬 Statistical Tests")
        print(f"  {'─' * 60}")
        print(f"    {n_sig}/{n_total} comparisons are statistically significant")
        print(f"    (paired t-test, α = {config.SIGNIFICANCE_LEVEL})")

    # Recalibration effectiveness
    print(f"\n  🔧 Recalibration Effectiveness (at {int(max_comp*100)}% compression)")
    print(f"  {'─' * 60}")

    mask_recal = (
        (df["group_attribute"] == "overall")
        & (df["compression_level"] == max_comp)
    )
    recal_data = df[mask_recal].copy()

    recal_summary = (
        recal_data
        .groupby(["dataset", "model", "recalibration"])
        .agg(ece_mean=("ece", "mean"))
        .reset_index()
    )

    for dataset in sorted(recal_summary["dataset"].unique()):
        ds_config = config.DATASETS[dataset]
        print(f"    {ds_config.name}:")
        ds_recal = recal_summary[recal_summary["dataset"] == dataset]

        for model in sorted(ds_recal["model"].unique()):
            model_recal = ds_recal[ds_recal["model"] == model]
            none_ece = model_recal[model_recal["recalibration"] == "none"]["ece_mean"].values[0]
            platt_ece = model_recal[model_recal["recalibration"] == "platt"]["ece_mean"].values[0]
            iso_ece = model_recal[model_recal["recalibration"] == "isotonic"]["ece_mean"].values[0]
            print(
                f"      {model}: None={none_ece:.4f} → "
                f"Platt={platt_ece:.4f} ({none_ece-platt_ece:+.4f}), "
                f"Isotonic={iso_ece:.4f} ({none_ece-iso_ece:+.4f})"
            )

    print(f"\n{separator}")
    print(f"  All results saved to: {config.RESULTS_DIR}")
    print(f"  All plots saved to:   {config.PLOTS_DIR}")
    print(f"{separator}\n")


# ══════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════
def main(results_path: Optional[str] = None) -> None:
    """Run the full analysis pipeline."""
    if results_path is None:
        results_path = config.RESULTS_DIR / "results_all.csv"
    else:
        results_path = Path(results_path)

    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        logger.error("Run 'python run_experiment.py' first to generate results.")
        sys.exit(1)

    df = load_results(results_path)

    # ── Generate tables ──────────────────────────────────────────────
    logger.info("\n" + "=" * 50 + " TABLES " + "=" * 50)

    table1 = table_overall_performance(df)
    logger.info(f"\nTable 1: Overall Performance\n{table1[['dataset', 'model', 'compression_level', 'Accuracy', 'ECE']].to_string(index=False)}\n")

    table2 = table_groupwise_ece(df)
    logger.info(f"Table 2: Group-wise ECE — {len(table2)} rows")

    table3 = table_recalibration_effectiveness(df)
    logger.info(f"Table 3: Recalibration — {len(table3)} rows")

    table4 = table_statistical_significance(df)
    logger.info(f"Table 4: Statistical Significance — {len(table4)} rows")

    # ── Generate plots ───────────────────────────────────────────────
    logger.info("\n" + "=" * 50 + " PLOTS " + "=" * 50)

    plot_calibration_degradation(df)
    plot_groupwise_calibration_gap(df)
    plot_reliability_diagrams(df)
    plot_recalibration_effectiveness(df)
    plot_ece_heatmap(df)
    plot_shap_comparison()
    plot_significance_matrix(table4)
    plot_accuracy_vs_ece(df)
    plot_efficiency_vs_calibration(df)

    # ── Print summary report ─────────────────────────────────────────
    print_summary_report(df, table4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze experiment results.")
    parser.add_argument(
        "--results-file",
        type=str,
        default=None,
        help="Path to results CSV. Default: results/results_all.csv",
    )
    args = parser.parse_args()
    main(args.results_file)
