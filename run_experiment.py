#!/usr/bin/env python3
"""
run_experiment.py — Main experiment orchestrator.

Runs the full experiment loop:
  for dataset in DATASETS:
      for seed in SEEDS:
          preprocess(dataset, seed)
          for model_class in [RF, GB, NN]:
              train baseline model
              for compression_level in COMPRESSION_LEVELS:
                  compress model
                  evaluate overall + group-wise calibration
                  apply recalibration (Platt, Isotonic)
                  re-evaluate calibration
                  store all results

All results are saved as structured CSVs in config.RESULTS_DIR.

Usage:
    # Full experiment (all datasets, all seeds)
    python run_experiment.py

    # Single dataset, single seed (for quick testing)
    python run_experiment.py --dataset adult --seed 42

    # Specific datasets
    python run_experiment.py --dataset adult --dataset german_credit
"""

import sys
import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.data_loader import load_dataset
from src.preprocessing import preprocess_dataset
from src.calibration_metrics import compute_overall_metrics, compute_groupwise_metrics
from src.recalibration import recalibrate_all_methods, recalibrate_groupwise
from src.efficiency import compute_efficiency_metrics
from src.explainability import compute_shap_values
from models.random_forest_model import RandomForestModelWrapper
from models.gradient_boosting_model import GradientBoostingModelWrapper
from models.neural_network_model import NeuralNetworkModelWrapper

# ──────────────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────────────
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────────────────────────────────
def create_model(model_name: str, input_dim: int, seed: int):
    """Instantiate a model wrapper by name.

    Parameters
    ----------
    model_name : str
        One of 'RandomForest', 'GradientBoosting', 'NeuralNetwork'.
    input_dim : int
        Number of input features (needed for NN architecture).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    BaseModelWrapper
        An untrained model wrapper.
    """
    if model_name == "RandomForest":
        return RandomForestModelWrapper(seed=seed)
    elif model_name == "GradientBoosting":
        return GradientBoostingModelWrapper(seed=seed)
    elif model_name == "NeuralNetwork":
        return NeuralNetworkModelWrapper(input_dim=input_dim, seed=seed)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ──────────────────────────────────────────────────────────────────────
# Single run (one model × one dataset × one seed)
# ──────────────────────────────────────────────────────────────────────
def run_single_experiment(
    dataset_key: str,
    model_name: str,
    seed: int,
    save_shap: bool = True,
    nn_mode: str = "prune+quantize",
) -> List[Dict[str, Any]]:
    """Run one model on one dataset with one seed across all compression levels.

    Parameters
    ----------
    dataset_key : str
        Key into config.DATASETS (e.g., 'adult').
    model_name : str
        One of config.MODEL_NAMES.
    seed : int
        Random seed.
    save_shap : bool
        Whether to compute and save SHAP values (slower).
    nn_mode : str
        Compression mode for NeuralNetwork ('prune+quantize', 'prune_only', 'quantize_only', 'fp16').

    Returns
    -------
    List[Dict[str, Any]]
        List of result rows, one per (compression_level, recalibration_method, group).
    """
    dataset_config = config.DATASETS[dataset_key]
    logger.info(
        f"Running: dataset={dataset_key}, model={model_name}, seed={seed}"
    )

    # ── Load and preprocess ──────────────────────────────────────────
    df = load_dataset(dataset_key)
    data = preprocess_dataset(df, dataset_key, seed)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_cal = data["X_cal"]
    y_cal = data["y_cal"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    groups_test = data["groups_test"]
    groups_cal = data["groups_cal"]
    feature_names = data["feature_names"]

    input_dim = X_train.shape[1]

    # ── Train baseline model ─────────────────────────────────────────
    logger.info(f"  Training baseline {model_name}...")
    baseline_model = create_model(model_name, input_dim, seed)
    if model_name == "NeuralNetwork":
        baseline_model.train(X_train, y_train)  # NN creates internal val split
    else:
        baseline_model.train(X_train, y_train, X_val=X_cal, y_val=y_cal)

    results: List[Dict[str, Any]] = []

    # ── Loop over compression levels ─────────────────────────────────
    for level in config.COMPRESSION_LEVELS:
        logger.info(f"  Compression level: {level:.0%}")

        if level == 0.0:
            current_model = baseline_model
        else:
            if model_name == "NeuralNetwork":
                current_model = baseline_model.compress(level, mode=nn_mode)
            else:
                current_model = baseline_model.compress(level)

        # Efficiency metrics
        eff_metrics = compute_efficiency_metrics(
            current_model, X_test, current_model.model_type
        )

        # Get predicted probabilities
        prob_test = current_model.predict_proba(X_test)
        prob_cal = current_model.predict_proba(X_cal)

        # ── Global Recalibration variants ────────────────────────────
        recalibrated_probs = recalibrate_all_methods(y_cal, prob_cal, prob_test)

        for recal_method, probs in recalibrated_probs.items():
            # Overall metrics
            overall = compute_overall_metrics(y_test, probs)
            results.append({
                "dataset": dataset_key,
                "model": model_name,
                "seed": seed,
                "compression_level": level,
                "recalibration": recal_method,
                "group_attribute": "overall",
                "group_value": "all",
                "group_size": len(y_test),
                **eff_metrics,
                **overall,
            })

            # Group-wise metrics for each sensitive attribute
            for attr_name, group_labels in groups_test.items():
                gw_df = compute_groupwise_metrics(y_test, probs, group_labels)
                for _, row in gw_df.iterrows():
                    results.append({
                        "dataset": dataset_key,
                        "model": model_name,
                        "seed": seed,
                        "compression_level": level,
                        "recalibration": recal_method,
                        "group_attribute": attr_name,
                        "group_value": row["group"],
                        "group_size": row["group_size"],
                        "ece": row["ece"],
                        "mce": row["mce"],
                        "brier": row["brier"],
                        "log_loss": row["log_loss"],
                        "accuracy": row["accuracy"],
                        **eff_metrics,
                    })

        # ── Group-Specific Recalibration variants ────────────────────
        for attr_name, group_labels in groups_test.items():
            for method in ["platt", "isotonic"]:
                try:
                    probs_gw = recalibrate_groupwise(
                        y_cal=y_cal,
                        prob_cal=prob_cal,
                        prob_test=prob_test,
                        groups_cal=groups_cal[attr_name],
                        groups_test=group_labels,
                        method=method,
                    )
                except Exception as exc:
                    logger.warning(
                        "Groupwise recalibration (%s, %s) failed: %s",
                        method,
                        attr_name,
                        exc,
                    )
                    continue

                gw_recal_method = f"{method}_groupwise"
                overall_gw = compute_overall_metrics(y_test, probs_gw)
                results.append({
                    "dataset": dataset_key,
                    "model": model_name,
                    "seed": seed,
                    "compression_level": level,
                    "recalibration": gw_recal_method,
                    "group_attribute": "overall",
                    "group_value": "all",
                    "group_size": len(y_test),
                    **eff_metrics,
                    **overall_gw,
                })

                gw_df = compute_groupwise_metrics(y_test, probs_gw, group_labels)
                for _, row in gw_df.iterrows():
                    results.append({
                        "dataset": dataset_key,
                        "model": model_name,
                        "seed": seed,
                        "compression_level": level,
                        "recalibration": gw_recal_method,
                        "group_attribute": attr_name,
                        "group_value": row["group"],
                        "group_size": row["group_size"],
                        "ece": row["ece"],
                        "mce": row["mce"],
                        "brier": row["brier"],
                        "log_loss": row["log_loss"],
                        "accuracy": row["accuracy"],
                        **eff_metrics,
                    })

        # ── SHAP explainability (baseline and max compression only) ──
        if save_shap and level in [0.0, config.COMPRESSION_LEVELS[-1]]:
            try:
                _save_shap_values(
                    current_model, X_test, feature_names,
                    dataset_key, model_name, level, seed,
                )
            except Exception as e:
                logger.warning(f"  SHAP computation failed: {e}")

    return results


def _save_shap_values(
    model, X_test, feature_names,
    dataset_key, model_name, level, seed,
):
    """Compute and save SHAP values for a model snapshot."""
    shap_dir = config.RESULTS_DIR / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)

    level_str = f"{int(level * 100)}"
    filename = f"shap_{dataset_key}_{model_name}_comp{level_str}_seed{seed}.npz"

    # Use a subsample for speed
    n_samples = min(500, X_test.shape[0])
    X_sample = X_test[:n_samples]

    shap_vals = compute_shap_values(
        model.get_model(), X_sample, feature_names,
        model_type=model.model_type,
    )

    if shap_vals is not None:
        np.savez(
            shap_dir / filename,
            shap_values=shap_vals.values if hasattr(shap_vals, "values") else shap_vals,
            feature_names=feature_names,
            X=X_sample,
        )
        logger.info(f"  SHAP values saved: {filename}")


# ──────────────────────────────────────────────────────────────────────
# Full experiment
# ──────────────────────────────────────────────────────────────────────
def run_full_experiment(
    dataset_keys: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    model_names: Optional[List[str]] = None,
    save_shap: bool = True,
    nn_mode: str = "prune+quantize",
) -> pd.DataFrame:
    """Run the complete experiment across all specified combinations.

    Parameters
    ----------
    dataset_keys : list of str, optional
        Which datasets to run. Defaults to all in config.DATASETS.
    seeds : list of int, optional
        Which seeds to use. Defaults to config.RANDOM_SEEDS.
    model_names : list of str, optional
        Which models to train. Defaults to config.MODEL_NAMES.
    save_shap : bool
        Whether to compute SHAP values.
    nn_mode : str
        Compression mode for NeuralNetwork ('prune+quantize', 'prune_only', 'quantize_only', 'fp16').

    Returns
    -------
    pd.DataFrame
        All results in a single tidy DataFrame.
    """
    if dataset_keys is None:
        dataset_keys = list(config.DATASETS.keys())
    if seeds is None:
        seeds = config.RANDOM_SEEDS
    if model_names is None:
        model_names = config.MODEL_NAMES

    all_results: List[Dict[str, Any]] = []
    total_runs = len(dataset_keys) * len(seeds) * len(model_names)
    run_idx = 0

    logger.info("=" * 70)
    logger.info("EXPERIMENT START")
    logger.info(f"  Datasets:    {dataset_keys}")
    logger.info(f"  Seeds:       {seeds}")
    logger.info(f"  Models:      {model_names}")
    logger.info(f"  Compression: {config.COMPRESSION_LEVELS}")
    logger.info(f"  NN Mode:     {nn_mode}")
    logger.info(f"  Total runs:  {total_runs}")
    logger.info("=" * 70)

    start_time = time.time()

    for dataset_key in dataset_keys:
        dataset_results: List[Dict[str, Any]] = []

        for seed in seeds:
            for model_name in model_names:
                run_idx += 1
                logger.info(
                    f"\n[{run_idx}/{total_runs}] "
                    f"dataset={dataset_key}, model={model_name}, seed={seed}"
                )

                try:
                    results = run_single_experiment(
                        dataset_key, model_name, seed, save_shap=save_shap, nn_mode=nn_mode
                    )
                    dataset_results.extend(results)
                    all_results.extend(results)
                except Exception as e:
                    logger.error(
                        f"  FAILED: {dataset_key}/{model_name}/seed={seed}: {e}",
                        exc_info=True,
                    )

        # Save per-dataset CSV
        if dataset_results:
            df_dataset = pd.DataFrame(dataset_results)
            csv_path = config.RESULTS_DIR / f"results_{dataset_key}.csv"
            df_dataset.to_csv(csv_path, index=False)
            logger.info(f"Saved {len(df_dataset)} rows to {csv_path}")

    # Save combined CSV
    df_all = pd.DataFrame(all_results)
    combined_path = config.RESULTS_DIR / "results_all.csv"
    df_all.to_csv(combined_path, index=False)

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info(f"EXPERIMENT COMPLETE — {len(df_all)} result rows in {elapsed:.1f}s")
    logger.info(f"Results saved to: {combined_path}")
    logger.info("=" * 70)

    return df_all


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run group-wise calibration experiments on compressed models."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=list(config.DATASETS.keys()),
        default=None,
        help="Dataset(s) to run. Repeat for multiple. Default: all.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        default=None,
        help="Seed(s) to use. Repeat for multiple. Default: all configured seeds.",
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=config.MODEL_NAMES,
        default=None,
        help="Model(s) to train. Repeat for multiple. Default: all.",
    )
    parser.add_argument(
        "--nn-mode",
        type=str,
        choices=config.NN_COMPRESSION_MODES,
        default="prune+quantize",
        help="Compression mode for NeuralNetwork. Default: prune+quantize.",
    )
    parser.add_argument(
        "--no-shap",
        action="store_true",
        help="Skip SHAP computation for faster runs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    df_results = run_full_experiment(
        dataset_keys=args.dataset,
        seeds=args.seed if args.seed else None,
        model_names=args.model,
        save_shap=not args.no_shap,
        nn_mode=args.nn_mode,
    )

    print(f"\nExperiment finished. {len(df_results)} result rows saved.")
    print(f"Results directory: {config.RESULTS_DIR}")
