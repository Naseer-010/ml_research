"""
Global configuration for the Group-Wise Calibration experiment.

All hyperparameters, paths, dataset metadata, and compression levels are
centralised here so that every module reads from a single source of truth.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_SAVE_DIR = RESULTS_DIR / "saved_models"

# Create directories if they don't exist
for _dir in [DATA_DIR, RESULTS_DIR, PLOTS_DIR, MODELS_SAVE_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────
RANDOM_SEEDS: List[int] = [42, 123, 456, 789, 1010, 2024, 3141, 4567, 5678, 7890]

# ──────────────────────────────────────────────────────────────────────
# Data splitting ratios
# ──────────────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.60
CALIBRATION_RATIO = 0.20  # held-out set for post-hoc recalibration
TEST_RATIO = 0.20

# ──────────────────────────────────────────────────────────────────────
# Compression levels (fraction of capacity *removed*)
# 0.0 = baseline (no compression), 0.9 = 90 % removed
# ──────────────────────────────────────────────────────────────────────
COMPRESSION_LEVELS: List[float] = [0.0, 0.30, 0.50, 0.70, 0.90]

# ──────────────────────────────────────────────────────────────────────
# Calibration metric settings
# ──────────────────────────────────────────────────────────────────────
ECE_NUM_BINS = 15

# ──────────────────────────────────────────────────────────────────────
# Neural Network hyperparameters
# ──────────────────────────────────────────────────────────────────────
NN_HIDDEN_SIZES: List[int] = [128, 64, 32]
NN_DROPOUT = 0.3
NN_LEARNING_RATE = 1e-3
NN_BATCH_SIZE = 256
NN_EPOCHS = 100
NN_EARLY_STOPPING_PATIENCE = 10
NN_COMPRESSION_MODES: List[str] = [
    "prune+quantize",
    "prune_only",
    "quantize_only",
    "fp16",
]

# ──────────────────────────────────────────────────────────────────────
# Random Forest hyperparameters
# ──────────────────────────────────────────────────────────────────────
RF_N_ESTIMATORS = 500
RF_MAX_DEPTH = None  # Unlimited by default; compression will cap it

# ──────────────────────────────────────────────────────────────────────
# Gradient Boosting hyperparameters
# ──────────────────────────────────────────────────────────────────────
GB_N_ESTIMATORS = 300
GB_MAX_DEPTH = 5
GB_LEARNING_RATE = 0.1


# ──────────────────────────────────────────────────────────────────────
# Dataset descriptors
# ──────────────────────────────────────────────────────────────────────
@dataclass
class DatasetConfig:
    """Metadata for each dataset used in the experiment."""
    name: str
    uci_id: int                          # UCI ML Repository ID
    target_column: str                   # Name of the binary target
    sensitive_attributes: List[str]      # Columns used for group-wise evaluation
    positive_label: Any = 1             # Value considered as positive class
    description: str = ""


DATASETS: Dict[str, DatasetConfig] = {
    "adult": DatasetConfig(
        name="Adult (Census Income)",
        uci_id=2,
        target_column="income",
        sensitive_attributes=["sex", "race"],
        positive_label=">50K",
        description="Predict whether income exceeds $50K/yr",
    ),
    "german_credit": DatasetConfig(
        name="Statlog German Credit",
        uci_id=144,
        target_column="class",
        sensitive_attributes=["sex", "age_group"],
        positive_label=1,
        description="Classify as good (1) or bad (2) credit risk",
    ),
    "credit_default": DatasetConfig(
        name="Default of Credit Card Clients",
        uci_id=350,
        target_column="default.payment" ,
        sensitive_attributes=["SEX", "EDUCATION", "MARRIAGE"],
        positive_label=1,
        description="Predict whether a client defaults next month",
    ),
    "bank_marketing": DatasetConfig(
        name="Bank Marketing",
        uci_id=222,
        target_column="y",
        sensitive_attributes=["age_group", "marital"],
        positive_label="yes",
        description="Predict if client subscribes to a term deposit",
    ),
}

# ──────────────────────────────────────────────────────────────────────
# Model registry (filled at runtime by each model module)
# ──────────────────────────────────────────────────────────────────────
MODEL_NAMES: List[str] = [
    "RandomForest",
    "GradientBoosting",
    "NeuralNetwork",
]

# ──────────────────────────────────────────────────────────────────────
# Recalibration methods
# ──────────────────────────────────────────────────────────────────────
RECALIBRATION_METHODS: List[str] = ["none", "platt", "isotonic"]

# ──────────────────────────────────────────────────────────────────────
# Statistical testing
# ──────────────────────────────────────────────────────────────────────
SIGNIFICANCE_LEVEL = 0.05

# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
