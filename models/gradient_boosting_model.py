"""Gradient Boosting classifier wrapper for group-wise calibration experiments.

Implements the BaseModelWrapper interface, providing uniform training,
probability estimation, post-training boosting stage truncation, and SHAP ensemble integration.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

# Ensure project root is in sys.path for robust imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from models.base import BaseModelWrapper
from src.compression import prune_gradient_boosting

logger = logging.getLogger(__name__)


class GradientBoostingModelWrapper(BaseModelWrapper):
    """Gradient Boosting classifier wrapper adhering to BaseModelWrapper."""

    def __init__(
        self,
        seed: int = 42,
        model: Optional[GradientBoostingClassifier] = None,
        compression_level: float = 0.0,
    ) -> None:
        """Initialize the Gradient Boosting model wrapper.

        Args:
            seed: Random seed for training reproducibility.
            model: Optional pre-trained or pruned GradientBoostingClassifier instance.
            compression_level: Compression level (0.0 for baseline, > 0.0 for compressed).
        """
        self.seed = seed
        self.compression_level = compression_level
        self.model: Optional[GradientBoostingClassifier] = model
        self._is_fitted: bool = model is not None and hasattr(model, "estimators_")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train the Gradient Boosting model on training data.

        Args:
            X_train: Training feature array of shape (n_samples, n_features).
            y_train: Binary training labels of shape (n_samples,).
            X_val: Optional validation feature array (kept for API consistency).
            y_val: Optional validation binary labels (kept for API consistency).
        """
        np.random.seed(self.seed)
        logger.info(
            "Training GradientBoostingClassifier with n_estimators=%d, max_depth=%d, lr=%.3f, seed=%d",
            config.GB_N_ESTIMATORS,
            config.GB_MAX_DEPTH,
            config.GB_LEARNING_RATE,
            self.seed,
        )

        self.model = GradientBoostingClassifier(
            n_estimators=config.GB_N_ESTIMATORS,
            max_depth=config.GB_MAX_DEPTH,
            learning_rate=config.GB_LEARNING_RATE,
            random_state=self.seed,
        )
        self.model.fit(X_train, y_train)
        self._is_fitted = True
        self.compression_level = 0.0

        n_stages = len(self.model.estimators_)
        logger.info("GradientBoosting trained with %d boosting stages.", n_stages)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return predicted positive class probabilities.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            1D array of probabilities for the positive class of shape (n_samples,).

        Raises:
            ValueError: If the model has not been trained yet.
        """
        if self.model is None or not self._is_fitted:
            raise ValueError("Model has not been trained yet. Call train() before predict_proba().")

        X_arr = np.asarray(X)
        proba = self.model.predict_proba(X_arr)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1].astype(np.float64)
        elif proba.ndim == 2 and proba.shape[1] == 1:
            return proba[:, 0].astype(np.float64)
        return proba.astype(np.float64)

    def compress(self, level: float) -> "GradientBoostingModelWrapper":
        """Return a new compressed copy at the given compression level.

        Truncates boosting rounds and scales tree depth proportionally.
        Does not mutate the current wrapper instance.

        Args:
            level: Compression level between 0.0 and 0.9.

        Returns:
            A new GradientBoostingModelWrapper instance containing the pruned model.

        Raises:
            ValueError: If model is not trained or level is invalid.
        """
        if self.model is None or not self._is_fitted:
            raise ValueError("Cannot compress an unfitted model. Call train() first.")

        pruned_gb = prune_gradient_boosting(self.model, level=level)
        new_wrapper = GradientBoostingModelWrapper(
            seed=self.seed,
            model=pruned_gb,
            compression_level=level,
        )
        return new_wrapper

    def get_model(self) -> object:
        """Return the underlying sklearn GradientBoostingClassifier model.

        Returns:
            The trained scikit-learn GradientBoostingClassifier.
        """
        return self.model

    def get_name(self) -> str:
        """Return human-readable model name.

        Returns:
            Model name string.
        """
        return "GradientBoosting"

    def get_compression_description(self) -> str:
        """Return description of the current compression state.

        Returns:
            Human-readable summary of compression status.
        """
        if self.model is None or not hasattr(self.model, "estimators_"):
            return "GradientBoosting (uninitialized)"

        n_stages = len(self.model.estimators_)
        depth = getattr(self.model, "max_depth", None)
        if self.compression_level == 0.0:
            return f"GradientBoosting baseline (uncompressed, {n_stages} rounds, max_depth={depth})"
        return (
            f"GradientBoosting compressed (level={self.compression_level:.2f}, "
            f"{n_stages} rounds, max_depth={depth})"
        )

    @property
    def model_type(self) -> str:
        """Return model type identifier for SHAP explainer selection.

        Returns:
            'tree' indicating TreeExplainer compatibility.
        """
        return "tree"
