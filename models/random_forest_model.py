"""Random Forest classifier wrapper for group-wise calibration experiments.

Implements the BaseModelWrapper interface, providing uniform training,
probability estimation, post-training ensemble pruning, and SHAP tree integration.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Ensure project root is in sys.path for robust imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from models.base import BaseModelWrapper
from src.compression import prune_random_forest

logger = logging.getLogger(__name__)


class RandomForestModelWrapper(BaseModelWrapper):
    """Random Forest classifier wrapper adhering to BaseModelWrapper."""

    def __init__(
        self,
        seed: int = 42,
        model: Optional[RandomForestClassifier] = None,
        compression_level: float = 0.0,
    ) -> None:
        """Initialize the Random Forest model wrapper.

        Args:
            seed: Random seed for training reproducibility.
            model: Optional pre-trained or pruned RandomForestClassifier instance.
            compression_level: Compression level (0.0 for baseline, > 0.0 for compressed).
        """
        self.seed = seed
        self.compression_level = compression_level
        self.model: Optional[RandomForestClassifier] = model
        self._is_fitted: bool = model is not None and hasattr(model, "estimators_")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train the Random Forest model on training data.

        Args:
            X_train: Training feature array of shape (n_samples, n_features).
            y_train: Binary training labels of shape (n_samples,).
            X_val: Optional validation feature array (unused by RF, kept for API consistency).
            y_val: Optional validation binary labels (unused by RF, kept for API consistency).
        """
        np.random.seed(self.seed)
        logger.info(
            "Training RandomForestClassifier with n_estimators=%d, max_depth=%s, seed=%d",
            config.RF_N_ESTIMATORS,
            config.RF_MAX_DEPTH,
            self.seed,
        )

        self.model = RandomForestClassifier(
            n_estimators=config.RF_N_ESTIMATORS,
            max_depth=config.RF_MAX_DEPTH,
            random_state=self.seed,
            n_jobs=-1,
            oob_score=True,
        )
        self.model.fit(X_train, y_train)
        self._is_fitted = True
        self.compression_level = 0.0

        if hasattr(self.model, "oob_score_") and self.model.oob_score_ is not None:
            logger.info("RandomForest trained. OOB Score: %.4f", self.model.oob_score_)

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

    def compress(self, level: float) -> "RandomForestModelWrapper":
        """Return a new compressed copy at the given compression level.

        Prunes the forest estimators and scales max_depth proportionally.
        Does not mutate the current wrapper instance.

        Args:
            level: Compression level between 0.0 and 0.9.

        Returns:
            A new RandomForestModelWrapper instance containing the pruned model.

        Raises:
            ValueError: If model is not trained or level is invalid.
        """
        if self.model is None or not self._is_fitted:
            raise ValueError("Cannot compress an unfitted model. Call train() first.")

        pruned_rf = prune_random_forest(self.model, level=level)
        new_wrapper = RandomForestModelWrapper(
            seed=self.seed,
            model=pruned_rf,
            compression_level=level,
        )
        return new_wrapper

    def get_model(self) -> object:
        """Return the underlying sklearn RandomForestClassifier model.

        Returns:
            The trained scikit-learn RandomForestClassifier.
        """
        return self.model

    def get_name(self) -> str:
        """Return human-readable model name.

        Returns:
            Model name string.
        """
        return "RandomForest"

    def get_compression_description(self) -> str:
        """Return description of the current compression state.

        Returns:
            Human-readable summary of compression status.
        """
        if self.model is None or not hasattr(self.model, "estimators_"):
            return "RandomForest (uninitialized)"

        n_trees = len(self.model.estimators_)
        depth = getattr(self.model, "max_depth", None)
        if self.compression_level == 0.0:
            return f"RandomForest baseline (uncompressed, {n_trees} trees, max_depth={depth})"
        return (
            f"RandomForest compressed (level={self.compression_level:.2f}, "
            f"{n_trees} trees, max_depth={depth})"
        )

    @property
    def model_type(self) -> str:
        """Return model type identifier for SHAP explainer selection.

        Returns:
            'tree' indicating TreeExplainer compatibility.
        """
        return "tree"
