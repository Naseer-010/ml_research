"""Base model wrapper interface for tabular models.

Defines the abstract interface that all model wrappers in the project must implement.
Ensures uniform training, inference, compression, and explainability hooks.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseModelWrapper(ABC):
    """Abstract base class for all model wrappers."""

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train the model.

        Args:
            X_train: Training features array of shape (n_samples, n_features).
            y_train: Training binary labels array of shape (n_samples,).
            X_val: Optional validation features array of shape (n_val_samples, n_features).
            y_val: Optional validation binary labels array of shape (n_val_samples,).
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return predicted probabilities for the positive class.

        Args:
            X: Feature array of shape (n_samples, n_features).

        Returns:
            1D array of predicted probabilities for the positive class of shape (n_samples,).
        """
        pass

    @abstractmethod
    def compress(self, level: float) -> "BaseModelWrapper":
        """Return a new compressed copy at the given compression level (0.0 to 0.9).

        Does NOT modify self. Returns a NEW wrapper with the compressed model.

        Args:
            level: Compression level between 0.0 and 0.9 representing the fraction
                of model capacity removed.

        Returns:
            A new instance of BaseModelWrapper wrapping the compressed model.
        """
        pass

    @abstractmethod
    def get_model(self) -> object:
        """Return the underlying sklearn/torch model for SHAP/explainability.

        Returns:
            Underlying trained model instance.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return human-readable model name.

        Returns:
            String model name.
        """
        pass

    @abstractmethod
    def get_compression_description(self) -> str:
        """Return description of current compression state.

        Returns:
            String describing current compression level and parameter state.
        """
        pass

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return 'tree', 'ensemble', or 'nn' for SHAP explainer selection.

        Returns:
            Model type string ('tree', 'ensemble', or 'nn').
        """
        pass
