"""PyTorch Neural Network wrapper for tabular group-wise calibration experiments.

Implements the BaseModelWrapper interface, providing uniform training with early
stopping, probability estimation, unstructured L1 pruning combined with dynamic
INT8 quantization, and SHAP explainer hooks.
"""

import copy
import logging
import random
import sys
from pathlib import Path
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is in sys.path for robust imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from models.base import BaseModelWrapper
from src.compression import compress_neural_network

logger = logging.getLogger(__name__)


class TabularMLP(nn.Module):
    """Small Multi-Layer Perceptron architecture for binary tabular classification.

    Architecture:
        Input(input_dim) -> Linear(input_dim, 128) -> BatchNorm1d -> ReLU -> Dropout(0.3)
        -> Linear(128, 64) -> BatchNorm1d -> ReLU -> Dropout(0.3)
        -> Linear(64, 32) -> BatchNorm1d -> ReLU -> Dropout(0.3)
        -> Linear(32, 1) -> Sigmoid
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Optional[List[int]] = None,
        dropout: float = 0.3,
    ) -> None:
        """Initialize the TabularMLP layers.

        Args:
            input_dim: Number of input features.
            hidden_sizes: List of hidden dimension sizes (default: [128, 64, 32]).
            dropout: Dropout probability.
        """
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = config.NN_HIDDEN_SIZES

        layers: List[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output probability tensor of shape (batch_size, 1).
        """
        return self.network(x)


class NeuralNetworkModelWrapper(BaseModelWrapper):
    """Neural Network wrapper adhering to BaseModelWrapper using PyTorch."""

    def __init__(
        self,
        seed: int = 42,
        model: Optional[nn.Module] = None,
        compression_level: float = 0.0,
        input_dim: Optional[int] = None,
        is_quantized: bool = False,
        compression_mode: str = "none",
    ) -> None:
        """Initialize the Neural Network model wrapper.

        Args:
            seed: Random seed for reproducibility.
            model: Optional pre-trained or compressed PyTorch module.
            compression_level: Compression level (0.0 for baseline, > 0.0 for compressed).
            input_dim: Dimension of input features if known.
            is_quantized: Flag indicating whether model has undergone INT8 quantization.
            compression_mode: Compression mode applied ('none', 'prune_only', 'quantize_only', 'prune+quantize', 'fp16').
        """
        self.seed = seed
        self.compression_level = compression_level
        self.model: Optional[nn.Module] = model
        self.input_dim = input_dim
        self.is_quantized = is_quantized
        self.compression_mode = compression_mode
        self._is_fitted: bool = model is not None

    def _set_seeds(self) -> None:
        """Set random seeds across random, numpy, and torch for CPU reproducibility."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train the neural network using Adam optimizer and early stopping.

        Args:
            X_train: Training features array of shape (n_samples, n_features).
            y_train: Training binary labels array of shape (n_samples,).
            X_val: Optional validation features array of shape (n_val_samples, n_features).
            y_val: Optional validation binary labels array of shape (n_val_samples,).
        """
        self._set_seeds()

        X_train_arr = np.asarray(X_train, dtype=np.float32)
        y_train_arr = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)
        self.input_dim = X_train_arr.shape[1]

        # Use passed validation split or create internal 15% validation split for early stopping
        if X_val is None or y_val is None:
            from sklearn.model_selection import train_test_split

            stratify = y_train_arr.flatten() if len(np.unique(y_train_arr)) > 1 else None
            X_tr, X_v, y_tr, y_v = train_test_split(
                X_train_arr,
                y_train_arr,
                test_size=0.15,
                random_state=self.seed,
                stratify=stratify,
            )
        else:
            X_tr = X_train_arr
            y_tr = y_train_arr
            X_v = np.asarray(X_val, dtype=np.float32)
            y_v = np.asarray(y_val, dtype=np.float32).reshape(-1, 1)

        train_dataset = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        val_dataset = TensorDataset(torch.from_numpy(X_v), torch.from_numpy(y_v))

        # Avoid batch norm failure when last batch has exactly 1 sample
        drop_last = (len(train_dataset) > config.NN_BATCH_SIZE) and (
            len(train_dataset) % config.NN_BATCH_SIZE == 1
        )

        pin_memory = torch.cuda.is_available()
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.NN_BATCH_SIZE,
            shuffle=True,
            drop_last=drop_last,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.NN_BATCH_SIZE,
            shuffle=False,
            pin_memory=pin_memory,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_label = (
            f"GPU: {torch.cuda.get_device_name(0)}"
            if device.type == "cuda"
            else "CPU"
        )
        self.model = TabularMLP(
            input_dim=self.input_dim,
            hidden_sizes=config.NN_HIDDEN_SIZES,
            dropout=config.NN_DROPOUT,
        ).to(device)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.NN_LEARNING_RATE)

        logger.info(
            "  Training NeuralNetwork on [%s] (input_dim=%d, batch_size=%d, lr=%.4f, max_epochs=%d, patience=%d)",
            device_label,
            self.input_dim,
            config.NN_BATCH_SIZE,
            config.NN_LEARNING_RATE,
            config.NN_EPOCHS,
            config.NN_EARLY_STOPPING_PATIENCE,
        )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(1, config.NN_EPOCHS + 1):
            self.model.train()
            total_train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device, non_blocking=pin_memory)
                batch_y = batch_y.to(device, non_blocking=pin_memory)

                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * len(batch_x)

            avg_train_loss = total_train_loss / len(train_dataset)

            # Evaluate on validation set
            self.model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device, non_blocking=pin_memory)
                    batch_y = batch_y.to(device, non_blocking=pin_memory)
                    preds = self.model(batch_x)
                    loss = criterion(preds, batch_y)
                    total_val_loss += loss.item() * len(batch_x)

            avg_val_loss = total_val_loss / len(val_dataset)

            if epoch % 25 == 0 or epoch == 1:
                logger.info(
                    "    Epoch %3d/%d | Train Loss: %.4f | Val Loss: %.4f",
                    epoch,
                    config.NN_EPOCHS,
                    avg_train_loss,
                    avg_val_loss,
                )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.NN_EARLY_STOPPING_PATIENCE:
                    logger.info(
                        "    Early stopping at epoch %d (Best Val Loss: %.4f)",
                        epoch,
                        best_val_loss,
                    )
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        self._is_fitted = True
        self.compression_level = 0.0
        self.is_quantized = False
        logger.info("  NeuralNetwork trained successfully. Best Val Loss: %.4f", best_val_loss)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return predicted probabilities for the positive class.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            1D array of positive class probabilities of shape (n_samples,).

        Raises:
            ValueError: If model has not been fitted yet.
        """
        if self.model is None or not self._is_fitted:
            raise ValueError("Model has not been trained yet. Call train() before predict_proba().")

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        self.model.eval()
        device = torch.device("cpu") if self.is_quantized else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model.to(device)
        except Exception:
            device = torch.device("cpu")
            self.model.to(device)

        batch_size = 4096 if (device.type == "cuda") else 2048
        all_probs: List[np.ndarray] = []
        is_half = any(p.dtype == torch.float16 for p in self.model.parameters())

        with torch.no_grad():
            for start_idx in range(0, len(X_arr), batch_size):
                end_idx = min(start_idx + batch_size, len(X_arr))
                batch_x = torch.from_numpy(X_arr[start_idx:end_idx]).to(device)
                if is_half:
                    batch_x = batch_x.half()
                preds = self.model(batch_x)
                probs = preds.squeeze(-1).cpu().numpy()
                if probs.ndim == 0:
                    probs = probs.reshape(1)
                all_probs.append(probs)

        return np.concatenate(all_probs, axis=0).astype(np.float64)

    def compress(self, level: float, mode: str = "prune+quantize") -> "NeuralNetworkModelWrapper":
        """Return a new compressed copy at the given compression level and mode.

        Args:
            level: Compression level between 0.0 and 0.9.
            mode: Compression mode ('prune_only', 'quantize_only', 'prune+quantize', 'fp16').

        Returns:
            A new NeuralNetworkModelWrapper instance wrapping the compressed model.

        Raises:
            ValueError: If model is not trained or level is invalid.
        """
        if self.model is None or not self._is_fitted:
            raise ValueError("Cannot compress an unfitted model. Call train() first.")

        compressed_model = compress_neural_network(self.model, level=level, mode=mode)
        is_quant = mode in ("quantize_only", "prune+quantize")
        new_wrapper = NeuralNetworkModelWrapper(
            seed=self.seed,
            model=compressed_model,
            compression_level=level,
            input_dim=self.input_dim,
            is_quantized=is_quant,
            compression_mode=mode,
        )
        return new_wrapper

    def get_model(self) -> object:
        """Return the underlying PyTorch nn.Module.

        Returns:
            The trained PyTorch model.
        """
        return self.model

    def get_name(self) -> str:
        """Return human-readable model name.

        Returns:
            Model name string.
        """
        return "NeuralNetwork"

    def get_compression_description(self) -> str:
        """Return description of the current compression state.

        Returns:
            Human-readable summary of compression status.
        """
        if self.model is None:
            return "NeuralNetwork (uninitialized)"
        if self.compression_level == 0.0 and self.compression_mode == "none":
            return "NeuralNetwork baseline (uncompressed FP32 MLP)"
        return (
            f"NeuralNetwork compressed (level={self.compression_level:.2f}, "
            f"mode={self.compression_mode})"
        )

    @property
    def model_type(self) -> str:
        """Return model type identifier for SHAP explainer selection.

        Returns:
            'nn' indicating DeepExplainer or GradientExplainer compatibility.
        """
        return "nn"
