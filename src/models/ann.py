"""
src/models/ann.py

Artificial Neural Network (ANN) Module using PyTorch Lightning.

Features:
---------
- Dynamic MLP architecture (Configurable hidden layers).
- Integrated Batch Normalization and Dropout.
- Standardized Lightning training and validation steps.
- Automatic hyperparameter logging.
- Scikit-learn compatible wrapper for seamless pipeline integration.

Author: Thesis Framework
"""

from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader


# ==========================================================
# 1. CORE LIGHTNING MODULE
# ==========================================================


class LitANN(pl.LightningModule):
    """
    LightningModule for Artificial Neural Network Classifier.

    Parameters
    ----------
    input_dim : int
        Number of input features (genes/probes).
    num_classes : int
        Number of output classes.
    hidden_dims : Tuple[int, ...], default=(128, 64)
        Dimensions of the hidden layers.
    dropout : float, default=0.3
        Dropout probability.
    learning_rate : float, default=0.01
        Learning rate for the Adam optimizer.
    use_batchnorm : bool, default=True
        Whether to include BatchNorm1d layers.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Tuple[int, ...] = (128, 64),
        dropout: float = 0.3,
        learning_rate: float = 0.01,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()

        # Saves arguments to self.hparams and logs them automatically to W&B
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # Build dynamic network architecture
        layers: List[nn.Module] = []
        in_features = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            in_features = hidden_dim

        # Final output layer
        layers.append(nn.Linear(in_features, num_classes))
        self.network = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize linear layers with Xavier Uniform and biases with zeros."""
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network to get raw logits."""
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get softmax probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self(x)
            return torch.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=1)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Standard Lightning training step."""
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Standard Lightning validation step."""
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        acc = (preds == y).float().mean()

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> dict:
        """Configure Adam optimizer and ReduceLROnPlateau scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


# ==========================================================
# 2. DATASET UTILITY
# ==========================================================


class NumpyDataset(Dataset):
    """Simple PyTorch Dataset wrapper for numpy arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 16,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Helper to create training and validation DataLoaders."""

    train_loader = DataLoader(
        NumpyDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        NumpyDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


# ==========================================================
# 3. SCIKIT-LEARN COMPATIBLE WRAPPER
# ==========================================================


class SklearnANNWrapper:
    """
    Scikit-learn style wrapper for LitANN.
    Allows seamless integration into the benchmark trainer pipeline.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Tuple[int, ...] = (128, 64),
        dropout: float = 0.3,
        learning_rate: float = 0.01,
        use_batchnorm: bool = True,
        batch_size: int = 16,
        max_epochs: int = 200,
        patience: int = 5,
        random_state: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.use_batchnorm = use_batchnorm
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state

        self.model: Optional[LitANN] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "SklearnANNWrapper":
        """Train the model using PyTorch Lightning Trainer with Early Stopping."""

        if X_val is None or y_val is None:
            raise ValueError(
                "X_val and y_val must be provided for Early Stopping in ANN."
            )

        # Ensure reproducibility
        pl.seed_everything(self.random_state, workers=True)

        # Initialize Lightning Module
        self.model = LitANN(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
            use_batchnorm=self.use_batchnorm,
        )

        # Prepare DataLoaders
        train_loader, val_loader = create_dataloaders(
            X_train, y_train, X_val, y_val, batch_size=self.batch_size
        )

        # Setup Early Stopping
        early_stop_callback = EarlyStopping(
            monitor="val_loss", patience=self.patience, mode="min", verbose=False
        )

        # Initialize Trainer
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator="auto",
            devices=1,
            callbacks=[early_stop_callback],
            logger=False,  # Trainer.py handles W&B logging
            enable_checkpointing=False,  # Trainer.py saves .joblib model
            enable_progress_bar=False,  # Keep console clean during K-Fold
            deterministic=True,
        )

        # Train
        trainer.fit(
            self.model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )

        # Prepare for inference
        self.model.eval()
        self.model.cpu()

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates for the test data X."""
        if self.model is None:
            raise RuntimeError("Model is not trained. Call fit() first.")

        X_tensor = torch.from_numpy(X).float()
        probs = self.model.predict_proba(X_tensor)
        return probs.numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for the given data X."""
        if self.model is None:
            raise RuntimeError("Model is not trained. Call fit() first.")

        X_tensor = torch.from_numpy(X).float()
        preds = self.model.predict(X_tensor)
        return preds.numpy()
