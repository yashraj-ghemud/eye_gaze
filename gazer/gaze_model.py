"""Gaze regression model — PyTorch NN with sklearn fallback.

Phase 2 upgrades:
- Wider network: 25 -> 128 -> 64 -> 32 -> 2
- BatchNorm1d + Dropout(0.2) after each hidden layer
- Tanh output (better edge behavior than Sigmoid)
- Weight decay (1e-4) in Adam optimizer
- Train/validation split (80/20) with early stopping (patience=15)
- Gaussian noise augmentation during training
- Mini-batch training (batch_size=32) for better generalization
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from gazer.face_tracker import FEATURE_DIM

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class GazeNet(nn.Module):
    """Regularized gaze regression network.

    Architecture: input -> 128 -> 64 -> 32 -> 2 (Tanh)
    Each hidden block: Linear -> BatchNorm -> ReLU -> Dropout
    """

    def __init__(self, input_dim: int = FEATURE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(32, 2),
            nn.Tanh(),   # output in [-1, 1], mapped to [0, 1] in GazeModel
        )

    def forward(self, x):
        return self.net(x)


class GazeModel:
    """Maps 25-dim feature vector to normalized screen (x, y) in [0, 1]."""

    def __init__(self, use_torch: bool = True, feature_dim: int = FEATURE_DIM):
        self.use_torch = use_torch and TORCH_AVAILABLE
        self._feature_dim = feature_dim
        self._scaler = StandardScaler()
        self._fitted = False
        self._torch_model: GazeNet | None = None
        self._sklearn_model: MLPRegressor | None = None
        self._device = "cpu"
        self._noise_std = 0.05       # Gaussian noise augmentation std
        self._early_stop_patience = 15
        self._weight_decay = 1e-4

        if self.use_torch:
            self._torch_model = GazeNet(input_dim=self._feature_dim).to(self._device)
        else:
            self._sklearn_model = MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                max_iter=500,
                warm_start=True,
                early_stopping=True,
                validation_fraction=0.2,
            )

    @property
    def is_trained(self) -> bool:
        return self._fitted

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def _prepare(self, X: np.ndarray, y: np.ndarray | None = None):
        if not self._fitted and y is not None:
            self._scaler.fit(X)
            self._fitted = True
        return self._scaler.transform(X)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 200, lr: float = 0.001) -> float:
        """Train on feature matrix X and target normalized coords y. Returns final val loss."""
        if len(X) < 10:
            return float("inf")

        self._scaler.fit(X)
        self._fitted = True
        Xs = self._scaler.transform(X)

        if self.use_torch and self._torch_model is not None:
            return self._train_torch(Xs, y, epochs, lr)
        return self._train_sklearn(Xs, y)

    def _train_torch(self, Xs: np.ndarray, y: np.ndarray, epochs: int, lr: float) -> float:
        """Training with mini-batches, noise augmentation, early stopping, val split."""
        assert self._torch_model is not None
        model = self._torch_model

        # Ensure model input dim matches data
        if model.net[0].in_features != Xs.shape[1]:
            logger.info("Recreating GazeNet for input_dim=%d", Xs.shape[1])
            self._torch_model = GazeNet(input_dim=Xs.shape[1]).to(self._device)
            model = self._torch_model

        # Transform targets: [0,1] -> [-1,1] for Tanh output
        yt = (2.0 * y - 1.0).astype(np.float32)

        # Train/validation split (80/20)
        n = len(Xs)
        if n < 20:
            # Too small for val split — use full data
            X_train, y_train = Xs, yt
            X_val, y_val = None, None
        else:
            perm = np.random.permutation(n)
            val_size = max(1, int(n * 0.2))
            train_idx = perm[val_size:]
            val_idx = perm[:val_size]
            X_train = Xs[train_idx].astype(np.float32)
            y_train = yt[train_idx]
            X_val = Xs[val_idx].astype(np.float32)
            y_val = yt[val_idx]

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=self._weight_decay)
        loss_fn = nn.MSELoss()

        # Mini-batch size
        batch_size = min(32, len(X_train))
        if batch_size < 4:
            batch_size = len(X_train)  # full batch for tiny datasets

        best_val_loss = float("inf")
        best_state: dict | None = None
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            perm_idx = np.random.permutation(len(X_train))
            X_shuf = X_train[perm_idx]
            y_shuf = y_train[perm_idx]

            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(X_train), batch_size):
                xb = torch.tensor(X_shuf[i:i + batch_size], dtype=torch.float32, device=self._device)
                yb = torch.tensor(y_shuf[i:i + batch_size], dtype=torch.float32, device=self._device)

                # Gaussian noise augmentation (only during training)
                if self._noise_std > 0 and model.training:
                    xb = xb + torch.randn_like(xb) * self._noise_std

                optimizer.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            # Validation check
            if X_val is not None and n_batches > 0:
                model.eval()
                with torch.no_grad():
                    xv = torch.tensor(X_val, dtype=torch.float32, device=self._device)
                    yv = torch.tensor(y_val, dtype=torch.float32, device=self._device)
                    val_pred = model(xv)
                    val_loss = loss_fn(val_pred, yv).item()

                # Early stopping
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self._early_stop_patience:
                        logger.debug("Early stop at epoch %d (best val_loss=%.6f)", epoch, best_val_loss)
                        break
            else:
                # No val set — track training loss
                best_val_loss = epoch_loss / max(n_batches, 1)
                if n_batches > 0:
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Restore best weights
        if best_state is not None:
            model.load_state_dict(best_state)
            # Move back to device
            model.to(self._device)

        model.eval()
        return best_val_loss

    def _train_sklearn(self, Xs: np.ndarray, y: np.ndarray) -> float:
        assert self._sklearn_model is not None
        self._sklearn_model.fit(Xs, y)
        pred = self._sklearn_model.predict(Xs)
        return float(np.mean((pred - y) ** 2))

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Incremental update (sklearn path) or mini retrain (torch path)."""
        if self.use_torch:
            self.train(X, y, epochs=50)
        else:
            Xs = self._prepare(X, y if not self._fitted else None)
            if not self._fitted:
                self._scaler.fit(X)
                self._fitted = True
                Xs = self._scaler.transform(X)
            assert self._sklearn_model is not None
            self._sklearn_model.partial_fit(Xs, y)

    def _tanh_to_norm(self, out: np.ndarray) -> np.ndarray:
        """Convert Tanh output [-1,1] to normalized [0,1]."""
        result = (out + 1.0) / 2.0
        return np.clip(result, 0.0, 1.0)

    def predict(self, features: np.ndarray) -> tuple[float, float] | None:
        if not self._fitted:
            return None
        x = features.reshape(1, -1)

        # Handle dimension mismatch (old profiles with 8-dim)
        if x.shape[1] != self._scaler.n_features_in_:
            return None

        xs = self._scaler.transform(x)

        if self.use_torch and self._torch_model is not None:
            self._torch_model.eval()
            with torch.no_grad():
                t = torch.tensor(xs, dtype=torch.float32, device=self._device)
                out = self._torch_model(t).cpu().numpy()[0]
            mapped = self._tanh_to_norm(out)
            return float(mapped[0]), float(mapped[1])

        assert self._sklearn_model is not None
        out = self._sklearn_model.predict(xs)[0]
        return float(out[0]), float(out[1])

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return np.zeros((len(X), 2))
        if X.shape[1] != self._scaler.n_features_in_:
            return np.zeros((len(X), 2))
        xs = self._scaler.transform(X)
        if self.use_torch and self._torch_model is not None:
            self._torch_model.eval()
            with torch.no_grad():
                t = torch.tensor(xs, dtype=torch.float32, device=self._device)
                out = self._torch_model(t).cpu().numpy()
            return self._tanh_to_norm(out)
        assert self._sklearn_model is not None
        return self._sklearn_model.predict(xs)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "use_torch": self.use_torch,
            "fitted": self._fitted,
            "feature_dim": self._feature_dim,
            "scaler_mean": self._scaler.mean_.tolist() if self._fitted else None,
            "scaler_scale": self._scaler.scale_.tolist() if self._fitted else None,
        }
        if self.use_torch and self._torch_model is not None:
            torch.save({"state_dict": self._torch_model.state_dict(), "meta": meta}, path)
        else:
            import joblib
            joblib.dump({"model": self._sklearn_model, "meta": meta}, path.with_suffix(".pkl"))

    def load(self, path: Path) -> bool:
        if not path.exists():
            pkl = path.with_suffix(".pkl")
            if not pkl.exists():
                return False
            path = pkl

        if path.suffix == ".pt" and self.use_torch:
            try:
                data = torch.load(path, map_location=self._device, weights_only=False)
            except TypeError:
                data = torch.load(path, map_location=self._device)
            meta = data["meta"]
            self._apply_meta(meta)

            loaded_dim = meta.get("feature_dim", 8)
            if loaded_dim != self._feature_dim:
                logger.warning(
                    "Model feature_dim=%d but current FEATURE_DIM=%d. "
                    "Recalibration recommended.",
                    loaded_dim, self._feature_dim,
                )
                self._feature_dim = loaded_dim

            assert self._torch_model is not None
            self._torch_model = GazeNet(input_dim=loaded_dim).to(self._device)
            self._torch_model.load_state_dict(data["state_dict"])
            return True

        import joblib
        data = joblib.load(path)
        meta = data["meta"]
        self.use_torch = False
        self._sklearn_model = data["model"]
        self._apply_meta(meta)
        return True

    def _apply_meta(self, meta: dict) -> None:
        self._fitted = meta["fitted"]
        loaded_dim = meta.get("feature_dim", 8)
        self._feature_dim = loaded_dim
        if meta["scaler_mean"] is not None:
            self._scaler.mean_ = np.array(meta["scaler_mean"])
            self._scaler.scale_ = np.array(meta["scaler_scale"])
            self._scaler.n_features_in_ = len(meta["scaler_mean"])