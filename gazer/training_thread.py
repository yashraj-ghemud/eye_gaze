"""Background training thread - keeps UI responsive during model training."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from gazer.gaze_model import GazeModel


class TrainingThread(QThread):
    """Trains a GazeModel in a background thread. Emits done() when finished."""

    done = pyqtSignal(object, float)  # (GazeModel, final_loss)

    def __init__(
        self,
        model: GazeModel,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 80,
        parent=None,
    ):
        super().__init__(parent)
        self._model = model
        self._X = X
        self._y = y
        self._epochs = epochs

    def run(self) -> None:
        try:
            loss = self._model.train(self._X, self._y, epochs=self._epochs)
            self.done.emit(self._model, loss)
        except Exception:
            self.done.emit(self._model, float("inf"))