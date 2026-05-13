"""Base class for all surrogate models."""
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class SurrogateModel(ABC):
    """Abstract base for surrogate models that predict horn response.

    A surrogate model takes a geometry parameter vector and returns
    a predicted SPL curve (and optionally uncertainty).
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SurrogateModel":
        """Train the surrogate on X (N designs × D params) → y (N × F SPL values).

        Returns self for chaining.
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict SPL for new designs.

        Args:
            X: M designs × D params

        Returns:
            (mean_predictions, uncertainties) — both M × F arrays.
            uncertainties are the per-point standard deviation.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this model type."""
