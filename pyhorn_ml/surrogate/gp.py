"""Gaussian Process surrogate model using scikit-learn."""
from typing import Tuple
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, WhiteKernel, ConstantKernel, Matern
)
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin

from pyhorn_ml.surrogate.base import SurrogateModel


class GPSurrogate(SurrogateModel, RegressorMixin):
    """Gaussian Process surrogate for horn SPL prediction.

    Uses scikit-learn's GaussianProcessRegressor with an RBF+Matern kernel
    combination and automatic relevance determination (ARD) for per-dimension
    lengthscales.

    The GP gives:
    - Mean prediction: smooth approximation of SPL vs. geometry
    - Uncertainty: per-point variance, used by the acquisition function

    Training is O(n³) in dataset size — switch to MLP when n > 500.
    """

    def __init__(
        self,
        initial_length_scale: float = 0.1,
        nu: float = 1.5,  # Matern nu=1.5 (once differentiable) or 2.5 (twice)
        alpha: float = 1e-6,  # noise regularisation
    ):
        self.initial_length_scale = initial_length_scale
        self.nu = nu
        self.alpha = alpha
        self._scaler = StandardScaler()
        self._gp: GaussianProcessRegressor | None = None

    @property
    def name(self) -> str:
        return f"GaussianProcess(nu={self.nu})"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GPSurrogate":
        """Fit GP to N designs × D params → N × F SPL values.

        Uses StandardScaler on both X and y so the GP kernel doesn't
        need to handle vastly different scales.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Scale inputs — zero mean, unit variance
        X_scaled = self._scaler.fit_transform(X)

        # Normalise targets per-frequency to have similar variance across the curve
        y_mean = y.mean(axis=0, keepdims=True)
        y_std = y.std(axis=0, keepdims=True) + 1e-8
        y_scaled = (y - y_mean) / y_std

        # Kernel: Constant × (RBF + Matern) + WhiteKernel for noise
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3)) *
            (
                RBF(length_scale=self.initial_length_scale, length_scale_bounds=(1e-4, 1e2))
                + Matern(length_scale=self.initial_length_scale, length_scale_bounds=(1e-4, 1e2), nu=self.nu)
            )
            + WhiteKernel(noise_level=self.alpha, noise_level_bounds=(1e-8, 1e1))
        )

        self._gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,  # noise is in the kernel
            n_restarts_optimizer=3,
            normalize_y=False,  # already normalised
            random_state=42,
        )
        self._gp.fit(X_scaled, y_scaled)

        # Store for denormalisation in predict
        self._y_mean = y_mean
        self._y_std = y_std

        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict SPL + uncertainty for new designs.

        Returns:
            mean: M × F predicted SPL values
            std:  M × F uncertainty (std dev per frequency)
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_scaled = self._scaler.transform(X)
        assert self._gp is not None, "GP not trained — call fit() first"
        y_scaled, y_cov = self._gp.predict(X_scaled, return_cov=True)

        # Denormalise
        mean = y_scaled * self._y_std + self._y_mean
        std = np.sqrt(np.maximum(y_cov, 0)) * self._y_std

        return mean, std

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convenience: fit then predict on training data (for sanity checking)."""
        self.fit(X, y)
        return self.predict(np.asarray(X, dtype=np.float64))
