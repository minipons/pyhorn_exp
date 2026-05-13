"""Multi-Layer Perceptron surrogate model using scikit-learn."""
from typing import Tuple
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin

from pyhorn_ml.surrogate.base import SurrogateModel


class MLPSurrogate(SurrogateModel, RegressorMixin):
    """MLP surrogate for horn SPL prediction.

    Faster than GP for large datasets (O(n) per prediction) but requires
    more training data (500+ designs) to train reliably.

    Architecture: 2 hidden layers of 128 neurons each with ReLU activation.
    Trained with Adam optimiser and early stopping on a held-out validation set.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, int] = (128, 128),
        max_iter: int = 1000,
        early_stopping: bool = True,
        validation_fraction: float = 0.15,
        random_state: int = 42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self._scaler_x = StandardScaler()
        self._scaler_y = StandardScaler()
        self._mlp: MLPRegressor | None = None

    @property
    def name(self) -> str:
        return f"MLP{self.hidden_layer_sizes}"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPSurrogate":
        """Fit MLP to N designs × D params → N × F SPL values."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        X_scaled = self._scaler_x.fit_transform(X)
        y_scaled = self._scaler_y.fit_transform(y)

        self._mlp = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=1e-4,  # L2 regularisation
            batch_size=min(256, X.shape[0]),
            learning_rate="adaptive",
            learning_rate_init=1e-3,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=20,
            random_state=self.random_state,
            verbose=False,
        )
        self._mlp.fit(X_scaled, y_scaled)
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict SPL for new designs.

        Note: MLP doesn't natively give uncertainty. We return a
        placeholder uncertainty of 1.0 (meaning: no uncertainty info).
        For proper uncertainty, use the GP surrogate.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_scaled = self._scaler_x.transform(X)
        assert self._mlp is not None, "MLP not trained — call fit() first"
        y_scaled = self._mlp.predict(X_scaled)

        mean = self._scaler_y.inverse_transform(y_scaled)
        # MLPs don't produce uncertainty — use residual std from training as proxy
        if self._mlp.best_validation_score_ is not None:
            # Negative MSE stored by sklearn as best_validation_score_
            uncertainty = np.sqrt(-self._mlp.best_validation_score_) * self._scaler_y.scale_
        else:
            uncertainty = np.ones_like(mean) * self._scaler_y.scale_

        return mean, uncertainty
