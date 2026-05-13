"""Bayesian optimisation loop with Gaussian Process surrogate."""
from __future__ import annotations
import time
from typing import List, Optional
import numpy as np

from pyhorn_ml.core.space import DesignSpace
from pyhorn_ml.core.target import TargetResponse
from pyhorn_ml.core.design_point import DesignPoint
from pyhorn_ml.surrogate.factory import auto_surrogate, SurrogateModel
from pyhorn_ml.optimization.acquisition import pick_best_candidate


class BayesianOptimizer:
    """Bayesian optimizer for horn geometry design.

    Uses a Gaussian Process surrogate with Expected Improvement acquisition.
    Latin Hypercube Sampling provides space-filling initialisation.

    ask/tell interface: ask() returns next design, tell(dp) updates the surrogate.
    """

    def __init__(
        self,
        design_space: DesignSpace,
        target: TargetResponse,
        n_initial: int = 20,
        acquisition: str = "ei",
        xi: float = 0.01,
        random_state: int = 42,
    ):
        self.space = design_space
        self.target = target
        self.n_initial = n_initial
        self.acquisition = acquisition
        self.xi = xi
        self.rng = np.random.default_rng(random_state)

        self._X: List[np.ndarray] = []
        self._y: List[np.ndarray] = []
        self._scores: List[float] = []
        self._surrogate: Optional[SurrogateModel] = None
        self._y_best: float = -np.inf

    @property
    def n_evaluated(self) -> int:
        return len(self._X)

    def ask(self) -> DesignPoint:
        """Return the next design to evaluate.

        Phase 1 (n < n_initial): Latin Hypercube random sample
        Phase 2 (n >= n_initial): GP-guided Expected Improvement maximisation
        """
        if self.n_evaluated < self.n_initial:
            params = self._lhs_sample(1)[0]
        else:
            params = self._ei_maximise()

        geom = self.space._decode(params)
        dp = DesignPoint(params=params, geometry_params=geom)
        return dp

    def tell(self, design: DesignPoint) -> None:
        """Update the surrogate with a completed evaluation.

        Skips designs where simulation returned NaN (invalid geometry
        or degenerate physics at extreme parameter combinations).
        """
        # Skip NaN simulations — don't pollute the GP training set
        if design.real_spl is None or not np.all(np.isfinite(design.real_spl)):
            print(f"  [warn] Skipping design {design.name}: non-finite SPL (NaN/inf)")
            return

        self._X.append(np.asarray(design.params, dtype=np.float64))
        self._y.append(np.asarray(design.real_spl, dtype=np.float64))
        self._scores.append(float(design.score))

        if design.score > self._y_best:
            self._y_best = float(design.score)

        self._fit_surrogate()

    def _lhs_sample(self, n: int) -> np.ndarray:
        """Latin Hypercube Sampling — space-filling random designs."""
        n_dims = self.space.n_dims()
        samples = np.zeros((n, n_dims))
        for d in range(n_dims):
            bins = np.linspace(0, 1, n + 1)
            offsets = self.rng.uniform(bins[:-1], bins[1:])
            self.rng.shuffle(offsets)
            samples[:, d] = offsets

        # Decode/re-encode to handle categorical parameters correctly
        decoded = [self.space._decode(samples[i]) for i in range(n)]
        encoded = np.array([self.space._encode(d) for d in decoded])
        return encoded

    def _ei_maximise(self, n_candidates: int = 500) -> np.ndarray:
        """Maximise Expected Improvement using the GP surrogate.

        Generates n_candidates space-filling candidates, evaluates EI,
        and returns the best one.
        """
        candidates = self._lhs_sample(n_candidates)

        assert self._surrogate is not None, "Surrogate not fitted — call fit() first"
        mean, std = self._surrogate.predict(candidates)

        # Use mean SPL in band as proxy for total score
        proxy_scores = np.mean(mean[:, 5:-5], axis=1)

        best_idx, _ = pick_best_candidate(
            candidates, proxy_scores, np.ones(n_candidates),
            acquisition=self.acquisition, y_best=self._y_best, xi=self.xi
        )
        return candidates[best_idx]

    def _fit_surrogate(self) -> None:
        """Fit or refit the GP/MLP surrogate to all evaluated points.

        Filters out rows where y contains NaN or inf as a safety net.
        """
        if len(self._X) < 2:
            return
        X_arr = np.array(self._X)
        y_arr = np.array(self._y)
        # Backstop: remove any rows with non-finite values
        valid = np.all(np.isfinite(y_arr), axis=1)
        if not np.any(valid):
            return
        X_arr = X_arr[valid]
        y_arr = y_arr[valid]
        self._surrogate = auto_surrogate(n_samples=len(X_arr))
        self._surrogate.fit(X_arr, y_arr)
