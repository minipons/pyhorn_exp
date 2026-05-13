"""Acquisition functions for Bayesian optimisation."""
import numpy as np
from typing import Tuple


def expected_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    y_best: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Expected Improvement (EI) acquisition function.

    EI trades off exploitation (near y_best) vs. exploration (high std).

    Args:
        mean: predicted mean for each candidate
        std: predicted uncertainty for each candidate
        y_best: best observed value so far
        xi: exploration bonus — higher = more exploration

    Returns:
        EI score per candidate (higher = more promising)
    """
    std = np.asarray(std)
    mean = np.asarray(mean)

    # Avoid division by zero
    std_safe = np.where(std > 1e-10, std, 1e-10)

    z = (mean - y_best - xi) / std_safe

    # Φ(z): standard normal CDF
    from scipy.stats import norm
    ei = (mean - y_best - xi) * norm.cdf(z) + std_safe * norm.pdf(z)

    # Zero out EI where std ≈ 0 (degenerate prediction)
    ei[std < 1e-10] = 0.0

    return ei


def upper_confidence_bound(
    mean: np.ndarray,
    std: np.ndarray,
    kappa: float = 2.0,
) -> np.ndarray:
    """Upper Confidence Bound (UCB) acquisition function.

    Simple and effective: mean + kappa * std.
    Higher kappa = more exploration.
    """
    return np.asarray(mean) + kappa * np.asarray(std)


def probability_of_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    y_best: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Probability of Improvement (PI) acquisition function.

    Returns P(y > y_best + xi) for each candidate.
    """
    from scipy.stats import norm

    std_safe = np.where(std > 1e-10, std, 1e-10)
    z = (mean - y_best - xi) / std_safe
    return norm.cdf(z)


def pick_best_candidate(
    X_candidates: np.ndarray,
    surrogate_mean: np.ndarray,
    surrogate_std: np.ndarray,
    acquisition: str = "ei",
    y_best: float = 0.0,
    xi: float = 0.01,
) -> Tuple[int, np.ndarray]:
    """Pick the best next candidate based on acquisition function.

    Args:
        X_candidates: M candidate designs (M × D)
        surrogate_mean: predicted means (M,)
        surrogate_std: predicted uncertainties (M,)
        acquisition: "ei", "ucb", or "pi"
        y_best: best score observed so far
        xi: exploration parameter for EI/PI

    Returns:
        (best_idx, acquisition_values) — best_idx into X_candidates
    """
    if acquisition == "ei":
        acq = expected_improvement(surrogate_mean, surrogate_std, y_best, xi)
    elif acquisition == "ucb":
        acq = upper_confidence_bound(surrogate_mean, surrogate_std)
    elif acquisition == "pi":
        acq = probability_of_improvement(surrogate_mean, surrogate_std, y_best, xi)
    else:
        raise ValueError(f"Unknown acquisition: {acquisition!r}")

    best_idx = int(np.argmax(acq))
    return best_idx, acq
