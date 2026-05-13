"""Score a design against acoustic targets."""

import numpy as np

from pyhorn_core.solver.scoring import (
    compute_response_metrics,
    cutoff_penalty,
    mean_spl_in_band,
)
from pyhorn_ml.core.target import TargetResponse


def score_design(
    spl: np.ndarray,
    freq: np.ndarray,
    target: TargetResponse,
) -> tuple[float, float, float, float, float]:
    """Compute per-objective and total scores for a design.

    Args:
        spl:  SPL values in dB, parallel to freq
        freq: frequency axis in Hz
        target: acoustic targets

    Returns:
        (flatness_score, sensitivity_score, bass_score, cutoff_penalty, total_score)
        All scores in 0.0–1.0, higher = better.
        cutoff_penalty encodes the hard cutoff constraint (1.0 = passes, <1 = fails).
    """
    metrics = compute_response_metrics(spl, freq, target.f_min, target.f_max)
    band_spl = spl[(freq >= target.f_min) & (freq <= target.f_max)]

    # ── Flatness: 1 - normalized std deviation in band ────────────────────
    if len(band_spl) < 2:
        flatness = 0.0
    else:
        std = metrics.flatness_db
        flatness = max(0.0, 1.0 - std / (target.target_spl * 0.5))

    # ── Sensitivity: closeness to target SPL in the horn-loading band ────
    #
    # Measure sensitivity in the frequency range where the horn does real loading work,
    # not in the driver's free-air resonance tail. Below ~200Hz the SPL is dominated by
    # the driver's Qt/vas roll-off, not the horn geometry — so peaking there boosts
    # midband averages without being genuine efficiency.
    #
    # f_min × 2.5 is a practical floor: for f_min=80Hz → 200Hz; for f_min=60Hz → 150Hz
    sens_band_lo = max(target.f_min * 2.5, 200.0)
    sens_mean = mean_spl_in_band(spl, freq, sens_band_lo, target.f_max)

    if sens_mean is not None:
        sensitivity = max(
            0.0, 1.0 - abs(sens_mean - target.target_spl) / target.target_spl
        )
    else:
        sensitivity = 0.0

    # ── Bass extension: bonus for holding SPL near f_min ───────────────────
    if metrics.bass_mean_spl is not None:
        bass_deficit = max(0.0, (target.target_spl - 12.0) - metrics.bass_mean_spl)
        bass = max(0.0, 1.0 - bass_deficit / 12.0)
    else:
        bass = 0.0

    # ── Hard cutoff constraint ─────────────────────────────────────────────
    penalty = cutoff_penalty(metrics.cutoff_frequency_hz, target.f_min)

    # ── Total score: weighted sum × cutoff penalty ─────────────────────────
    total = (
        target.flatness * flatness
        + target.sensitivity * sensitivity
        + target.bass_extension * bass
    )
    weight_sum = target.flatness + target.sensitivity + target.bass_extension
    if weight_sum > 0:
        total /= weight_sum

    # Apply hard cutoff as a multiplier (zeroes out designs that fail cutoff)
    total *= penalty

    return flatness, sensitivity, bass, penalty, total


def dominated(design_a: dict, design_b: dict) -> bool:
    """Check if design_a is dominated by design_b.

    A dominates B if B is at least as good on all objectives
    and strictly better on at least one.
    """
    at_least_as_good = all(
        design_b.get(k, 0) >= design_a.get(k, 0)
        for k in ("flatness_score", "sensitivity_score", "bass_score")
    )
    strictly_better = any(
        design_b.get(k, 0) > design_a.get(k, 0)
        for k in ("flatness_score", "sensitivity_score", "bass_score")
    )
    return at_least_as_good and strictly_better
