"""Run pyhorn simulation for a given design and return the response."""

from functools import lru_cache
from pathlib import Path
from typing import Tuple

import numpy as np

from pyhorn_core.config.parser import parse_driver_specs
from pyhorn_core.solver.design import build_horn_from_params
from pyhorn_core.solver.models import horn_response
from pyhorn_ml.core.design_point import DesignPoint
from pyhorn_ml.core.score import score_design
from pyhorn_ml.core.target import TargetResponse


def build_horn_geometry(params: dict, n_segments: int):
    """Build a HornGeometry instance from design params."""
    return build_horn_from_params(params, n_segments=n_segments)


@lru_cache(maxsize=1)
def _cached_driver(driver_path: str):
    return parse_driver_specs(Path(driver_path))


def simulate(
    geometry_params: dict,
    driver_path: str,
    fmin: float = 20.0,
    fmax: float = 5000.0,
    n_points: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run pyhorn simulation and return (freq, spl, z).

    Args:
        geometry_params: decoded geometry dict
        driver_path: path to driver YAML
        fmin: min frequency Hz
        fmax: max frequency Hz
        n_points: frequency points

    Returns:
        (freq, spl, z) — each as numpy arrays
    """
    geometry = build_horn_geometry(
        geometry_params,
        int(geometry_params.get("n_segments", 20)),
    )
    driver = _cached_driver(driver_path)

    freqs = np.linspace(fmin, fmax, n_points)
    result = horn_response(freqs, driver, geometry)

    return result.freqs, result.spl, result.impedance


def evaluate_design(
    design: DesignPoint,
    driver_path: str,
    target: TargetResponse,
    fmin: float = 20.0,
    fmax: float = 5000.0,
    n_points: int = 200,
) -> DesignPoint:
    """Run full pyhorn simulation for a DesignPoint and fill in scores.

    Modifies design in-place: fills real_spl, real_z, freq, scores.
    """
    freq, spl, z = simulate(
        design.geometry_params,
        driver_path,
        fmin=fmin,
        fmax=fmax,
        n_points=n_points,
    )

    design.real_spl = spl
    design.real_z = z
    design.freq = freq

    flatness, sensitivity, bass, cutoff_penalty, total = score_design(spl, freq, target)
    design.flatness_score = flatness
    design.sensitivity_score = sensitivity
    design.bass_score = bass
    design.cutoff_penalty = cutoff_penalty
    design.score = total

    return design
