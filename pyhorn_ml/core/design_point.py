"""A candidate horn design with its predicted/acquired response."""
from dataclasses import dataclass, field
from typing import Optional, Any, Dict
import numpy as np


@dataclass
class DesignPoint:
    """A single horn design candidate.

    Attributes:
        params: raw float vector (encoded form used by optimizer).
        geometry_params: decoded dict with physical parameters.
        predicted_spl: predicted SPL curve (frequency × dB).
        predicted_z: predicted impedance curve (frequency × Ohms).
        real_spl: actual simulated SPL (filled in after evaluation).
        real_z: actual simulated impedance.
        score: scalar fitness (0.0–1.0, higher = better).
        dominated: True if another design beats it on all objectives.
        pareto_rank: Pareto front rank (0 = on best front).
        uncertainty: predicted SPL uncertainty from GP surrogate (if available).
    """

    # Encoded form (for ML models)
    params: np.ndarray = field(default_factory=lambda: np.array([]))

    # Decoded physical parameters
    geometry_params: Dict[str, Any] = field(default_factory=dict)

    # Predicted response (from surrogate)
    predicted_spl: Optional[np.ndarray] = None
    predicted_z: Optional[np.ndarray] = None
    uncertainty: Optional[np.ndarray] = None  # std dev per frequency point

    # Filled after real evaluation
    real_spl: Optional[np.ndarray] = None
    real_z: Optional[np.ndarray] = None
    freq: Optional[np.ndarray] = None  # frequency axis for SPL/Z

    # Optional name label (used in plots/reports); if None, auto-derived from geometry_params
    _name: Optional[str] = None

    # Scores per objective
    flatness_score: float = 0.0
    sensitivity_score: float = 0.0
    bass_score: float = 0.0
    cutoff_penalty: float = 1.0  # hard cutoff constraint (1.0 = passes)
    score: float = 0.0  # weighted sum × cutoff_penalty

    # Pareto state
    dominated: bool = False
    pareto_rank: int = -1

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DesignPoint):
            return NotImplemented
        return id(self) == id(other)

    def __hash__(self) -> int:
        return hash(id(self))

    @property
    def name(self) -> str:
        if self._name:
            return self._name
        p = self.geometry_params
        profile = p.get("profile_type", "?")
        segs = p.get("n_segments", "?")
        path = p.get("path_length", 0)
        return f"{profile}_s{segs}_L{round(path*1000)}mm"

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    def to_dict(self) -> dict:
        """Serializable dict for YAML/JSON export — all native Python types."""
        gp = self.geometry_params
        return {
            "name": self.name,
            "score": float(round(self.score, 4)),
            "flatness": float(round(self.flatness_score, 4)),
            "sensitivity": float(round(self.sensitivity_score, 4)),
            "bass": float(round(self.bass_score, 4)),
            "cutoff_penalty": float(round(self.cutoff_penalty, 4)),
            "dominated": bool(self.dominated),
            "pareto_rank": int(self.pareto_rank),
            "throat_area": float(gp.get("throat_area", 0)),
            "throat_area_cm2": float(round(gp.get("throat_area", 0) * 10_000, 1)),
            "mouth_area": float(gp.get("mouth_area", 0)),
            "mouth_area_cm2": float(round(gp.get("mouth_area", 0) * 10_000, 1)),
            "path_length_m": float(round(gp.get("path_length", 0), 3)),
            "profile_type": str(gp.get("profile_type", "unknown")),
            "n_segments": int(gp.get("n_segments", 0)),
            "fold_style": str(gp.get("fold_style", "straight")),
        }
