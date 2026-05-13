"""The parameter space to search over."""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
import numpy as np

from pyhorn_core.config.design_space import (
    FOLD_STYLES,
    ML_LRC_RANGE,
    ML_PROFILE_TYPES,
    ML_VTC_RANGE,
    MOUTH_AREA_RANGE,
    N_SEGMENTS_RANGE,
    PATH_LENGTH_RANGE,
    THROAT_AREA_RANGE,
)


# Acoustic features for each fold style — replaces one-hot encoding.
# These map geometry directly to the acoustic effects that matter:
# bend count → number of series inertance steps in the LEM model;
# mean angle → average corner volume per bend.
_FOLD_BEND_COUNT: Dict[str, float] = {
    "straight": 0.0,
    "W": 2.0,
    "pi": 3.0,
}
_FOLD_MEAN_ANGLE: Dict[str, float] = {  # radians
    "straight": 0.0,
    "W": 1.5708,   # π/2 — two 90° turns
    "pi": 2.3562,  # ~2π/3 — three wider turns (pi-fold is 3 bends, ~120° each)
}
_FOLD_KEYS = list(FOLD_STYLES)  # ["straight", "W", "pi"]


@dataclass
class DesignSpace:
    """Defines the parameter space for horn geometry optimization.

    All continuous ranges are (min, max) tuples.
    Discrete choices are lists of strings.

    The fold_style categorical is encoded as two continuous acoustic features
    (bend_count, mean_angle) rather than one-hot, so the GP/MLP surrogate can
    learn smooth responses across fold geometries rather than treating them
    as isolated points.

    Example:
        DesignSpace(
            throat_area_m2=(0.005, 0.025),
            mouth_area_m2=(0.03, 0.15),
            path_length_m=(0.3, 1.5),
            profile_type=["conical", "exponential", "hyperbolic"],
            n_segments=(10, 60),
            fold_style=["straight", "W", "pi"],
            lrc_m=(0.01, 0.5),
            vtc_m3=(0.0, 0.005),
        )
    """

    throat_area_m2: Tuple[float, float] = THROAT_AREA_RANGE  # 50–300 cm²
    mouth_area_m2: Tuple[float, float] = MOUTH_AREA_RANGE  # 200–2000 cm²
    path_length_m: Tuple[float, float] = PATH_LENGTH_RANGE

    # Rear chamber (lrc) and throat chamber volume (vtc)
    # vtc minimum ~1L for the Fostex FE166NV2 basket volume; upper bound 5L
    lrc_m: Tuple[float, float] = ML_LRC_RANGE  # rear chamber length (m)
    # throat chamber volume (m³); min ~2L for driver clearance
    vtc_m3: Tuple[float, float] = ML_VTC_RANGE

    # Discrete choices
    profile_type: List[str] = field(default_factory=lambda: list(ML_PROFILE_TYPES))
    n_segments: Tuple[int, int] = N_SEGMENTS_RANGE
    fold_style: List[str] = field(default_factory=lambda: list(FOLD_STYLES))

    def n_dims(self) -> int:
        """Total optimization dimensions.

        Returns 11: 5 continuous + 3 profile one-hot + 1 n_segments
        + 2 fold acoustic features (bend_count, mean_angle).
        """
        return 5 + len(self.profile_type) + 1 + 2

    def _fold_features(self, fold_style: str) -> Tuple[float, float]:
        """Convert fold_style string to (bend_count, mean_angle)."""
        return (
            _FOLD_BEND_COUNT.get(fold_style, 0.0),
            _FOLD_MEAN_ANGLE.get(fold_style, 0.0),
        )

    def _fold_from_features(self, bend_count: float, mean_angle: float) -> str:
        """Map continuous bend features back to nearest fold_style."""
        best, best_dist = "straight", float("inf")
        for key in _FOLD_KEYS:
            bc, ma = _FOLD_BEND_COUNT[key], _FOLD_MEAN_ANGLE[key]
            dist = (bend_count - bc) ** 2 + (mean_angle - ma) ** 2
            if dist < best_dist:
                best, best_dist = key, dist
        return best

    def sample_random(self, n: int = 1) -> List[Dict[str, Any]]:
        """Generate n random designs using Latin Hypercube Sampling.

        Returns list of dicts with keys: throat_area, mouth_area, path_length,
        lrc, vtc, profile_type, n_segments, fold_style.
        """
        designs = []
        for _ in range(n):
            throat = np.random.uniform(*self.throat_area_m2)
            mouth = np.random.uniform(*self.mouth_area_m2)
            path = np.random.uniform(*self.path_length_m)
            lrc = np.random.uniform(*self.lrc_m)
            vtc = np.random.uniform(*self.vtc_m3)
            profile = self.profile_type[np.random.randint(len(self.profile_type))]
            n_seg = np.random.randint(self.n_segments[0], self.n_segments[1] + 1)
            fold = self.fold_style[np.random.randint(len(self.fold_style))]
            designs.append(
                {
                    "throat_area": throat,
                    "mouth_area": mouth,
                    "path_length": path,
                    "lrc": lrc,
                    "vtc": vtc,
                    "profile_type": profile,
                    "n_segments": n_seg,
                    "fold_style": fold,
                }
            )
        return designs

    def bounds(self) -> List[Tuple[float, float]]:
        """Return continuous bounds for Bayesian optimizer.

        Fold style is now encoded as 2 continuous acoustic features
        (bend_count, mean_angle) instead of one-hot, giving the
        surrogate smooth interpolation across fold geometries.
        """
        bounds = [
            self.throat_area_m2,  # 0
            self.mouth_area_m2,  # 1
            self.path_length_m,  # 2
            self.lrc_m,  # 3
            self.vtc_m3,  # 4
        ]
        # One-hot encoding for profile_type
        bounds.extend([(0.0, 1.0)] * len(self.profile_type))
        # n_segments encoded uniformly in [0, 1] then scaled
        bounds.append((float(self.n_segments[0]), float(self.n_segments[1])))
        # Fold style: two continuous acoustic features
        bounds.append((0.0, 3.0))   # bend_count range
        bounds.append((0.0, 3.0))  # mean_angle range (radians)
        return bounds

    def _encode(self, geom: dict) -> np.ndarray:
        """Encode a geometry dict into a float parameter vector.

        The vector matches the order produced by bounds().
        Fold style is encoded as (bend_count, mean_angle) — two continuous
        features that map directly to the acoustic effects of folding.
        Used to build the X matrix for surrogate training.
        """
        b = self.bounds()

        throat = self._scale(geom["throat_area"], *b[0])
        mouth = self._scale(geom["mouth_area"], *b[1])
        path = self._scale(geom["path_length"], *b[2])
        lrc = self._scale(geom["lrc"], *b[3])
        vtc = self._scale(geom["vtc"], *b[4])

        vec = [throat, mouth, path, lrc, vtc]

        # One-hot profile_type
        for p in self.profile_type:
            vec.append(1.0 if geom["profile_type"] == p else 0.0)

        n_profile = len(self.profile_type)
        n_seg_idx = 5 + n_profile
        vec.append(self._scale(geom["n_segments"], *b[n_seg_idx]))

        # Fold style: continuous acoustic features
        bend_count, mean_angle = self._fold_features(geom.get("fold_style", "straight"))
        vec.append(self._scale(bend_count, *b[n_seg_idx + 1]))
        vec.append(self._scale(mean_angle, *b[n_seg_idx + 2]))

        return np.array(vec, dtype=np.float64)

    def _decode(self, params: np.ndarray) -> dict:
        """Decode a float parameter vector back to a geometry dict.

        Reverse of _encode(). Fold features map back to nearest fold_style.
        """
        b = self.bounds()
        n_profile = len(self.profile_type)
        n_cont = 5  # throat, mouth, path, lrc, vtc

        throat = self._unscale(params[0], *b[0])
        mouth = self._unscale(params[1], *b[1])
        path = self._unscale(params[2], *b[2])
        lrc = self._unscale(params[3], *b[3])
        vtc = self._unscale(params[4], *b[4])

        # Profile: argmax of one-hot block
        profile_onehot = params[n_cont : n_cont + n_profile]
        profile = self.profile_type[int(np.argmax(profile_onehot))]

        # n_segments
        n_seg_idx = n_cont + n_profile
        n_seg_raw = params[n_seg_idx]
        n_seg = int(round(self._unscale(n_seg_raw, *b[n_seg_idx])))
        n_seg = np.clip(n_seg, self.n_segments[0], self.n_segments[1])

        # Fold style: decode continuous features → nearest fold_style
        bend_count = self._unscale(params[n_seg_idx + 1], *b[n_seg_idx + 1])
        mean_angle = self._unscale(params[n_seg_idx + 2], *b[n_seg_idx + 2])
        fold = self._fold_from_features(bend_count, mean_angle)

        return {
            "throat_area": float(throat),
            "mouth_area": float(mouth),
            "path_length": float(path),
            "lrc": float(lrc),
            "vtc": float(vtc),
            "profile_type": profile,
            "n_segments": int(n_seg),
            "fold_style": fold,
        }

    @staticmethod
    def _scale(val: float, lo: float, hi: float) -> float:
        """Scale val from [lo, hi] to [0, 1]."""
        return (val - lo) / (hi - lo + 1e-12)

    @staticmethod
    def _unscale(val: float, lo: float, hi: float) -> float:
        """Unscale val from [0, 1] to [lo, hi]."""
        return lo + val * (hi - lo)