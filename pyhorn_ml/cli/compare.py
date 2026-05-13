"""Compare multiple designs on one SPL plot."""

import glob
from pathlib import Path

import numpy as np
import yaml

from pyhorn_ml.core.design_point import DesignPoint
from pyhorn_ml.pipeline.evaluate import simulate
from pyhorn_ml.plot.response import plot_response_overlay


def run_compare(
    designs: str,
    driver: str,
    output: str,
    fmin: float = 20.0,
    fmax: float = 5000.0,
) -> None:
    """Load designs and compare their SPL curves."""
    raw_paths = [p.strip() for p in designs.split(",") if p.strip()]
    paths: list[str] = []
    for p in raw_paths:
        if "*" in p or "?" in p:
            paths.extend(glob.glob(p))
        else:
            paths.append(p)

    dps: list[DesignPoint] = []
    for p in paths:
        name = Path(p).stem
        with open(p) as f:
            data = yaml.safe_load(f)

        geom = data.get("geometry") or data
        params = {
            "throat_area": geom["throat_area"],
            "mouth_area": geom["mouth_area"],
            "path_length": geom["path_length"],
            "profile_type": geom.get("profile_type", "conical"),
            "n_segments": geom.get("n_segments", 40),
            "fold_style": geom.get("fold_style", "straight"),
        }

        print(f"  Simulating {name}...", end=" ", flush=True)
        freq, spl, _z = simulate(params, driver, fmin=fmin, fmax=fmax, n_points=300)
        dp = DesignPoint(geometry_params=params, real_spl=spl, freq=freq)
        dp.name = name
        dp.score = data.get("score", 0.0)
        dps.append(dp)
        print(f"done  SPL={np.mean(spl):.1f}dB")

    plot_response_overlay(
        dps,
        output,
        f_min=fmin,
        f_max=fmax,
        title="SPL Comparison",
    )
    print(f"\n  Saved: {output}")
