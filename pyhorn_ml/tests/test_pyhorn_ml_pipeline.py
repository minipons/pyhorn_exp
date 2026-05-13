from pathlib import Path

import numpy as np

from pyhorn_core.config.models import HornGeometry
from pyhorn_core.solver.design import build_horn_from_params
from pyhorn_ml.pipeline.evaluate import build_horn_geometry, simulate


def test_build_horn_geometry_returns_shared_horn_geometry():
    params = {
        "throat_area": 0.01,
        "mouth_area": 0.08,
        "path_length": 1.1,
        "profile_type": "exponential",
        "n_segments": 24,
        "lrc": 0.08,
        "vtc": 0.001,
    }

    horn = build_horn_geometry(params, n_segments=24)
    shared = build_horn_from_params(params, profile_type="exponential", n_segments=24)

    assert isinstance(horn, HornGeometry)
    assert horn == shared
    assert horn.n_segments == 24


def test_simulate_returns_finite_response_arrays():
    params = {
        "throat_area": 0.01,
        "mouth_area": 0.08,
        "path_length": 1.1,
        "profile_type": "conical",
        "n_segments": 18,
        "lrc": 0.08,
        "vtc": 0.001,
    }
    driver_path = Path(__file__).parent.parent.parent / "drivers" / "FE166NV2.yaml"

    freq, spl, impedance = simulate(
        params,
        str(driver_path),
        fmin=80.0,
        fmax=500.0,
        n_points=24,
    )

    assert freq.shape == (24,)
    assert spl.shape == (24,)
    assert impedance.shape == (24,)
    assert np.all(np.isfinite(freq))
    assert np.all(np.isfinite(spl))
    assert np.all(np.isfinite(impedance.real))
    assert np.all(np.isfinite(impedance.imag))
