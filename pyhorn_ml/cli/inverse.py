"""Inverse design: find geometry that matches a target SPL curve."""

import os
import time

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize, differential_evolution

from pyhorn_ml.core.space import DesignSpace
from pyhorn_ml.core.target import TargetResponse
from pyhorn_ml.core.design_point import DesignPoint
from pyhorn_ml.core.score import score_design
from pyhorn_ml.pipeline.evaluate import evaluate_design, simulate
from pyhorn_ml.data.dataset import DesignDataset


def _mse_in_band(
    pred_spl: np.ndarray,
    freq: np.ndarray,
    target_freq: np.ndarray,
    target_spl: np.ndarray,
    f_min: float,
    f_max: float,
) -> float:
    """MSE loss for inverse design: interpolate target onto simulation freq axis."""
    # Interpolate target SPL onto the simulation frequency grid
    mask = (freq >= f_min) & (freq <= f_max)
    f_band = freq[mask]
    pred_band = pred_spl[mask]
    target_interp = np.interp(f_band, target_freq, target_spl)
    return float(np.mean((pred_band - target_interp) ** 2))


def run_inverse(
    driver: str,
    target_curve: str,
    budget: int,
    output: str,
    model_path: str | None = None,
    fmin: float = 80.0,
    fmax: float = 5000.0,
) -> None:
    """Inverse design: find geometry matching a target SPL curve.

    Two modes:
    1. With trained surrogate + model_path → Bayesian-inversion mode
       Uses scipy.optimize on the surrogate loss landscape, then
       runs a real pyhorn simulation on the best result.
    2. Without surrogate → Differential Evolution on real simulations
       (slower but doesn't require a trained model — can be used cold)

    The result is always evaluated with a real pyhorn simulation and
    scored against the target using the standard scoring pipeline.
    """
    t_start = time.time()
    print(f"[pyhorn-ml] Inverse design")
    print(f"  driver:     {driver}")
    print(f"  target:     {target_curve}")
    print(f"  budget:     {budget} function evaluations")
    print(f"  f_range:    {fmin}–{fmax} Hz")

    # ── Load target curve ─────────────────────────────────────────────────
    df = pd.read_csv(target_curve)
    freq_target = df["freq"].values.astype(np.float64)
    spl_target = df["spl"].values.astype(np.float64)
    print(
        f"  target:     {len(freq_target)} points, "
        f"{freq_target.min():.0f}–{freq_target.max():.0f} Hz"
    )

    space = DesignSpace()
    os.makedirs(output, exist_ok=True)

    best_dp: DesignPoint | None = None
    best_mse = float("inf")

    # ── Target response object for scoring ───────────────────────────────
    target = TargetResponse(f_min=fmin, f_max=fmax)

    # ── Mode 1: Surrogate-guided optimization ────────────────────────────
    if model_path and os.path.exists(model_path):
        import joblib

        print(f"\n  [mode] Surrogate-guided inversion using {model_path}")
        surrogate = joblib.load(model_path)
        n_segments = space.n_segments[1]  # use upper bound for initial guess

        def loss_fn(params: np.ndarray) -> float:
            """Surrogate loss on normalized parameter space."""
            geom = space._decode(params)
            geom["n_segments"] = n_segments
            enc = space._encode(geom)
            pred_spl, _ = surrogate.predict(enc.reshape(1, -1))
            freq_sim = np.linspace(fmin, fmax, 200)
            return _mse_in_band(pred_spl[0], freq_sim, freq_target, spl_target, fmin, fmax)

        # Initial guess: best from dataset (if any), else random
        dataset = None
        dataset_path = os.path.join(output, "..", "data", ".dataset")
        if os.path.exists(os.path.dirname(dataset_path.rstrip("/.dataset"))):
            try:
                dataset = DesignDataset(dataset_path)
                X, y = dataset.X_y()
                if len(X) > 5:
                    # Use best from dataset as starting point
                    scores = [score_design(y[i], np.linspace(fmin, fmax, y.shape[1]), target)[-1] for i in range(len(y))]
                    best_idx = int(np.argmax(scores))
                    x0 = X[best_idx]
                    print(f"  [warm start] Using design #{best_idx} from dataset as initial guess")
                else:
                    x0 = space._encode(space.sample_random(1)[0])
            except Exception:
                x0 = space._encode(space.sample_random(1)[0])
        else:
            x0 = space._encode(space.sample_random(1)[0])

        print(f"  [optimize] Running Nelder-Mead on surrogate surface ({budget} iters)")
        result = minimize(
            loss_fn,
            x0,
            method="Nelder-Mead",
            options={"maxiter": budget, "xatol": 1e-5, "fatol": 1e-4},
        )

        best_geom = space._decode(result.x)
        best_geom["n_segments"] = n_segments
        best_mse = float(result.fun)
        print(f"  [surrogate] Best surrogate MSE: {best_mse:.4f}")
        print(f"  [geometry]  {best_geom}")

        # ── Run real pyhorn simulation on best geometry ─────────────────
        print(f"\n  [validate] Running real pyhorn simulation on best geometry...")
        dp = DesignPoint(
            params=space._encode(best_geom),
            geometry_params=best_geom,
        )
        dp = evaluate_design(dp, driver, target, fmin=fmin, fmax=fmax, n_points=200)
        best_dp = dp

    # ── Mode 2: Differential Evolution on real simulations ────────────────
    else:
        print(f"\n  [mode] Direct optimization (no surrogate) — {budget} real simulations")
        print(f"  [info] This is slow (~{budget * 0.5:.0f}s). Train a surrogate first for speed.")

        # Define bounds for differential evolution
        bounds = space.bounds()

        def objective(params: np.ndarray) -> float:
            geom = space._decode(params)
            n_seg = geom["n_segments"]
            dp = DesignPoint(params=params, geometry_params=geom)
            dp = evaluate_design(dp, driver, target, fmin=fmin, fmax=fmax, n_points=200)
            # Return negative score (DE minimizes)
            return -dp.score

        print(f"  [optimize] Running Differential Evolution...")
        result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=budget // 20,
            popsize=10,
            tol=1e-4,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            polish=True,
            workers=1,
        )

        best_geom = space._decode(result.x)
        print(f"  [best score] {-result.fun:.4f}")
        print(f"  [geometry]  {best_geom}")

        dp = DesignPoint(params=result.x, geometry_params=best_geom)
        dp = evaluate_design(dp, driver, target, fmin=fmin, fmax=fmax, n_points=200)
        best_dp = dp
        best_mse = float(np.mean((dp.real_spl - np.interp(dp.freq, freq_target, spl_target)) ** 2))

    # ── Save results ───────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    out_path = os.path.join(output, "inverse_best.yaml")

    yaml_dict = {
        "name": "inverse_best",
        "score": round(float(best_dp.score), 4),
        "scores": {
            "flatness": round(float(best_dp.flatness_score), 4),
            "sensitivity": round(float(best_dp.sensitivity_score), 4),
            "bass": round(float(best_dp.bass_score), 4),
        },
        "cutoff_penalty": round(float(best_dp.cutoff_penalty), 4),
        "geometry": best_dp.geometry_params,
        "mse_vs_target": round(best_mse, 4),
        "elapsed_seconds": round(elapsed, 1),
        "mode": "surrogate" if model_path else "direct",
    }

    with open(out_path, "w") as f:
        yaml.dump(yaml_dict, f, allow_unicode=True)
    print(f"\n  → {out_path}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Final score: {best_dp.score:.4f}")
    print(f"  SPL at 1m: {np.mean(best_dp.real_spl):.1f} dB")

    # Save the SPL curve
    import csv
    csv_path = os.path.join(output, "inverse_best_response.csv")
    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["freq", "spl"])
        for f_val, spl_val in zip(best_dp.freq, best_dp.real_spl):
            writer.writerow([round(f_val, 2), round(spl_val, 2)])
    print(f"  → {csv_path}")