"""pyhorn-ml optimise command — Bayesian multi-objective optimisation."""

import dataclasses
import math
import os
import time
from pathlib import Path

import numpy as np
import yaml

from pyhorn_core.solver.profiles import profile_area_at_distance

from pyhorn_ml.core.space import DesignSpace
from pyhorn_ml.core.target import TargetResponse
from pyhorn_ml.core.design_point import DesignPoint
from pyhorn_ml.core.score import score_design
from pyhorn_ml.pipeline.evaluate import evaluate_design
from pyhorn_ml.optimization.bayesian import BayesianOptimizer
from pyhorn_ml.optimization.pareto import extract_pareto_front, assign_pareto_ranks
from pyhorn_ml.plot.pareto import plot_pareto_front, plot_pareto_3d
from pyhorn_ml.plot.convergence import plot_convergence
from pyhorn_ml.plot.response import plot_response_overlay
from pyhorn_ml.data.dataset import DesignDataset


def _plain(obj):
    """Convert numpy types to native Python for clean YAML serialization."""
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [_plain(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_plain(x) for x in obj]
    return obj




def run_optimize(
    driver: str,
    target_str: str,
    fmin: float,
    fmax: float,
    budget: int,
    output: str,
    n_initial: int = 20,
    acquisition: str = "ei",
    seed: int = 42,
) -> None:
    """Run Bayesian multi-objective optimisation and save results.

    Loop:
        Phase 1 (n_initial evals):   Latin Hypercube random sampling
        Phase 2 (remaining evals):  Bayesian optimisation with GP surrogate
        → Expected Improvement acquisition → Pareto front extraction
    """
    t_start = time.time()

    # ── Resolve relative paths (workdir may differ from project root) ─────
    _ROOT = Path(os.path.expanduser("~/pyhorn"))
    driver_path = (
        str((_ROOT / driver).resolve()) if not Path(driver).is_absolute() else driver
    )

    print(f"[pyhorn-ml] Starting optimisation")
    print(f"  driver:   {driver_path}")
    print(f"  targets:  {target_str}")
    print(
        f"  budget:   {budget} evaluations  ({n_initial} initial + {budget-n_initial} BO)"
    )
    print(f"  output:   {output}")
    print(f"  acq:     {acquisition.upper()}")
    print()

    # ── Parse inputs ───────────────────────────────────────────────────────
    target = TargetResponse.from_str(target_str)
    target.f_min = fmin
    target.f_max = fmax

    space = DesignSpace()

    # ── Folder structure ────────────────────────────────────────────────────
    data_dir = os.path.join(output, "data")
    top_dir = os.path.join(output, "top")
    plots_dir = os.path.join(output, "plots")
    os.makedirs(output, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(top_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # ── Dataset ────────────────────────────────────────────────────────────
    dataset = DesignDataset(os.path.join(data_dir, ".dataset"))
    dataset.init(dataclasses.asdict(target))

    # ── Optimiser ──────────────────────────────────────────────────────────
    opt = BayesianOptimizer(
        design_space=space,
        target=target,
        n_initial=n_initial,
        acquisition=acquisition,
        random_state=seed,
    )

    all_designs: list[DesignPoint] = []

    def print_design(dp: DesignPoint, elapsed: float) -> None:
        print(
            f"  [{len(all_designs):3d}/{budget}] "
            f"{dp.name}  "
            f"SPL={np.mean(dp.real_spl or 0):.1f}dB  "
            f"score={dp.score:.3f}  "
            f"({elapsed:.1f}s)"
        )

    # ── Phase 1 + 2 combined: ask/tell loop ─────────────────────────────────
    print(f"[{time.strftime('%H:%M:%S')}] Phase 1: {n_initial} random evaluations...")
    for i in range(budget):
        dp = opt.ask()
        t0 = time.time()
        evaluate_design(dp, driver_path, target, fmin=fmin, fmax=fmax, n_points=200)
        elapsed = time.time() - t0

        opt.tell(dp)
        dataset.add(dp)
        all_designs.append(dp)

        print_design(dp, elapsed)

        if i == n_initial - 1 and budget > n_initial:
            elapsed_total = time.time() - t_start
            print(
                f"\n[{time.strftime('%H:%M:%S')}] Phase 2: {budget - n_initial} Bayesian evaluations..."
            )
            print(
                f"  surrogate: {opt._surrogate.name if opt._surrogate else 'fitting...'}"
            )
            print(f"  elapsed so far: {elapsed_total:.0f}s")

    # ── Pareto analysis ─────────────────────────────────────────────────────
    assign_pareto_ranks(all_designs)
    pareto_front = [d for d in all_designs if not d.dominated]
    top5 = sorted(all_designs, key=lambda x: -x.score)[:5]

    elapsed_total = time.time() - t_start
    print(f"\n[{time.strftime('%H:%M:%S')}] Done in {elapsed_total:.0f}s")
    print(f"  Evaluated: {len(all_designs)} designs")
    print(f"  Pareto front: {len(pareto_front)} non-dominated designs")
    print(f"  Best score: {max(d.score for d in all_designs):.3f}")

    # ── Save all designs ────────────────────────────────────────────────────
    all_yaml = {
        "optimisation": {
            "budget": budget,
            "n_initial": n_initial,
            "acquisition": acquisition,
            "target": _plain(dataclasses.asdict(target)),
            "driver": driver,
            "elapsed_seconds": round(elapsed_total, 1),
        },
        "designs": [
            _plain(d.to_dict()) for d in sorted(all_designs, key=lambda x: -x.score)
        ],
    }
    all_path = os.path.join(data_dir, "all_designs.yaml")
    with open(all_path, "w") as f:
        yaml.dump(all_yaml, f, allow_unicode=True)
    print(f"  → {all_path}")

    # ── Save top-5 as individual YAMLs ─────────────────────────────────────
    for rank, dp in enumerate(top5):
        geom = dp.geometry_params
        throat_r = float(geom["throat_area"]) / math.pi
        mouth_r = float(geom["mouth_area"]) / math.pi
        compression_ratio = float(geom["mouth_area"]) / float(geom["throat_area"])
        profile = geom["profile_type"]

        flare_constant = None
        if profile == "exponential":
            m = math.log(float(geom["mouth_area"]) / float(geom["throat_area"]))
            flare_constant = round(m / float(geom["path_length"]), 4)
        elif profile == "hyperbolic":
            ratio = math.sqrt(float(geom["mouth_area"]) / float(geom["throat_area"]))
            if ratio > 1:
                xh = float(geom["path_length"]) / math.acosh(ratio)
                flare_constant = round(1.0 / xh, 4)

        seg_areas = []
        seg_positions_mm = []
        seg_diameters_cm = []
        path_len = float(geom["path_length"])
        t_area = float(geom["throat_area"])
        m_area = float(geom["mouth_area"])
        for i in range(int(geom["n_segments"])):
            x_dist = path_len * (i + 1) / float(geom["n_segments"])
            s = profile_area_at_distance(profile, t_area, m_area, path_len, x_dist)
            seg_areas.append(round(s, 6))
            seg_positions_mm.append(round(x_dist * 1000, 1))
            seg_diameters_cm.append(round(2 * math.sqrt(s / math.pi) * 100, 2))

        # Horn-acoustic metrics from profiles module
        from pyhorn_core.solver.profiles import horn_profile_metrics
        metrics = horn_profile_metrics(profile, t_area, m_area, path_len)

        yaml_dict = {
            "name": dp.name,
            "score": round(float(dp.score), 4),
            "scores": {
                "flatness": round(float(dp.flatness_score), 4),
                "sensitivity": round(float(dp.sensitivity_score), 4),
                "bass": round(float(dp.bass_score), 4),
            },
            "throat_area_m2": round(float(geom["throat_area"]), 6),
            "throat_diameter_cm": round(2 * math.sqrt(throat_r) * 100, 2),
            "mouth_area_m2": round(float(geom["mouth_area"]), 6),
            "mouth_diameter_cm": round(2 * math.sqrt(mouth_r) * 100, 2),
            "path_length_m": round(float(geom["path_length"]), 3),
            "compression_ratio": round(compression_ratio, 3),
            "profile_type": profile,
            "flare_constant_per_m": flare_constant,
            "n_segments": int(geom["n_segments"]),
            "fold_style": geom.get("fold_style", "straight"),
            "cutoff_penalty": round(float(dp.cutoff_penalty), 4),
            "lrc_m": round(float(geom.get("lrc", 0.0)), 4),
            "vtc_m3": round(float(geom.get("vtc", 0.0)), 6),
            "pareto_rank": int(dp.pareto_rank),
            "dominated": bool(dp.dominated),
            "segment_areas_m2": seg_areas,
            "segment_positions_mm": seg_positions_mm,
            "segment_diameters_cm": seg_diameters_cm,
            # Horn-acoustic metrics
            "cutoff_hz": round(float(metrics["cutoff_hz"]), 1),
            "tl_tuning_hz": round(float(metrics["tl_tuning_hz"]), 1),
            "krm": round(float(metrics["krm"]), 3),
            "kaL": round(float(metrics["kaL"]), 3),
            "mouth_rating": metrics["mouth_rating"],
            "mouth_krm_min_hz": round(float(metrics["mouth_krm_min_hz"]), 1),
            "mouth_ko_cm": round(float(metrics["mouth_ko"]) * 100, 1),
        }
        path = os.path.join(top_dir, f"design_{rank+1:03d}_{dp.name}.yaml")
        with open(path, "w") as f:
            yaml.dump(yaml_dict, f, allow_unicode=True)
    print(f"  → {top_dir}/design_001_*.yaml ...")

    # ── Save Pareto front ───────────────────────────────────────────────────
    pareto_yaml = {
        "n_designs": len(pareto_front),
        "designs": [
            _plain(d.to_dict()) for d in sorted(pareto_front, key=lambda x: -x.score)
        ],
    }
    pf_path = os.path.join(data_dir, "pareto_front.yaml")
    with open(pf_path, "w") as f:
        yaml.dump(pareto_yaml, f, allow_unicode=True)
    print(f"  → {pf_path}")

    # ── Plots ───────────────────────────────────────────────────────────────
    print("\n[pyhorn-ml] Generating plots...")

    plot_convergence(
        all_designs,
        os.path.join(plots_dir, "convergence.png"),
        title="Bayesian Optimisation — Score Convergence",
    )

    plot_pareto_front(
        all_designs,
        pareto_front,
        os.path.join(plots_dir, "pareto_front.png"),
        title="Pareto Front — Horn Design Optimisation",
    )

    if len(pareto_front) >= 4:
        plot_pareto_3d(
            pareto_front,
            os.path.join(plots_dir, "pareto_front_3d.png"),
        )

    if top5:
        for dp in top5:
            if dp.freq is None:
                dp.freq = np.linspace(fmin, fmax, 200)
        plot_response_overlay(
            top5,
            os.path.join(plots_dir, "top5_spl.png"),
            target_spl=target.target_spl,
            f_min=fmin,
            f_max=fmax,
            title="Top 5 Designs — SPL Response",
        )

    # ── Report ─────────────────────────────────────────────────────────────
    from pyhorn_ml.cli.report import write_report

    write_report(
        output_dir=output,
        driver=driver,
        target_str=target_str,
        fmin=fmin,
        fmax=fmax,
        budget=budget,
        n_initial=n_initial,
        acquisition=acquisition,
        seed=seed,
        elapsed_seconds=elapsed_total,
        all_designs=all_designs,
        pareto_front=pareto_front,
        top5=top5,
    )

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n[pyhorn-ml] Complete in {elapsed_total:.0f}s")
    print(f"  Results:     {output}/")
    print(f"  README.md            — run report with command + top-5 specs")
    print(f"  config.yaml          — exact params for reproducibility")
    print(f"  data/all_designs.yaml   — all {len(all_designs)} designs")
    print(f"  data/pareto_front.yaml  — {len(pareto_front)} non-dominated")
    print(f"  top/                   — top-5 as individual YAMLs")
    print(f"  plots/                 — convergence, Pareto, SPL plots")
