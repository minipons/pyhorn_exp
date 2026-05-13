"""Build a human-readable report from optimisation results."""
import os
import math
from datetime import datetime

from pyhorn_core.solver.profiles import profile_area_at_distance
from pyhorn_ml.core.design_point import DesignPoint


def write_report(
    output_dir: str,
    driver: str,
    target_str: str,
    fmin: float,
    fmax: float,
    budget: int,
    n_initial: int,
    acquisition: str,
    seed: int,
    elapsed_seconds: float,
    all_designs: list[DesignPoint],
    pareto_front: list[DesignPoint],
    top5: list[DesignPoint],
) -> None:
    """Write README.md and config.yaml into the output directory."""
    os.makedirs(output_dir, exist_ok=True)

    # ── config.yaml: exact reproducibility params ─────────────────────────
    config = {
        "driver": driver,
        "target_str": target_str,
        "fmin": fmin,
        "fmax": fmax,
        "budget": budget,
        "n_initial": n_initial,
        "acquisition": acquisition,
        "seed": seed,
    }
    import yaml
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump({"run": config, "elapsed_seconds": round(elapsed_seconds, 1)}, f)

    # ── README.md ───────────────────────────────────────────────────────────
    best = max(all_designs, key=lambda d: d.score)
    n_pareto = len(pareto_front)

    lines = [
        f"# pyhorn-ml Optimisation Report",
        f"",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"## Command",
        f"",
        f"```bash",
        f"pyhorn-ml optimize \\",
        f"  --driver {driver} \\",
        f"  --target {target_str} \\",
        f"  --fmin {fmin} --fmax {fmax} \\",
        f"  --budget {budget} \\",
        f"  --n-initial {n_initial} \\",
        f"  --acquisition {acquisition} \\",
        f"  --seed {seed} \\",
        f"  --output {output_dir}",
        f"```",
        f"",
        f"## Summary",
        f"",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total evaluations | {len(all_designs)} |",
        f"| LHS initial | {n_initial} |",
        f"| BO evaluations | {len(all_designs) - n_initial} |",
        f"| Pareto front size | {n_pareto} |",
        f"| Best score | {best.score:.4f} |",
        f"| Best design | `{best.name}` |",
        f"| Elapsed | {elapsed_seconds:.1f}s |",
        f"| f_min | {fmin} Hz |",
        f"| f_max | {fmax} Hz |",
        f"",
        f"## Top-5 Designs",
        f"",
    ]

    for rank, dp in enumerate(top5, 1):
        g = dp.geometry_params
        throat_r = float(g["throat_area"]) / math.pi
        mouth_r = float(g["mouth_area"]) / math.pi
        cr = float(g["mouth_area"]) / float(g["throat_area"])
        lines.append(f"### {rank}. `{dp.name}`")
        lines.append(f"")
        lines.append(f"| Parameter | Value |")
        lines.append(f"|---|---|")
        lines.append(f"| Score | {dp.score:.4f} |")
        lines.append(f"| Profile | {g['profile_type']} |")
        lines.append(f"| Throat Ø | {2*math.sqrt(throat_r)*100:.1f} cm |")
        lines.append(f"| Mouth Ø | {2*math.sqrt(mouth_r)*100:.1f} cm |")
        lines.append(f"| Path length | {float(g['path_length'])*1000:.0f} mm |")
        lines.append(f"| Segments | {g['n_segments']} |")
        lines.append(f"| Compression ratio | {cr:.2f} |")
        lines.append(f"| Fold style | {g.get('fold_style','straight')} |")
        lines.append(f"| lrc | {float(g.get('lrc',0)):.4f} m |")
        lines.append(f"| vtc | {float(g.get('vtc',0))*1e6:.1f} cm³ |")
        lines.append(f"| Cutoff penalty | {dp.cutoff_penalty:.2f} |")
        lines.append(f"| Pareto rank | {dp.pareto_rank} |")
        lines.append(f"")
        lines.append(f"| Objective | Score |")
        lines.append(f"|---|---:|")
        lines.append(f"| Flatness | {dp.flatness_score:.3f} |")
        lines.append(f"| Sensitivity | {dp.sensitivity_score:.3f} |")
        lines.append(f"| Bass | {dp.bass_score:.3f} |")
        lines.append(f"")
        lines.append(f"```yaml")
        lines.append(f"name: {dp.name}")
        lines.append(f"throat_area_m2: {float(g['throat_area']):.6f}")
        lines.append(f"mouth_area_m2: {float(g['mouth_area']):.6f}")
        lines.append(f"path_length_m: {float(g['path_length']):.3f}")
        lines.append(f"lrc_m: {float(g.get('lrc',0)):.4f}")
        lines.append(f"vtc_m3: {float(g.get('vtc',0)):.6f}")
        lines.append(f"profile_type: {g['profile_type']}")
        lines.append(f"n_segments: {g['n_segments']}")
        lines.append(f"fold_style: {g.get('fold_style','straight')}")
        lines.append(f"")

        # ── Segment table ─────────────────────────────────────────────────
        profile_geo = g["profile_type"]
        path_geo = float(g["path_length"])
        t_area = float(g["throat_area"])
        m_area = float(g["mouth_area"])
        n_seg = int(g["n_segments"])

        seg_data = []
        for i in range(n_seg):
            x_dist = path_geo * (i + 1) / n_seg
            s = profile_area_at_distance(profile_geo, t_area, m_area, path_geo, x_dist)
            seg_data.append((i+1, round(x_dist * 1000, 1), round(s, 6), round(2*math.sqrt(s/math.pi)*100, 2)))

        lines.append(f"### Segment geometry")
        lines.append(f"")
        lines.append(f"| # | x(mm) | Area(m²) | Ø(cm) |")
        lines.append(f"|---|---|---|---|")
        for seg_num, x_mm, area, diam in seg_data:
            lines.append(f"| {seg_num} | {x_mm} | {area} | {diam} |")
        lines.append(f"")

    lines += [
        f"## Files",
        f"",
        f"| File | Description |",
        f"|---|---|",
        f"| `config.yaml` | Exact parameters for reproducibility |",
        f"| `data/all_designs.yaml` | All {len(all_designs)} designs ranked |",
        f"| `data/pareto_front.yaml` | {n_pareto} non-dominated designs |",
        f"| `top/design_001_*.yaml` ... | Top-5 as individual specs |",
        f"| `plots/convergence.png` | Score convergence over evaluations |",
        f"| `plots/pareto_front.png` | 2D Pareto projections |",
        f"| `plots/pareto_front_3d.png` | 3D Pareto scatter |",
        f"| `plots/top5_spl.png` | SPL curves of top-5 |",
        f"| `.dataset/` | Raw JSONL data (for --resume) |",
    ]

    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write("\n".join(lines))

    print(f"  → {output_dir}/README.md")
