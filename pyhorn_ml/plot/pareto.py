"""Pareto front visualization — 2D and 3D slices of the objective space."""
import os
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from pyhorn_ml.core.design_point import DesignPoint


def plot_pareto_front(
    designs: List[DesignPoint],
    pareto_front: List[DesignPoint],
    output_path: str,
    title: str = "Pareto Front — Horn Design Optimisation",
) -> None:
    """Plot the 3-objective Pareto front with 2D scatter projections.

    Creates a 3-panel figure showing flatness vs sensitivity, flatness vs bass,
    and sensitivity vs bass. Pareto-optimal designs are highlighted.
    """
    all_dp = list(designs)
    pf_set = set(id(d) for d in pareto_front)

    def get_scores(dp: DesignPoint) -> tuple:
        return dp.flatness_score, dp.sensitivity_score, dp.bass_score

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    pairs = [
        ("flatness_score", "sensitivity_score", "Flatness", "Sensitivity"),
        ("flatness_score", "bass_score", "Flatness", "Bass Extension"),
        ("sensitivity_score", "bass_score", "Sensitivity", "Bass Extension"),
    ]

    for ax, (x_key, y_key, x_label, y_label) in zip(axes, pairs):
        # Non-Pareto
        non_pf = [d for d in all_dp if id(d) not in pf_set]
        xs_npf = [getattr(d, x_key) for d in non_pf]
        ys_npf = [getattr(d, y_key) for d in non_pf]
        ax.scatter(xs_npf, ys_npf, c="steelblue", alpha=0.4, s=30, label="Dominated", zorder=1)

        # Pareto
        xs_pf = [getattr(d, x_key) for d in pareto_front]
        ys_pf = [getattr(d, y_key) for d in pareto_front]
        ax.scatter(xs_pf, ys_pf, c="crimson", s=60, zorder=3, label="Pareto front")

        # Connect Pareto points with a line
        order = np.argsort(xs_pf)
        ax.plot([xs_pf[i] for i in order], [ys_pf[i] for i in order], c="crimson", linewidth=1.5, alpha=0.6, zorder=2)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.set_title(f"{x_label} vs {y_label}")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Pareto front plot: {output_path}")


def plot_pareto_3d(
    pareto_front: List[DesignPoint],
    output_path: str,
    title: str = "Pareto Front (3D Objective Space)",
) -> None:
    """Interactive-style 3D scatter of the Pareto front in objective space."""
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    xs = [d.flatness_score for d in pareto_front]
    ys = [d.sensitivity_score for d in pareto_front]
    zs = [d.bass_score for d in pareto_front]

    # Color by total score
    colors = [d.score for d in pareto_front]
    sc = ax.scatter(xs, ys, zdir="z", zs=zs, c=colors, cmap="plasma", s=60, zorder=3)  # type: ignore[reportArgumentType]

    # Connect in order of one objective
    order = np.argsort(xs)
    ax.plot([xs[i] for i in order], [ys[i] for i in order], [zs[i] for i in order],
            c="grey", linewidth=1, alpha=0.5, zorder=2)

    ax.set_xlabel("Flatness")
    ax.set_ylabel("Sensitivity")
    ax.set_zlabel("Bass Extension")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_zlim(0, 1.05)
    ax.set_title(title)
    fig.colorbar(sc, label="Total Score", shrink=0.6)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  3D Pareto plot: {output_path}")
