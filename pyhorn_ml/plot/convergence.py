"""Convergence plot — score vs. evaluation number."""
import os
from typing import List
import numpy as np
import matplotlib.pyplot as plt

from pyhorn_ml.core.design_point import DesignPoint


def plot_convergence(
    designs: List[DesignPoint],
    output_path: str,
    title: str = "Optimisation Convergence",
) -> None:
    """Plot best score and hypervolume proxy vs. evaluation number.

    Shows:
    - Upper envelope (best score so far) — should trend upward
    - Per-iteration scores — shows search diversity
    """
    n = len(designs)
    scores = [d.score for d in designs]
    indices = list(range(1, n + 1))

    # Running best (upper envelope)
    best_so_far = [max(scores[:i]) for i in range(1, n + 1)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: all scores + best-so-far
    ax = axes[0]
    ax.scatter(indices, scores, c="steelblue", alpha=0.5, s=25, label="Per evaluation")
    ax.plot(indices, best_so_far, c="crimson", linewidth=2, label="Best so far", zorder=5)
    ax.set_xlabel("Evaluation number")
    ax.set_ylabel("Score (0–1)")
    ax.set_title("Score per Evaluation")
    ax.legend()
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, n + 1)
    ax.set_ylim(0, 1.05)

    # Right: Pareto rank over time
    ax2 = axes[1]
    pareto_ranks = [d.pareto_rank if d.pareto_rank >= 0 else -1 for d in designs]
    pareto_ranks_arr = np.array(pareto_ranks, dtype=float)
    pareto_ranks_arr[pareto_ranks_arr < 0] = np.nan

    ax2.scatter(indices, pareto_ranks_arr, c="steelblue", alpha=0.5, s=25)
    ax2.set_xlabel("Evaluation number")
    ax2.set_ylabel("Pareto Rank (0 = best front)")
    ax2.set_title("Pareto Rank per Evaluation")
    ax2.grid(True, alpha=0.2)
    ax2.set_xlim(0, n + 1)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Convergence plot: {output_path}")
