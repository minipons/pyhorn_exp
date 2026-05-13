"""SPL response overlay — compare multiple designs on one plot."""
import os
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt

from pyhorn_ml.core.design_point import DesignPoint


COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def plot_response_overlay(
    designs: List[DesignPoint],
    output_path: str,
    target_spl: Optional[float] = None,
    f_min: float = 20.0,
    f_max: float = 5000.0,
    title: str = "SPL Response Comparison",
) -> None:
    """Plot SPL curves for multiple designs on one axis.

    Args:
        designs: list of DesignPoints (must have real_spl + freq set)
        output_path: where to save the PNG
        target_spl: optional horizontal reference line
        f_min: x-axis lower bound
        f_max: x-axis upper bound
    """
    fig, ax = plt.subplots(figsize=(13, 6))

    for i, dp in enumerate(designs):
        if dp.freq is None or dp.real_spl is None:
            continue
        color = COLORS[i % len(COLORS)]
        label = f"{dp.name}  (score={dp.score:.3f})"
        ax.plot(dp.freq, dp.real_spl, color=color, linewidth=1.5, label=label, alpha=0.85)

    if target_spl is not None:
        ax.axhline(target_spl, color="grey", linewidth=1, linestyle="--", alpha=0.7, label=f"Target {target_spl:.0f} dB")

    ax.set_xscale("log")
    ax.set_xlim(f_min, f_max)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("SPL (dB re 2.83V @ 1m)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best", framealpha=0.8)
    ax.grid(True, which="both", alpha=0.25)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  SPL overlay: {output_path}")
