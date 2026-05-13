"""
pyhorn_wavefront — 2D Wavefront Simulator for pyhorn.

Extracted from pyhorn_core/solver/wavefront.py (commit: 735ac92 + subsequent phases).

Public API
----------
WavefrontGrid       — 2-D rectangular simulation grid container
solve_2d_wave       — direct sparse Helmholtz solver
solve_2d_wave_pml   — PML-damped Helmholtz solver
boundary_condition_mask  — absorbing mask for horn walls
load_horn_geometry  — load geometry from YAML
edit_horn_geometry  — interactive wall editor
ka_warning          — k·a > 0.5 caution flag
plot_wavefront      — wavefront overlay on horn geometry
plot_pressure_amplitude  — pressure colormap
animate_wave_propagation — steady-state phase cycle animation
plot_wavefront_polar    — mouth polar directivity from 2-D pressure field
"""

from pyhorn_wavefront.wavefront import (
    # Core classes and functions
    WavefrontGrid,
    solve_2d_wave,
    solve_2d_wave_pml,
    boundary_condition_mask,
    load_horn_geometry,
    edit_horn_geometry,
    edit_horn_geometry_from_yaml,
    edit_horn_geometry_and_simulate,
    pml_damping_mask,
    compute_pressure_field,
    ka_warning,
    # Plotting
    plot_wavefront,
    plot_pressure_amplitude,
    plot_amplitude_db,
    animate_wave_propagation,
    plot_animation_frames,
    plot_wavefront_polar,
    # Time domain
    solve_2d_wave_time_domain,
    solve_2d_wave_time_domain_pml,
    WavefrontGrid_animate,
)

__all__ = [
    "WavefrontGrid",
    "solve_2d_wave",
    "solve_2d_wave_pml",
    "boundary_condition_mask",
    "load_horn_geometry",
    "edit_horn_geometry",
    "edit_horn_geometry_from_yaml",
    "edit_horn_geometry_and_simulate",
    "pml_damping_mask",
    "compute_pressure_field",
    "ka_warning",
    "plot_wavefront",
    "plot_pressure_amplitude",
    "plot_amplitude_db",
    "animate_wave_propagation",
    "plot_animation_frames",
    "plot_wavefront_polar",
    "solve_2d_wave_time_domain",
    "solve_2d_wave_time_domain_pml",
    "WavefrontGrid_animate",
]
