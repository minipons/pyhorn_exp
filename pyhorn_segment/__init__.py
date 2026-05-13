"""
pyhorn_segment — Medial-axis / auto-segment computation for pyhorn.

This package provides the Medial Axis centerline extraction and cross-section
width computation used to convert Onshape sketch exports into horn geometry.

Public API
----------
generate_auto_segments(json_path, output_yaml, ...)
    Main entry point: read a Pyhorn Auto-Segment JSON export, compute the
    Medial Axis centerline and widths, emit a Pyhorn YAML config.

rectangular_segments_to_sections(rectangular_segments, width)
    Convert rectangular segment lists to chained-sections format.

_reduce_stair_points(...)
    Drop stations that don't meaningfully change direction or width.

_remove_duplicate_stations(...)
    Collapse stations that collapse to the same location.
"""

from pyhorn_segment.segment import (
    generate_auto_segments,
    rectangular_segments_to_sections,
    _reduce_stair_points,
    _remove_duplicate_stations,
    _distance_to_nearest_wall,
    _perpendicular_width_at_point,
    _build_wall_lines,
    _infer_profile_type,
    _generate_graph_method,
    _generate_voronoi_method,
)

__all__ = [
    "generate_auto_segments",
    "rectangular_segments_to_sections",
    "_reduce_stair_points",
    "_remove_duplicate_stations",
    "_distance_to_nearest_wall",
    "_perpendicular_width_at_point",
    "_build_wall_lines",
    "_infer_profile_type",
    "_generate_graph_method",
    "_generate_voronoi_method",
]
