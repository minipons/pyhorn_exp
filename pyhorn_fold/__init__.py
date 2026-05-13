"""pyhorn_fold — SLSQP-based horn folding (extracted from pyhorn_core.solver.folding).

Re-exports the full public API so callers can use either:
    from pyhorn_fold import extrapolate_folded_horn, throat_chamber_side_length
    from pyhorn_fold.folding import _point_along_segment, _throat_attachment_point, ...
"""

from pyhorn_fold.folding import (
    _point_along_segment,
    _throat_attachment_point,
    _fold_segment_heights,
    _fold_path_polygons,
    _build_vertical_fold_path_from_lanes,
    _build_vertical_boundary_terminal_l_path,
    _build_vertical_fold_path_optimized,
    _build_fold_path,
    _reflect_lanes_from_mirrors,
    _segment_height_limit,
    throat_chamber_side_length,
    extrapolate_folded_horn,
)

__all__ = [
    # Public API
    "throat_chamber_side_length",
    "extrapolate_folded_horn",
    # Private helpers (used in tests)
    "_point_along_segment",
    "_throat_attachment_point",
    "_segment_height_limit",
    "_fold_segment_heights",
    "_fold_path_polygons",
    "_build_vertical_fold_path_from_lanes",
    "_build_vertical_boundary_terminal_l_path",
    "_build_vertical_fold_path_optimized",
    "_build_fold_path",
    "_reflect_lanes_from_mirrors",
]
