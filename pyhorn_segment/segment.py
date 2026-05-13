"""
Medial-axis / auto-segment computation for pyhorn.

Public API
----------
generate_auto_segments(json_path, output_yaml, ...)
    Read a Pyhorn Auto-Segment JSON export, compute the Medial Axis
    centerline and cross-section widths, and emit a Pyhorn YAML config.

Helper functions (public for testing)
--------------------------------------
rectangular_segments_to_sections
    Convert rectangular segments to chained-sections format.

_reduce_stair_points
    Keep only stations where direction or local width changes materially.

_remove_duplicate_stations
    Drop consecutive stations that collapse to the same location.
"""

import json
import logging
import yaml
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon, LineString, Point, MultiLineString
from shapely.ops import linemerge, polygonize, unary_union
from scipy.spatial import Voronoi
from scipy.signal import savgol_filter
import networkx as nx

from typing import Optional, List

logger = logging.getLogger(__name__)


# ─── Sections format converter ─────────────────────────────────────────────────

_AREA_RATIO_STRAIGHT_THRESHOLD = 1.1  # area ratio below this → "straight" profile


def _infer_profile_type(start_area: float, end_area: float) -> str:
    """Infer profile type from first/last segment area.

    Rules (from BACKLOG.md agreed format):
      - area_ratio < 1.1  → "straight"  (near-constant cross-section)
      - otherwise          → "exponential"  (expanding flare)

    Hyperbolic is a higher-level design choice not inferable from geometry alone.
    """
    if start_area <= 0 or end_area <= 0:
        return "exponential"
    ratio = max(start_area, end_area) / min(start_area, end_area)
    return "straight" if ratio < _AREA_RATIO_STRAIGHT_THRESHOLD else "exponential"


def rectangular_segments_to_sections(
    rectangular_segments: List[List[float]],
    width: float,
) -> List[dict]:
    """Convert rectangular segments to chained-sections format.

    Each rectangular segment is:
        [width_m, height_start_m, width_end_m, height_end_m, length_m]

    Consecutive segments with similar expansion ratios are grouped into one section.
    The section's profile_type is inferred from the overall area change:
      - area_ratio < 1.1  → "straight"  (near-constant area)
      - otherwise          → "exponential"  (expanding flare)

    Returns a list of section dicts matching the ``sections`` YAML format:
        {name, profile_type, length, start_area, end_area}
    """
    if not rectangular_segments:
        return []

    # ── Height smoothing (throat region only) ───────────────────────────────────
    # Fragmented Onshape sketch edges cause oscillating heights in segments 5-8
    # (47% to -23% per-segment height changes → standing-wave resonances at
    # 200-500 Hz).  Apply Savitzky-Golay smoothing ONLY to the throat portion
    # (first ~15 boundaries, roughly first 150mm of path) where fragmentation
    # is concentrated.  The rest of the profile (main horn flare, mouth) is
    # left unsmoothed to preserve genuine geometric discontinuities.
    n_segs = len(rectangular_segments)
    boundary_heights = [rectangular_segments[0][1]]  # h[0] = first segment's start height
    for seg in rectangular_segments:
        boundary_heights.append(seg[3])             # h[i] = each segment's end height

    # Throat region: first 15 boundaries (or fewer if total segments < 15)
    n_throat = min(15, n_segs + 1)
    # SG requires win > order (poly_order=3 → need win >= 5)
    if n_throat >= 5:
        win = min(5, n_throat)
        if win % 2 == 0:
            win -= 1
        if win > 3:  # win=5 is valid for order=3 (5 > 3)
            throat_h = boundary_heights[:n_throat]
            smoothed_throat = savgol_filter(np.array(throat_h), win, 3).tolist()
            # Re-clamp to positive and merge with unsmoothed tail
            smoothed_throat = [max(h, 0.001) for h in smoothed_throat]
            boundary_heights = smoothed_throat + boundary_heights[n_throat:]

    # Compute area at start and end of each segment
    # area = width × height  (rectangular cross-section)
    segment_areas: List[List[float]] = []  # [area_start, area_end] per segment
    for i_seg, seg in enumerate(rectangular_segments):
        w = seg[0]          # width (constant per segment, but keep for clarity)
        h_start = boundary_heights[i_seg]      # smoothed start height
        _w_end = seg[2]     # width at segment end (should == w for most segments)
        h_end = boundary_heights[i_seg + 1]   # smoothed end height
        area_start = w * h_start
        area_end = w * h_end
        segment_areas.append([area_start, area_end])

    # Group consecutive segments by expansion ratio similarity
    groups: List[List[int]] = []  # list of segment-index lists
    current_group: List[int] = [0]

    for i in range(1, len(segment_areas)):
        prev_area = segment_areas[i - 1]
        curr_area = segment_areas[i]

        # Expansion ratio for each segment: end_area / start_area
        # Use the inner ratio (end of prev / start of prev, end of curr / start of curr)
        prev_ratio = prev_area[1] / prev_area[0] if prev_area[0] > 0 else 1.0
        curr_ratio = curr_area[1] / curr_area[0] if curr_area[0] > 0 else 1.0

        # Group if ratios are within ~5% of each other
        if prev_ratio > 0:
            ratio_diff = abs(curr_ratio - prev_ratio) / prev_ratio
        else:
            ratio_diff = 1.0

        if ratio_diff < 0.05:
            current_group.append(i)
        else:
            groups.append(current_group)
            current_group = [i]

    groups.append(current_group)

    # Build sections from groups
    sections: List[dict] = []
    for group_idx, group in enumerate(groups):
        # Cumulative length
        total_length = sum(rectangular_segments[i][4] for i in group)

        # Start area = area at start of first segment in group
        start_area = segment_areas[group[0]][0]

        # End area = area at end of last segment in group
        end_area = segment_areas[group[-1]][1]

        # Infer profile type
        profile_type = _infer_profile_type(start_area, end_area)

        sections.append({
            "name": f"segment_{group_idx + 1}",
            "profile_type": profile_type,
            "length": round(total_length, 6),
            "start_area": round(start_area, 8),
            "end_area": round(end_area, 8),
        })

    return sections


# ─── Core medial-axis helpers ──────────────────────────────────────────────────

def _distance_to_nearest_wall(
    pt: tuple[float, float],
    wall_lines: list[LineString],
) -> float:
    """Perpendicular Euclidean distance from a 2D point to the nearest wall segment."""
    px, py = pt
    min_dist = float("inf")
    for line in wall_lines:
        coords = np.array(line.coords)
        p1, p2 = coords[0], coords[-1]
        vx, vy = p2[0] - p1[0], p2[1] - p1[1]
        wx, wy = px - p1[0], py - p1[1]
        denom = vx * vx + vy * vy
        if denom < 1e-12:
            t = 0.0
        else:
            t = max(0.0, min(1.0, (wx * vx + wy * vy) / denom))
        proj_x = p1[0] + t * vx
        proj_y = p1[1] + t * vy
        d = np.hypot(px - proj_x, py - proj_y)
        if d < min_dist:
            min_dist = d
    return min_dist


def _perpendicular_width_at_point(
    pt: tuple[float, float],
    tangent: tuple[float, float],
    wall_lines: list[LineString],
    max_ray: float = 0.5,
) -> Optional[float]:
    """
    Measure the full cross-section width perpendicular to the path at a given point.

    Shoots two rays in the ±normal direction from the point and finds the nearest
    wall intersection in each direction.  Returns 2 × min(intersection distances),
    or None if either ray misses all walls within max_ray.

    Parameters
    ----------
    pt       : (x, y) centerline point
    tangent  : (dx, dy) unit tangent vector along path at pt
    wall_lines: list of wall LineStrings
    max_ray  : maximum ray length in metres

    Returns
    -------
    Width in metres, or None if perpendicular approach fails at this point.
    """
    tx, ty = tangent
    nx_, ny_ = -ty, tx  # normal (perpendicular to tangent)

    half_widths = []
    for sign in (-1, 1):
        ox = pt[0] + sign * 1e-4 * tx
        oy = pt[1] + sign * 1e-4 * ty
        rdx, rdy = nx_ * sign, ny_ * sign
        best_t = float("inf")

        for wall in wall_lines:
            wc = np.array(wall.coords)
            for j in range(len(wc) - 1):
                w1, w2 = wc[j], wc[j + 1]
                wx, wy = w2[0] - w1[0], w2[1] - w1[1]
                wx_, wy_ = ox - w1[0], oy - w1[1]
                denom = rdx * wy - rdy * wx
                if abs(denom) < 1e-12:
                    continue
                t = (wx * wy_ - wy * wx_) / denom
                u = (rdx * wy_ - rdy * wx_) / denom
                if 0.0 < t < best_t and 0.0 <= u <= 1.0:
                    best_t = t

        if best_t < max_ray:
            half_widths.append(best_t)

    if not half_widths:
        return None
    return 2.0 * min(half_widths)


def _reduce_stair_points(
    points: list[list[float]],
    widths: list[float],
    distances: Optional[list[float]] = None,
    angle_threshold_deg: float = 25.0,
    width_rel_threshold: float = 0.10,
    width_abs_threshold: float = 0.004,
    min_segment_length: float = 0.0,
    max_segment_length: Optional[float] = None,
) -> tuple[list[list[float]], list[float], list[float]]:
    """Keep only stations where direction or local width changes materially."""
    if len(points) <= 2:
        if distances is None:
            distances = [0.0 for _ in points]
        return points, widths, distances

    if distances is None:
        distances = [float(i) for i in range(len(points))]

    reduced_points = [points[0]]
    reduced_widths = [widths[0]]
    reduced_distances = [distances[0]]
    cos_threshold = np.cos(np.deg2rad(angle_threshold_deg))

    for i in range(1, len(points) - 1):
        prev_pt = np.array(points[i - 1], dtype=float)
        curr_pt = np.array(points[i], dtype=float)
        next_pt = np.array(points[i + 1], dtype=float)

        v1 = curr_pt - prev_pt
        v2 = next_pt - curr_pt
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)

        keep_for_turn = False
        strong_turn = False
        if len1 > 1e-10 and len2 > 1e-10:
            turn_cos = np.clip(np.dot(v1, v2) / (len1 * len2), -1.0, 1.0)
            keep_for_turn = turn_cos < cos_threshold
            strong_turn = turn_cos < np.cos(np.deg2rad(45.0))

        prev_w = widths[i - 1]
        curr_w = widths[i]
        next_w = widths[i + 1]
        width_jump = max(abs(curr_w - prev_w), abs(next_w - curr_w))
        ref_w = max(prev_w, curr_w, next_w, 1e-9)
        keep_for_width = (
            width_jump >= width_abs_threshold
            and width_jump / ref_w >= width_rel_threshold
        )

        keep_for_length = min(len1, len2) >= min_segment_length
        keep_for_spacing = (
            max_segment_length is not None
            and distances[i] - reduced_distances[-1] >= max_segment_length
        )

        if (
            keep_for_spacing
            or keep_for_width
            or ((keep_for_turn and keep_for_length) or strong_turn)
        ):
            reduced_points.append(points[i])
            reduced_widths.append(widths[i])
            reduced_distances.append(distances[i])

    reduced_points.append(points[-1])
    reduced_widths.append(widths[-1])
    reduced_distances.append(distances[-1])
    return reduced_points, reduced_widths, reduced_distances


def _remove_duplicate_stations(
    points: list[list[float]],
    widths: list[float],
    distances: list[float],
    min_distance: float = 1e-4,
) -> tuple[list[list[float]], list[float], list[float]]:
    """Drop consecutive stations that collapse to the same location."""
    if not points:
        return points, widths, distances

    filtered_points = [points[0]]
    filtered_widths = [widths[0]]
    filtered_distances = [distances[0]]

    for point, width, distance in zip(points[1:], widths[1:], distances[1:]):
        prev = np.array(filtered_points[-1], dtype=float)
        curr = np.array(point, dtype=float)
        if np.linalg.norm(curr - prev) >= min_distance:
            filtered_points.append(point)
            filtered_widths.append(width)
            filtered_distances.append(distance)
        else:
            filtered_widths[-1] = width
            filtered_distances[-1] = distance

    return filtered_points, filtered_widths, filtered_distances


def _build_wall_lines(
    edges: list[list[list[float]]],
) -> list[LineString]:
    """Build a list of wall LineStrings from raw edge data.

    Handles both 2D (y,z) and 3D (x,y,z) edges by dropping the x coordinate.
    Skips zero-length edges (where start == end) to avoid empty LineStrings.
    """
    wall_lines = []
    for edge in edges:
        if len(edge) >= 2:
            # Check dimensionality
            pt_sample = edge[0]
            if len(pt_sample) == 3:
                # Drop x coordinate (which is 0 for 2D section data)
                coords_2d = [(pt[1], pt[2]) for pt in edge]
                ls = LineString(coords_2d)
            else:
                ls = LineString(edge)
            # Skip zero-length (degenerate) edges
            if ls.is_empty or ls.length < 1e-12:
                continue
            wall_lines.append(ls)
    return wall_lines


def generate_auto_segments(
    json_path: Optional[Path],
    output_yaml: Path,
    n_segments: int = 20,
    flip_x: bool = False,
    flip_y: bool = False,
    from_clipboard: bool = False,
    geometry_aware: bool = False,
    preserve_breaks: bool = False,
    method: str = "voronoi",
    center: bool = True,
    output_format: str = "sections",
):
    """
    Reads a Pyhorn Auto-Segment JSON export, computes the Medial Axis,
    and generates a Pyhorn YAML configuration.

    Parameters
    ----------
    method : {"voronoi", "graph"}
        "voronoi"  — polygon-based Voronoi medial axis (original, needs closed sketch).
        "graph"    — direct wall-graph + perpendicular width (default, more robust
                      for fragmented Onshape exports).
    center : bool
        If True (default), shifts all coordinates so the bounding box starts at (0, 0).
        This corrects sketches drawn in the negative quadrant of the Onshape sketch plane.
    output_format : {"sections", "legacy"}
        "sections"  — emit ``sections`` list with chained profile sections
                      (straight / exponential profile_type, start/end areas, cumulative lengths).
                      This is the default and preferred format for the ``HornGeometry.sections`` field.
        "legacy"   — emit ``rectangular_segments`` or ``conical_segments``
                      (the original per-segment format).
    """
    if from_clipboard:
        import subprocess

        try:
            json_str = subprocess.run(
                ["pbpaste"], capture_output=True, text=True, check=True
            ).stdout
            data = json.loads(json_str)
        except Exception as e:
            raise ValueError(f"Failed to read valid JSON from clipboard: {e}")
    else:
        if json_path is None:
            raise ValueError("json_path is required when --from-clipboard is not used")
        with open(json_path, "r") as f:
            data = json.load(f)

    width = float(data.get("width", 0.2))

    raw_throat = np.array(data["throat"])
    raw_mouth = np.array(data["mouth"])
    raw_edges = [np.array(e) for e in data["boundary_edges"]]

    # 0. Project 3D → 2D if needed
    if raw_throat.shape[-1] == 3:
        all_pts = np.vstack([raw_throat, raw_mouth] + raw_edges)
        var = np.var(all_pts, axis=0)
        drop_idx = np.argmin(var)
        throat_pts = np.delete(raw_throat, drop_idx, axis=1).tolist()
        mouth_pts = np.delete(raw_mouth, drop_idx, axis=1).tolist()
        edges = [np.delete(e, drop_idx, axis=1).tolist() for e in raw_edges]
    else:
        throat_pts = raw_throat.tolist()
        mouth_pts = raw_mouth.tolist()
        edges = [e.tolist() for e in raw_edges]

    wall_lines = _build_wall_lines(edges)

    # ── Common geometry ────────────────────────────────────────────────────────
    throat_line = LineString(throat_pts)
    mouth_line = LineString(mouth_pts)
    t_center = (
        (throat_pts[0][0] + throat_pts[1][0]) / 2.0,
        (throat_pts[0][1] + throat_pts[1][1]) / 2.0,
    )
    m_center = (
        (mouth_pts[0][0] + mouth_pts[1][0]) / 2.0,
        (mouth_pts[0][1] + mouth_pts[1][1]) / 2.0,
    )
    t_rad = float(throat_line.length) / 2.0
    m_rad = float(mouth_line.length) / 2.0

    # Store original centers for test reference (before path centering)
    orig_t_center = t_center
    orig_m_center = m_center

    # ── Graph method (default, more robust) ───────────────────────────────────
    if method == "graph":
        result = _generate_graph_method(
            wall_lines,
            throat_pts,
            mouth_pts,
            t_center,
            m_center,
            t_rad,
            m_rad,
            width,
            n_segments,
            geometry_aware,
            preserve_breaks,
        )
    else:
        result = _generate_voronoi_method(
            edges,
            throat_pts,
            mouth_pts,
            width,
            n_segments,
            geometry_aware,
            preserve_breaks,
        )

    # ── Bend angles ───────────────────────────────────────────────────────────
    if geometry_aware:
        from pyhorn_core.solver.geometry_discretise import compute_bend_angles

        coords = result["coordinates"]
        bend_angles = [round(a, 4) for a in compute_bend_angles(coords)]
        result["bend_angles"] = bend_angles

    # ── Flip coordinates ───────────────────────────────────────────────────────
    coords = result["coordinates"]
    if flip_x or flip_y:
        coords_arr = np.array(coords)
        # Use bounding box of centerline
        xmin, ymin = coords_arr.min(axis=0)
        xmax, ymax = coords_arr.max(axis=0)
        if flip_x:
            coords_arr[:, 0] = (xmin + xmax) - coords_arr[:, 0]
        if flip_y:
            coords_arr[:, 1] = (ymin + ymax) - coords_arr[:, 1]
        coords = [[round(float(c[0]), 4), round(float(c[1]), 4)] for c in coords_arr]
        result["coordinates"] = coords

    # ── Center coordinates in positive quadrant ───────────────────────────────
    # Shift so the bounding box always starts at (0, 0).  This corrects sketches
    # drawn in the negative quadrant of the Onshape sketch plane.
    # Store the centering offset before modifying coordinates so it can be
    # recorded in the output YAML for test reference.
    center_offset: Optional[list[float]] = None
    if center:
        coords_arr = np.array(coords)
        xmin = float(np.min(coords_arr[:, 0]))
        ymin = float(np.min(coords_arr[:, 1]))
        if xmin != 0.0 or ymin != 0.0:
            center_offset = [round(xmin, 6), round(ymin, 6)]
            coords_arr[:, 0] -= xmin
            coords_arr[:, 1] -= ymin
            coords = [[round(float(c[0]), 4), round(float(c[1]), 4)] for c in coords_arr]
            result["coordinates"] = coords

    # ── Bounding box ───────────────────────────────────────────────────────────
    # bbox is computed AFTER centering so the rect always sits at the origin
    coords_arr = np.array(coords)
    xmin, ymin = float(np.min(coords_arr[:, 0])), float(np.min(coords_arr[:, 1]))
    xmax, ymax = float(np.max(coords_arr[:, 0])), float(np.max(coords_arr[:, 1]))
    enc_depth = xmax - xmin
    enc_height = ymax - ymin

    out_dict = {
        "enclosure_type": "BLH",
        "width": float(result.get("width", round(width, 4))),
        "enclosure_dims": [float(round(enc_depth, 4)), float(round(enc_height, 4))],
        "coordinates": coords,
        # Original throat/mouth centers from JSON (before centering offset is applied)
        "_throat_center": [round(orig_t_center[0], 6), round(orig_t_center[1], 6)],
        "_mouth_center": [round(orig_m_center[0], 6), round(orig_m_center[1], 6)],
        # Centering offset — the amount subtracted from coordinates (pre-centering min)
        "_center_offset": center_offset if center_offset is not None else [0.0, 0.0],
    }

    if output_format == "sections":
        # Convert rectangular segments to chained-sections format.
        # rectangular_segments: [width, h_start, width_end, h_end, length]
        # sections: {name, profile_type, length, start_area, end_area}
        rect_segs = result.get("rectangular_segments")
        horn_width = float(result.get("width", width))
        if rect_segs:
            out_dict["sections"] = rectangular_segments_to_sections(rect_segs, horn_width)
        else:
            # Fall back to conical segments if no rectangular segments available
            con_segs = result.get("conical_segments", [])
            # Convert conical [h_start, h_end, length] to sections format
            # For conical, area = width × height → but width is unknown here.
            # Use h as proxy (circular equivalent: area ~ pi*(h/2)^2).
            # Since we don't have the full width, convert conical heights directly
            # as approximate areas: area = h^2 * pi/4 (circular) — just use h as proxy.
            # Actually, for the sections format we need absolute areas.
            # Conical segments from the medial axis method use height directly.
            # Build equivalent sections treating each conical seg as a straight section.
            sections_from_conical = []
            for idx, seg in enumerate(con_segs):
                h_start, h_end, seg_len = seg[0], seg[1], seg[2]
                # Approximate: treat height as the relevant dimension
                # (conical segments from medial_axis are height-based, not area-based)
                sections_from_conical.append({
                    "name": f"segment_{idx + 1}",
                    "profile_type": _infer_profile_type(h_start, h_end),
                    "length": round(seg_len, 6),
                    "start_area": round(h_start, 8),
                    "end_area": round(h_end, 8),
                })
            out_dict["sections"] = sections_from_conical
    else:
        # Legacy format — emit rectangular_segments or conical_segments
        if preserve_breaks or result.get("rectangular_segments"):
            out_dict["rectangular_segments"] = result.get("rectangular_segments", [])
        else:
            out_dict["conical_segments"] = result.get("conical_segments", [])

    if geometry_aware:
        out_dict["discretisation"] = "geometry"
        if "bend_angles" in result:
            out_dict["bend_angles"] = result["bend_angles"]

    with open(output_yaml, "w") as f:
        yaml.dump(out_dict, f, default_flow_style=None, sort_keys=False)

    return out_dict


# ─── Graph method (new, more robust) ──────────────────────────────────────────


def _generate_graph_method(
    wall_lines: list[LineString],
    throat_pts: list[list[float]],
    mouth_pts: list[list[float]],
    t_center: tuple[float, float],
    m_center: tuple[float, float],
    t_rad: float,
    m_rad: float,
    width: float,
    n_segments: int,
    geometry_aware: bool,
    preserve_breaks: bool,
    MIN_HORN_HEIGHT: float = 0.001,  # 1mm minimum height to avoid zero-area segments
) -> dict:
    """
    Robust centerline extraction using wall-graph + perpendicular widths.

    Works directly from wall segments without needing a closed polygon,
    making it suitable for fragmented Onshape exports.
    """
    from shapely import set_precision
    from scipy.interpolate import interp1d

    # 1. Build a combined boundary by merging all wall lines
    lines = [set_precision(ln, grid_size=1e-4) for ln in wall_lines]
    merged = linemerge(lines)

    # If merged is a proper polygon, use its interior; otherwise use convex-hull buffer
    if isinstance(merged, Polygon):
        poly = merged
    elif isinstance(merged, MultiLineString):
        # MultiPolygon or GeometryCollection — take the largest
        polys = [g for g in merged.geoms if isinstance(g, Polygon) and g.area > 1e-6]
        if polys:
            poly = max(polys, key=lambda p: p.area)
        else:
            # No valid polygon — fall back to merged.convex_hull.buffer
            poly = merged.convex_hull.buffer(0.005)
            logger.warning(
                "No valid polygon from wall merge; using buffered convex hull."
            )
    else:
        poly = merged.convex_hull.buffer(0.005)

    boundary = poly.boundary if hasattr(poly, "boundary") else poly.exterior
    throat_line = LineString(throat_pts)
    mouth_line = LineString(mouth_pts)

    # 2. Dense boundary points for Voronoi sites
    n_sites = min(int(boundary.length * 200), 5000)
    b_pts = [boundary.interpolate(i / n_sites, normalized=True) for i in range(n_sites)]
    site_coords = np.array([[p.x, p.y] for p in b_pts])

    # 3. Voronoi of boundary sites → medial axis graph
    vor = Voronoi(site_coords)
    G = nx.Graph()
    for vpair in vor.ridge_vertices:
        if vpair[0] >= 0 and vpair[1] >= 0:
            p1 = vor.vertices[vpair[0]]
            p2 = vor.vertices[vpair[1]]
            mid = Point((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            if poly.contains(mid):
                dist = float(np.linalg.norm(p1 - p2))
                G.add_edge(vpair[0], vpair[1], weight=dist)

    if len(G.nodes) == 0:
        raise ValueError(
            "Graph method: no internal Voronoi edges found. "
            "Check that the boundary forms an enclosed shape."
        )

    nodes = list(G.nodes)
    node_coords = vor.vertices[nodes]

    # 4. Nearest graph nodes to throat and mouth
    t_arr = np.array(t_center)
    m_arr = np.array(m_center)

    t_idx = nodes[int(np.argmin(np.linalg.norm(node_coords - t_arr, axis=1)))]
    m_cands = [
        n
        for n in nodes
        if float(np.linalg.norm(vor.vertices[n] - m_arr))
        < float(mouth_line.length) * 1.5
    ]
    if m_cands:
        m_idx = max(m_cands, key=lambda n: boundary.distance(Point(vor.vertices[n])))
    else:
        m_idx = nodes[int(np.argmin(np.linalg.norm(node_coords - m_arr, axis=1)))]

    # 5. Shortest throat→mouth path through the graph
    try:
        path = nx.shortest_path(G, source=t_idx, target=m_idx, weight="weight")
    except nx.NetworkXNoPath:
        raise ValueError(
            "Graph method: no path found from throat to mouth. "
            "The boundary may not form a connected interior."
        )

    path_coords = vor.vertices[path]

    # 6. Extend to exact throat/mouth centers
    full_path = (
        [[t_center[0], t_center[1]]]
        + path_coords.tolist()
        + [[m_center[0], m_center[1]]]
    )

    # 6b. Smooth the path to suppress jaggedness from fragmented Onshape edges.
    #     Savitzky-Golay (window=5, order=2) removes high-frequency jitter while
    #     preserving geometry shape.  Only used when path is long enough.
    if len(full_path) >= 5:
        path_arr = np.array(full_path)
        # window_length must be odd and <= len; use min(5, len) adjusted for len
        win = min(5, len(full_path))
        if win % 2 == 0:
            win -= 1
        if win >= 3:
            path_arr[:, 0] = savgol_filter(path_arr[:, 0], win, 2)
            path_arr[:, 1] = savgol_filter(path_arr[:, 1], win, 2)
            full_path = path_arr.tolist()

    # Cumulative distance
    cum_dist = [0.0]
    for i in range(1, len(full_path)):
        p1 = np.array(full_path[i - 1])
        p2 = np.array(full_path[i])
        cum_dist.append(cum_dist[-1] + float(np.linalg.norm(p2 - p1)))

    # 7. Perpendicular-to-wall width at each interior path point
    #    (endpoints use explicit throat/mouth widths)
    raw_widths = []
    for i, pt in enumerate(full_path):
        if i == 0:
            raw_widths.append(2.0 * t_rad)  # throat width
        elif i == len(full_path) - 1:
            raw_widths.append(2.0 * m_rad)  # mouth width
        else:
            # Try perpendicular width first
            if i > 0 and i < len(full_path) - 1:
                tangent = (
                    float(full_path[i + 1][0] - full_path[i - 1][0]),
                    float(full_path[i + 1][1] - full_path[i - 1][1]),
                )
                t_len = np.linalg.norm(tangent)
                if t_len > 1e-9:
                    tangent = (float(tangent[0] / t_len), float(tangent[1] / t_len))
                pw = _perpendicular_width_at_point(
                    (float(pt[0]), float(pt[1])),
                    tangent,
                    wall_lines,
                    max_ray=0.5,
                )
                if pw is not None and pw > 0.002:
                    raw_widths.append(float(pw))
                    continue
            # Fallback: perpendicular distance to wall
            r = _distance_to_nearest_wall((float(pt[0]), float(pt[1])), wall_lines)
            raw_widths.append(2.0 * r)

    # 8. Discretise
    # Clamp all widths to minimum before reduction to avoid zero-area segments
    raw_widths = [max(w, MIN_HORN_HEIGHT) for w in raw_widths]

    # 8b. Smooth widths to suppress oscillation from fragmented Onshape wall edges.
    #     Savitzky-Golay (window=7, order=3) removes medium-frequency width jitter
    #     (typically 3-5 path-points wide) while preserving the overall flare profile.
    #     Without this, the 46-fragment Onshape throat sketch causes oscillating
    #     heights in segments 5-8 of the throat, which creates standing-wave
    #     resonances at 200-500 Hz → ragged SPL in that band.
    #     Only applies when the path is long enough to support the window (>= 7 pts).
    if len(raw_widths) >= 7:
        win = 7  # must be odd; order=3 is below window size
        raw_widths = savgol_filter(np.array(raw_widths), win, 3).tolist()
        # Re-clamp after smoothing (polynomial fit may undershoot near discontinuities)
        raw_widths = [max(w, MIN_HORN_HEIGHT) for w in raw_widths]

    if preserve_breaks:
        min_seg_len = cum_dist[-1] / max(n_segments * 2, 1)
        max_seg_len = cum_dist[-1] / max(n_segments, 1)
        coord_list, true_widths, coord_distances = _reduce_stair_points(
            full_path,
            raw_widths,
            cum_dist,
            min_segment_length=min_seg_len,
            max_segment_length=max_seg_len,
        )
        coord_list, true_widths, coord_distances = _remove_duplicate_stations(
            coord_list, true_widths, coord_distances
        )
        segment_points = coord_list
        result_coords = [c for c in segment_points]
        rect_segs = []
        for i in range(len(segment_points) - 1):
            p1 = segment_points[i]
            p2 = segment_points[i + 1]
            seg_len = coord_distances[i + 1] - coord_distances[i]
            h_start = max(true_widths[i], MIN_HORN_HEIGHT)
            h_end = max(true_widths[i + 1], MIN_HORN_HEIGHT)
            rect_segs.append(
                [
                    round(width, 4),
                    round(h_start, 4),
                    round(width, 4),
                    round(h_end, 4),
                    round(seg_len, 4),
                ]
            )
        return {
            "coordinates": [[round(c[0], 4), round(c[1], 4)] for c in result_coords],
            "rectangular_segments": rect_segs,
            "conical_segments": [],
        }

    else:
        # Uniform n_segments sampling
        segment_points = []
        for i in range(n_segments + 1):
            t_param = i / n_segments
            d_target = t_param * cum_dist[-1]
            # Find index in cum_dist
            idx = next(
                (j for j, d in enumerate(cum_dist) if d >= d_target), len(cum_dist) - 1
            )
            if idx == 0:
                pt = full_path[0]
            elif idx >= len(full_path):
                pt = full_path[-1]
            else:
                t0, t1 = cum_dist[idx - 1], cum_dist[idx]
                alpha = (d_target - t0) / (t1 - t0) if t1 > t0 else 0.0
                pt = [
                    full_path[idx - 1][0]
                    + alpha * (full_path[idx][0] - full_path[idx - 1][0]),
                    full_path[idx - 1][1]
                    + alpha * (full_path[idx][1] - full_path[idx - 1][1]),
                ]
            segment_points.append(pt)

        result_coords = segment_points
        interp1d(cum_dist, raw_widths, kind="linear")

        rect_segs = []
        for i in range(len(segment_points) - 1):
            p1, p2 = segment_points[i], segment_points[i + 1]
            d1 = (i / n_segments) * cum_dist[-1]
            d2 = ((i + 1) / n_segments) * cum_dist[-1]
            seg_len = float(np.linalg.norm(np.array(p2) - np.array(p1)))
            h_start = float(
                np.interp(np.clip(d1, cum_dist[0], cum_dist[-1]), cum_dist, raw_widths)
            )
            h_end = float(
                np.interp(np.clip(d2, cum_dist[0], cum_dist[-1]), cum_dist, raw_widths)
            )
            h_start = max(h_start, MIN_HORN_HEIGHT)
            h_end = max(h_end, MIN_HORN_HEIGHT)
            rect_segs.append(
                [
                    round(width, 4),
                    round(h_start, 4),
                    round(width, 4),
                    round(h_end, 4),
                    round(seg_len, 4),
                ]
            )
        return {
            "coordinates": [[round(c[0], 4), round(c[1], 4)] for c in result_coords],
            "rectangular_segments": rect_segs,
            "conical_segments": [],
        }


# ─── Voronoi / polygon method (original) ──────────────────────────────────────


def _generate_voronoi_method(
    edges: list[list[list[float]]],
    throat_pts: list[list[float]],
    mouth_pts: list[list[float]],
    width: float,
    n_segments: int,
    geometry_aware: bool,
    preserve_breaks: bool,
) -> dict:
    """Original polygon-based Voronoi medial axis method."""
    from shapely import set_precision
    from scipy.interpolate import interp1d
    from pyhorn_core.solver.geometry_discretise import compute_perpendicular_sections

    lines = []
    for edge in edges:
        if len(edge) >= 2:
            ls = set_precision(LineString(edge), grid_size=1e-4)
            if not ls.is_empty and ls.length >= 1e-12:
                lines.append(ls)

    merged = linemerge(lines)
    polys = list(polygonize(merged))
    if not polys:
        noded = unary_union(lines)
        polys = list(polygonize(noded))

    if not polys:
        raise ValueError(
            "Could not form a closed polygon from the boundary edges. "
            "Ensure the sketch face is closed, or try method='graph'."
        )

    poly = max(polys, key=lambda p: p.area)
    boundary = poly.boundary

    n_sites = min(int(boundary.length * 200), 5000)
    b_pts = [boundary.interpolate(i / n_sites, normalized=True) for i in range(n_sites)]
    pts = np.array([[p.x, p.y] for p in b_pts])

    vor = Voronoi(pts)

    G = nx.Graph()
    for vpair in vor.ridge_vertices:
        if vpair[0] >= 0 and vpair[1] >= 0:
            p1 = vor.vertices[vpair[0]]
            p2 = vor.vertices[vpair[1]]
            mid = Point((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            if poly.contains(mid):
                dist = float(np.linalg.norm(p1 - p2))
                G.add_edge(vpair[0], vpair[1], weight=dist)

    if len(G.nodes) == 0:
        raise ValueError("Failed to build internal Voronoi graph.")

    t_center = Point(
        (throat_pts[0][0] + throat_pts[1][0]) / 2,
        (throat_pts[0][1] + throat_pts[1][1]) / 2,
    )
    m_center = Point(
        (mouth_pts[0][0] + mouth_pts[1][0]) / 2,
        (mouth_pts[0][1] + mouth_pts[1][1]) / 2,
    )
    throat_line = LineString(throat_pts)
    mouth_line = LineString(mouth_pts)

    nodes = list(G.nodes)
    node_coords = vor.vertices[nodes]

    t_idx = nodes[
        int(
            np.argmin(
                np.linalg.norm(node_coords - np.array([t_center.x, t_center.y]), axis=1)
            )
        )
    ]

    m_cands = [
        n
        for n in nodes
        if float(np.linalg.norm(vor.vertices[n] - np.array([m_center.x, m_center.y])))
        < float(mouth_line.length) * 1.5
    ]
    if m_cands:
        m_idx = max(m_cands, key=lambda n: boundary.distance(Point(vor.vertices[n])))
    else:
        m_idx = nodes[
            int(
                np.argmin(
                    np.linalg.norm(
                        node_coords - np.array([m_center.x, m_center.y]), axis=1
                    )
                )
            )
        ]

    try:
        path = nx.shortest_path(G, source=t_idx, target=m_idx, weight="weight")
    except nx.NetworkXNoPath:
        raise ValueError(
            "Could not find a continuous path from throat to mouth inside the polygon."
        )

    path_coords = vor.vertices[path]

    # Width via perpendicular distance to actual wall segments (more accurate than
    # polygon boundary.distance for fragmented/sketch-style boundaries)
    wall_lines = lines  # already a list of wall LineStrings built above
    radii = [_distance_to_nearest_wall(tuple(p), wall_lines) for p in path_coords]

    t_rad = float(throat_line.length) / 2.0
    m_rad = float(mouth_line.length) / 2.0

    full_path = (
        [[t_center.x, t_center.y]] + path_coords.tolist() + [[m_center.x, m_center.y]]
    )
    full_radii = [t_rad] + radii + [m_rad]

    cum_dist = [0.0]
    for i in range(1, len(full_path)):
        p1_arr = np.array(full_path[i - 1])
        p2_arr = np.array(full_path[i])
        cum_dist.append(cum_dist[-1] + float(np.linalg.norm(p2_arr - p1_arr)))

    rad_interp = interp1d(cum_dist, full_radii, kind="linear", fill_value="extrapolate")

    raw_centerline = LineString(full_path)

    if geometry_aware:
        centerline = raw_centerline
    else:
        centerline = raw_centerline.simplify(0.005, preserve_topology=True)

    conical_segments = []
    coordinates = []
    true_widths = None
    rectangular_segments = []
    coord_distances: Optional[list[float]] = None

    if geometry_aware:
        if preserve_breaks:
            sampled_coords = [list(pt) for pt in full_path]
            sampled_widths = [float(r) * 2.0 for r in full_radii]
            sampled_distances = cum_dist
            # Clamp minimum width to avoid zero-area segments
            MIN_HORN_HEIGHT = 0.001
            sampled_widths = [max(w, MIN_HORN_HEIGHT) for w in sampled_widths]
            sampled_widths[0] = float(throat_line.length)
            sampled_widths[-1] = float(mouth_line.length)
            min_seg_len = raw_centerline.length / max(n_segments * 2, 1)
            max_seg_len = raw_centerline.length / max(n_segments, 1)
            coord_list, true_widths, coord_distances = _reduce_stair_points(
                sampled_coords,
                sampled_widths,
                sampled_distances,
                min_segment_length=float(min_seg_len),
                max_segment_length=float(max_seg_len),
            )
            coord_list, true_widths, coord_distances = _remove_duplicate_stations(
                coord_list, true_widths, coord_distances
            )
            segment_points = [Point(p[0], p[1]) for p in coord_list]
        else:
            segment_points = [
                centerline.interpolate(i / n_segments, normalized=True)
                for i in range(n_segments + 1)
            ]
            coord_list = [[p.x, p.y] for p in segment_points]
            true_widths = compute_perpendicular_sections(poly, coord_list)
            true_widths[0] = float(throat_line.length)
            true_widths[-1] = float(mouth_line.length)
    else:
        segment_points = [
            centerline.interpolate(i / n_segments, normalized=True)
            for i in range(n_segments + 1)
        ]

    for i in range(len(segment_points) - 1):
        p1 = segment_points[i]
        p2 = segment_points[i + 1]

        if geometry_aware:
            if true_widths is None:
                raise ValueError("Failed to compute geometry-aware cross-sections")
            h_start = true_widths[i]
            h_end = true_widths[i + 1]
        else:
            d1 = raw_centerline.project(p1)
            d2 = raw_centerline.project(p2)
            h_start = float(rad_interp(d1)) * 2
            h_end = float(rad_interp(d2)) * 2

        seg_len = float(p1.distance(p2))
        if preserve_breaks and geometry_aware:
            assert coord_distances is not None
            seg_len = coord_distances[i + 1] - coord_distances[i]

        conical_segments.append([round(h_start, 4), round(h_end, 4), round(seg_len, 4)])
        rectangular_segments.append(
            [
                round(width, 4),
                round(h_start, 4),
                round(width, 4),
                round(h_end, 4),
                round(seg_len, 4),
            ]
        )
        coordinates.append([round(p1.x, 4), round(p1.y, 4)])

    p_last = segment_points[-1]
    coordinates.append([round(p_last.x, 4), round(p_last.y, 4)])

    return {
        "coordinates": coordinates,
        "rectangular_segments": rectangular_segments,
        "conical_segments": conical_segments,
        "width": round(width, 4),
    }
