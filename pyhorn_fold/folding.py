"""Horn folding — maps a continuous-flare horn onto a folded rectangular-segment layout.

Extrapolates an optimized continuous horn profile (exponential, conical, etc.)
into a folded constant-width enclosure using lane-based path planning with
SLSQP optimization and Shapely polygon overlap checking.
"""

import math
from typing import Optional, List, Tuple

import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon as ShapelyPolygon, box
from shapely.ops import unary_union

from pyhorn_core.config.models import HornGeometry
from pyhorn_core.solver.profiles import profile_area_at_distance


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _point_along_segment(
    start: Tuple[float, float],
    end: Tuple[float, float],
    distance: float,
) -> Tuple[float, float]:
    total = math.hypot(end[0] - start[0], end[1] - start[1])
    if total <= 0:
        return start
    ratio = distance / total
    return (
        start[0] + (end[0] - start[0]) * ratio,
        start[1] + (end[1] - start[1]) * ratio,
    )


def throat_chamber_side_length(horn: HornGeometry, enclosure_width: float) -> float:
    """Approximate square side length of the side-view throat chamber around the driver."""
    chamber_volume = horn.vtc + 1e-3
    if chamber_volume <= 0 or enclosure_width <= 0:
        return 0.0
    chamber_area_2d = chamber_volume / enclosure_width
    if chamber_area_2d <= 0:
        return 0.0
    return math.sqrt(chamber_area_2d)


def _throat_attachment_point(
    enclosure_dims: Tuple[float, float],
    driver_coord: Tuple[float, float],
    chamber_side: float,
) -> Tuple[float, float]:
    """Return the throat anchor on the bottom face of the throat chamber."""
    depth, height = enclosure_dims
    driver_x, driver_y = driver_coord
    if chamber_side <= 0:
        return driver_coord

    chamber_side = min(chamber_side, depth, height)
    chamber_y0 = max(0.0, driver_y - chamber_side / 2.0)
    chamber_y0 = min(chamber_y0, height - chamber_side)

    if driver_x <= depth / 2.0:
        chamber_x0 = 0.0
    else:
        chamber_x0 = depth - chamber_side

    return (chamber_x0 + chamber_side / 2.0, chamber_y0 + chamber_side)


def _build_vertical_fold_path_from_lanes(
    throat_start: Tuple[float, float],
    lane_positions: List[float],
    y_min: float,
    y_max: float,
    major_direction: int,
    mouth_x: float,
    path_length: float,
    use_boundary_terminal_l: bool = False,
    boundary_left_x: Optional[float] = None,
    boundary_right_x: Optional[float] = None,
) -> List[Tuple[Tuple[float, float], Tuple[float, float], float]]:
    segments: List[Tuple[Tuple[float, float], Tuple[float, float], float]] = []
    throat_x, throat_y = throat_start
    current = throat_start
    remaining = path_length
    current_major_direction = major_direction

    for lane_index, lane_x in enumerate(lane_positions):
        if remaining <= 1e-9:
            break

        is_last_lane = lane_index == len(lane_positions) - 1
        if (
            is_last_lane
            and use_boundary_terminal_l
            and boundary_left_x is not None
            and boundary_right_x is not None
        ):
            for target in (
                (current[0], y_min),
                (boundary_left_x, y_min),
                (boundary_left_x, y_max),
                (boundary_right_x, y_max),
            ):
                capacity = math.hypot(target[0] - current[0], target[1] - current[1])
                if capacity <= 1e-9:
                    continue
                take = min(remaining, capacity)
                end = _point_along_segment(current, target, take)
                segments.append((current, end, take))
                current = end
                remaining -= take
                if remaining <= 1e-9:
                    break
            break

        if is_last_lane:
            mouth_gap = abs(mouth_x - current[0])
            if mouth_gap > 1e-9 and remaining <= mouth_gap + 1e-9:
                end = _point_along_segment(current, (mouth_x, current[1]), remaining)
                segments.append((current, end, remaining))
                remaining = 0.0
                break

        turn_y = y_max if current_major_direction > 0 else y_min
        major_target = (lane_x, turn_y)
        major_capacity = math.hypot(
            major_target[0] - current[0], major_target[1] - current[1]
        )
        if major_capacity > 1e-9:
            take = min(remaining, major_capacity)
            if is_last_lane:
                mouth_gap = abs(mouth_x - current[0])
                take = min(take, max(0.0, remaining - mouth_gap))
            if take > 1e-9:
                end = _point_along_segment(current, major_target, take)
                segments.append((current, end, take))
                current = end
                remaining -= take
                if remaining <= 1e-9:
                    break

        if is_last_lane:
            secondary_target = (mouth_x, current[1])
        else:
            secondary_target = (lane_positions[lane_index + 1], current[1])

        secondary_capacity = math.hypot(
            secondary_target[0] - current[0], secondary_target[1] - current[1]
        )
        if secondary_capacity > 1e-9:
            take = min(remaining, secondary_capacity)
            end = _point_along_segment(current, secondary_target, take)
            segments.append((current, end, take))
            current = end
            remaining -= take
            if remaining <= 1e-9:
                break

        current_major_direction *= -1

    if remaining > 1e-9:
        raise ValueError("Could not fit requested path length inside folded enclosure")

    return segments


def _build_vertical_boundary_terminal_l_path(
    throat_start: Tuple[float, float],
    x_min: float,
    x_right: float,
    y_min: float,
    y_max: float,
    path_length: float,
) -> Tuple[List[Tuple[Tuple[float, float], Tuple[float, float], float]], float]:
    """Build a terminal path that uses the full left wall, then the full bottom edge."""
    start_x, start_y = throat_start
    wall_length = y_max - y_min
    bottom_length = x_right - x_min
    dy = start_y - y_min
    sx = start_x - x_min

    min_total = math.hypot(sx, dy) + wall_length + bottom_length
    max_total = (
        math.hypot(x_right - start_x, dy) + bottom_length + wall_length + bottom_length
    )
    if path_length < min_total - 1e-9 or path_length > max_total + 1e-9:
        raise ValueError(
            "Boundary terminal L path is not feasible for requested length"
        )

    budget = path_length - wall_length - bottom_length
    if abs(budget - sx) < 1e-9:
        anchor_offset = 0.0
    else:
        numerator = budget**2 - sx**2 - dy**2
        denominator = 2.0 * (budget - sx)
        anchor_offset = numerator / denominator if abs(denominator) > 1e-9 else 0.0
    anchor_offset = float(np.clip(anchor_offset, 0.0, x_right - x_min))
    anchor_x = x_min + anchor_offset

    segments: List[Tuple[Tuple[float, float], Tuple[float, float], float]] = []
    current = throat_start
    remaining = path_length
    for target in (
        (anchor_x, y_min),
        (x_min, y_min),
        (x_min, y_max),
        (x_right, y_max),
    ):
        capacity = math.hypot(target[0] - current[0], target[1] - current[1])
        if capacity <= 1e-9:
            continue
        take = min(remaining, capacity)
        end = _point_along_segment(current, target, take)
        segments.append((current, end, take))
        current = end
        remaining -= take
        if remaining <= 1e-9:
            break

    if remaining > 1e-9:
        raise ValueError("Boundary terminal L path could not consume requested length")

    return segments, anchor_x - x_min


def _reflect_lanes_from_mirrors(
    throat_x: float,
    mirror_positions: List[float],
) -> List[float]:
    """Derive successive lane x-positions by reflecting the centerline across fold lines."""
    lanes: List[float] = []
    current_x = throat_x
    for mirror_x in mirror_positions:
        current_x = 2.0 * mirror_x - current_x
        lanes.append(current_x)
    return lanes


def _fold_segment_heights(
    path_segments: List[Tuple[Tuple[float, float], Tuple[float, float], float]],
    profile_type: str,
    throat_area: float,
    mouth_area: float,
    path_length: float,
    enclosure_width: float,
    hyperbolic_t: float = 1.0,
) -> List[Tuple[float, float]]:
    heights: List[Tuple[float, float]] = []
    cumulative = 0.0
    for _, _, seg_length in path_segments:
        area_start = profile_area_at_distance(
            profile_type,
            throat_area,
            mouth_area,
            path_length,
            cumulative,
            hyperbolic_t,
        )
        area_end = profile_area_at_distance(
            profile_type,
            throat_area,
            mouth_area,
            path_length,
            cumulative + seg_length,
            hyperbolic_t,
        )
        heights.append((area_start / enclosure_width, area_end / enclosure_width))
        cumulative += seg_length
    return heights


def _fold_path_polygons(
    path_segments: List[Tuple[Tuple[float, float], Tuple[float, float], float]],
    heights: List[Tuple[float, float]],
) -> List[ShapelyPolygon]:
    polygons: List[ShapelyPolygon] = []
    for (start, end, _), (h_start, h_end) in zip(path_segments, heights):
        p1 = np.array(start)
        p2 = np.array(end)
        vec = p2 - p1
        length = np.linalg.norm(vec)
        if length <= 1e-9:
            continue
        direction = vec / length
        normal = np.array([-direction[1], direction[0]])
        w1_inner = p1 + normal * (h_start / 2.0)
        w1_outer = p1 - normal * (h_start / 2.0)
        w2_inner = p2 + normal * (h_end / 2.0)
        w2_outer = p2 - normal * (h_end / 2.0)
        polygon = ShapelyPolygon([w1_inner, w2_inner, w2_outer, w1_outer])
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if not polygon.is_empty:
            polygons.append(polygon)
    return polygons


def _build_vertical_fold_path_optimized(
    enclosure_dims: Tuple[float, float],
    driver_coord: Tuple[float, float],
    path_length: float,
    throat_offset: float,
    major_direction: int,
    near_x_margin: float,
    far_x_margin: float,
    y_margin: float,
    profile_type: str,
    throat_area: float,
    mouth_area: float,
    enclosure_width: float,
    use_boundary_terminal_l: bool = False,
) -> Tuple[List[Tuple[Tuple[float, float], Tuple[float, float], float]], float]:
    depth, height = enclosure_dims
    driver_x, driver_y = driver_coord
    x_min = near_x_margin
    x_max = depth - far_x_margin
    y_min = y_margin
    y_max = height - y_margin
    throat_start = _throat_attachment_point(enclosure_dims, driver_coord, throat_offset)
    throat_x = max(x_min, throat_start[0])
    throat_y = min(max(throat_start[1], y_min), y_max)
    throat_start = (throat_x, throat_y)
    mouth_x = x_min if driver_x <= depth / 2.0 else x_max
    far_x = x_max if mouth_x == x_min else x_min

    if use_boundary_terminal_l and driver_x <= depth / 2.0:
        return _build_vertical_boundary_terminal_l_path(
            throat_start,
            x_min,
            depth - near_x_margin,
            y_min,
            y_max,
            path_length,
        )

    if throat_x >= x_max - 1e-9:
        raise ValueError("Throat chamber offset leaves no room for folded horn path")

    available_secondary = abs(far_x - throat_x)
    if available_secondary <= 1e-9:
        raise ValueError(
            "Driver position leaves no room to add folds in this direction"
        )

    first_major = (y_max - throat_y) if major_direction > 0 else (throat_y - y_min)
    full_major = y_max - y_min
    remaining_after_first = max(0.0, path_length - first_major - available_secondary)
    lane_count = 1 + math.ceil(remaining_after_first / max(full_major, 1e-9))
    initial_lanes = [
        throat_x + (far_x - throat_x) * ((i + 1) / lane_count)
        for i in range(lane_count)
    ]
    initial_mirrors: List[float] = []
    current_x = throat_x
    for lane_x in initial_lanes:
        initial_mirrors.append((current_x + lane_x) / 2.0)
        current_x = lane_x
    initial_guess = np.array(initial_mirrors, dtype=float)

    low = min(throat_x, far_x) + 1e-4
    high = max(throat_x, far_x) - 1e-4
    bounds = [(low, high) for _ in range(lane_count)]
    enclosure_polygon = box(0.0, 0.0, depth, height)
    order_sign = 1.0 if far_x >= throat_x else -1.0

    def _lane_positions(values: np.ndarray) -> List[float]:
        return _reflect_lanes_from_mirrors(
            throat_x,
            [float(v) for v in values],
        )

    def _objective(values: np.ndarray) -> float:
        lanes = _lane_positions(values)
        try:
            path_segments = _build_vertical_fold_path_from_lanes(
                throat_start,
                lanes,
                y_min,
                y_max,
                major_direction,
                mouth_x,
                path_length,
                use_boundary_terminal_l=use_boundary_terminal_l,
                boundary_left_x=x_min,
                boundary_right_x=depth - near_x_margin,
            )
        except ValueError:
            return 1e9

        heights = _fold_segment_heights(
            path_segments,
            profile_type,
            throat_area,
            mouth_area,
            path_length,
            enclosure_width,
        )
        polygons = _fold_path_polygons(path_segments, heights)
        if not polygons:
            return 1e9

        union = unary_union(polygons)
        union_area = union.area if not union.is_empty else 0.0
        total_area = sum(poly.area for poly in polygons)
        overlap_area = max(0.0, total_area - union_area)
        outside_area = (
            union.difference(enclosure_polygon).area if not union.is_empty else 0.0
        )
        mouth_distance = abs(path_segments[-1][1][0] - mouth_x)
        spacing = np.diff(np.array([throat_x] + lanes + [far_x], dtype=float))
        spacing_penalty = float(np.var(np.abs(spacing))) if len(spacing) > 1 else 0.0
        return (
            1e6 * outside_area
            + 1e6 * overlap_area
            + 100.0 * mouth_distance
            + spacing_penalty
        )

    constraints = []
    for i in range(lane_count - 1):
        if order_sign > 0:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda values, i=i: _lane_positions(values)[i + 1]
                    - _lane_positions(values)[i]
                    - 1e-4,
                }
            )
        else:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda values, i=i: _lane_positions(values)[i]
                    - _lane_positions(values)[i + 1]
                    - 1e-4,
                }
            )

    for i in range(lane_count):
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda values, i=i: _lane_positions(values)[i] - low,
            }
        )
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda values, i=i: high - _lane_positions(values)[i],
            }
        )

    result = minimize(
        _objective,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-8},
    )

    candidate_values = result.x if result.success else initial_guess
    lane_positions = _lane_positions(candidate_values)
    path_segments = _build_vertical_fold_path_from_lanes(
        throat_start,
        lane_positions,
        y_min,
        y_max,
        major_direction,
        mouth_x,
        path_length,
        use_boundary_terminal_l=use_boundary_terminal_l,
        boundary_left_x=x_min,
        boundary_right_x=depth - near_x_margin,
    )
    heights = _fold_segment_heights(
        path_segments,
        profile_type,
        throat_area,
        mouth_area,
        path_length,
        enclosure_width,
    )
    polygons = _fold_path_polygons(path_segments, heights)
    union = unary_union(polygons) if polygons else None
    outside_area = (
        union.difference(enclosure_polygon).area
        if union and not union.is_empty
        else 0.0
    )
    overlap_area = (
        max(0.0, sum(poly.area for poly in polygons) - union.area)
        if union and not union.is_empty
        else 0.0
    )
    if outside_area > 1e-8 or overlap_area > 1e-8:
        raise ValueError("Optimized folded path still overlaps enclosure constraints")

    pitch = (
        float(np.mean(np.abs(np.diff(lane_positions))))
        if len(lane_positions) > 1
        else available_secondary
    )
    return path_segments, pitch


def _build_fold_path(
    enclosure_dims: Tuple[float, float],
    driver_coord: Tuple[float, float],
    path_length: float,
    throat_offset: float,
    primary_axis: str,
    major_direction: int,
    secondary_direction: int,
    near_x_margin: float,
    far_x_margin: float,
    y_margin: float,
    use_boundary_terminal_l: bool = False,
) -> Tuple[List[Tuple[Tuple[float, float], Tuple[float, float], float]], float]:
    depth, height = enclosure_dims
    segments: List[Tuple[Tuple[float, float], Tuple[float, float], float]] = []
    driver_x, driver_y = driver_coord

    x_min = near_x_margin
    x_max = depth - far_x_margin
    y_min = y_margin
    y_max = height - y_margin
    throat_start = _throat_attachment_point(enclosure_dims, driver_coord, throat_offset)
    throat_x = max(x_min, throat_start[0])
    throat_y = min(max(throat_start[1], y_min), y_max)
    throat_start = (throat_x, throat_y)
    mouth_x = x_min if driver_x <= depth / 2.0 else x_max
    far_x = x_max if mouth_x == x_min else x_min

    if throat_x >= x_max - 1e-9:
        raise ValueError("Throat chamber offset leaves no room for folded horn path")

    if primary_axis == "horizontal":
        available_secondary = (
            (y_max - driver_y) if secondary_direction > 0 else (driver_y - y_min)
        )
        if available_secondary <= 1e-9:
            raise ValueError(
                "Driver position leaves no room to add folds in this direction"
            )
        first_major = abs(far_x - throat_x)
        full_major = x_max - x_min
        remaining_after_first = max(
            0.0, path_length - first_major - available_secondary
        )
        lane_count = 1 + math.ceil(remaining_after_first / max(full_major, 1e-9))
        if lane_count % 2 == 0:
            lane_count += 1
        pitch = available_secondary / max(lane_count - 1, 1)
        lane_positions = [
            driver_y + secondary_direction * pitch * i for i in range(lane_count)
        ]
        lane_positions[0] = throat_y
        start = (throat_x, lane_positions[0])
        current_major_direction = 1 if far_x >= throat_x else -1
    else:
        available_secondary = abs(far_x - throat_x)
        if available_secondary <= 1e-9:
            raise ValueError(
                "Driver position leaves no room to add folds in this direction"
            )
        first_major = (y_max - throat_y) if major_direction > 0 else (throat_y - y_min)
        full_major = y_max - y_min
        remaining_after_first = max(
            0.0, path_length - first_major - available_secondary
        )
        lane_count = 1 + math.ceil(remaining_after_first / max(full_major, 1e-9))
        pitch = available_secondary / lane_count
        lane_positions = [
            throat_x + (far_x - throat_x) * ((i + 1) / lane_count)
            for i in range(lane_count)
        ]
        start = throat_start
        current_major_direction = major_direction

    current = start
    remaining = path_length

    for lane_index, lane_position in enumerate(lane_positions):
        if remaining <= 1e-9:
            break

        is_last_lane = lane_index == len(lane_positions) - 1

        if (
            primary_axis == "vertical"
            and is_last_lane
            and use_boundary_terminal_l
            and driver_x <= depth / 2.0
        ):
            boundary_segments, pitch = _build_vertical_boundary_terminal_l_path(
                throat_start,
                x_min,
                depth - near_x_margin,
                y_min,
                y_max,
                path_length,
            )
            return boundary_segments, pitch

        if primary_axis == "vertical" and is_last_lane:
            mouth_gap = abs(mouth_x - current[0])
            if mouth_gap > 1e-9 and remaining <= mouth_gap + 1e-9:
                end = _point_along_segment(current, (mouth_x, current[1]), remaining)
                segments.append((current, end, remaining))
                current = end
                remaining = 0.0
                break

        if primary_axis == "horizontal":
            turn_x = x_max if current_major_direction > 0 else x_min
            major_target = (turn_x, lane_position)
        else:
            turn_y = y_max if current_major_direction > 0 else y_min
            major_target = (lane_position, turn_y)

        major_capacity = math.hypot(
            major_target[0] - current[0], major_target[1] - current[1]
        )
        if major_capacity > 1e-9:
            take = min(remaining, major_capacity)
            if primary_axis == "vertical" and is_last_lane:
                mouth_gap = abs(mouth_x - current[0])
                take = min(take, max(0.0, remaining - mouth_gap))
            end = _point_along_segment(current, major_target, take)
            if take > 1e-9:
                segments.append((current, end, take))
                current = end
                remaining -= take
                if remaining <= 1e-9:
                    break

        if is_last_lane:
            secondary_target = (mouth_x, current[1])
        else:
            next_lane = lane_positions[lane_index + 1]
            if primary_axis == "horizontal":
                secondary_target = (current[0], next_lane)
            else:
                secondary_target = (next_lane, current[1])

        secondary_capacity = math.hypot(
            secondary_target[0] - current[0], secondary_target[1] - current[1]
        )
        if secondary_capacity > 1e-9:
            take = min(remaining, secondary_capacity)
            end = _point_along_segment(current, secondary_target, take)
            segments.append((current, end, take))
            current = end
            remaining -= take
            if remaining <= 1e-9:
                break

        current_major_direction *= -1

    if remaining > 1e-9:
        raise ValueError("Could not fit requested path length inside folded enclosure")

    return segments, pitch


def _segment_height_limit(
    segment: Tuple[Tuple[float, float], Tuple[float, float], float],
    enclosure_dims: Tuple[float, float],
    fill_factor: float,
) -> float:
    (x0, y0), (x1, y1), _ = segment
    depth, height = enclosure_dims
    if abs(y1 - y0) <= 1e-9:
        boundary_limit = 2.0 * min(y0, height - y0)
        limit = boundary_limit
    else:
        boundary_limit = 2.0 * min(x0, depth - x0)
        limit = boundary_limit
    return fill_factor * max(limit, 1e-6)


# ─── Main entry point ───────────────────────────────────────────────────────


def extrapolate_folded_horn(
    horn: HornGeometry,
    enclosure_dims: Tuple[float, float],
    driver_coord: Tuple[float, float],
    enclosure_width: float,
    fill_factor: float = 0.8,
) -> HornGeometry:
    """Map an optimized continuous flare onto a folded constant-width layout."""
    if horn.profile_type is None:
        raise ValueError("Folded extrapolation requires a continuous profile horn")
    profile_type = horn.profile_type

    depth, height = enclosure_dims
    driver_x, driver_y = driver_coord
    if depth <= 0 or height <= 0:
        raise ValueError("Enclosure dimensions must be > 0")
    if enclosure_width <= 0:
        raise ValueError("Enclosure width must be > 0")
    throat_offset = throat_chamber_side_length(horn, enclosure_width)
    throat_chamber_height = throat_offset

    base_margin = max(0.01, 0.05 * min(depth, height))
    if not (0.0 <= driver_x <= depth):
        raise ValueError("driver_x must lie on or inside the enclosure boundary")
    chamber_margin = max(base_margin, throat_chamber_height / 2.0)
    if not (chamber_margin < driver_y < height - chamber_margin):
        raise ValueError(
            "driver_y must lie inside the enclosure with throat chamber clearance"
        )
    if throat_chamber_height > 0 and not (
        throat_chamber_height / 2.0 < driver_y < height - throat_chamber_height / 2.0
    ):
        raise ValueError("Throat chamber around driver exceeds enclosure height")

    primary_axis = "vertical" if height > depth else "horizontal"

    def _select_candidate(
        candidate_paths: List[
            Tuple[
                float,
                List[Tuple[Tuple[float, float], Tuple[float, float], float]],
                float,
                bool,
            ]
        ],
        x_boundary_margin: float,
        y_boundary_margin: float,
        max_width: Optional[float] = None,
        prefer_boundary_terminal: bool = False,
    ):
        target_x = (
            x_boundary_margin if driver_x <= depth / 2.0 else depth - x_boundary_margin
        )
        pool = candidate_paths
        if max_width is not None:
            feasible = [item for item in candidate_paths if item[0] <= max_width + 1e-9]
            if feasible:
                pool = feasible
        best_overall = min(pool, key=lambda item: item[0])

        if prefer_boundary_terminal:
            boundary_candidates = [item for item in pool if item[3]]
            if boundary_candidates:
                return min(boundary_candidates, key=lambda item: item[0])

        front_facing = []
        same_side = []
        for item in pool:
            _, path_segments, _, _ = item
            if not path_segments:
                continue
            end_x = path_segments[-1][1][0]
            last_start, last_end, _ = path_segments[-1]
            facing_front = (last_end[0] - last_start[0]) > 1e-9
            facing_side = (
                (last_end[0] - last_start[0]) < -1e-9
                if driver_x <= depth / 2.0
                else (last_end[0] - last_start[0]) > 1e-9
            )
            dist_target = abs(end_x - target_x)
            if facing_front:
                front_facing.append((item[0], dist_target, item))
            if facing_side:
                same_side.append((item[0], dist_target, item))

        competitive_width_limit = best_overall[0] * 1.1 + 1e-9
        front_facing = [
            pair for pair in front_facing if pair[0] <= competitive_width_limit
        ]
        same_side = [pair for pair in same_side if pair[0] <= competitive_width_limit]

        if front_facing:
            return min(front_facing, key=lambda pair: (pair[0], pair[1]))[2]
        if same_side:
            return min(same_side, key=lambda pair: (pair[0], pair[1]))[2]
        return best_overall

    def _evaluate_candidates(
        near_x_boundary_margin: float,
        far_x_boundary_margin: float,
        y_boundary_margin: float,
    ):
        candidate_paths: List[
            Tuple[
                float,
                List[Tuple[Tuple[float, float], Tuple[float, float], float]],
                float,
                bool,
            ]
        ] = []

        if primary_axis == "horizontal":
            direction_pairs = [(1, -1), (1, 1)]
        else:
            direction_pairs = [(-1, 1), (1, 1)]

        for major_direction, secondary_direction in direction_pairs:
            terminal_patterns = [False]
            if primary_axis == "vertical" and driver_x <= depth / 2.0:
                terminal_patterns.append(True)
            for use_boundary_terminal_l in terminal_patterns:
                try:
                    if primary_axis == "vertical":
                        path_segments, pitch = _build_vertical_fold_path_optimized(
                            enclosure_dims,
                            driver_coord,
                            horn.path_length,
                            throat_offset,
                            major_direction,
                            near_x_boundary_margin,
                            far_x_boundary_margin,
                            y_boundary_margin,
                            profile_type,
                            horn.throat_area,
                            horn.mouth_area,
                            enclosure_width,
                            use_boundary_terminal_l=use_boundary_terminal_l,
                        )
                    else:
                        path_segments, pitch = _build_fold_path(
                            enclosure_dims,
                            driver_coord,
                            horn.path_length,
                            throat_offset,
                            primary_axis,
                            major_direction,
                            secondary_direction,
                            near_x_boundary_margin,
                            far_x_boundary_margin,
                            y_boundary_margin,
                            use_boundary_terminal_l=use_boundary_terminal_l,
                        )
                except ValueError:
                    if primary_axis == "vertical":
                        try:
                            path_segments, pitch = _build_fold_path(
                                enclosure_dims,
                                driver_coord,
                                horn.path_length,
                                throat_offset,
                                primary_axis,
                                major_direction,
                                secondary_direction,
                                near_x_boundary_margin,
                                far_x_boundary_margin,
                                y_boundary_margin,
                                use_boundary_terminal_l=use_boundary_terminal_l,
                            )
                        except ValueError:
                            continue
                    else:
                        continue

                cumulative = 0.0
                required_width = 0.0
                for segment in path_segments:
                    _, _, seg_length = segment
                    h_limit = _segment_height_limit(
                        segment, enclosure_dims, fill_factor
                    )
                    area_start = profile_area_at_distance(
                        profile_type,
                        horn.throat_area,
                        horn.mouth_area,
                        horn.path_length,
                        cumulative,
                        horn.hyperbolic_t,
                    )
                    area_end = profile_area_at_distance(
                        profile_type,
                        horn.throat_area,
                        horn.mouth_area,
                        horn.path_length,
                        cumulative + seg_length,
                        horn.hyperbolic_t,
                    )
                    required_width = max(
                        required_width,
                        area_start / h_limit,
                        area_end / h_limit,
                    )
                    cumulative += seg_length

                candidate_paths.append(
                    (
                        required_width,
                        path_segments,
                        pitch,
                        use_boundary_terminal_l,
                    )
                )

        return candidate_paths

    x_margin = base_margin
    far_x_margin = max(
        base_margin, horn.mouth_area / (2.0 * enclosure_width * fill_factor)
    )
    y_margin = max(base_margin, horn.mouth_area / (2.0 * enclosure_width * fill_factor))
    candidates = _evaluate_candidates(x_margin, far_x_margin, y_margin)

    if not candidates:
        raise ValueError("Could not fit a folded path inside the requested enclosure")

    resolved_width = enclosure_width

    required_width, path_segments, pitch, _ = _select_candidate(
        candidates,
        x_margin,
        y_margin,
        max_width=resolved_width,
        prefer_boundary_terminal=(
            primary_axis == "vertical" and driver_x <= depth / 2.0
        ),
    )
    if resolved_width + 1e-9 < required_width:
        raise ValueError(
            f"Enclosure width {resolved_width:.4f} m is too small; requires at least {required_width:.4f} m"
        )

    coordinates: List[Tuple[float, float]] = []
    conical_segments: List[Tuple[float, float, float]] = []
    rectangular_segments: List[Tuple[float, float, float, float, float]] = []

    cumulative = 0.0
    for start, end, seg_length in path_segments:
        area_start = profile_area_at_distance(
            profile_type,
            horn.throat_area,
            horn.mouth_area,
            horn.path_length,
            cumulative,
            horn.hyperbolic_t,
        )
        area_end = profile_area_at_distance(
            profile_type,
            horn.throat_area,
            horn.mouth_area,
            horn.path_length,
            cumulative + seg_length,
            horn.hyperbolic_t,
        )
        h_limit = _segment_height_limit(
            segment=(start, end, seg_length),
            enclosure_dims=enclosure_dims,
            fill_factor=fill_factor,
        )
        h_start = area_start / resolved_width
        h_end = area_end / resolved_width
        if h_start > h_limit + 1e-9 or h_end > h_limit + 1e-9:
            raise ValueError(
                "Fixed folded width produces a segment taller than enclosure clearance"
            )
        coordinates.append((float(start[0]), float(start[1])))
        conical_segments.append((float(h_start), float(h_end), float(seg_length)))
        rectangular_segments.append(
            (
                float(resolved_width),
                float(h_start),
                float(resolved_width),
                float(h_end),
                float(seg_length),
            )
        )
        cumulative += seg_length

    coordinates.append((float(path_segments[-1][1][0]), float(path_segments[-1][1][1])))

    return HornGeometry(
        throat_area=horn.throat_area,
        mouth_area=horn.mouth_area,
        path_length=horn.path_length,
        enclosure_type=horn.enclosure_type,
        path_diff=horn.path_diff,
        ang=horn.ang,
        vrc=horn.vrc,
        lrc=horn.lrc,
        fr_rc=horn.fr_rc,
        vtc=horn.vtc,
        atc=horn.atc,
        fr_tc=horn.fr_tc,
        profile_type=None,
        hyperbolic_t=horn.hyperbolic_t,
        n_segments=horn.n_segments,
        width=float(resolved_width),
        conical_segments=conical_segments,
        rectangular_segments=rectangular_segments,
        coordinates=coordinates,
        enclosure_dims=(float(depth), float(height)),
        driver_coord=(float(driver_x), float(driver_y)),
        discretisation=horn.discretisation,
        bend_angles=horn.bend_angles,
        lem_step_model=horn.lem_step_model,
        lem_step_strength=horn.lem_step_strength,
        lem_step_resistance=horn.lem_step_resistance,
    )
