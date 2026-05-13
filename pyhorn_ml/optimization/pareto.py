"""Pareto front extraction and multi-objective ranking."""
from typing import List
import numpy as np

from pyhorn_ml.core.design_point import DesignPoint


def dominated_scores(a: dict, b: dict) -> bool:
    """True if b dominates a (b is better on all, strictly better on one).

    Objectives: flatness, sensitivity, bass (higher = better for all).
    """
    objectives = ["flatness_score", "sensitivity_score", "bass_score"]
    at_least_as_good = all(
        b.get(k, 0) >= a.get(k, 0) for k in objectives
    )
    strictly_better = any(
        b.get(k, 0) > a.get(k, 0) for k in objectives
    )
    return at_least_as_good and strictly_better


def extract_pareto_front(designs: List[DesignPoint]) -> List[DesignPoint]:
    """Extract the non-dominated Pareto front from a list of designs.

    O(n²) domination filter — suitable for up to ~10,000 designs.
    """
    front: List[DesignPoint] = []
    for candidate in designs:
        is_dominated = False
        for other in designs:
            if other is candidate:
                continue
            if dominated_scores(
                {"flatness_score": candidate.flatness_score,
                 "sensitivity_score": candidate.sensitivity_score,
                 "bass_score": candidate.bass_score},
                {"flatness_score": other.flatness_score,
                 "sensitivity_score": other.sensitivity_score,
                 "bass_score": other.bass_score},
            ):
                is_dominated = True
                break
        candidate.dominated = is_dominated
        if not is_dominated:
            front.append(candidate)
    return front


def assign_pareto_ranks(designs: List[DesignPoint]) -> None:
    """Assign Pareto front ranks using NSGA-II-like non-dominated sorting.

    Rank 0 = first front (Pareto optimal), rank 1 = second front, etc.
    """
    remaining = list(designs)
    rank = 0
    while remaining:
        front = extract_pareto_front(remaining)
        front_ids = {id(dp) for dp in front}
        for dp in list(remaining):
            if id(dp) in front_ids:
                dp.pareto_rank = rank
                remaining.remove(dp)
        rank += 1


def crowding_distance(front: List[DesignPoint]) -> dict[int, float]:
    """Compute crowding distance for NSGA-II selection.

    Higher distance = more diverse in objective space = better for selection.

    Returns: dp.id → crowding_distance map
    """
    if len(front) <= 2:
        return {id(dp): float("inf") for dp in front}

    distances = {id(dp): 0.0 for dp in front}
    objectives = ["flatness_score", "sensitivity_score", "bass_score"]

    for obj in objectives:
        # Sort front by this objective
        sorted_front = sorted(front, key=lambda d: getattr(d, obj))
        obj_min = getattr(sorted_front[0], obj)
        obj_max = getattr(sorted_front[-1], obj)
        range_ = obj_max - obj_min if obj_max > obj_min else 1.0

        # Boundary points get infinite distance
        distances[id(sorted_front[0])] = float("inf")
        distances[id(sorted_front[-1])] = float("inf")

        # Interior points
        for i in range(1, len(sorted_front) - 1):
            prev = getattr(sorted_front[i - 1], obj)
            next_ = getattr(sorted_front[i + 1], obj)
            distances[id(sorted_front[i])] += (next_ - prev) / range_

    return distances
