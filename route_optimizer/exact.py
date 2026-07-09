"""Exact solver: brute force over all orderings.

Only tractable for small N (<= ~10), where it provides the ground-truth optimum
used to measure how close the heuristics get.
"""

from __future__ import annotations

from itertools import permutations
from typing import List, Optional, Tuple

from .data import DeliveryInstance
from .evaluate import evaluate_route

DEFAULT_MAX_N = 10


def brute_force_exact(
    instance: DeliveryInstance,
    distance_matrix: List[List[float]],
    time_matrix: List[List[float]],
    max_n: int = DEFAULT_MAX_N,
) -> Tuple[Optional[List[int]], Optional[float]]:
    """Return the minimum-distance feasible route and its distance.

    Raises ``ValueError`` if the instance is too large to enumerate. Returns
    ``(None, None)`` if no feasible ordering exists.
    """
    n = instance.n_stops
    if n > max_n:
        raise ValueError(
            f"brute force is limited to N <= {max_n} (got N = {n}); "
            "use solve() for larger instances"
        )

    best_route: Optional[List[int]] = None
    best_distance: Optional[float] = None

    for perm in permutations(instance.stop_indices):
        result = evaluate_route(list(perm), instance, distance_matrix, time_matrix)
        if not result.feasible:
            continue
        if best_distance is None or result.distance < best_distance:
            best_distance = result.distance
            best_route = list(perm)

    return best_route, best_distance
