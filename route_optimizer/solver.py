"""End-to-end solve pipeline: construct -> improve, with a full report."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

from .bounds import mst_lower_bound
from .construct import earliest_due_date, nearest_neighbour, time_oriented_nn
from .data import DeliveryInstance
from .distance import build_distance_matrix, build_time_matrix
from .evaluate import RouteResult, evaluate_route, is_feasible
from .improve import build_neighbour_lists, local_search, repair


@dataclass
class SolveReport:
    """Everything a caller needs to inspect or benchmark a solve."""

    route: List[int]
    result: RouteResult              # evaluation of the final route
    construction_distance: float     # distance right after nearest neighbour
    lower_bound: float               # MST lower bound on distance
    elapsed_seconds: float

    @property
    def distance(self) -> float:
        return self.result.distance

    @property
    def feasible(self) -> bool:
        return self.result.feasible

    @property
    def gap_vs_lower_bound(self) -> Optional[float]:
        """Fractional gap above the MST lower bound (e.g. 0.18 == +18%)."""
        if self.lower_bound <= 0:
            return None
        return self.distance / self.lower_bound - 1.0

    @property
    def improvement_over_construction(self) -> Optional[float]:
        """Fraction of construction distance removed by local search."""
        if self.construction_distance <= 0:
            return None
        return 1.0 - self.distance / self.construction_distance


def _violation_cost(
    route: List[int],
    instance: DeliveryInstance,
    distance_matrix: List[List[float]],
    time_matrix: List[List[float]],
) -> tuple:
    """Lexicographic (constraint_violation, distance) cost of a route."""
    res = evaluate_route(route, instance, distance_matrix, time_matrix)
    violation = res.total_lateness
    if instance.capacity is not None:
        violation += max(0.0, res.load - instance.capacity)
    return (violation, res.distance)


def _least_infeasible(
    a: List[int],
    b: List[int],
    instance: DeliveryInstance,
    distance_matrix: List[List[float]],
    time_matrix: List[List[float]],
) -> List[int]:
    """Return whichever route has the smaller (violation, distance) cost."""
    cost_a = _violation_cost(a, instance, distance_matrix, time_matrix)
    cost_b = _violation_cost(b, instance, distance_matrix, time_matrix)
    return a if cost_a <= cost_b else b


def solve(
    instance: DeliveryInstance,
    *,
    neighbour_k: int = 10,
    improve: bool = True,
    repair_passes: int = 40,
) -> SolveReport:
    """Solve an instance and return a detailed report.

    For pure-distance (TSP) instances the route is built greedily by nearest
    neighbour. For time-windowed / capacitated instances a distance-greedy seed
    tends to strand stops behind closed windows, so we build three cheap seeds
    (time-oriented, distance-greedy, earliest-due-date), keep the least
    infeasible, and run a penalty-guided repair as a safety net. Distance is
    then polished by 2-opt / Or-opt local search under feasibility constraints.
    """
    start = time.perf_counter()

    distance_matrix = build_distance_matrix(instance)
    time_matrix = build_time_matrix(distance_matrix, instance.speed_kmph)
    neighbours = build_neighbour_lists(instance, distance_matrix, k=neighbour_k)

    constrained = instance.use_time_windows or instance.capacity is not None

    if constrained:
        # Cheap, diverse seeds; keep whichever violates constraints least.
        seeds = [
            time_oriented_nn(instance, distance_matrix, time_matrix),
            nearest_neighbour(instance, distance_matrix, time_matrix),
            earliest_due_date(instance),
        ]
        route = seeds[0]
        for alt in seeds[1:]:
            route = _least_infeasible(
                route, alt, instance, distance_matrix, time_matrix
            )
    else:
        route = nearest_neighbour(instance, distance_matrix, time_matrix)

    construction_distance = evaluate_route(
        route, instance, distance_matrix, time_matrix
    ).distance

    if constrained and not is_feasible(
        route, instance, distance_matrix, time_matrix
    ):
        route = repair(
            route, instance, distance_matrix, time_matrix,
            neighbours, max_passes=repair_passes,
        )

    if improve:
        route = local_search(
            route, instance, distance_matrix, time_matrix, neighbours
        )

    result = evaluate_route(route, instance, distance_matrix, time_matrix)
    elapsed = time.perf_counter() - start
    lower_bound = mst_lower_bound(distance_matrix)

    return SolveReport(
        route=route,
        result=result,
        construction_distance=construction_distance,
        lower_bound=lower_bound,
        elapsed_seconds=elapsed,
    )
