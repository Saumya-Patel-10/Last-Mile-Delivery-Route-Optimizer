"""Construction heuristic: nearest neighbour.

From the current node, go to the closest unvisited stop whose time window can
still be met (and that respects capacity). If no such stop exists, fall back to
the least-late option so a full route is always produced; the returned route is
then flagged infeasible by the evaluator.
"""

from __future__ import annotations

from typing import List, Optional

from .data import DeliveryInstance


def earliest_due_date(instance: DeliveryInstance) -> List[int]:
    """Order stops by earliest due time (ties broken by ready time).

    A simple, feasibility-oriented seed that handles time-critical stops first;
    used as a fallback when nearest-neighbour strands stops behind windows.
    """
    stops = instance.stops
    return sorted(
        instance.stop_indices,
        key=lambda s: (stops[s].due, stops[s].ready),
    )


def time_oriented_nn(
    instance: DeliveryInstance,
    distance_matrix: List[List[float]],
    time_matrix: List[List[float]],
) -> List[int]:
    """Feasibility-oriented construction for time-windowed instances.

    From the current node, among the unvisited stops that are still reachable
    before their window closes (and fit capacity), pick the most urgent one
    (earliest due time), breaking ties by distance. Serving urgent-yet-reachable
    stops first avoids stranding them, so this rarely produces an infeasible
    route. Distance is cleaned up afterwards by local search.
    """
    stops = instance.stops
    unvisited = set(instance.stop_indices)
    route: List[int] = []

    current = 0
    clock = 0.0
    load = 0.0

    while unvisited:
        best = None
        best_key = None
        fallback = None
        fallback_key = None

        for s in unvisited:
            stop = stops[s]
            arrival = clock + time_matrix[current][s]
            start = max(arrival, stop.ready)
            lateness = max(0.0, start - stop.due)
            over_capacity = (
                instance.capacity is not None
                and load + stop.demand > instance.capacity
            )
            if lateness == 0.0 and not over_capacity:
                key = (stop.due, distance_matrix[current][s])
                if best_key is None or key < best_key:
                    best_key, best = key, s
            fb_key = (lateness, distance_matrix[current][s])
            if fallback_key is None or fb_key < fallback_key:
                fallback_key, fallback = fb_key, s

        chosen = best if best is not None else fallback
        stop = stops[chosen]
        arrival = clock + time_matrix[current][chosen]
        clock = max(arrival, stop.ready) + stop.service
        load += stop.demand

        route.append(chosen)
        unvisited.remove(chosen)
        current = chosen

    return route


def nearest_neighbour(
    instance: DeliveryInstance,
    distance_matrix: List[List[float]],
    time_matrix: List[List[float]],
) -> List[int]:
    """Build an initial route visiting every stop exactly once."""
    stops = instance.stops
    unvisited = set(instance.stop_indices)
    route: List[int] = []

    current = 0          # depot
    clock = 0.0          # minutes
    load = 0.0

    while unvisited:
        best: Optional[int] = None
        best_key = None
        fallback: Optional[int] = None
        fallback_key = None

        for s in unvisited:
            travel = time_matrix[current][s]
            arrival = clock + travel
            stop = stops[s]

            start = max(arrival, stop.ready) if instance.use_time_windows else arrival
            lateness = max(0.0, start - stop.due) if instance.use_time_windows else 0.0
            over_capacity = (
                instance.capacity is not None
                and load + stop.demand > instance.capacity
            )

            feasible = lateness == 0.0 and not over_capacity
            dist = distance_matrix[current][s]

            if feasible:
                if best_key is None or dist < best_key:
                    best_key, best = dist, s
            # Track a fallback ranked by (lateness, distance) for dead ends.
            key = (lateness, dist)
            if fallback_key is None or key < fallback_key:
                fallback_key, fallback = key, s

        chosen = best if best is not None else fallback
        assert chosen is not None

        stop = stops[chosen]
        arrival = clock + time_matrix[current][chosen]
        if instance.use_time_windows:
            arrival = max(arrival, stop.ready)
        clock = arrival + stop.service
        load += stop.demand

        route.append(chosen)
        unvisited.remove(chosen)
        current = chosen

    return route
