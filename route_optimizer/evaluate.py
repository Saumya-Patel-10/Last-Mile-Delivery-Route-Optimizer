"""Evaluation layer: simulate a route's schedule and check feasibility.

A *route* is a list of stop indices in visiting order, excluding the depot.
The vehicle always leaves the depot (node 0) at time 0 and returns to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .data import DeliveryInstance

# Numerical slack (minutes) so floating-point noise never flips feasibility.
EPS = 1e-6


@dataclass(frozen=True)
class RouteResult:
    """Outcome of simulating a route."""

    route: List[int]
    distance: float          # total kilometres, depot -> ... -> depot
    duration: float          # total minutes incl. waiting and service
    feasible: bool           # respects all time windows and capacity
    total_lateness: float    # sum of window violations (minutes); 0 if feasible
    load: float              # total demand carried
    arrivals: List[float]    # arrival time at each stop, aligned with ``route``


def evaluate_route(
    route: List[int],
    instance: DeliveryInstance,
    distance_matrix: List[List[float]],
    time_matrix: List[List[float]],
) -> RouteResult:
    """Simulate ``route`` and return distance, timing and feasibility."""
    stops = instance.stops
    t = 0.0            # clock, in minutes, starting at the depot
    prev = 0           # depot
    total_distance = 0.0
    total_lateness = 0.0
    load = 0.0
    arrivals: List[float] = []
    feasible = True

    for s in route:
        total_distance += distance_matrix[prev][s]
        t += time_matrix[prev][s]
        arrivals.append(t)

        stop = stops[s]
        if instance.use_time_windows:
            if t < stop.ready:
                t = stop.ready              # wait for the window to open
            if t > stop.due + EPS:
                feasible = False
                total_lateness += t - stop.due
        t += stop.service
        load += stop.demand
        prev = s

    # Return leg to the depot.
    total_distance += distance_matrix[prev][0]
    t += time_matrix[prev][0]

    if instance.capacity is not None and load > instance.capacity + EPS:
        feasible = False

    return RouteResult(
        route=list(route),
        distance=total_distance,
        duration=t,
        feasible=feasible,
        total_lateness=total_lateness,
        load=load,
        arrivals=arrivals,
    )


def route_distance(
    route: List[int], distance_matrix: List[List[float]]
) -> float:
    """Total distance of a route, including depot at both ends."""
    if not route:
        return 0.0
    total = distance_matrix[0][route[0]]
    for a, b in zip(route, route[1:]):
        total += distance_matrix[a][b]
    total += distance_matrix[route[-1]][0]
    return total


def is_feasible(
    route: List[int],
    instance: DeliveryInstance,
    distance_matrix: List[List[float]],
    time_matrix: List[List[float]],
) -> bool:
    """Convenience wrapper returning only the feasibility flag."""
    return evaluate_route(route, instance, distance_matrix, time_matrix).feasible


def feasible_prefix_time(
    route: List[int],
    instance: DeliveryInstance,
    time_matrix: List[List[float]],
    start_index: int = 0,
    start_time: float = 0.0,
    start_node: int = 0,
) -> Optional[float]:
    """Check time-window feasibility of ``route[start_index:]``.

    Simulates only the tail of a route, given the clock/node state when the
    tail begins. Returns the departure time after the last stop's service if
    the tail is feasible, or ``None`` on the first violation. This lets the
    local-search moves re-check only the portion of the route they changed.
    """
    if not instance.use_time_windows:
        return start_time  # nothing to violate

    stops = instance.stops
    t = start_time
    prev = start_node
    for s in route[start_index:]:
        t += time_matrix[prev][s]
        stop = stops[s]
        if t < stop.ready:
            t = stop.ready
        if t > stop.due + EPS:
            return None
        t += stop.service
        prev = s
    return t
