"""Improvement heuristics: 2-opt and Or-opt local search.

Both operators use candidate *neighbour lists* (each node's k nearest stops) so
they scale to large instances, and both accept a move only if it (a) shortens
the total distance and (b) keeps the route feasible with respect to the time
windows and capacity.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .data import DeliveryInstance
from .evaluate import EPS, evaluate_route, is_feasible, route_distance


def _repair_cost(
    route: List[int],
    instance: DeliveryInstance,
    distance_matrix: List[List[float]],
    time_matrix: List[List[float]],
) -> tuple:
    """Lexicographic cost (constraint_violation, distance) used for repair."""
    res = evaluate_route(route, instance, distance_matrix, time_matrix)
    violation = res.total_lateness
    if instance.capacity is not None:
        violation += max(0.0, res.load - instance.capacity)
    return (violation, res.distance)


def repair(
    route: Sequence[int],
    instance: DeliveryInstance,
    distance_matrix: List[List[float]],
    time_matrix: List[List[float]],
    neighbours: Optional[Dict[int, List[int]]] = None,
    max_passes: int = 200,
) -> List[int]:
    """Relocate stops to reduce time-window / capacity violations.

    Uses a lexicographic objective: drive total violation to zero first, then
    prefer shorter distance. Returns the best route found; if the instance has
    no feasible ordering, it returns the least-infeasible route.
    """
    route = list(route)
    best_cost = _repair_cost(route, instance, distance_matrix, time_matrix)
    if best_cost[0] <= EPS:
        return route  # already feasible

    scan_all = len(route) <= 300
    if neighbours is None:
        neighbours = build_neighbour_lists(instance, distance_matrix, k=20)

    for _ in range(max_passes):
        improved = False
        for seg_len in (1, 2):
            i = 0
            while i + seg_len <= len(route):
                seg = route[i : i + seg_len]
                reduced = route[:i] + route[i + seg_len :]

                if scan_all:
                    positions = range(len(reduced) + 1)
                else:
                    reduced_pos = {node: idx for idx, node in enumerate(reduced)}
                    cand = set()
                    for endpoint in (seg[0], seg[-1]):
                        for v in neighbours[endpoint]:
                            p = reduced_pos.get(v)
                            if p is not None:
                                cand.add(p)
                                cand.add(p + 1)
                    positions = sorted(cand)

                best_route = None
                for p in positions:
                    candidate = reduced[:p] + seg + reduced[p:]
                    cost = _repair_cost(
                        candidate, instance, distance_matrix, time_matrix
                    )
                    if cost < best_cost:
                        best_cost, best_route = cost, candidate

                if best_route is not None:
                    route = best_route
                    improved = True
                    if best_cost[0] <= EPS:
                        return route
                else:
                    i += 1
        if not improved:
            break
    return route


def build_neighbour_lists(
    instance: DeliveryInstance,
    distance_matrix: List[List[float]],
    k: int = 10,
) -> Dict[int, List[int]]:
    """For every node, the ``k`` nearest *stops* (depot excluded as a target)."""
    stop_indices = instance.stop_indices
    neighbours: Dict[int, List[int]] = {}
    for node in range(len(instance.stops)):
        ranked = sorted(
            (s for s in stop_indices if s != node),
            key=lambda s: distance_matrix[node][s],
        )
        neighbours[node] = ranked[:k]
    return neighbours


def two_opt(
    route: Sequence[int],
    instance: DeliveryInstance,
    distance_matrix: List[List[float]],
    time_matrix: List[List[float]],
    neighbours: Optional[Dict[int, List[int]]] = None,
    max_passes: int = 60,
) -> List[int]:
    """Improve a route by reversing segments (2-opt), keeping feasibility."""
    route = list(route)
    n = len(route)
    if n < 3:
        return route

    if neighbours is None:
        neighbours = build_neighbour_lists(instance, distance_matrix)

    pos = {node: idx for idx, node in enumerate(route)}
    D = distance_matrix

    for _ in range(max_passes):
        improved = False
        for p in range(n):
            a = route[p - 1] if p > 0 else 0     # node before the segment
            b = route[p]
            for c in neighbours[a]:
                q = pos.get(c)
                if q is None or q <= p:
                    continue
                d = route[q]
                e = route[q + 1] if q + 1 < n else 0
                # Remove edges (a,b) and (d,e); add (a,d) and (b,e).
                delta = D[a][d] + D[b][e] - D[a][b] - D[d][e]
                if delta < -EPS:
                    route[p : q + 1] = route[p : q + 1][::-1]
                    if is_feasible(route, instance, distance_matrix, time_matrix):
                        for idx in range(p, q + 1):
                            pos[route[idx]] = idx
                        improved = True
                        break
                    route[p : q + 1] = route[p : q + 1][::-1]  # revert
            if improved:
                break
        if not improved:
            break
    return route


def or_opt(
    route: Sequence[int],
    instance: DeliveryInstance,
    distance_matrix: List[List[float]],
    time_matrix: List[List[float]],
    neighbours: Optional[Dict[int, List[int]]] = None,
    segment_lengths: Sequence[int] = (1, 2, 3),
    max_passes: int = 60,
) -> List[int]:
    """Relocate short segments to better positions (Or-opt)."""
    route = list(route)
    if len(route) < 3:
        return route

    if neighbours is None:
        neighbours = build_neighbour_lists(instance, distance_matrix)

    D = distance_matrix

    for _ in range(max_passes):
        improved = False
        for seg_len in segment_lengths:
            i = 0
            while i + seg_len <= len(route):
                seg = route[i : i + seg_len]
                prev = route[i - 1] if i > 0 else 0
                nxt = route[i + seg_len] if i + seg_len < len(route) else 0
                removal_gain = D[prev][seg[0]] + D[seg[-1]][nxt] - D[prev][nxt]

                reduced = route[:i] + route[i + seg_len :]
                reduced_pos = {node: idx for idx, node in enumerate(reduced)}

                best_delta = -EPS
                best_insert_at: Optional[int] = None

                # Insert the segment just after a near neighbour of its head...
                for v in neighbours[seg[0]]:
                    p = reduced_pos.get(v)
                    if p is None:
                        continue
                    x = v
                    y = reduced[p + 1] if p + 1 < len(reduced) else 0
                    insertion = D[x][seg[0]] + D[seg[-1]][y] - D[x][y]
                    delta = insertion - removal_gain
                    if delta < best_delta:
                        best_delta, best_insert_at = delta, p + 1

                # ...or just before a near neighbour of its tail.
                for v in neighbours[seg[-1]]:
                    p = reduced_pos.get(v)
                    if p is None:
                        continue
                    x = reduced[p - 1] if p > 0 else 0
                    y = v
                    insertion = D[x][seg[0]] + D[seg[-1]][y] - D[x][y]
                    delta = insertion - removal_gain
                    if delta < best_delta:
                        best_delta, best_insert_at = delta, p

                applied = False
                if best_insert_at is not None:
                    candidate = (
                        reduced[:best_insert_at] + seg + reduced[best_insert_at:]
                    )
                    if is_feasible(candidate, instance, distance_matrix, time_matrix):
                        route = candidate
                        improved = applied = True

                if not applied:
                    i += 1  # only advance when nothing moved at this position
        if not improved:
            break
    return route


def local_search(
    route: Sequence[int],
    instance: DeliveryInstance,
    distance_matrix: List[List[float]],
    time_matrix: List[List[float]],
    neighbours: Optional[Dict[int, List[int]]] = None,
    rounds: int = 5,
) -> List[int]:
    """Alternate 2-opt and Or-opt until neither improves the distance."""
    if neighbours is None:
        neighbours = build_neighbour_lists(instance, distance_matrix)

    best = list(route)
    best_dist = route_distance(best, distance_matrix)

    for _ in range(rounds):
        best = two_opt(best, instance, distance_matrix, time_matrix, neighbours)
        best = or_opt(best, instance, distance_matrix, time_matrix, neighbours)
        dist = route_distance(best, distance_matrix)
        if dist > best_dist - EPS:  # no meaningful improvement this round
            best_dist = dist
            break
        best_dist = dist
    return best
