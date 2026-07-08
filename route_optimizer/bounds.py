"""Lower bounds for the routing distance.

A tour that visits every node, minus one edge, is a spanning tree, so the
weight of a minimum spanning tree (MST) is a valid lower bound on the optimal
tour length. It ignores time windows, so it bounds the distance relaxation.
"""

from __future__ import annotations

from typing import List


def mst_lower_bound(distance_matrix: List[List[float]]) -> float:
    """Weight of the MST over all nodes, via Prim's algorithm (O(n^2))."""
    n = len(distance_matrix)
    if n <= 1:
        return 0.0

    in_tree = [False] * n
    best_edge = [float("inf")] * n
    best_edge[0] = 0.0
    total = 0.0

    for _ in range(n):
        u = -1
        u_cost = float("inf")
        for v in range(n):
            if not in_tree[v] and best_edge[v] < u_cost:
                u_cost, u = best_edge[v], v
        in_tree[u] = True
        total += u_cost
        row = distance_matrix[u]
        for v in range(n):
            if not in_tree[v] and row[v] < best_edge[v]:
                best_edge[v] = row[v]

    return total
