"""Distance layer: great-circle distance and travel-time matrices."""

from __future__ import annotations

from typing import List

from .data import DeliveryInstance
from .geo import haversine_km  # re-exported for a stable public API

__all__ = ["haversine_km", "build_distance_matrix", "build_time_matrix"]


def build_distance_matrix(instance: DeliveryInstance) -> List[List[float]]:
    """Symmetric distance matrix (km) over all nodes, depot included."""
    stops = instance.stops
    n = len(stops)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_km(stops[i].lat, stops[i].lon, stops[j].lat, stops[j].lon)
            dist[i][j] = d
            dist[j][i] = d
    return dist


def build_time_matrix(
    distance_matrix: List[List[float]], speed_kmph: float
) -> List[List[float]]:
    """Travel-time matrix (minutes) derived from distances and speed."""
    if speed_kmph <= 0:
        raise ValueError("speed_kmph must be positive")
    factor = 60.0 / speed_kmph  # km -> minutes
    return [[d * factor for d in row] for row in distance_matrix]
