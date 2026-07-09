"""Data layer: synthetic delivery instances.

Index 0 is always the depot. Stops are indices 1..N.

Units:
    - coordinates : degrees (lat, lon)
    - time        : minutes from the start of the planning horizon
    - demand      : abstract "package weight" units

Time windows are generated so that a feasible route is guaranteed to exist:
we lay out a random seed route, simulate its arrival times, and place each
stop's window around its arrival time with random slack. This mirrors how
solvable VRPTW benchmark instances (e.g. Solomon) are built, and keeps the
"percent above optimal" metric meaningful instead of frequently infeasible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .geo import haversine_km

# Rough bounding box for the Dallas / Fort Worth metro area.
DALLAS_BBOX = {
    "lat_min": 32.60,
    "lat_max": 33.00,
    "lon_min": -97.00,
    "lon_max": -96.60,
}


@dataclass(frozen=True)
class Stop:
    """A single node in the problem (depot or delivery point)."""

    index: int
    lat: float
    lon: float
    ready: float          # earliest service time (minutes)
    due: float            # latest service time (minutes)
    service: float        # time spent servicing the stop (minutes)
    demand: float         # package weight / units

    @property
    def is_depot(self) -> bool:
        return self.index == 0


@dataclass(frozen=True)
class DeliveryInstance:
    """A complete VRPTW instance for one vehicle."""

    stops: List[Stop]
    speed_kmph: float = 30.0            # average urban travel speed
    capacity: Optional[float] = None    # None disables the capacity constraint
    use_time_windows: bool = True       # False turns the problem into a plain TSP
    horizon: float = 480.0              # length of the planning day (minutes)

    @property
    def n_stops(self) -> int:
        """Number of delivery stops, excluding the depot."""
        return len(self.stops) - 1

    @property
    def depot(self) -> Stop:
        return self.stops[0]

    @property
    def stop_indices(self) -> List[int]:
        """Indices of all non-depot stops."""
        return [s.index for s in self.stops[1:]]


def generate_instance(
    n_stops: int,
    *,
    seed: Optional[int] = None,
    speed_kmph: float = 30.0,
    use_time_windows: bool = True,
    capacity: Optional[float] = None,
    service_minutes: float = 5.0,
    window_slack: Tuple[float, float] = (30.0, 180.0),
    bbox: Optional[dict] = None,
) -> DeliveryInstance:
    """Generate a random instance with ``n_stops`` delivery points.

    ``window_slack`` gives the (min, max) minutes of slack sampled on each side
    of a stop's seed arrival time; wider slack means looser windows and more
    room for the optimizer to shorten the route.
    """
    if n_stops < 1:
        raise ValueError("n_stops must be >= 1")

    rng = random.Random(seed)
    box = bbox or DALLAS_BBOX

    depot_lat = (box["lat_min"] + box["lat_max"]) / 2
    depot_lon = (box["lon_min"] + box["lon_max"]) / 2

    # Node coordinates (index 0 = depot).
    coords: List[Tuple[float, float]] = [(depot_lat, depot_lon)]
    for _ in range(n_stops):
        coords.append(
            (
                rng.uniform(box["lat_min"], box["lat_max"]),
                rng.uniform(box["lon_min"], box["lon_max"]),
            )
        )

    demands = [0.0] + [
        rng.uniform(1.0, 10.0) if capacity is not None else 0.0
        for _ in range(n_stops)
    ]

    if not use_time_windows:
        # Plain TSP: open windows across a nominal day.
        horizon = 480.0
        stops = [Stop(0, depot_lat, depot_lon, 0.0, horizon, 0.0, 0.0)]
        for i in range(1, n_stops + 1):
            stops.append(
                Stop(i, coords[i][0], coords[i][1], 0.0, horizon,
                     service_minutes, demands[i])
            )
        return DeliveryInstance(stops, speed_kmph, capacity, False, horizon)

    # --- Feasibility-guaranteed time windows -------------------------------
    minutes_per_km = 60.0 / speed_kmph
    seed_order = list(range(1, n_stops + 1))
    rng.shuffle(seed_order)

    arrival: Dict[int, float] = {}
    t = 0.0
    prev = 0
    for s in seed_order:
        t += haversine_km(*coords[prev], *coords[s]) * minutes_per_km
        arrival[s] = t
        t += service_minutes
        prev = s
    t += haversine_km(*coords[prev], *coords[0]) * minutes_per_km  # return leg

    lo_slack, hi_slack = window_slack
    windows: Dict[int, Tuple[float, float]] = {}
    max_due = 0.0
    for s in seed_order:
        back = rng.uniform(lo_slack, hi_slack)
        fwd = rng.uniform(lo_slack, hi_slack)
        ready = max(0.0, arrival[s] - back)
        due = arrival[s] + fwd
        windows[s] = (ready, due)
        max_due = max(max_due, due)

    horizon = max(t, max_due) + max(hi_slack, service_minutes)

    stops = [Stop(0, depot_lat, depot_lon, 0.0, horizon, 0.0, 0.0)]
    for i in range(1, n_stops + 1):
        ready, due = windows[i]
        stops.append(
            Stop(i, coords[i][0], coords[i][1], ready, due,
                 service_minutes, demands[i])
        )

    return DeliveryInstance(stops, speed_kmph, capacity, True, horizon)
