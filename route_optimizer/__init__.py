"""Last-Mile Delivery Route Optimizer (VRPTW).

A small, dependency-free toolkit for the Vehicle Routing Problem with
Time Windows on a single vehicle:

    - data     : synthetic instance generation
    - distance : haversine distance / travel-time matrices
    - evaluate : schedule simulation + feasibility (time windows, capacity)
    - construct: nearest-neighbour construction heuristic
    - improve  : 2-opt and Or-opt local search
    - exact    : brute-force exact solver (ground truth for small N)
    - bounds   : minimum-spanning-tree lower bound
    - solver   : end-to-end solve() pipeline
"""

from .data import DeliveryInstance, Stop, generate_instance
from .distance import haversine_km, build_distance_matrix, build_time_matrix
from .evaluate import RouteResult, evaluate_route
from .construct import earliest_due_date, nearest_neighbour, time_oriented_nn
from .improve import two_opt, or_opt, local_search, repair
from .exact import brute_force_exact
from .bounds import mst_lower_bound
from .solver import solve, SolveReport

__all__ = [
    "DeliveryInstance",
    "Stop",
    "generate_instance",
    "haversine_km",
    "build_distance_matrix",
    "build_time_matrix",
    "RouteResult",
    "evaluate_route",
    "nearest_neighbour",
    "time_oriented_nn",
    "earliest_due_date",
    "two_opt",
    "or_opt",
    "local_search",
    "repair",
    "brute_force_exact",
    "mst_lower_bound",
    "solve",
    "SolveReport",
]
