Last-Mile Delivery Route Optimizer (VRPTW)

A small, **dependency-free** Python toolkit for planning a single delivery
vehicle's route through a depot and *N* customer stops, each with a delivery
**time window**, so that total driving distance is minimized while every window
(and an optional weight capacity) is respected.

The core solver is pure standard library. `matplotlib` is used *only* by the
optional `--plot` flags in `demo.py` and `benchmark.py`; nothing in the
`route_optimizer` package imports it.

---

## The problem

This is the **Vehicle Routing Problem with Time Windows (VRPTW)** for one
vehicle — equivalently, a Travelling Salesman Problem where each stop may only
be served inside `[ready, due]`, the vehicle waits if it arrives early, and an
arrival after `due` is infeasible. With windows switched off it reduces to a
plain distance-minimizing TSP. Both are NP-hard, so the aim is high-quality
routes fast, with honest measures of how far from optimal we are.

---

## Architecture

The package is a thin pipeline of single-responsibility modules:

| Module        | Responsibility |
|---------------|----------------|
| `geo.py`      | `haversine_km` great-circle distance (shared by data + distance, kept separate to avoid a circular import). |
| `data.py`     | `Stop`, `DeliveryInstance`, and `generate_instance` — synthetic Dallas-metro instances. |
| `distance.py` | `build_distance_matrix` (symmetric haversine) and `build_time_matrix` (km → minutes at the vehicle speed). |
| `evaluate.py` | `evaluate_route` simulates the schedule from the depot at *t=0*, waiting when early, and returns a `RouteResult` (distance, duration, feasibility, lateness, load, per-stop arrivals). |
| `construct.py`| Three construction heuristics: `nearest_neighbour` (distance-greedy), `time_oriented_nn` (urgency-greedy), `earliest_due_date` (sort by deadline). |
| `improve.py`  | `two_opt`, `or_opt`, and `local_search` (distance polish under feasibility), plus `repair` (penalty-guided relocation to fix violations). |
| `exact.py`    | `brute_force_exact` — optimal route by permutation for N ≤ 10, the ground truth for quality checks. |
| `bounds.py`   | `mst_lower_bound` — Prim's minimum spanning tree, a valid lower bound on any tour's distance. |
| `solver.py`   | `solve` — the end-to-end construct → repair → improve pipeline, returning a `SolveReport`. |

### Key data structures

- **`Stop`** — a frozen dataclass: `index, lat, lon, ready, due, service, demand`.
- **`DeliveryInstance`** — the depot (index 0) plus the stop list, vehicle
  `speed_kmph`, optional `capacity`, and the `use_time_windows` flag.
- **Distance / time matrices** — plain `list[list[float]]`; symmetric, computed
  once per solve and threaded through every routine (no globals, no hidden
  state, fully deterministic for a given seed).
- **A route** — a `list[int]` of stop indices in visiting order; the depot at
  both ends is implicit.

### How `solve()` works

1. **Construct.** For a pure TSP, greedy nearest-neighbour. For a time-windowed
   or capacitated instance, build three cheap seeds (time-oriented,
   distance-greedy, earliest-due-date) and keep the one that violates the
   constraints least.
2. **Repair (safety net).** If that seed is still infeasible, relocate stops
   using a lexicographic objective — drive total violation to zero first, then
   prefer shorter distance — until feasible.
3. **Improve.** Polish distance with 2-opt and Or-opt, using per-node
   *neighbour lists* (each node's *k* nearest stops) so the moves stay cheap at
   scale. Every move is accepted only if it both shortens the route and keeps
   it feasible.

---

## Quick start

```bash
# from the project root (the directory containing this README)
python demo.py                      # 20 stops, time windows, seed 42
python demo.py --stops 8            # small enough to also print the exact optimum
python demo.py --stops 40 --capacity 300 --plot route.png
python demo.py --no-windows --stops 200   # plain TSP

python benchmark.py                 # quality-vs-speed tables for both tracks
python benchmark.py --plot bench.png --seeds 5

python tests_scratch.py             # correctness / regression checks
```

`demo.py` prints the full schedule — arrival time, window, wait, service, and
running load for every stop — plus distance, duration, the MST lower bound, and
(for N ≤ 10) the gap above the brute-force optimum.

---

## The quality-vs-speed story

Two things are measured as *N* grows: how long a solve takes, and how good the
route is. Quality is reported two ways — against the **MST lower bound** (always
available, but loose: a spanning tree is cheaper than any tour) and against the
**brute-force optimum** (exact, but only computable for N ≤ 10).

Representative run (`python benchmark.py`, 3 seeds/size):

```
TSP track (single route, open windows)
    N     dist(km)   +MST%    +opt%    mean(ms)
   10        121.4    41.2     0.00        0.5
   50        243.8    22.6      -          7.2
  100        340.1    20.8      -         23.1
  500        702.4    17.4      -       1103.0
 1000        983.3    17.0      -       4922.4

VRPTW track (single vehicle, hard time windows)
    N   feasible   dist(km)   +opt%    mean(ms)
   10        3/3      154.4    0.28        1.1
   50        3/3      900.4     -         42.3
  100        3/3     1783.2     -        301.6
  250        3/3     4011.1     -       6244.7
```

What this shows:

- **Near-optimal where we can check.** The heuristic is within ~1% of the true
  optimum at N = 10 (0.00% TSP, 0.28% VRPTW), and worst-case over 30 seeds it
  never beats the optimum and stays ~1% for TSP.
- **The MST gap is an over-estimate.** It settles around 17% for TSP not because
  the routes are 17% too long, but because a spanning tree is a genuinely weaker
  bound than a tour — the real gap is far smaller.
- **TSP scales to 1000 stops in ~5 s.** Neighbour-list-restricted 2-opt/Or-opt
  keeps the improvement phase near-linear in practice.
- **Two operators earn their keep.** Local search removes ~15–30% of the
  distance the construction heuristic leaves on the table.

### Why the two tracks stop at different sizes — an honest limitation

The TSP track runs to 1000 stops; the time-windowed track stops around 250.
That is deliberate. What guarantees VRPTW feasibility here is a **full-scan
repair** (it considers every re-insertion position, which neighbour-restricted
repair was measured to be unable to match on feasibility). Full-scan repair is
roughly *O(N³)*, so it is the cost driver at the top of the VRPTW track.

More fundamentally, a *single* vehicle physically cannot serve many hundreds of
tightly-windowed stops in one shift — past a few hundred stops the problem stops
being "route one vehicle well" and becomes "how many vehicles do I need," i.e. a
fleet-sizing problem, which is out of scope here. So the VRPTW track is swept
across the range where a single-vehicle answer is meaningful.

### Generating solvable time-windowed instances

Randomly-drawn time windows are almost never jointly satisfiable by one vehicle
as *N* grows, which would make "percent above optimal" meaningless. Instead, in
the spirit of the classic Solomon benchmarks, `generate_instance` lays down a
random seed route, simulates the arrival time at each stop, and centres that
stop's window on its arrival (with random slack). A feasible route therefore
provably exists — the seed itself — so feasibility rates and optimality gaps are
honest, well-defined quantities.

---

## Testing

`tests_scratch.py` checks, with zero failures:

- Heuristic vs **exact optimum** over 30 seeds (both TSP and VRPTW): the
  heuristic is always feasible and never beats the optimum.
- **Local search** never worsens distance and preserves every stop exactly once.
- **Time-window and capacity** constraints are genuinely enforced.
- **100% feasibility** across N ∈ {10, 25, 50, 100, 250} on the VRPTW track.
- **Determinism**: a fixed seed always produces the identical route.
- **Scaling**: N up to 1000 solves quickly with every stop present.

---

## Design choices at a glance

- **Dependency-free core** — trivial to drop into any environment; `matplotlib`
  is optional and confined to plotting entry points.
- **Everything is deterministic and explicit** — matrices and neighbour lists
  are passed in, never stashed globally, so results are reproducible and the
  pieces are unit-testable in isolation.
- **Feasibility first, then distance** — repair uses a lexicographic
  (violation, distance) objective; local search only ever accepts feasible,
  distance-reducing moves.
- **Always measured against a ground truth** — exact optimum for small N, a
  valid MST bound for all N — so quality claims are checkable, not asserted.
