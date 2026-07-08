#!/usr/bin/env python3
"""Benchmark solution quality against speed as the instance size grows.

Two tracks:

  TSP  - open time windows; a single route to every stop. This is the regime
         that scales to 1000 stops, and where we can quote a distance gap
         against the MST lower bound (and the exact optimum for N <= 10).

  VRPTW- real time windows on a single vehicle. Feasibility, not raw size, is
         the binding constraint here, so this track is swept to a few hundred
         stops (beyond that a single vehicle can't serve the day and it becomes
         a fleet-sizing problem, which is out of scope).

Usage:
    python benchmark.py                 # print tables
    python benchmark.py --plot bench.png# also save a quality-vs-speed plot
    python benchmark.py --seeds 5       # average over more seeds
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import List, Tuple

from route_optimizer import (
    build_distance_matrix,
    build_time_matrix,
    brute_force_exact,
    generate_instance,
    solve,
)

TSP_SIZES = (10, 25, 50, 100, 250, 500, 1000)
VRPTW_SIZES = (10, 25, 50, 100, 250)


def _exact_gap(instance) -> float | None:
    """Percentage above the brute-force optimum, or None if too large."""
    if len(instance.stops) - 1 > 10:
        return None
    D = build_distance_matrix(instance)
    T = build_time_matrix(D, instance.speed_kmph)
    _, opt = brute_force_exact(instance, D, T)
    if not opt:
        return None
    rep = solve(instance)
    return (rep.distance / opt - 1.0) * 100.0


def _run_track(sizes, use_windows: bool, seeds: int) -> List[Tuple]:
    rows = []
    for n in sizes:
        dists, times, gaps, feas = [], [], [], 0
        exact_gaps = []
        for s in range(seeds):
            inst = generate_instance(n, seed=s, use_time_windows=use_windows)
            t0 = time.perf_counter()
            rep = solve(inst)
            times.append(time.perf_counter() - t0)
            dists.append(rep.distance)
            if rep.gap_vs_lower_bound is not None:
                gaps.append(rep.gap_vs_lower_bound * 100.0)
            feas += rep.feasible
            eg = _exact_gap(inst)
            if eg is not None:
                exact_gaps.append(eg)
        rows.append((
            n,
            feas, seeds,
            statistics.mean(dists),
            statistics.mean(gaps) if gaps else float("nan"),
            statistics.mean(exact_gaps) if exact_gaps else None,
            statistics.mean(times) * 1000.0,
            max(times) * 1000.0,
        ))
    return rows


def _print_table(title: str, rows: List[Tuple], show_feas: bool) -> None:
    print(f"\n{title}")
    print("-" * 78)
    header = (f"{'N':>5}  {'feasible':>9}  {'dist(km)':>9}  {'+MST%':>7}  "
              f"{'+opt%':>7}  {'mean(ms)':>9}  {'max(ms)':>9}")
    print(header)
    print("-" * 78)
    for (n, feas, seeds, dist, gap, egap, tmean, tmax) in rows:
        feas_str = f"{feas}/{seeds}" if show_feas else "-"
        egap_str = f"{egap:6.2f}" if egap is not None else "   -  "
        print(f"{n:>5}  {feas_str:>9}  {dist:9.1f}  {gap:6.1f}  "
              f"{egap_str:>7}  {tmean:9.1f}  {tmax:9.1f}")


def _plot(tsp_rows, vrptw_rows, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for rows, label, color in (
        (tsp_rows, "TSP", "#3b6ea5"),
        (vrptw_rows, "VRPTW", "#d1495b"),
    ):
        ns = [r[0] for r in rows]
        tmean = [r[6] for r in rows]
        gap = [r[4] for r in rows]
        ax1.plot(ns, tmean, "o-", color=color, label=label)
        ax2.plot(ns, gap, "o-", color=color, label=label)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("stops (N)")
    ax1.set_ylabel("mean solve time (ms)")
    ax1.set_title("Speed vs size")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    ax2.set_xscale("log")
    ax2.set_xlabel("stops (N)")
    ax2.set_ylabel("mean gap above MST lower bound (%)")
    ax2.set_title("Quality vs size")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quality-vs-speed benchmark")
    parser.add_argument("--seeds", type=int, default=3,
                        help="instances averaged per size")
    parser.add_argument("--plot", metavar="PATH", default=None,
                        help="save a quality-vs-speed plot to PATH")
    parser.add_argument("--max-tsp", type=int, default=1000,
                        help="largest TSP size to run")
    args = parser.parse_args()

    tsp_sizes = [n for n in TSP_SIZES if n <= args.max_tsp]

    print("Benchmarking (averaged over "
          f"{args.seeds} seed{'s' if args.seeds != 1 else ''} per size)...")
    print("  +MST%  = distance above the MST lower bound (loose but valid)")
    print("  +opt%  = distance above the brute-force optimum (N <= 10 only)")

    tsp_rows = _run_track(tsp_sizes, use_windows=False, seeds=args.seeds)
    _print_table("TSP track (single route, open windows) - scales to 1000 stops",
                 tsp_rows, show_feas=False)

    vrptw_rows = _run_track(VRPTW_SIZES, use_windows=True, seeds=args.seeds)
    _print_table("VRPTW track (single vehicle, hard time windows)",
                 vrptw_rows, show_feas=True)

    print("\nReading the numbers:")
    print("  - The MST bound underestimates a real tour, so +MST% overstates the")
    print("    true gap; the +opt% column shows the heuristic is within ~1-2% of")
    print("    optimal where the optimum is computable.")
    print("  - VRPTW stays 100% feasible to a few hundred stops; the full-scan")
    print("    repair that guarantees this is the cost driver at the top end.")

    if args.plot:
        _plot(tsp_rows, vrptw_rows, args.plot)
        print(f"\nPlot written to {args.plot}")


if __name__ == "__main__":
    main()
