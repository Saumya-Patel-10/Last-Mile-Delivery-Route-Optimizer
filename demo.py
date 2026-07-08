#!/usr/bin/env python3
"""Solve one delivery instance and print a human-readable route + schedule.

Usage:
    python demo.py                 # 20 stops, time windows on, seed 42
    python demo.py --stops 40      # bigger instance
    python demo.py --no-windows    # plain TSP (open windows)
    python demo.py --plot route.png

The optional --plot flag draws the route with matplotlib (the only place the
package touches a third-party library; the solver itself is pure stdlib).
"""

from __future__ import annotations

import argparse

from route_optimizer import (
    build_distance_matrix,
    build_time_matrix,
    brute_force_exact,
    evaluate_route,
    generate_instance,
    solve,
)


def _fmt_clock(minutes: float) -> str:
    """Minutes-since-depot-open -> HH:MM on a nominal 08:00 start."""
    total = int(round(minutes))
    h, m = divmod(total, 60)
    return f"{8 + h:02d}:{m:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Route optimizer demo")
    parser.add_argument("--stops", type=int, default=20, help="number of stops")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--no-windows", action="store_true", help="disable time windows (TSP)"
    )
    parser.add_argument(
        "--capacity", type=float, default=None, help="vehicle capacity (weight)"
    )
    parser.add_argument(
        "--plot", metavar="PATH", default=None, help="save a route plot to PATH"
    )
    args = parser.parse_args()

    instance = generate_instance(
        args.stops,
        seed=args.seed,
        use_time_windows=not args.no_windows,
        capacity=args.capacity,
    )

    report = solve(instance)
    res = report.result

    kind = "TSP (no time windows)" if args.no_windows else "VRPTW (time windows)"
    print(f"\n{kind} - {args.stops} stops, seed {args.seed}")
    print("=" * 64)
    print(f"  feasible          : {res.feasible}")
    print(f"  total distance    : {report.distance:8.2f} km")
    print(f"  route duration    : {res.duration:8.1f} min "
          f"({_fmt_clock(res.duration)} back at depot)")
    if instance.capacity is not None:
        print(f"  load / capacity   : {res.load:8.1f} / {instance.capacity:.1f}")
    print(f"  MST lower bound   : {report.lower_bound:8.2f} km")
    if report.gap_vs_lower_bound is not None:
        print(f"  gap vs MST bound  : {report.gap_vs_lower_bound * 100:7.1f} %")
    print(f"  saved by 2opt/oropt: {report.improvement_over_construction * 100:6.1f} % "
          f"(from {report.construction_distance:.1f} km)")
    print(f"  solve time        : {report.elapsed_seconds * 1000:8.1f} ms")

    # Exact optimum for small instances, so the demo shows true quality.
    if args.stops <= 10:
        D = build_distance_matrix(instance)
        T = build_time_matrix(D, instance.speed_kmph)
        _, opt = brute_force_exact(instance, D, T)
        if opt:
            print(f"  brute-force optimum: {opt:8.2f} km  "
                  f"(heuristic is +{(report.distance / opt - 1) * 100:.2f}%)")

    print("\n  stop   arrive   window          wait  service  load")
    print("  " + "-" * 54)
    stops = instance.stops
    for node, arrive in zip(report.route, res.arrivals):
        s = stops[node]
        start = max(arrive, s.ready)
        wait = start - arrive
        win = f"[{_fmt_clock(s.ready)}-{_fmt_clock(s.due)}]"
        print(f"  {node:4d}  {_fmt_clock(arrive)}  {win:15s}"
              f"  {wait:4.0f}  {s.service:6.0f}  {s.demand:5.1f}")
    print()

    if args.plot:
        _plot(instance, report.route, args.plot)
        print(f"  route plot written to {args.plot}\n")


def _plot(instance, route, path: str) -> None:
    """Draw stops and the visiting order; depot highlighted. Needs matplotlib."""
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    stops = instance.stops
    order = [0] + list(route) + [0]
    xs = [stops[i].lon for i in order]
    ys = [stops[i].lat for i in order]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(xs, ys, "-", color="#3b6ea5", linewidth=1.2, zorder=1)
    ax.scatter(
        [s.lon for s in stops[1:]], [s.lat for s in stops[1:]],
        c="#3b6ea5", s=30, zorder=2, label="stops",
    )
    ax.scatter(
        [stops[0].lon], [stops[0].lat],
        c="#d1495b", s=160, marker="*", zorder=3, label="depot",
    )
    for rank, node in enumerate(route, start=1):
        ax.annotate(str(rank), (stops[node].lon, stops[node].lat),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")

    ax.set_title(f"Route: {len(route)} stops, {sum_dist(instance, route):.1f} km")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def sum_dist(instance, route) -> float:
    D = build_distance_matrix(instance)
    return evaluate_route(route, instance, D,
                          build_time_matrix(D, instance.speed_kmph)).distance


if __name__ == "__main__":
    main()
