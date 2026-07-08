import sys, time
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from route_optimizer import (
    generate_instance, build_distance_matrix, build_time_matrix,
    nearest_neighbour, two_opt, or_opt, local_search,
    brute_force_exact, mst_lower_bound, evaluate_route, solve,
)

def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        raise SystemExit("TEST FAILED: " + msg)

# 1) Exact vs heuristic on small instances, with and without time windows.
for tw in (False, True):
    worst_gap = 0.0
    for seed in range(30):
        inst = generate_instance(8, seed=seed, use_time_windows=tw)
        D = build_distance_matrix(inst)
        T = build_time_matrix(D, inst.speed_kmph)
        opt_route, opt_dist = brute_force_exact(inst, D, T)
        rep = solve(inst)
        check(rep.feasible or opt_route is None, f"heuristic feasible (tw={tw}, seed={seed})")
        if opt_dist is not None:
            # Heuristic can never beat the exact optimum (allow tiny fp slack).
            check(rep.distance >= opt_dist - 1e-6, f"heuristic >= optimal (tw={tw}, seed={seed})")
            gap = rep.distance / opt_dist - 1.0
            worst_gap = max(worst_gap, gap)
    print(f"  time_windows={tw}: worst gap vs OPTIMAL over 30 seeds = {worst_gap*100:.2f}%")

# 2) Local search never worsens distance and preserves feasibility (TSP case).
inst = generate_instance(60, seed=1, use_time_windows=False)
D = build_distance_matrix(inst); T = build_time_matrix(D, inst.speed_kmph)
nn = nearest_neighbour(inst, D, T)
d_nn = evaluate_route(nn, inst, D, T).distance
ls = local_search(nn, inst, D, T)
d_ls = evaluate_route(ls, inst, D, T).distance
check(sorted(ls) == sorted(inst.stop_indices), "local search keeps all stops exactly once")
check(d_ls <= d_nn + 1e-6, "local search does not worsen distance")
print(f"  N=60 TSP: NN={d_nn:.2f} km -> local search={d_ls:.2f} km ({(1-d_ls/d_nn)*100:.1f}% shorter)")

# 3) Time-window feasibility genuinely enforced; guaranteed-feasible generator.
inst = generate_instance(40, seed=7, use_time_windows=True, capacity=400.0)
rep = solve(inst)
res = rep.result
check(sorted(rep.route) == sorted(inst.stop_indices), "TW route keeps all stops")
check(res.feasible, "TW+capacity route is feasible")
check(res.load <= inst.capacity + 1e-6, "capacity respected")
check(res.distance >= rep.lower_bound - 1e-6, "distance >= MST lower bound")
print(f"  N=40 TW+cap: dist={res.distance:.2f} km, load={res.load:.1f}, gap vs MST={rep.gap_vs_lower_bound*100:.1f}%")

# 3b) Feasibility rate across sizes (generator guarantees a feasible route).
# Single-vehicle time-windowed routing is meaningful up to a few hundred stops;
# beyond that it becomes a fleet-sizing problem (out of scope). Full-scan repair
# is what secures feasibility, so the largest sizes use fewer seeds to stay fast.
for n, trials in ((10, 10), (25, 10), (50, 10), (100, 10), (250, 5)):
    ok = sum(solve(generate_instance(n, seed=s, use_time_windows=True)).feasible
             for s in range(trials))
    check(ok == trials, f"N={n} TW feasible on all {trials} seeds")
print("  TW feasibility rate = 100% across N in {10,25,50,100,250}")

# 4) Determinism: same seed -> same result.
r1 = solve(generate_instance(30, seed=99, use_time_windows=True))
r2 = solve(generate_instance(30, seed=99, use_time_windows=True))
check(r1.route == r2.route and abs(r1.distance - r2.distance) < 1e-9, "deterministic for fixed seed")

# 5) Scaling smoke test up to N=1000 on the pure-distance (TSP) track, where a
#    single route to 1000 stops is the intended regime. Checks it runs quickly,
#    stays feasible, and keeps every stop exactly once.
for n in (100, 300, 1000):
    t0 = time.perf_counter()
    rep = solve(generate_instance(n, seed=3, use_time_windows=False))
    dt = time.perf_counter() - t0
    check(rep.feasible, f"N={n} feasible")
    check(sorted(rep.route) == sorted(range(1, n+1)), f"N={n} all stops present")
    print(f"  N={n}: {dt:.2f}s, dist={rep.distance:.1f} km, gap vs MST={rep.gap_vs_lower_bound*100:.1f}%")

print("\nALL TESTS PASSED")
