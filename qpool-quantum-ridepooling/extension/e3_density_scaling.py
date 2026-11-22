"""Extension E2 — does the dispatch core outgrow the device when demand gets denser?

E1 showed that, at the paper's demand level (the first 240 requests of the public Sioux Falls trace, about
125 requests per hour), exact reduction settles 68 % of the dispatch periods and the core that reaches the
device has a median of five binaries.  A real city is denser.  E3 replays the same pipeline at k = 1, 2, 4, 8
and 12.5 times the demand density: the first 240 k requests of the trace (3 000 at k = 12.5, the whole
trace) with arrival times divided by k, so that the horizon stays at about 1.9 h, and a fleet scaled with
demand (16 k and 32 k vehicles), for 60-s and 300-s dispatch periods.  Every period is frozen and analysed
as in E1:

  * conflict-graph size before and after exact reduction, largest connected component;
  * whether the period fits the SPQC-550 model (550 spins, signed 8-bit couplings) at a usable precision
    (rho >= 1 on the paper's power-of-two ladder), component by component;
  * the exact optimum with HiGHS (time limit 60 s) and its solve time, and the LP-relaxation bound of the
    clique-constrained packing (integrality gap), i.e. how hard the core is for a classical solver.

Requires the public trace (scripts/fetch_example_data.py --prefix 3000 --out data/full_example).
The submission's reduction routine rescans the whole period after every removal; a work-list version of the same
rule is used here and checked to give identical kept / fixed / removed sets on every period with at most 600
candidates.
Run:  python extension/e3_density_scaling.py      (resumable: finished runs are kept in results/extension/_e3_work/done)
"""
from __future__ import annotations
import json, os, shutil, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import milp, linprog, Bounds, LinearConstraint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import qpool.replay as R                                                       # noqa: E402
from qpool.replay import run_replay, Config                                    # noqa: E402
from qpool.qubo import PackingInstance, connected_components, induced          # noqa: E402
from qpool.competition import reduce_packing                                   # noqa: E402
from qpool.solvers import packing_constraints                                  # noqa: E402
from e1_precision_audit import ladder_rho                                      # noqa: E402

SRC = ROOT / "data" / "full_example"
WORK = ROOT / "results" / "extension" / "_e3_work"
OUT = ROOT / "results" / "extension"
DENSITY = (1, 2, 4, 8, 12.5)
FLEET_PER_240 = (16, 32)
EPOCHS = (60, 300)


def scaled_trace(k: float) -> Path:
    d = WORK / f"trace_k{k}"; d.mkdir(parents=True, exist_ok=True)
    for f in SRC.iterdir():
        if f.name != "requests.csv":
            shutil.copy(f, d / f.name)
    r = pd.read_csv(SRC / "requests.csv").head(int(round(240 * k))).copy()
    r["arrival"] = np.rint(r.arrival / k).astype(int)
    r.to_csv(d / "requests.csv", index=False)
    return d


def fast_reduce_packing(inst: PackingInstance):
    """The submission's exact reduction (qpool.competition.reduce_packing) with a work list instead of a full rescan.

    The reference removes, at every step, the smallest-index active vertex that is removable (non-positive weight,
    isolated, or dominated in the closed neighbourhood by a neighbour u with w_u >= w_v, ties by index) and then
    rescans from the start, which is cubic in the period size.  Removing x can only change the removability of
    vertices within distance two of x, so only those are re-examined; popping the smallest index from the
    set of re-examined vertices therefore removes exactly the same vertex as the reference at every step.
    Equality of kept, fixed and removed sets with the reference is asserted on every period with at most
    VERIFY_MAX candidates (see analyse)."""
    import heapq
    n = len(inst.weights); w = inst.weights; adj = inst.adjacency(); active = [True] * n
    heap = list(range(n)); heapq.heapify(heap); queued = [True] * n
    fixed, removed = [], []

    def removable(v):
        if w[v] <= 0:
            return "removed"
        nb = {u for u in adj[v] if active[u]}
        if not nb:
            return "fixed"
        Nv = nb | {v}
        for u in sorted(nb):
            if w[u] < w[v]:
                continue
            Nu = {x for x in adj[u] if active[x]} | {u}
            if Nu <= Nv and (w[u] > w[v] or Nu != Nv or u < v):
                return "removed"
        return None

    while heap:
        v = heapq.heappop(heap); queued[v] = False
        if not active[v]:
            continue
        r = removable(v)
        if r is None:
            continue
        active[v] = False
        (fixed if r == "fixed" else removed).append(v)
        touched = set()
        for u in adj[v]:
            if active[u]:
                touched.add(u)
                touched |= {x for x in adj[u] if active[x]}
        for x in touched:
            if not queued[x]:
                heapq.heappush(heap, x); queued[x] = True
    kept = [v for v in range(n) if active[v]]
    return induced(inst, kept), kept, sorted(fixed), sorted(removed)


def event_reduce_packing(inst: PackingInstance):
    """Same reduction and the same removal order as the reference, driven by dominance counts.

    For every ordered adjacent pair (u, v) the count c[u -> v] = |N[u] minus N[v]| of the active closed
    neighbourhoods is kept up to date; u dominates v exactly when c[u -> v] = 0.  Removing x lowers c[u -> v]
    by one precisely when u is an active neighbour of x and v is an active neighbour of u that is not adjacent
    to x, and no other removal event can make a vertex removable except losing its last neighbour.  A vertex
    therefore only re-enters the work list when one of its counts reaches zero or it becomes isolated, and
    popping the smallest index from the work list removes the same vertex as the reference's full rescan.
    Each removal costs a few vectorised operations over the neighbourhoods of the removed vertex's neighbours.
    Equality of kept, fixed and removed sets with the reference is asserted on every period with at most
    VERIFY_MAX candidates (see analyse)."""
    import heapq
    n = len(inst.weights); w = np.asarray(inst.weights, float)
    if n == 0:
        return induced(inst, []), [], [], []
    E = np.asarray(inst.edges, dtype=np.int64).reshape(-1, 2)
    key = np.unique(np.concatenate([E[:, 0] * n + E[:, 1], E[:, 1] * n + E[:, 0]]))
    a, nbr = key // n, key % n                                   # directed pairs, sorted by (a, nbr)
    ptr = np.searchsorted(a, np.arange(n + 1)); deg = np.diff(ptr)
    rev = np.searchsorted(key, nbr * n + a)                      # position of (nbr -> a)
    words = (n + 63) // 64                                       # packed adjacency rows for common-neighbour counts
    bits = np.zeros((n, words), np.uint64)
    np.bitwise_or.at(bits, (a, nbr // 64), np.left_shift(np.uint64(1), (nbr % 64).astype(np.uint64)))
    common = np.empty(len(a), np.int64)
    for s0 in range(0, len(a), 20000):
        sl = slice(s0, s0 + 20000)
        common[sl] = np.bitwise_count(bits[a[sl]] & bits[nbr[sl]]).sum(axis=1)
    del bits
    c = deg[a] - 1 - common                                      # |N[a] \ N[nbr]| for adjacent a, nbr
    active = np.ones(n, bool); deg_act = deg.copy(); in_nx = np.zeros(n, bool)
    heap = list(range(n)); heapq.heapify(heap); queued = np.ones(n, bool)
    fixed, removed = [], []

    def status(v):
        if w[v] <= 0:
            return "removed"
        if deg_act[v] == 0:
            return "fixed"
        P = slice(ptr[v], ptr[v + 1]); us = nbr[P]
        ok = active[us] & (c[rev[P]] == 0) & (w[us] >= w[v]) & ((w[us] > w[v]) | (c[P] != 0) | (us < v))
        return "removed" if ok.any() else None

    while heap:
        v = heapq.heappop(heap); queued[v] = False
        if not active[v]:
            continue
        r = status(v)
        if r is None:
            continue
        active[v] = False
        (fixed if r == "fixed" else removed).append(v)
        nx = nbr[ptr[v]:ptr[v + 1]]; nx = nx[active[nx]]
        if len(nx) == 0:
            continue
        deg_act[nx] -= 1
        events = [nx[deg_act[nx] == 0]]
        lens = deg[nx]; tot = int(lens.sum())
        idx = np.repeat(ptr[nx] - np.concatenate(([0], np.cumsum(lens)[:-1])), lens) + np.arange(tot)
        vs = nbr[idx]
        in_nx[nx] = True
        m = active[vs] & ~in_nx[vs]
        in_nx[nx] = False
        pos = idx[m]; c[pos] -= 1
        events.append(vs[m][c[pos] == 0])
        ev = np.unique(np.concatenate(events))
        for x in ev[~queued[ev]].tolist():
            heapq.heappush(heap, x); queued[x] = True
    kept = np.flatnonzero(active).tolist()
    return induced(inst, kept), kept, sorted(fixed), sorted(removed)


VERIFY_MAX = 600


def exact_and_lp(inst: PackingInstance):
    n = len(inst.weights)
    if n == 0:
        return 0.0, 0.0, 0.0
    A = packing_constraints(inst); c = -inst.weights
    t0 = time.perf_counter()
    res = milp(c, integrality=np.ones(n), bounds=Bounds(0, 1), constraints=[LinearConstraint(A, -np.inf, 1)],
               options={"time_limit": 60, "mip_rel_gap": 0.0})
    t_mip = time.perf_counter() - t0
    lp = linprog(c, A_ub=A, b_ub=np.ones(A.shape[0]), bounds=(0, 1), method="highs")       # sparse constraint matrix
    opt = -res.fun if res.status == 0 else np.nan
    return float(opt), float(-lp.fun), float(t_mip)


def analyse_instance(inst: PackingInstance, tag: str, k: int) -> dict:
    """Reduction, device fit, exact optimum and LP bound of one frozen period."""
    red_inst, kept, fixed, removed = event_reduce_packing(inst)
    verified = len(inst.weights) <= VERIFY_MAX
    if verified:                                              # identical to the submission's implementation
        ref = reduce_packing(inst)
        assert (ref.kept, ref.fixed, ref.removed) == (kept, fixed, removed), (tag, k)
    comps = connected_components(red_inst)
    sizes = sorted((len(c) for c in comps), reverse=True)
    fit_all, max_spins = True, 0
    for ids in comps:
        if len(ids) < 2:
            continue
        if len(ids) + 1 > 550:                                # plus the reference spin: exceeds SPQC-550 at any precision
            fit_all = False
            continue
        r_pass, _, f = ladder_rho(induced(red_inst, ids))
        if r_pass is None or r_pass < 1:
            fit_all = False
        else:
            max_spins = max(max_spins, len(f[1]))
    opt, lp, t_mip = exact_and_lp(inst)
    return dict(run=tag, period=k, n_candidates=len(inst.weights), n_edges=len(inst.edges),
                n_after_reduction=len(kept), n_components=len(comps), largest_component=sizes[0] if sizes else 0,
                reduction_verified_against_reference=verified,
                fits_device_rho_ge_1=fit_all, max_spins=max_spins, opt=opt, lp_bound=lp,
                lp_gap_rel=(lp - opt) / max(abs(opt), 1e-9) if np.isfinite(opt) else np.nan, mip_seconds=t_mip)


def analyse(snapshots, tag):
    """Analysis of stored snapshots (run_replay(..., capture=True)); main() streams the same instances instead."""
    return [analyse_instance(PackingInstance(np.array(s["weights"], float), [tuple(e) for e in s["edges"]], s.get("labels")), tag, k)
            for k, s in enumerate(snapshots)]


def replay_and_analyse(trace, out, cfg, tag):
    """Run the classical replay and analyse every period's packing instance as it is solved.

    The instance handed to the replay's MILP is exactly the one run_replay(capture=True) would store as the period's
    snapshot (same weights, edges and labels), so the rows equal analyse(snapshots); streaming avoids holding tens of
    millions of conflict edges in memory at the higher densities.  The analysis runs before the replay's own MILP
    call, whose time budget is measured inside solve_milp, so the dispatch decisions are unchanged."""
    rows = []
    real = R.solve_milp

    def solve_and_record(inst, budget):
        if len(inst.weights):
            rows.append(analyse_instance(inst, tag, len(rows)))
        return real(inst, budget)

    R.solve_milp = solve_and_record
    try:
        m = run_replay(trace, out, "myopic_mip", cfg, capture=False)
    finally:
        R.solve_milp = real
    return m, rows


def main():
    if not (SRC / "requests.csv").exists():
        sys.exit("fetch the trace first: python scripts/fetch_example_data.py --accept-source-license --prefix 3000 --out data/full_example")
    done_dir = WORK / "done"; done_dir.mkdir(parents=True, exist_ok=True)     # finished runs, so an interrupted job resumes
    defer = set(filter(None, os.environ.get("E3_DEFER", "").split(",")))     # runs to leave for a later invocation
    skip = set(filter(None, os.environ.get("E3_SKIP", "").split(",")))       # runs left out of the final tables (recorded)
    runs, periods = [], []
    for k in DENSITY:
        trace = None
        for f240 in FLEET_PER_240:
            for e in EPOCHS:
                fleet = int(round(f240 * k)); tag = f"k{k}_f{fleet}_e{e}"
                rec = done_dir / f"{tag}.json"
                if rec.exists():
                    r = json.loads(rec.read_text()); runs.append(r["run"]); periods += r["periods"]
                    print("cached", tag, flush=True)
                    continue
                if tag in defer or tag in skip:
                    print("deferred" if tag in defer else "skipped", tag, flush=True)
                    continue
                trace = trace or scaled_trace(k)
                out = WORK / tag
                t0 = time.time()
                m, rows = replay_and_analyse(trace, out, Config(fleet=fleet, epoch=e, seed=11, solver_budget=1.0, cost_weight=.08), tag)
                d = pd.DataFrame(rows)
                run = dict(run=tag, density=k, requests=int(round(240 * k)), fleet=fleet, epoch=e, service_rate=m["service_rate"],
                           periods=len(d), share_solved_by_reduction=float((d.n_after_reduction == 0).mean()),
                           median_candidates=float(d.n_candidates.median()), p95_candidates=float(d.n_candidates.quantile(.95)),
                           median_core=float(d.n_after_reduction.median()), p95_core=float(d.n_after_reduction.quantile(.95)),
                           max_core=int(d.n_after_reduction.max()), max_component=int(d.largest_component.max()),
                           share_fit_device=float(d.fits_device_rho_ge_1.mean()), max_spins=int(d.max_spins.max()),
                           share_lp_integral=float((d.lp_gap_rel.abs() < 1e-7).mean()), max_lp_gap_rel=float(d.lp_gap_rel.max()),
                           median_mip_s=float(d.mip_seconds.median()), max_mip_s=float(d.mip_seconds.max()),
                           wall_s=time.time() - t0)
                rec.write_text(json.dumps({"run": run, "periods": rows}, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
                runs.append(run); periods += rows
                print(run, flush=True)
                shutil.rmtree(out, ignore_errors=True)
    if defer - {r["run"] for r in runs}:
        print("deferred runs remain; final tables not written", flush=True)
        return
    note = OUT / "e3_density_skipped.json"
    if skip:
        note.write_text(json.dumps({"skipped_runs": sorted(skip), "reason": os.environ.get("E3_SKIP_REASON", "")}, indent=1))
    else:
        note.unlink(missing_ok=True)
    pd.DataFrame(runs).to_csv(OUT / "e3_density_runs.csv", index=False)
    pd.DataFrame(periods).to_csv(OUT / "e3_density_periods.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    if not skip:
        shutil.rmtree(WORK, ignore_errors=True)


if __name__ == "__main__":
    main()
