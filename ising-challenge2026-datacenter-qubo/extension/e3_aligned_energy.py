"""Extension E3 — does an energy that agrees with the objective close the gap to the certified optimum?

The competition report found that, across neighbourhood families, the QUBO energy was almost perfectly
correlated with the action cost and *negatively* correlated with the recourse cost (r = -0.98).  The final
schedule therefore came from recourse-aware polishing outside the QUBO, and E1 showed it to be 1.6 % above
the certified optimum at the same peak.  E3 asks whether the gap is a property of the Ising route or of the
energy function, by giving the same annealer three energies and scoring every decoded schedule with the
exact objective (expected + CVaR95 recourse, the evaluator of the frozen package).

Objective.  For slot loads L (MW) the recourse of scenario s is a sum over slots of
    price_st (L_t - R_st)^+  +  5 (R_st - L_t)^+  +  kappa_t (f_st - (ref_t - L_t)^+)^+,
and with the five scenario probabilities CVaR95 equals the worst scenario cost, so
    F(L) = sum_s p_s C_s(L) + max_s C_s(L).
Constraints of the exact model: one real action per job (no dummy) and the action-cost guard
sum_j (c_j - min c_j) <= 1 USD; optionally the submitted schedule's peak as a cap.

Energies (all quadratic in the action bits, i.e. QUBO-representable, since L_t is linear in them)
  action cost     sum of action costs (the base formulation; its minimiser is the cheapest schedule)
  linear recourse first-order expansion of F around an anchor load profile: slot marginal costs, the
                  form of the submission's 'aligned recourse' family
  quadratic       per slot, a convex quadratic a_t L_t^2 + b_t L_t fitted by least squares to the slot's
  recourse        contribution to F (expected part plus the anchor's worst scenario) on a trust window
                  around the anchor; refitted around each new solution with a halving window (sequential
                  quadratic surrogate, four rounds)
and, as a search-quality ceiling, the same annealer on F itself (not QUBO-representable).

Annealer.  Simulated annealing with one-hot-preserving moves (reassign one job); the guard and the
optional peak cap are enforced by rejecting violating moves, as a constraint-preserving mixer would.
Eight seeds per energy; 4 x 10^6 proposed moves per run.  Energy and F are recorded along the trajectories
to measure how well each energy ranks schedules.
Run:  python extension/e3_aligned_energy.py
"""
from __future__ import annotations
import importlib.util, json, sys, tempfile, time
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "extension"; OUT.mkdir(parents=True, exist_ok=True)
SEEDS = range(8)
ROUNDS = 12                 # trust-region rounds of the sequential surrogates
MOVES_ROUND = 1_000_000     # proposed moves per round (the single-shot energies get ROUNDS x this)
W0 = 0.005                  # initial and maximal trust-region half-width per slot (MW; ~7 typical jobs)
T0_ROUND, T0_EXACT, T1 = 0.02, 0.05, 1e-5


def load_instance():
    spec = importlib.util.spec_from_file_location("ecb", ROOT / "src/07_solve_isling/build_exact_controlled_baseline.py")
    ecb = importlib.util.module_from_spec(spec); sys.modules["ecb"] = ecb; spec.loader.exec_module(ecb)
    with tempfile.TemporaryDirectory() as td:                     # the selection table is only used for neighbourhood
        tmp = Path(td) / "site.parquet"                           # sampling; the packaged validation table is identical
        pd.read_csv(ROOT / "deliverables/final_solution_package/validation_data/site_timeseries_for_validation.csv",
                    encoding="utf-8-sig").to_parquet(tmp)
        ecb.SELECTION_SITE_PATH = tmp
        data = ecb.load_frozen_data()
    return data


# ----------------------------------------------------------------------------- exact objective (numba)
@njit(cache=True)
def scen_costs(L, R, price, target, pen, ref):
    S, T = R.shape
    c = np.zeros(S)
    for s in range(S):
        tot = 0.0
        for t in range(T):
            l = L[t]; a = R[s, t]
            if l > a:
                tot += (l - a) * price[s, t]
            else:
                tot += 5.0 * (a - l)
            pot = ref[t] - l
            if pot < 0.0:
                pot = 0.0
            if target[s, t] > pot:
                tot += (target[s, t] - pot) * pen[t]
        c[s] = tot
    return c


@njit(cache=True)
def objective(L, R, price, target, pen, ref, prob):
    c = scen_costs(L, R, price, target, pen, ref)
    return np.dot(prob, c) + c.max()


# ----------------------------------------------------------------------------- annealer (numba)
@njit(cache=True)
def anneal(asg0, starts, ends, P, dc, slack, lo, hi, mode, qa, qb, R, price, target, pen, ref, prob,
           n_moves, t0, t1, seed, record_every):
    """mode 0: energy sum_t qa_t L_t^2 + qb_t L_t (qa = 0 gives a linear energy); mode 1: the exact objective F;
    mode 2: action cost.  Moves that break the guard or leave the per-slot load box [lo, hi] (peak cap and/or
    trust region) are rejected.  Returns the best assignment by the annealed energy and a trajectory sample."""
    np.random.seed(seed)
    J = starts.shape[0]; T = P.shape[1]
    asg = asg0.copy()
    L = np.zeros(T); extra = 0.0
    for j in range(J):
        L += P[asg[j]]; extra += dc[asg[j]]
    def_energy = 0.0
    if mode == 0:
        for t in range(T):
            def_energy += qa[t] * L[t] * L[t] + qb[t] * L[t]
    elif mode == 1:
        def_energy = objective(L, R, price, target, pen, ref, prob)
    else:
        def_energy = extra
    E = def_energy
    best_E = E; best = asg.copy()
    n_rec = n_moves // record_every
    rec_E = np.zeros(n_rec); rec_F = np.zeros(n_rec)
    multi = np.zeros(J, np.int64); m = 0
    for j in range(J):
        if ends[j] - starts[j] > 1:
            multi[m] = j; m += 1
    ratio = (t1 / t0) ** (1.0 / n_moves)
    temp = t0
    d = np.zeros(T)
    for it in range(n_moves):
        j = multi[np.random.randint(m)]
        a_old = asg[j]
        a_new = starts[j] + np.random.randint(ends[j] - starts[j] - 1)
        if a_new >= a_old:
            a_new += 1
        new_extra = extra + dc[a_new] - dc[a_old]
        ok = new_extra <= slack + 1e-12
        if ok:
            for t in range(T):
                v = L[t] + P[a_new, t] - P[a_old, t]
                if v > hi[t] + 1e-12 or v < lo[t] - 1e-12:
                    ok = False
                    break
        if ok:
            for t in range(T):
                d[t] = P[a_new, t] - P[a_old, t]
            if mode == 0:
                dE = 0.0
                for t in range(T):
                    dE += qa[t] * (2.0 * L[t] * d[t] + d[t] * d[t]) + qb[t] * d[t]
            elif mode == 1:
                dE = objective(L + d, R, price, target, pen, ref, prob) - E
            else:
                dE = dc[a_new] - dc[a_old]
            if dE <= 0.0 or np.random.random() < np.exp(-dE / temp):
                asg[j] = a_new; extra = new_extra; E += dE
                for t in range(T):
                    L[t] += d[t]
                if E < best_E - 1e-15:
                    best_E = E; best[:] = asg
        temp *= ratio
        if (it + 1) % record_every == 0:
            k = (it + 1) // record_every - 1
            rec_E[k] = E; rec_F[k] = objective(L, R, price, target, pen, ref, prob)
    return best, best_E, rec_E, rec_F


# ----------------------------------------------------------------------------- surrogates
def slot_functions(inst, s_star):
    """phi_t(l): slot-t contribution to sum_s p_s C_s + C_{s*} as a function of the slot load l."""
    R, price, target, pen, ref, prob = inst["R"], inst["price"], inst["target"], inst["pen"], inst["ref"], inst["prob"]
    w = prob.copy(); w[s_star] += 1.0

    def phi(t, l):
        l = np.asarray(l, float)
        g = np.zeros_like(l)
        for s in range(R.shape[0]):
            v = np.where(l > R[s, t], (l - R[s, t]) * price[s, t], 5.0 * (R[s, t] - l))
            v = v + np.maximum(target[s, t] - np.maximum(ref[t] - l, 0.0), 0.0) * pen[t]
            g = g + w[s] * v
        return g
    return phi


def fit_surrogate(inst, L_anchor, width, quadratic=True):
    c = scen_costs(L_anchor, inst["R"], inst["price"], inst["target"], inst["pen"], inst["ref"])
    phi = slot_functions(inst, int(np.argmax(c)))
    T = len(L_anchor); qa, qb = np.zeros(T), np.zeros(T)
    for t in range(T):
        grid = np.linspace(max(L_anchor[t] - width[t], 0.0), L_anchor[t] + width[t], 81)
        y = phi(t, grid)
        if quadratic:                                              # curvature of either sign: the flex-penalty term
            A = np.column_stack([grid ** 2, grid, np.ones_like(grid)])   # saturates above the reference load
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            qa[t], qb[t] = coef[0], coef[1]
        else:                                                      # first-order: least-squares slope on the same window
            A = np.column_stack([grid, np.ones_like(grid)])
            qb[t] = np.linalg.lstsq(A, y, rcond=None)[0][0]
    return qa, qb


def main():
    data = load_instance()
    ev = data.evaluator
    real = data.actions.loc[~data.actions.is_dummy].sort_values(["job_id", "action_id"]).reset_index(drop=True)
    jobs, starts = np.unique(real.job_id.to_numpy(), return_index=True)
    ends = np.r_[starts[1:], len(real)]
    P = np.vstack(real._power.to_list()).astype(float)
    cost = real.action_cost.to_numpy(float)
    cmin = np.minimum.reduceat(cost, starts)
    dc = cost - np.repeat(cmin, ends - starts)
    slack = float(data.action_cost_guard_usd - cmin.sum())
    T = P.shape[1]
    inp = ev.recourse_inputs(data.site, data.reference_load_mw); sc = ev.recourse_scenarios(T)
    inst = dict(R=np.array([inp["base_green_mw"] * np.asarray(s["renewable_factor"], float) for s in sc]),
                price=np.array([inp["lmp"] * np.asarray(s["price_factor"], float) for s in sc]),
                target=np.array([inp["flex_target_base"] * float(s["flex_factor"]) for s in sc]),
                pen=np.asarray(inp["penalty_short"], float), ref=np.asarray(data.reference_load_mw, float),
                prob=np.array([s["probability"] for s in sc], float))
    args = (inst["R"], inst["price"], inst["target"], inst["pen"], inst["ref"], inst["prob"])
    aid = {a: i for i, a in enumerate(real.action_id)}

    def assignment(action_ids):
        a = np.sort(np.array([aid[x] for x in action_ids]))           # one action per job, job-major order
        assert len(a) == len(starts) and np.all(np.searchsorted(starts, a, side="right") - 1 == np.arange(len(starts)))
        return a

    def load_of(asg):
        return P[asg].sum(axis=0)

    def score(asg):
        L = load_of(asg); r = ev.evaluate_recourse(L, data.reference_load_mw, data.site, alpha=0.95)
        return dict(F=r["expected_recourse_cost_usd"] + r["cvar_recourse_cost_usd"], expected=r["expected_recourse_cost_usd"],
                    cvar=r["cvar_recourse_cost_usd"], peak=float(L.max()), extra_cost=float(dc[asg].sum()))

    # references: cheapest schedule, submitted schedule, certified optima (E1)
    e1 = json.loads((OUT / "e1_full_exact.json").read_text())
    base = starts + np.array([np.argmin(dc[s:e]) for s, e in zip(starts, ends)])
    refs = {"cheapest schedule": base,
            "submitted": assignment(data.incumbent.action_id.astype(str)),
            "certified optimum, peak capped": assignment(e1["peak_guarded_selected_action_ids"]),
            "certified optimum": assignment(e1["selected_action_ids"])}
    ref_scores = {k: score(v) for k, v in refs.items()}
    # the numba objective must reproduce the package evaluator exactly
    for k, v in refs.items():
        assert abs(objective(load_of(v), *args) - ref_scores[k]["F"]) < 1e-9, k
    cap_sub = ref_scores["submitted"]["peak"]
    print(json.dumps({k: round(v["F"], 4) for k, v in ref_scores.items()}))

    rows, traj = [], []
    zero = np.zeros(T); inf = np.full(T, np.inf)

    def sequential(quadratic, start, cap_hi, seed, energy):
        """Trust-region loop: fit the surrogate on [anchor - w, anchor + w], anneal inside that box, keep the
        result if the exact objective improved (then widen w), otherwise halve w."""
        anchor, f_anchor, w = start.copy(), score(start)["F"], np.full(T, W0)
        for k in range(ROUNDS):
            La = load_of(anchor)
            qa, qb = fit_surrogate(inst, La, w, quadratic=quadratic)
            lo, hi = np.maximum(La - w, 0.0), np.minimum(La + w, cap_hi)
            best, _, rE, rF = anneal(anchor, starts, ends, P, dc, slack, lo, hi, 0, qa, qb, *args,
                                     MOVES_ROUND, T0_ROUND, T1, 1000 + 97 * seed + k, 10_000)
            if k == 0:
                traj.append(pd.DataFrame(dict(cap=cap_name, seed=seed, energy=energy, E=rE, F=rF)))
            f_new = score(best)["F"]
            if f_new < f_anchor - 1e-12:
                anchor, f_anchor, w = best, f_new, np.minimum(2 * w, W0)
            else:
                w = w / 2
            rows.append(dict(cap=cap_name, seed=seed, energy=energy, round=k, **score(anchor)))

    for cap_name, cap in (("no peak cap", np.inf), ("peak capped at submitted", cap_sub)):
        # every run starts from the cheapest schedule unless it violates the cap, then from the submitted one
        start0 = base if load_of(base).max() <= cap + 1e-12 else refs["submitted"]
        hi_cap = np.full(T, cap)
        for seed in SEEDS:
            # (1) action-cost energy (the base formulation)
            best, _, rE, rF = anneal(start0, starts, ends, P, dc, slack, zero, hi_cap, 2, zero, zero, *args,
                                     MOVES_ROUND * ROUNDS, T0_ROUND, T1, 1000 + seed, 10_000 * ROUNDS)
            rows.append(dict(cap=cap_name, seed=seed, energy="action cost", round=ROUNDS - 1, **score(best)))
            traj.append(pd.DataFrame(dict(cap=cap_name, seed=seed, energy="action cost", E=rE, F=rF)))
            # (2) sequential linear and (3) sequential quadratic recourse surrogates
            sequential(False, start0, hi_cap, seed, "linear recourse")
            sequential(True, start0, hi_cap, seed, "quadratic recourse")
            # (4) annealing on the exact objective with the same total number of moves (search-quality ceiling)
            best, _, rE, rF = anneal(start0, starts, ends, P, dc, slack, zero, hi_cap, 1, zero, zero, *args,
                                     MOVES_ROUND * ROUNDS, T0_EXACT, T1, 1000 + seed, 10_000 * ROUNDS)
            rows.append(dict(cap=cap_name, seed=seed, energy="exact objective (not QUBO)", round=ROUNDS - 1, **score(best)))
            print(cap_name, seed, {r["energy"]: round(r["F"], 3) for r in rows if r["seed"] == seed and r["cap"] == cap_name
                                   and r["round"] == ROUNDS - 1}, flush=True)
    # ---------------------------------------------------------------- landscape diagnosis
    # (a) where the optimum puts its load: slots above the reference load ("sacrificial" slots, in which the
    #     flex shortfall is already at its cap, so further load costs only energy)
    flex_ramp_lo = inst["ref"] - inst["target"].max(axis=0)
    struct = {}
    for k in ("submitted", "certified optimum, peak capped", "certified optimum"):
        L = load_of(refs[k])
        struct[k] = dict(slots_above_reference=[int(t) for t in np.nonzero(L > inst["ref"] + 1e-12)[0]],
                         slots_with_any_flex_shortfall=int(np.sum(L > flex_ramp_lo + 1e-12)),
                         load_kw=[round(float(v) * 1e3, 2) for v in L])
    # (b) is the submitted schedule a local optimum of F?  Steepest descent with single-job moves (guard and cap
    #     respected) from the submitted and from the cheapest schedule
    job_of = np.repeat(np.arange(len(starts)), ends - starts)

    def steepest(asg, cap):
        asg = asg.copy(); L = load_of(asg); F = objective(L, *args); extra = float(dc[asg].sum()); moves = 0
        while True:
            best_d, best_a = 0.0, -1
            for a in range(len(P)):
                j = job_of[a]; o = asg[j]
                if a == o or extra + dc[a] - dc[o] > slack + 1e-12:
                    continue
                Ln = L + P[a] - P[o]
                if Ln.max() > cap + 1e-12:
                    continue
                d_ = objective(Ln, *args) - F
                if d_ < best_d - 1e-12:
                    best_d, best_a = d_, a
            if best_a < 0:
                return dict(F=float(F), moves=moves, peak=float(L.max()))
            j = job_of[best_a]; o = asg[j]; L = L + P[best_a] - P[o]; extra += dc[best_a] - dc[o]; asg[j] = best_a; F += best_d; moves += 1
    descent = {f"from {k}, {cn}": steepest(refs[k], c) for k in ("submitted", "cheapest schedule")
               for cn, c in (("no peak cap", np.inf), ("peak capped at submitted", cap_sub))
               if not (k == "cheapest schedule" and c < load_of(base).max())}
    # (c) an upper bound on the barrier between the submitted schedule and the optimum along single-job paths:
    #     best-first switching of the differing jobs; the barrier is the largest climb above the running minimum
    barrier = {}
    for target_name in ("certified optimum, peak capped", "certified optimum"):
        cur = refs["submitted"].copy(); goal = refs[target_name]
        todo = list(np.nonzero(cur != goal)[0]); L = load_of(cur)
        path, extra, peak, loads = [objective(L, *args)], [float(dc[cur].sum())], [float(L.max())], [L.copy()]
        while todo:
            cand = [objective(L - P[cur[j]] + P[goal[j]], *args) for j in todo]
            i = int(np.argmin(cand)); j = todo.pop(i)
            L = L - P[cur[j]] + P[goal[j]]; cur[j] = goal[j]
            path.append(cand[i]); extra.append(float(dc[cur].sum())); peak.append(float(L.max())); loads.append(L.copy())
        path = np.array(path); climb = path - np.minimum.accumulate(path); k_ = int(np.argmax(climb))
        k0 = int(np.flatnonzero(path[:k_ + 1] == np.minimum.accumulate(path)[k_])[0])          # where the climb starts
        loads = 1e3 * np.array(loads)
        slots_kw = {f"slot{t}": {"reference": round(float(1e3 * inst["ref"][t]), 2), "start": round(float(loads[0, t]), 2),
                                 "climb_start": round(float(loads[k0, t]), 2), "barrier_top": round(float(loads[k_, t]), 2),
                                 "end": round(float(loads[-1, t]), 2)} for t in (6, 12)}
        barrier[target_name] = dict(jobs_switched=len(path) - 1, start_F=float(path[0]), end_F=float(path[-1]),
                                    uphill_steps=int(np.sum(np.diff(path) > 1e-9)), downhill_steps=int(np.sum(np.diff(path) < -1e-9)),
                                    barrier_usd=float(climb.max()), barrier_from_F=float(np.minimum.accumulate(path)[k_]),
                                    barrier_at_step=k_, climb_start_step=k0, slots_kw=slots_kw,
                                    slots_above_reference=[int(np.sum(loads[k] > 1e3 * inst["ref"] + 1e-9)) for k in (0, k0, k_, len(loads) - 1)],
                                    max_extra_action_cost=float(max(extra)), max_peak_along_path=float(max(peak)),
                                    path_F=[round(float(v), 4) for v in path])
    df = pd.DataFrame(rows); df.to_csv(OUT / "e3_aligned_energy_runs.csv", index=False)
    tr = pd.concat(traj); tr.to_csv(OUT / "e3_aligned_energy_trajectories.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})

    final = df[df["round"] == ROUNDS - 1]
    opt = {"no peak cap": ref_scores["certified optimum"]["F"], "peak capped at submitted": ref_scores["certified optimum, peak capped"]["F"]}
    summ = {"references": ref_scores, "moves_total_per_run": MOVES_ROUND * ROUNDS, "seeds": len(list(SEEDS)), "rounds": ROUNDS,
            "trust_region_halfwidth_mw": W0,
            "action_cost_slack_usd": slack, "results": {}, "rank_agreement": {}, "structure": struct, "steepest_descent": descent,
            "barrier": barrier}
    for (cap_name, en), g in final.groupby(["cap", "energy"]):
        gap = 100 * (g.F - opt[cap_name]) / opt[cap_name]
        summ["results"][f"{cap_name} | {en}"] = dict(best_F=float(g.F.min()), median_F=float(g.F.median()), best_gap_pct=float(gap.min()),
                                                    median_gap_pct=float(gap.median()), max_extra_cost=float(g.extra_cost.max()),
                                                    max_peak=float(g.peak.max()))
    qr = df[df.energy.isin(["linear recourse", "quadratic recourse"])].groupby(["cap", "energy", "round"]).F.median()
    summ["sequential_by_round_median_F"] = {f"{c} | {e} | round {r}": float(v) for (c, e, r), v in qr.items()}
    for (cap_name, en), g in tr.groupby(["cap", "energy"]):
        summ["rank_agreement"][f"{cap_name} | {en}"] = float(stats.spearmanr(g.E, g.F)[0])
    (OUT / "e3_aligned_energy_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ["results"], indent=1)); print(json.dumps(summ["rank_agreement"], indent=1))


if __name__ == "__main__":
    t = time.time(); main(); print("wall", round(time.time() - t, 1))
