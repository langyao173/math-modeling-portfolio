"""Extension E1/E2 — precision audit of the SPQC-550 compilation over every
frozen dispatch epoch, and a finer precision search.

For each of the 435 frozen one-period packing instances recorded during the
classical replays (8 fleet/period conditions, seed 11), the QPool pipeline is
applied component-wise: exact dominance reduction -> connected components ->
finest passing resolution rho (power-of-two ladder, as in the paper) -> signed
8-bit Ising check.  For every component that fits the hardware model we compute
exactly, by MILP,
  OPT_w           original-utility optimum,
  OPT_q           quantized optimum,
  [U_min, U_max]  original utility over ALL quantized-optimal solutions
                  (the tie range that a hardware solver may return),
  U_polish        QPool's equal-quantized-reward exchange polish started from
                  the worst tie (a stress test of the classical post-processing),
  K/rho           the a-priori loss bound of the paper (K = max cardinality).
E2 replaces the power-of-two ladder by a fine search over rho (step 1/8 between
the passing and failing ladder values) and repeats the exact tie analysis.
Utility differences convert to planned vehicle time via dT = dU * T0 / lambda
= dU * 125 vehicle-minutes (report Eq. 12).
"""
from __future__ import annotations
import glob, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import milp, Bounds, LinearConstraint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qpool.qubo import PackingInstance, connected_components, induced            # noqa: E402
from qpool.competition import reduce_packing, compile_integer_packing, polish_equal_quantized_reward  # noqa: E402
from qpool.solvers import packing_constraints                                      # noqa: E402

OUT = ROOT / "results" / "extension"; OUT.mkdir(parents=True, exist_ok=True)
LADDER = (32, 16, 8, 4, 2, 1, .5, .25)
VEH_MIN_PER_UTILITY = 600 / 0.08 / 60   # 125


def fits(inst: PackingInstance, rho: float):
    q = PackingInstance(np.rint(inst.weights * rho), inst.edges, inst.labels)
    if q.weights.max(initial=0) <= 0:
        return None
    try:
        _, I, meta = compile_integer_packing(q)
    except ValueError:
        return None
    return (q, I) if meta["preflight"]["valid"] else None


def ladder_rho(inst):
    prev_fail = None
    for r in LADDER:
        f = fits(inst, r)
        if f is not None:
            return r, prev_fail, f
        prev_fail = r
    return None, prev_fail, None


def fine_rho(inst, r_pass, r_fail, step=0.125):
    """Finest passing rho strictly between the ladder's passing and failing values.
    When the top of the ladder (32) already passes, the ladder choice is kept."""
    if r_fail is None:
        return r_pass, fits(inst, r_pass)
    hi = r_fail
    best = (r_pass, fits(inst, r_pass))
    for r in np.arange(r_pass + step, hi - 1e-9, step):
        f = fits(inst, float(r))
        if f is not None:
            best = (float(r), f)
    return best


def _milp(c, A, n, extra=None):
    cons = [LinearConstraint(A, -np.inf, 1)]
    if extra is not None:
        cons.append(extra)
    res = milp(c, integrality=np.ones(n), bounds=Bounds(0, 1), constraints=cons,
               options={"time_limit": 30, "mip_rel_gap": 0.0})
    assert res.status == 0, res.message
    return np.rint(res.x).astype(int)


def tie_analysis(orig: PackingInstance, quant: PackingInstance):
    n = len(orig.weights); A = packing_constraints(orig)
    w, q = orig.weights, quant.weights
    x_w = _milp(-w, A, n); opt_w = float(w @ x_w)
    x_q = _milp(-q, A, n); opt_q = float(q @ x_q)
    tie = LinearConstraint(q.reshape(1, -1), opt_q - 1e-6, np.inf)
    x_hi = _milp(-w, A, n, tie); x_lo = _milp(w, A, n, tie)
    x_pol, _ = polish_equal_quantized_reward(orig, quant, x_lo, time_limit=0.5)
    K = int(_milp(-np.ones(n), A, n).sum())
    return dict(opt_w=opt_w, u_max=float(w @ x_hi), u_min=float(w @ x_lo), u_polish=float(w @ x_pol), K=K)


def main():
    rows = []
    files = sorted(glob.glob(str(ROOT / "results/stage2/contest_replay/*_myopic_mip/snapshots.json")))
    t0 = time.time()
    for f in files:
        cond = Path(f).parent.name.replace("_myopic_mip", "")
        for k, snap in enumerate(json.load(open(f))):
            inst = PackingInstance(np.array(snap["weights"]), [tuple(e) for e in snap["edges"]], snap.get("labels"))
            red = reduce_packing(inst)
            comps = connected_components(red.instance)
            if not comps:
                rows.append(dict(condition=cond, epoch_index=k, n_candidates=len(inst.weights),
                                 n_after_reduction=len(red.kept), component=-1, n_vars=0))
            for c, ids in enumerate(connected_components(red.instance)):
                sub = induced(red.instance, ids)
                row = dict(condition=cond, epoch_index=k, n_candidates=len(inst.weights),
                           n_after_reduction=len(red.kept), component=c, n_vars=len(ids),
                           n_edges=len(sub.edges), max_degree=max((len(a) for a in sub.adjacency()), default=0))
                r_pass, r_fail, f_pass = ladder_rho(sub)
                row["rho_ladder"] = r_pass
                if r_pass is None or len(ids) < 2:
                    rows.append(row); continue
                q, I = f_pass
                row["spins"] = len(I)
                ta = tie_analysis(sub, q)
                row.update({f"{k2}_ladder": v for k2, v in ta.items()})
                r_fine, (qf, If) = fine_rho(sub, r_pass, r_fail)
                row["rho_fine"] = r_fine
                tf = tie_analysis(sub, qf)
                row.update({f"{k2}_fine": v for k2, v in tf.items() if k2 != "opt_w"})
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "e1_precision_audit_components.csv", index=False)

    comp = df[df.n_vars >= 2]
    ok = comp[comp.rho_ladder.notna()].copy()
    for tag in ["ladder", "fine"]:
        rho = ok["rho_ladder"] if tag == "ladder" else ok["rho_fine"]
        ok[f"tie_range_{tag}"] = ok[f"u_max_{tag}"] - ok[f"u_min_{tag}"]
        ok[f"worst_loss_{tag}"] = ok["opt_w_ladder"] - ok[f"u_min_{tag}"]
        ok[f"bound_{tag}"] = ok[f"K_{tag}"] / rho
        ok[f"polish_gap_{tag}"] = ok["opt_w_ladder"] - ok[f"u_polish_{tag}"]
    ok["precision_class"] = np.where(ok.rho_ladder < 1, "degenerate (rho<1)", np.where(ok.rho_ladder >= 32, "ladder top (rho=32)", "intermediate (1<=rho<32)"))
    def block(d):
        return {
            "components": int(len(d)),
            "median_vars": float(d.n_vars.median()), "max_vars": int(d.n_vars.max()),
            "share_with_ties_ladder": float((d.tie_range_ladder > 1e-9).mean()),
            "share_with_ties_fine": float((d.tie_range_fine > 1e-9).mean()),
            "share_upper_tie_is_original_opt_ladder": float((d.opt_w_ladder - d.u_max_ladder < 1e-9).mean()),
            "share_upper_tie_is_original_opt_fine": float((d.opt_w_ladder - d.u_max_fine < 1e-9).mean()),
            "worst_tie_loss_vehmin_ladder": float(d.worst_loss_ladder.sum() * VEH_MIN_PER_UTILITY),
            "worst_tie_loss_vehmin_fine": float(d.worst_loss_fine.sum() * VEH_MIN_PER_UTILITY),
            "best_tie_loss_vehmin_ladder": float((d.opt_w_ladder - d.u_max_ladder).sum() * VEH_MIN_PER_UTILITY),
            "best_tie_loss_vehmin_fine": float((d.opt_w_ladder - d.u_max_fine).sum() * VEH_MIN_PER_UTILITY),
            "after_polish_loss_vehmin_ladder": float(d.polish_gap_ladder.sum() * VEH_MIN_PER_UTILITY),
            "after_polish_loss_vehmin_fine": float(d.polish_gap_fine.sum() * VEH_MIN_PER_UTILITY),
            "median_rho_gain_fine_over_ladder": float((d.rho_fine / d.rho_ladder).median()),
            "median_bound_over_actual_worst_loss": float((d.bound_ladder / d.worst_loss_ladder.where(d.worst_loss_ladder > 1e-9)).median()),
            "polish_reaches_tie_max_share_ladder": float(((d.u_max_ladder - d.u_polish_ladder) < 1e-9).mean()),
        }
    inst = df.groupby(["condition", "epoch_index"]).agg(n=("n_candidates", "first"), k=("n_after_reduction", "first"))
    summary = {
        "frozen_instances": int(len(inst)),
        "instances_fully_solved_by_reduction": int((inst.k == 0).sum()),
        "median_candidates": float(inst.n.median()), "median_vars_after_reduction": float(inst.k.median()),
        "components_with_>=2_vars": int(len(comp)),
        "components_fitting_550_spins_and_int8_with_rho>=1": int((ok.rho_ladder >= 1).sum()),
        "largest_component_vars": int(comp.n_vars.max()),
        "rho_ladder_distribution": {str(k): int(v) for k, v in ok.rho_ladder.value_counts().sort_index().items()},
        "by_precision_class": {c: block(d) for c, d in ok.groupby("precision_class")},
        "all_nondegenerate": block(ok[ok.rho_ladder >= 1]),
        "runtime_s": time.time() - t0,
    }
    nd = ok[ok.rho_ladder >= 1]
    paired = {}
    for name, a, b in [("after_polish", "u_polish_fine", "u_polish_ladder"), ("best_tie", "u_max_fine", "u_max_ladder"),
                       ("worst_tie", "u_min_fine", "u_min_ladder")]:
        d = nd[b] - nd[a]          # >0: fine search loses more utility than the ladder
        paired[name] = {"fine_better": int((d < -1e-9).sum()), "equal": int((d.abs() <= 1e-9).sum()), "fine_worse": int((d > 1e-9).sum())}
    summary["paired_fine_vs_ladder_components"] = paired
    df["status"] = np.where(df.n_vars < 2, "trivial", np.where(df.rho_ladder >= 1, "fits", "needs_conditioning"))
    dep = (df.groupby(["condition", "epoch_index"]).status
           .apply(lambda s: "needs_conditioning" if (s == "needs_conditioning").any() else "fits_whole_period")
           .reset_index().groupby("condition").status.value_counts().unstack(fill_value=0))
    dep["share_whole_period_fits"] = dep["fits_whole_period"] / dep.sum(axis=1)
    dep.to_csv(OUT / "e1_deployability_by_condition.csv")
    summary["deployability_by_condition"] = dep["share_whole_period_fits"].round(4).to_dict()
    (OUT / "e1_precision_audit_summary.json").write_text(json.dumps(summary, indent=2))
    ok.to_csv(OUT / "e1_precision_audit_fitting.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
