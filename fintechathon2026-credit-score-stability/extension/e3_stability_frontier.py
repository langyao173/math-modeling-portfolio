"""Extension E3 — is the stability gate necessary, and is +-1 pp the right radius?

The final score keeps a customer's score within a radius (1 pp) of the previous score only when the
customer's account flows pass a stability gate (`seven_05`), and then restores the base model's top-k
review list exactly (capacity guard).  Two design choices are therefore open: the gate, and the radius.
The generic alternatives from the prediction-churn literature apply the smoothing to everyone: clipping
every update to the radius, or exponentially smoothing every score towards the previous one
(as in anchor / distillation approaches to churn reduction, e.g. Fard et al. 2016; Jiang et al. 2022).

E3 replays all variants through the identical sequential procedure of the submission (phases 0 and 1,
customer-out-of-fold batches, 10 % capacity with the same per-fold quotas, capacity guard), so every variant
reviews exactly the same customers as its base model; only stability and discrimination move.

  base model      reference-scale model (`base_reference`, the submission's base) or the strong risk
                  ensemble the paper compares against (`strong_S`)
  stabiliser      none | gated clipping, radius r (the submission: r = 0.01) | clipping of every update,
                  radius r | exponential smoothing of every update, z = b q_prev + (1 - b) p
Metrics: AUC on the 5 467 labelled states of phases 0 and 1; tail stability as in the paper, the mean of the
largest 5 % absolute score changes (CVaR95, pp) in the three "unchanged-risk" cohorts (gate-stable,
similar repayment state, similar payment structure) and over all transitions; the share of gated updates
outside +-1 pp; distinct early-delinquency customers covered by the two 10 % review rounds; and responsiveness,
the mean same-month score change on updates with a new payment shortfall and on updates followed by an early
delinquency (what a stabiliser must not suppress).

Run:  python extension/e3_stability_frontier.py <path-to-043-package> results/extension
(the package's row-level inputs are not redistributed; outputs are aggregate tables only)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve(); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "code"))
import capacity_guard as cg                    # noqa: E402

COHORTS = ["stable_05", "repayment_state", "flow_structure"]


def cvar95(x):
    """Mean of the largest 5 % (fractional boundary), as in the submission's tail metric."""
    a = np.sort(np.asarray(x, float))[::-1]; m = .05 * len(a); k = int(m)
    return float((a[:k].sum() + (m - k) * a[k]) / m) if len(a) else np.nan


def replay(p, meta, edges, *, mode="clip", gate_rule="seven_05", radius=.01, beta=.5, capacity=.1):
    """The submission's apply_sequence with the stabiliser and the gate as parameters."""
    p = np.asarray(p, float); q = p.copy()
    folds = sorted(meta.loc[meta.fold.ge(0), "fold"].unique())
    for phase in (0, 1):
        m = meta.phase.eq(phase) & meta.fold.ge(0)
        sizes = np.array([(m & meta.fold.eq(f)).sum() for f in folds])
        ideal = capacity * sizes; v = np.floor(ideal).astype(int); n = int(np.ceil(capacity * sizes.sum())) - v.sum()
        v[np.lexsort((np.arange(len(folds)), -(ideal - v)))[:n]] += 1
        quota = dict(zip(folds, v))
        for fold in folds:
            ix = np.flatnonzero(m & meta.fold.eq(fold)); position = {r: i for i, r in enumerate(ix)}
            e = edges[edges.phase_after.eq(phase) & edges.fold.eq(fold)]
            prev = p[ix].copy(); has_prev = np.zeros(len(ix), bool); gate = np.zeros(len(ix), bool)
            for a, b, g in zip(e.row_before, e.row_after, e.seven_05):
                z = position[int(b)]; prev[z] = q[int(a)]; has_prev[z] = True; gate[z] = bool(g)
            if gate_rule == "all":
                gate = has_prev.copy()
            base = p[ix]
            if mode == "none":
                z = base.copy()
            elif mode == "clip":
                z = np.where(gate, np.clip(base, prev - radius, prev + radius), base)
            elif mode == "ewma":
                z = np.where(gate, beta * prev + (1 - beta) * base, base)
            else:
                raise ValueError(mode)
            qq, _ = cg.protect_membership(base, np.clip(z, 0, 1), meta.iloc[ix].row_id.to_numpy(), [int(quota[fold])])
            q[ix] = qq
    return q


def evaluate(q, meta, edges, capacity=.1):
    ev = meta.phase.isin([0, 1]) & meta.y.notna()
    out = {"auc_phase0_1": float(roc_auc_score(meta.y[ev], q[ev]))}
    for ph in (0, 1):
        mm = meta.phase.eq(ph) & meta.y.notna()
        out[f"auc_phase{ph}"] = float(roc_auc_score(meta.y[mm], q[mm]))
    worst = 0.0
    for ph in (0, 1):
        for group in COHORTS + ["all_transitions"]:
            e = edges[edges.phase_after.eq(ph) & edges.fold.ge(0) & edges[group]]
            d = 100 * np.abs(q[e.row_after.to_numpy(int)] - q[e.row_before.to_numpy(int)])
            out[f"cvar95_pp_phase{ph}_{group}"] = cvar95(d)
            if group in COHORTS:
                worst = max(worst, out[f"cvar95_pp_phase{ph}_{group}"])
        g = edges[edges.phase_after.eq(ph) & edges.fold.ge(0) & edges.seven_05]
        d = 100 * np.abs(q[g.row_after.to_numpy(int)] - q[g.row_before.to_numpy(int)])
        out[f"gated_share_outside_1pp_phase{ph}"] = float(np.mean(d > 1 + 1e-9))
    out["worst_cohort_cvar95_pp"] = worst
    # responsiveness: how much of a genuine deterioration reaches the score in the same month
    e = edges[edges.phase_after.isin([0, 1]) & edges.fold.ge(0)]
    i, j = e.row_before.to_numpy(int), e.row_after.to_numpy(int)
    dq = 100 * (q[j] - q[i])
    sf = e.new_payment_shortfall.to_numpy(bool)
    ev_after = meta.y.to_numpy()[j] == 1
    out["shortfall_updates"] = int(sf.sum()); out["shortfall_mean_change_pp"] = float(dq[sf].mean())
    out["event_updates"] = int(ev_after.sum()); out["event_mean_change_pp"] = float(dq[ev_after].mean())
    out["event_share_score_up"] = float(np.mean(dq[ev_after] > 0))
    # two review rounds at 10 %: distinct customers with an early delinquency who are on a list
    hit = set()
    folds = sorted(meta.loc[meta.fold.ge(0), "fold"].unique())
    for ph in (0, 1):
        m = meta.phase.eq(ph) & meta.fold.ge(0)
        sizes = np.array([(m & meta.fold.eq(f)).sum() for f in folds])
        ideal = capacity * sizes; v = np.floor(ideal).astype(int); n = int(np.ceil(capacity * sizes.sum())) - v.sum()
        v[np.lexsort((np.arange(len(folds)), -(ideal - v)))[:n]] += 1
        for f, k in zip(folds, v):
            ix = np.flatnonzero(m & meta.fold.eq(f))
            sel = ix[cg.order(q[ix], meta.iloc[ix].row_id.to_numpy())[:k]]
            hit |= set(meta.iloc[sel].loc[meta.iloc[sel].y.eq(1), "customer_id"].astype(int))
    out["distinct_event_customers_covered"] = len(hit)
    return out


def wcvar95(x, w):
    o = np.argsort(-x); x, w = x[o], w[o]; m = .05 * w.sum(); c = np.cumsum(w)
    k = np.searchsorted(c, m)                                  # first index where the cumulative weight reaches m
    return float((np.sum(x[:k] * w[:k]) + (m - (c[k - 1] if k else 0.0)) * x[k]) / m)


def paired_bootstrap(qa, qb, meta, edges, reps=1000, seed=2026):
    """Customer-level paired bootstrap of (AUC_b - AUC_a, worst-cohort CVaR95_b - CVaR95_a)."""
    rng = np.random.default_rng(seed)
    cust = meta.customer_id.to_numpy(); uc, inv = np.unique(cust, return_inverse=True)
    ev = (meta.phase.isin([0, 1]) & meta.y.notna()).to_numpy(); y = meta.y.to_numpy()
    groups = [(ph, g) for ph in (0, 1) for g in COHORTS]
    eix = {k: edges[edges.phase_after.eq(k[0]) & edges.fold.ge(0) & edges[k[1]]] for k in groups}
    ecust = {k: np.searchsorted(uc, e.customer_id.to_numpy()) for k, e in eix.items()}
    dq = {lab: {k: 100 * np.abs(q[e.row_after.to_numpy(int)] - q[e.row_before.to_numpy(int)]) for k, e in eix.items()} for lab, q in (("a", qa), ("b", qb))}
    out = []
    for _ in range(reps):
        wc = rng.multinomial(len(uc), np.full(len(uc), 1 / len(uc))).astype(float)
        w = wc[inv]
        auc = {lab: roc_auc_score(y[ev], q[ev], sample_weight=w[ev]) for lab, q in (("a", qa), ("b", qb))}
        worst = {lab: max(wcvar95(dq[lab][k], wc[ecust[k]]) for k in groups if wc[ecust[k]].sum() > 0) for lab in ("a", "b")}
        out.append((auc["b"] - auc["a"], worst["b"] - worst["a"]))
    out = np.array(out)
    return {"auc_diff_ci95": np.percentile(out[:, 0], [2.5, 97.5]).tolist(), "tail_diff_ci95": np.percentile(out[:, 1], [2.5, 97.5]).tolist()}


def main():
    meta = pd.read_parquet(ROOT / "evidence/current/joint/predictions.parquet")
    edges = pd.read_parquet(ROOT / "evidence/current/joint/edges.parquet")
    # the parameterised replay must reproduce the published final score exactly
    q_sub = replay(meta.base_reference.to_numpy(), meta, edges)
    repro = float(np.abs(q_sub - meta.guard10.to_numpy()).max())
    assert repro == 0.0, repro
    variants = [("none", {}, "none")]
    variants += [(f"gated clip r={r:g}", dict(mode="clip", gate_rule="seven_05", radius=r), "gated clip") for r in (.0025, .005, .01, .02, .04)]
    variants += [(f"clip all r={r:g}", dict(mode="clip", gate_rule="all", radius=r), "clip all") for r in (.0025, .005, .01, .02, .04)]
    variants += [(f"smooth all b={b:g}", dict(mode="ewma", gate_rule="all", beta=b), "smooth all") for b in (.25, .5, .75, .9)]
    variants += [(f"smooth gated b={b:g}", dict(mode="ewma", gate_rule="seven_05", beta=b), "smooth gated") for b in (.5, .9)]
    rows, scores = [], {}
    for base_name in ("base_reference", "strong_S"):
        p = meta[base_name].to_numpy()
        for name, kw, family in variants:
            q = p.copy() if family == "none" else replay(p, meta, edges, **kw)
            scores[(base_name, name)] = q
            rows.append({"base_model": base_name, "variant": name, "family": family, **kw, **evaluate(q, meta, edges)})
            print(base_name, name, {k: round(v, 4) for k, v in rows[-1].items() if k in ("auc_phase0_1", "worst_cohort_cvar95_pp", "distinct_event_customers_covered")}, flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT / "e3_stability_frontier.csv", index=False)
    sub = df[(df.base_model == "base_reference") & (df.variant == "gated clip r=0.01")].iloc[0]
    summ = {"replay_reproduces_published_final_score_max_abs_diff": repro,
            "submitted": {k: float(sub[k]) for k in ("auc_phase0_1", "worst_cohort_cvar95_pp", "gated_share_outside_1pp_phase0",
                                                       "gated_share_outside_1pp_phase1", "distinct_event_customers_covered")}}
    # dominance: does any variant match the submitted tail stability with higher AUC, or vice versa?
    ref = df[df.base_model == "base_reference"]
    better = ref[(ref.auc_phase0_1 >= sub.auc_phase0_1 - 1e-12) & (ref.worst_cohort_cvar95_pp <= sub.worst_cohort_cvar95_pp + 1e-12) &
                 ((ref.auc_phase0_1 > sub.auc_phase0_1 + 1e-12) | (ref.worst_cohort_cvar95_pp < sub.worst_cohort_cvar95_pp - 1e-12))]
    summ["variants_dominating_submitted"] = better.variant.tolist()
    summ["paired_bootstrap_vs_submitted"] = {}
    for key in (("base_reference", "clip all r=0.01"), ("base_reference", "clip all r=0.02"), ("base_reference", "smooth gated b=0.9"),
                ("strong_S", "none"), ("strong_S", "clip all r=0.02")):
        summ["paired_bootstrap_vs_submitted"][" | ".join(key)] = paired_bootstrap(scores[("base_reference", "gated clip r=0.01")], scores[key], meta, edges)
        print(key, summ["paired_bootstrap_vs_submitted"][" | ".join(key)], flush=True)
    summ["frontier"] = df[["base_model", "variant", "auc_phase0_1", "worst_cohort_cvar95_pp", "distinct_event_customers_covered"]].round(5).to_dict(orient="records")
    (OUT / "e3_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k != "frontier"}, indent=1))


if __name__ == "__main__":
    main()
