"""Extension E1 — exact minimum-change capacity protection.

The submitted guard (code/capacity_guard.py::protect_membership) keeps the base
model's top-k membership by clipping the stability candidate z at a FIXED
threshold, the midpoint of the k-th and (k+1)-th base scores.  Any threshold
that separates the protected set P from the rest U is admissible, so the
minimum-change problem
        min_f ||f - z||_2^2   s.t.  min_{i in P} f_i > max_{j in U} f_j
is solved by f_i = max(z_i, tau+) (i in P), f_j = min(z_j, tau) (j in U) with the
tau that minimises the convex piecewise-quadratic cost
        C(tau) = sum_P (tau - z_i)_+^2 + sum_U (z_j - tau)_+^2 .
C'(tau) is monotone, so tau* is found exactly by bisection on C'.  Because the
fixed threshold is one admissible tau, the optimal guard never changes the
scores more (in L2) than the submitted one; it is applied in exactly the same
sequential replay (phases, folds, gates, radius 0.01, capacity 10 %).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(sys.argv[1]).resolve()            # path to the 043 package
OUT = Path(sys.argv[2]).resolve(); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "code"))
import capacity_guard as cg                    # noqa: E402

_fixed = cg.protect_membership


def optimal_protect(base, target, row_ids, quotas):
    p = np.asarray(base, float); z = np.asarray(target, float); ids = np.asarray(row_ids, dtype=str)
    n = len(p); cuts = sorted(set(int(k) for k in quotas if 0 < k < n))
    if len(cuts) != 1:
        return _fixed(base, target, row_ids, quotas)     # nested multi-capacity: keep submitted rule
    k = cuts[0]; o = cg.order(p, ids); P = np.zeros(n, bool); P[o[:k]] = True
    zp, zu = z[P], z[~P]
    if zp.min() > zu.max():
        q = z.copy()
    else:
        def dC(t): return np.clip(t - zp, 0, None).sum() - np.clip(zu - t, 0, None).sum()
        lo, hi = zu.min() if False else min(zp.min(), zu.min()), max(zp.max(), zu.max())
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if dC(mid) > 0: hi = mid
            else: lo = mid
        tau = 0.5 * (lo + hi); up = np.nextafter(tau, 1.0)
        q = z.copy(); q[P] = np.maximum(z[P], up); q[~P] = np.minimum(z[~P], tau)
    oq = cg.order(q, ids)
    assert set(o[:k]) == set(oq[:k]), "membership protection failed"
    return q, dict(lower=None, upper=None, thresholds=[], boundary_tie=False, guard_changed=np.abs(q - z) > 1e-12)


_fixed_score = cg.score_batch


def promise_score_batch(base, previous, gate, row_ids, *, quota, radius=.01):
    """Choose the separating threshold that minimises, lexicographically,
    (number of gated customers pushed outside the stability radius,
     total excess over the radius, total L1 change).  Single capacity only."""
    cuts = [quota] if isinstance(quota, (int, np.integer)) else list(quota)
    if len(cuts) != 1:
        return _fixed_score(base, previous, gate, row_ids, quota=quota, radius=radius)
    p = np.asarray(base, float); old = np.asarray(previous, float); g = np.asarray(gate, bool)
    ids = np.asarray(row_ids, dtype=str); n = len(p); k = int(cuts[0])
    z = np.where(g, np.clip(p, old - radius, old + radius), p)
    o = cg.order(p, ids); P = np.zeros(n, bool); P[o[:k]] = True
    if z[P].min() > z[~P].max():
        q = z.copy()
    else:
        lo_b, hi_b = z[~P].max(), z[P].min()          # violation band: hi_b < lo_b
        cand = np.unique(np.r_[z[(z >= hi_b) & (z <= lo_b)], hi_b, lo_b])
        cand = np.unique(np.r_[cand, 0.5 * (cand[1:] + cand[:-1])])
        best = None
        for t in cand:
            up = np.nextafter(t, 1.0)
            f = z.copy(); f[P] = np.maximum(z[P], up); f[~P] = np.minimum(z[~P], t)
            exc = np.where(g, np.maximum(0, np.abs(f - old) - radius), 0)
            key = (int((exc > 1e-12).sum()), float(exc.sum()), float(np.abs(f - z).sum()))
            if best is None or key < best[0]: best = (key, f)
        q = best[1]
    oq = cg.order(q, ids); assert set(o[:k]) == set(oq[:k]), "membership protection failed"
    rec = dict(temporal_candidate=z, final_probability=q, gate=g, guard_changed=np.abs(q - z) > 1e-12,
               unavoidable_excess=np.where(g, np.maximum(0, np.abs(q - old) - radius), 0), boundary_tie=False,
               base_order=o, final_order=oq)
    return q, rec


def run(protect, scorer=None):
    cg.protect_membership = protect
    if scorer is not None: cg.score_batch = scorer
    meta = pd.read_parquet(ROOT / "evidence/current/joint/predictions.parquet")
    edges = pd.read_parquet(ROOT / "evidence/current/joint/edges.parquet")
    q, rec = cg.apply_sequence(meta.base_reference.to_numpy(), meta, edges, capacities=(.1,), radius=.01)
    cg.protect_membership = _fixed; cg.score_batch = _fixed_score
    return meta, q, pd.DataFrame(rec)


def summarise(meta, q, rec, label):
    r = rec.copy(); r["dev"] = r.final - r.temporal_candidate
    ev = meta.phase.ge(0) & meta.y.notna()
    s = {"rule": label,
         "rows_scored": int(len(r)),
         "rows_changed_by_guard": int(r.guard_changed.sum()),
         "gated_rows_changed_by_guard": int((r.guard_changed & r.gate).sum()),
         "l2_change_from_stability_candidate": float(np.sqrt((r.dev ** 2).sum())),
         "l1_change_pp": float(100 * r.dev.abs().sum()),
         "gated_rows_with_excess_over_radius": int((r.excess_pp > 1e-9).sum()),
         "total_excess_over_radius_pp": float(r.excess_pp.sum()),
         "max_excess_over_radius_pp": float(r.excess_pp.max()),
         "auc_phase0_1": float(roc_auc_score(meta.y[ev], q[ev]))}
    for ph in [0, 1]:
        m = meta.phase.eq(ph) & meta.y.notna()
        s[f"auc_phase{ph}"] = float(roc_auc_score(meta.y[m], q[m]))
    return s


def main():
    m1, q1, r1 = run(_fixed); m2, q2, r2 = run(optimal_protect); m3, q3, r3 = run(_fixed, promise_score_batch)
    ref = pd.read_parquet(ROOT / "evidence/current/joint/predictions.parquet").guard10.to_numpy()
    out = {"submitted_guard_reproduces_published_guard10_max_abs_diff": float(np.abs(q1 - ref).max()),
           "fixed_threshold (submitted)": summarise(m1, q1, r1, "fixed midpoint threshold"),
           "L2_optimal_threshold": summarise(m2, q2, r2, "exact L2 minimum-change threshold"),
           "promise_optimal_threshold (extension)": summarise(m3, q3, r3, "threshold minimising broken stability promises")}
    for ph in [0, 1]:   # top-10 % membership is identical by construction; report overlap check
        m = m1.phase.eq(ph).to_numpy()
        out[f"phase{ph}_same_final_order_share_L2_vs_submitted"] = float(np.mean(np.argsort(-q1[m]) == np.argsort(-q2[m])))
        out[f"phase{ph}_same_final_order_share_promise_vs_submitted"] = float(np.mean(np.argsort(-q1[m]) == np.argsort(-q3[m])))
    (OUT / "e1_optimal_guard_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
