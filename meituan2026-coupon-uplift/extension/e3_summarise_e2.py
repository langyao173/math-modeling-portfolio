"""Summary of E2 (e2_uplift_benchmark.py): learner means and paired comparisons per campaign.

For every campaign and learner: mean Qini coefficient, uplift at 30 %, share of the total increment captured
by the top 30 %, and the CATE calibration error. Paired differences between learners use the same held-out
halves; their 95 % intervals use the corrected resampled t-test of Nadeau & Bengio (2003), which inflates the
split-to-split variance by (1/J + n_test/n_train) because repeated random splits share training data.

Run:  python extension/e3_summarise_e2.py   (after e2_uplift_benchmark.py)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "extension"
CAMPAIGNS = ("hillstrom", "x5", "lenta")
LEARNERS = ("response", "S-learner", "T-learner", "transformed outcome", "X-learner", "DR-learner",
            "causal forest", "random")
UPLIFT = ("S-learner", "T-learner", "transformed outcome", "X-learner", "DR-learner", "causal forest")


def corrected_ci(d: np.ndarray, test_over_train: float = 1.0) -> tuple[float, float, float]:
    """Mean and Nadeau-Bengio corrected 95 % interval of paired split-level differences."""
    d = np.asarray(d, float); j = len(d); m = float(d.mean())
    se = float(np.sqrt((1 / j + test_over_train) * d.var(ddof=1)))
    h = float(stats.t.ppf(0.975, j - 1) * se)
    return m, m - h, m + h


def summarise(name: str) -> dict | None:
    f = RES / f"e2_{name}_scores.csv"
    if not f.exists():
        return None
    s = pd.read_csv(f)
    info = json.loads((RES / f"e2_{name}_info.json").read_text())
    cal = pd.read_csv(RES / f"e2_{name}_calibration.csv")
    wide = {m: s.pivot(index="rep", columns="learner", values=m) for m in ("qini", "uplift_at_30", "share_top30")}
    out = {"info": info, "splits": int(s.rep.nunique()),
           "mean": {m: wide[m][list(LEARNERS)].mean().round(5).to_dict() for m in wide},
           "calibration_error": cal.groupby("learner").calibration_error.mean().round(5).to_dict()}
    q = wide["qini"]
    best_uplift = max(UPLIFT, key=lambda c: q[c].mean())
    out["best_uplift_learner"] = best_uplift
    pairs = {}
    for a, b in [(best_uplift, "response"), ("T-learner", "response"), (best_uplift, "T-learner"),
                 ("X-learner", "response"), ("DR-learner", "response"), ("causal forest", "response")]:
        key = f"{a} minus {b}"
        if key in pairs:
            continue
        d = (q[a] - q[b]).to_numpy()
        m, lo, hi = corrected_ci(d)
        pairs[key] = {"qini_mean": round(m, 5), "ci95": [round(lo, 5), round(hi, 5)], "wins": int((d > 0).sum()),
                      "splits": len(d)}
    out["paired_qini"] = pairs
    c = cal.pivot(index="rep", columns="learner", values="calibration_error")
    d = (c["T-learner + isotonic (proposal)"] - c["T-learner"]).to_numpy()
    m, lo, hi = corrected_ci(d)
    out["isotonic_minus_raw_calibration"] = {"mean": round(m, 5), "ci95": [round(lo, 5), round(hi, 5)],
                                             "improves_in": int((d < 0).sum()), "splits": len(d)}
    out["isotonic_ranking_identical_to_T"] = bool(np.allclose(
        s[s.learner == "T-learner"].qini.to_numpy(), s[s.learner == "T-learner + isotonic (proposal)"].qini.to_numpy()))
    return out


def main() -> None:
    summary = {name: r for name in CAMPAIGNS if (r := summarise(name)) is not None}
    (RES / "e2_summary.json").write_text(json.dumps(summary, indent=1))
    for name, r in summary.items():
        print(name, r["info"]["n"], "best uplift:", r["best_uplift_learner"])
        print("  qini:", {k: round(v, 4) for k, v in r["mean"]["qini"].items()})
        for k, v in r["paired_qini"].items():
            print("  ", k, v)
        print("  calibration:", r["calibration_error"], r["isotonic_minus_raw_calibration"])


if __name__ == "__main__":
    main()
