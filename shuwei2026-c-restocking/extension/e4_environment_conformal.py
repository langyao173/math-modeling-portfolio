"""Extension E4 — does the calibrated 90 % prediction ball of Problem 2 hold in a new environment?

Problem 2 calibrates the radius multiplier of the prediction ball by split-conformal quantiles of the normalised
error z = |error| / sd on the validation windows, pooling all windows of the four validation environments, and
reports 0.90 coverage on the test windows pooled over the four test environments.  The marginal guarantee treats
windows as exchangeable; windows of one environment share an operator, a layout and a task mix, so the
operationally relevant question is coverage *in a new environment*.  The submission's own table
(results/p2_by_env.csv) already shows environment-level coverage between 0.79 and 0.98.

E4 uses the eight environments that played no part in fitting the state model (four validation + four test)
and holds out one environment at a time (leave-one-environment-out, LOEO).  The radius multiplier is calibrated
on the other seven by three rules, per horizon, at nominal 90 %:
  windows pooled     the submission's rule: quantile over all windows (large environments dominate)
  environments pooled  equal weight for every environment: quantile of the average of the seven per-environment
                     empirical CDFs (the CDF-pooling method for two-layer hierarchical data of Dunn, Wasserman &
                     Ramdas 2022, which targets coverage for a new group)
  worst environment  the largest of the seven per-environment 90 % quantiles (guards every calibration
                     environment; the conservative end)
A fourth rule keeps the windows-pooled calibration as the starting point and adapts it on line inside the new
environment (adaptive conformal inference, Gibbs & Candes 2021), using only outcomes that are already known when
the next onset arrives; the step size gamma is not tuned (0.005, 0.01, 0.02 are all reported).  When the adapted
level exceeds 1 (which would mean an infinite radius), the multiplier is clipped to the largest calibration score.
Reported per held-out environment: coverage of the error at all horizons up to 3 s (pooled), at 0.5 s and at 3 s,
and the mean radius (mm).  The light +-0.1 s smoothing of the submission is applied to the static rules.

Requires interim/frames.npz (step s02_build_dataset of the pipeline).  About 2 minutes.
Run:  python extension/e4_environment_conformal.py
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from features import Data                                                    # noqa: E402
from statemodel import assign_phases, load_frozen_state, run_list, DT       # noqa: E402
import p2a_masking as p2a                                                    # noqa: E402

OUT = ROOT / "results" / "extension"; OUT.mkdir(parents=True, exist_ok=True)
P = 0.90
H_REPORT = {"0.5 s": 14, "3 s": 89}
GAMMAS = (0.005, 0.01, 0.02)


def smooth(q):
    return np.array([np.median(q[max(0, h - 3):h + 4]) for h in range(len(q))])


def pooled_cdf_quantile(groups, p):
    """Smallest t with mean_k F_k(t) >= p, F_k the empirical CDF of group k (equal weight per group)."""
    allv = np.unique(np.concatenate(groups))
    F = np.mean([np.searchsorted(np.sort(g), allv, side="right") / len(g) for g in groups], axis=0)
    return float(allv[np.searchsorted(F, p - 1e-12)])


def multipliers(Z, env, cal_envs):
    """Per-horizon multipliers of the three rules from the calibration environments."""
    H = Z.shape[1]
    q_win, q_env, q_max = np.zeros(H), np.zeros(H), np.zeros(H)
    for h in range(H):
        col = Z[:, h]
        groups = [col[(env == e) & ~np.isnan(col)] for e in cal_envs]
        groups = [g for g in groups if len(g)]
        pooled = np.concatenate(groups)
        n = len(pooled)
        q_win[h] = np.quantile(pooled, min(1.0, np.ceil((n + 1) * P) / n), method="higher")   # split-conformal order statistic
        q_env[h] = pooled_cdf_quantile(groups, P)
        q_max[h] = max(np.quantile(g, P) for g in groups)
    return {"windows pooled": smooth(q_win), "environments pooled": smooth(q_env), "worst environment": smooth(q_max)}


def aci_coverage(Z_cal, z_new, gamma, alpha=1 - P, step_frames=15):
    """Adaptive conformal inference (Gibbs & Candes 2021) inside the held-out environment: per horizon the
    miscoverage level is updated, alpha_t+1 = alpha_t + gamma (alpha - miss_t), from the realised outcome of earlier
    onsets.  The outcome at horizon h becomes known ceil((h + 1) / 15) onsets later (onsets are 15 frames apart),
    so updates are applied with that delay.  Returns the coverage indicators and multipliers (onsets x horizons)."""
    n_on, H = z_new.shape
    cover = np.full((n_on, H), np.nan); mult = np.full((n_on, H), np.nan)
    for h in range(H):
        cal = np.sort(Z_cal[:, h][~np.isnan(Z_cal[:, h])]); n = len(cal)
        a = alpha; delay = int(np.ceil((h + 1) / step_frames)); pending = []
        for i in range(n_on):
            while pending and pending[0][0] <= i:
                a += gamma * (alpha - pending.pop(0)[1])
            level = 1 - a                      # a level above 1 would mean an infinite radius; it is clipped to the
            q = cal[-1] if level >= 1 else (0.0 if level <= 0 else cal[min(n - 1, int(np.ceil(n * level)) - 1)])   # largest calibration score
            if np.isnan(z_new[i, h]):
                continue
            c = float(z_new[i, h] <= q); cover[i, h] = c; mult[i, h] = q
            pending.append((i + delay, 1.0 - c))
    return cover, mult


def main():
    t0 = time.time()
    D = Data(); fz, hmm, par = load_frozen_state(); runs = run_list(D); phase, _ = assign_phases(D, runs, hmm)
    main_c = fz["candidates"][fz["model"]]
    R = {s: p2a.collect(D, runs[s], phase, hmm, main_c) for s in ("validation", "test")}
    err = np.vstack([R[s]["err"] for s in R]); sd = np.vstack([R[s]["sd"] for s in R])
    env = np.concatenate([D.env_of_rid[R[s]["meta"][:, 0]] for s in R])
    split = np.concatenate([np.full(len(R[s]["err"]), s) for s in R])
    rec = np.concatenate([R[s]["meta"][:, 0] for s in R]); hand = np.concatenate([R[s]["meta"][:, 1] for s in R])
    frame = np.concatenate([R[s]["meta"][:, 2] for s in R])                       # meta = (record id, hand, frame)
    Z = err / sd
    envs = sorted(np.unique(env))
    print("onsets", len(Z), "environments", len(envs), "(%.0fs)" % (time.time() - t0), flush=True)
    # the submission's frozen rule (validation windows pooled) reproduced on the same onsets
    frozen = np.array(json.loads((ROOT / "results" / "frozen_p2_calibration.json").read_text())["multiplier"]["0.9"])
    rows = []
    for e in envs:
        cal = [c for c in envs if c != e]
        mult = multipliers(Z, env, cal)
        if split[env == e][0] == "test":
            mult["submitted (validation windows)"] = frozen
        m = env == e
        # adaptive calibration inside the held-out environment, starting from the windows-pooled calibration
        idx = np.flatnonzero(m)
        order = idx[np.lexsort((hand[idx], frame[idx], rec[idx]))]        # chronological within records
        calm = np.isin(env, cal)
        for gamma in GAMMAS:
            cov_a, mul_a = aci_coverage(Z[calm], Z[order], gamma)
            ok = ~np.isnan(cov_a); rad = mul_a * sd[order]
            capped = mul_a >= np.array([np.nanmax(Z[calm][:, h]) for h in range(Z.shape[1])])[None, :] - 1e-12
            r = dict(environment_id=e, split=split[m][0], rule=f"windows pooled + online adaptation (gamma={gamma:g})", onsets=int(m.sum()),
                     coverage_all_horizons=float(np.nanmean(cov_a)), mean_radius_mm=float(1e3 * np.nanmean(rad)),
                     share_at_largest_calibration_score=float(np.mean(capped[ok])))
            for lab, h in H_REPORT.items():
                r[f"coverage_{lab}"] = float(np.nanmean(cov_a[:, h])); r[f"radius_mm_{lab}"] = float(1e3 * np.nanmean(rad[:, h]))
            rows.append(r)
        for rule, q in mult.items():
            rad = q[None, :] * sd[m]; ok = ~np.isnan(err[m])
            cover = err[m] <= rad
            r = dict(environment_id=e, split=split[m][0], rule=rule, onsets=int(m.sum()),
                     coverage_all_horizons=float(cover[ok].mean()), mean_radius_mm=float(1e3 * rad[ok].mean()))
            for lab, h in H_REPORT.items():
                okh = ok[:, h]
                r[f"coverage_{lab}"] = float(cover[okh, h].mean()); r[f"radius_mm_{lab}"] = float(1e3 * rad[okh, h].mean())
            rows.append(r)
    df = pd.DataFrame(rows); df.to_csv(OUT / "e4_environment_conformal.csv", index=False)
    summ = {"nominal": P, "environments": envs, "onsets": int(len(Z)), "by_rule": {}}
    for rule, g in df[df.rule != "submitted (validation windows)"].groupby("rule"):
        summ["by_rule"][rule] = {"mean_coverage": float(g.coverage_all_horizons.mean()), "min_coverage": float(g.coverage_all_horizons.min()),
                                 "envs_below_0.85": int((g.coverage_all_horizons < 0.85).sum()),
                                 "min_coverage_3s": float(g["coverage_3 s"].min()), "mean_radius_mm": float(g.mean_radius_mm.mean()),
                                 "mean_radius_mm_3s": float(g["radius_mm_3 s"].mean())}
    sub = df[df.rule == "submitted (validation windows)"]
    summ["submitted_on_test_envs"] = sub[["environment_id", "coverage_all_horizons", "coverage_3 s", "mean_radius_mm"]].round(4).to_dict(orient="records")
    (OUT / "e4_environment_conformal_summary.json").write_text(json.dumps(summ, indent=1))
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False)); print(json.dumps(summ["by_rule"], indent=1))


if __name__ == "__main__":
    main()
