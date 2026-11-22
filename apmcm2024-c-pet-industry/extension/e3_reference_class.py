"""Extension E3 — how much can five annual observations tell us?  A reference-class benchmark.

Problem 1 forecast 2024-2026 from five annual values (2019-2023) with trend extrapolation, a grey model,
an ARIMA and an inverse-MSE combination of the three.  E1 showed the forecasts drifting away from what
was later observed.  One Chinese series per species cannot say whether that was bad luck or a property of
the method, so E3 evaluates the same method family on a reference class of real annual series and uses
the class to calibrate prediction intervals.

Reference class.  The 1 088 annual *Demographic* series of the M4 forecasting competition (Makridakis,
Spiliotis & Assimakopoulos 2020, IJF 36:54-74; train and test files of the official repository).  For
each series one window is taken at its end: the first L values are the training data (L = 5 as in the
paper, L = 7 as for a forecaster standing in 2025), the next three are the targets (h = 1, 2, 3).  One window
per series keeps the evaluation units independent.  All 23 000 annual M4 series are a robustness class.

Methods (fitted to the L training values only)
  naive        last value
  drift        random walk with drift
  linear       OLS linear trend in time (the paper's linear regression)
  CAGR         compound growth from the first to the last value (the paper's average growth rate)
  GM(1,1)      grey model, least-squares background-value form
  ARIMA        lowest AIC among ARIMA(0,1,0), (0,1,0)+drift, (1,1,0), (0,1,1)
  Theta        Assimakopoulos & Nikolopoulos (2000), statsmodels ThetaModel, no seasonality
  inv-MSE combination   w_i proportional to 1 / in-sample MSE over {linear, GM(1,1), ARIMA} (paper, Sec. 4.1.4)
  equal combination     simple mean of the same three

Accuracy: median absolute percentage error per horizon, the median absolute error scaled by the in-window naive
MAE, the share of windows in which a method beats the naive forecast, and the share in which the sign of the
two-year change is predicted correctly.  Intervals: the empirical quantiles of
log(actual / forecast) per method and horizon, estimated on a random half of the series and tested on the other
half (split-conformal across series; the assumption is exchangeability of windows within the class).  OLS
prediction intervals of the linear trend (t with L - 2 df) are the model-based comparator.

Finally the class-calibrated intervals are applied to China's urban cat and dog series (White Paper, 10^4
animals): from origin 2023 (what the paper knew) and from origin 2025.
Run:  python extension/e3_reference_class.py
"""
from __future__ import annotations
import json, urllib.request, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data" / "external", ROOT / "results" / "extension"
DATA.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
M4 = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/"
FILES = {"Yearly-train.csv": M4 + "Train/Yearly-train.csv", "Yearly-test.csv": M4 + "Test/Yearly-test.csv", "M4-info.csv": M4 + "M4-info.csv"}
H = 3
SEED = 2024
# China, urban pet cats and dogs, 10^4 animals: 2019-2023 Annex 1 of the problem (= White Paper);
# 2024 and 2025 from the 2025 and 2026 White Paper editions (see e1_forecast_check.py)
CHINA = {"cat": {2019: 4412, 2020: 4862, 2021: 5806, 2022: 6536, 2023: 6980, 2024: 7153, 2025: 7289},
         "dog": {2019: 5503, 2020: 5222, 2021: 5429, 2022: 5119, 2023: 5175, 2024: 5258, 2025: 5343}}
PAPER = {"cat": {2024: 7297, 2026: 8386}, "dog": {2024: 5110, 2026: 5061}}


def fetch(name):
    p = DATA / name
    if not p.exists():
        with urllib.request.urlopen(FILES[name], timeout=600) as r:
            p.write_bytes(r.read())
    return p


def load_m4():
    info = pd.read_csv(fetch("M4-info.csv")).set_index("M4id")
    tr = pd.read_csv(fetch("Yearly-train.csv")).set_index("V1")
    te = pd.read_csv(fetch("Yearly-test.csv")).set_index("V1")
    series = {}
    for sid in tr.index:
        a = tr.loc[sid].dropna().to_numpy(float); b = te.loc[sid].dropna().to_numpy(float)
        series[sid] = np.concatenate([a, b])
    return series, info.category


# ----------------------------------------------------------------------------- methods
def f_naive(y, h):
    return np.repeat(y[-1], h), np.zeros(len(y) - 1)


def f_drift(y, h):
    d = np.diff(y).mean()
    return y[-1] + d * np.arange(1, h + 1), np.diff(y) - d


def f_linear(y, h):
    t = np.arange(len(y)); b, a = np.polyfit(t, y, 1)
    fit = a + b * t
    return a + b * (len(y) - 1 + np.arange(1, h + 1)), (y - fit)[1:]


def f_cagr(y, h):
    if y[0] <= 0 or y[-1] <= 0:
        return f_drift(y, h)
    r = (y[-1] / y[0]) ** (1 / (len(y) - 1))
    fit = y[0] * r ** np.arange(len(y))
    return y[-1] * r ** np.arange(1, h + 1), (y - fit)[1:]


def f_gm11(y, h):
    x1 = np.cumsum(y); z = 0.5 * (x1[1:] + x1[:-1])
    B = np.column_stack([-z, np.ones_like(z)])
    (a, b), *_ = np.linalg.lstsq(B, y[1:], rcond=None)
    if abs(a) < 1e-9 or not np.isfinite(a):
        return f_linear(y, h)
    k = np.arange(len(y) + h)
    x1h = (y[0] - b / a) * np.exp(-a * k) + b / a
    x0h = np.concatenate([[y[0]], np.diff(x1h)])
    out = x0h[len(y):]
    if not np.all(np.isfinite(out)):
        return f_linear(y, h)
    return out, (y - x0h[:len(y)])[1:]


def f_arima(y, h):
    from statsmodels.tsa.arima.model import ARIMA
    best = None
    for order, trend in (((0, 1, 0), "n"), ((0, 1, 0), "t"), ((1, 1, 0), "n"), ((0, 1, 1), "n")):
        try:
            r = ARIMA(y, order=order, trend=trend).fit()
        except Exception:
            continue
        if np.isfinite(r.aic) and (best is None or r.aic < best.aic):
            best = r
    if best is None:
        return f_drift(y, h)
    fc = np.asarray(best.forecast(h), float)
    if not np.all(np.isfinite(fc)):
        return f_drift(y, h)
    return fc, np.asarray(best.resid, float)[1:]                 # first residual is the diffuse level


def f_theta(y, h):
    from statsmodels.tsa.forecasting.theta import ThetaModel
    try:
        fc = np.asarray(ThetaModel(y, period=1, deseasonalize=False).fit().forecast(h), float)
        if np.all(np.isfinite(fc)):
            return fc, None
    except Exception:
        pass
    return f_drift(y, h)[0], None


METHODS = {"naive": f_naive, "drift": f_drift, "linear": f_linear, "CAGR": f_cagr, "GM(1,1)": f_gm11, "ARIMA": f_arima, "Theta": f_theta}
COMBO = ("linear", "GM(1,1)", "ARIMA")


def forecast_all(y, h=H):
    fc, res = {}, {}
    for k, f in METHODS.items():
        fc[k], res[k] = f(y, h)
    mse = np.array([np.mean(res[k] ** 2) for k in COMBO])
    w = 1 / np.maximum(mse, 1e-12); w /= w.sum()
    fc["inv-MSE combination"] = sum(wi * fc[k] for wi, k in zip(w, COMBO))
    fc["equal combination"] = np.mean([fc[k] for k in COMBO], axis=0)
    return fc, dict(zip(COMBO, w))


def ols_interval(y, h, level):
    """Classical OLS prediction interval of the linear trend (t distribution, L - 2 df)."""
    n = len(y); t = np.arange(n); b, a = np.polyfit(t, y, 1)
    s = np.sqrt(np.sum((y - a - b * t) ** 2) / (n - 2)); tq = stats.t.ppf(0.5 + level / 2, n - 2)
    t0 = n - 1 + np.arange(1, h + 1)
    se = s * np.sqrt(1 + 1 / n + (t0 - t.mean()) ** 2 / np.sum((t - t.mean()) ** 2))
    m = a + b * t0
    return m - tq * se, m + tq * se


# ----------------------------------------------------------------------------- evaluation
def evaluate(series, cats, L, keep):
    rows = []
    for sid, z in series.items():
        if cats[sid] not in keep or len(z) < L + H:
            continue
        w = z[-(L + H):]; y, a = w[:L], w[L:]
        if np.any(w <= 0):
            continue
        scale = np.mean(np.abs(np.diff(y)))
        fc, wts = forecast_all(y)
        r80 = ols_interval(y, H, 0.80); r95 = ols_interval(y, H, 0.95)
        for k, f in fc.items():
            for j in range(H):
                rows.append(dict(sid=sid, category=cats[sid], L=L, method=k, h=j + 1, actual=a[j], forecast=f[j],
                                 ape=abs(f[j] / a[j] - 1), ase=abs(f[j] - a[j]) / scale if scale > 0 else np.nan,
                                 logratio=np.log(a[j] / f[j]) if f[j] > 0 else np.nan,
                                 sign_ok=float(np.sign(f[1] - y[-1]) == np.sign(a[1] - y[-1])) if j == 1 else np.nan,
                                 ols80=float(r80[0][j] <= a[j] <= r80[1][j]) if k == "linear" else np.nan,
                                 ols95=float(r95[0][j] <= a[j] <= r95[1][j]) if k == "linear" else np.nan,
                                 r2_linear=float(1 - np.sum((y - np.polyval(np.polyfit(np.arange(L), y, 1), np.arange(L))) ** 2)
                                                 / np.sum((y - y.mean()) ** 2)) if np.var(y) > 0 else np.nan,
                                 w_linear=wts["linear"], w_gm=wts["GM(1,1)"], w_arima=wts["ARIMA"]))
    return pd.DataFrame(rows)


def accuracy_table(df):
    """Median APE, median scaled error (the mean MASE is dominated by near-constant windows with a tiny
    naive scale) and the share of windows in which the method beats the naive forecast."""
    naive = df[df.method == "naive"].set_index(["sid", "h"]).ape
    d = df.join(naive.rename("ape_naive"), on=["sid", "h"])
    d["beats_naive"] = (d.ape < d.ape_naive - 1e-12).astype(float)
    return (d.groupby(["method", "h"]).agg(median_ape=("ape", "median"), mean_ape=("ape", "mean"), median_ase=("ase", "median"),
                                           share_beats_naive=("beats_naive", "mean")).reset_index())


def conformal(df, rng, levels=(0.80, 0.95)):
    """Quantiles of log(actual/forecast) on a calibration half of the series; coverage on the other half."""
    sids = df.sid.unique(); cal = set(rng.choice(sids, len(sids) // 2, replace=False))
    out, quant = [], {}
    for (m, h), g in df.dropna(subset=["logratio"]).groupby(["method", "h"]):
        gc, gt = g[g.sid.isin(cal)], g[~g.sid.isin(cal)]
        for lv in levels:
            lo, hi = np.quantile(gc.logratio, [(1 - lv) / 2, (1 + lv) / 2])
            cov = float(((gt.logratio >= lo) & (gt.logratio <= hi)).mean())
            out.append(dict(method=m, h=h, level=lv, q_lo=lo, q_hi=hi, test_coverage=cov,
                            median_width_pct=float(100 * (np.exp(hi) - np.exp(lo)))))
        full = g.logratio.to_numpy()
        quant[(m, h)] = {lv: np.quantile(full, [(1 - lv) / 2, (1 + lv) / 2]) for lv in levels}
    return pd.DataFrame(out), quant


def china(quant, method_list, L):
    rows = []
    for sp, ser in CHINA.items():
        yrs = sorted(ser)
        for origin in (2023, 2025):
            tr = [y for y in yrs if y <= origin][-L:]
            if len(tr) < L:
                continue
            y = np.array([ser[t] for t in tr], float)
            fc, wts = forecast_all(y)
            for m in method_list:
                for j in range(H):
                    yr = origin + j + 1; f = fc[m][j]
                    r = dict(species=sp, origin=origin, L=L, method=m, year=yr, forecast=round(float(f), 1),
                             observed=ser.get(yr, np.nan))
                    for lv in (0.80, 0.95):
                        lo, hi = quant[(m, j + 1)][lv]
                        r[f"lo{int(lv * 100)}"] = round(float(f * np.exp(lo)), 1); r[f"hi{int(lv * 100)}"] = round(float(f * np.exp(hi)), 1)
                    rows.append(r)
    return pd.DataFrame(rows)


def main():
    series, cats = load_m4()
    rng = np.random.default_rng(SEED)
    summary, all_tabs = {}, []
    for L in (5, 7):
        demo = evaluate(series, cats, L, {"Demographic"})
        demo.to_csv(OUT / f"e3_demographic_L{L}_windows.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        tab = accuracy_table(demo)
        tab["L"] = L; tab["class"] = "Demographic"; all_tabs.append(tab)
        cov, quant = conformal(demo, rng)
        cov["L"] = L; cov.to_csv(OUT / f"e3_interval_coverage_L{L}.csv", index=False)
        lin = demo[(demo.method == "linear")]
        summary[f"L{L}"] = dict(
            windows=int(demo.sid.nunique()),
            ols_linear_coverage={f"h{h}": {"80": float(g.ols80.mean()), "95": float(g.ols95.mean())} for h, g in lin.groupby("h")},
            sign_of_two_year_change_correct={m: float(g.sign_ok.mean()) for m, g in demo[demo.h == 2].groupby("method")},
            spearman_r2_vs_ape_h3=float(stats.spearmanr(lin[lin.h == 3].r2_linear, lin[lin.h == 3].ape, nan_policy="omit")[0]),
            mean_weights_inv_mse={"linear": float(lin.w_linear.mean()), "GM(1,1)": float(lin.w_gm.mean()), "ARIMA": float(lin.w_arima.mean())})
        # in-sample fit as a guide to out-of-sample error: does a high R^2 of the linear trend predict accuracy?
        b = lin[lin.h == 3].dropna(subset=["r2_linear"]).copy()
        b["r2_bin"] = pd.cut(b.r2_linear, [-0.01, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0])
        summary[f"L{L}"]["median_ape_h3_by_linear_r2"] = {str(k): float(v) for k, v in b.groupby("r2_bin", observed=True).ape.median().items()}
        ch = china(quant, list(demo.method.unique()), L); ch.to_csv(OUT / f"e3_china_L{L}.csv", index=False)
    # robustness class: all annual M4 series (sampled for the slower ARIMA/Theta fits)
    sample_ids = set(rng.choice(list(series), 4000, replace=False))
    allc = evaluate({k: v for k, v in series.items() if k in sample_ids}, cats, 5, set(cats.unique()))
    tab = accuracy_table(allc)
    tab["L"] = 5; tab["class"] = "all annual (4 000 sampled)"; all_tabs.append(tab)
    pd.concat(all_tabs).round(4).to_csv(OUT / "e3_accuracy.csv", index=False)
    summary["robustness_all_annual_L5_windows"] = int(allc.sid.nunique())
    (OUT / "e3_summary.json").write_text(json.dumps(summary, indent=1))
    print(pd.concat(all_tabs).pivot_table(index=["class", "L", "method"], columns="h", values=["median_ape", "median_ase"]).round(3).to_string())
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
