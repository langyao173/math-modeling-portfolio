"""Extension E1 — check the paper's Problem-1 forecasts against figures published later.

Forecasts (paper, Sec. 5, combined model, urban China, unit 10^4 animals): cats 7 297 (2024) -> 8 386 (2026),
CAGR ~7.3 %; dogs 5 110 (2024) -> 5 061 (2026).  2025 values are interpolated geometrically.
Observed: China Pet Industry White Paper (Petdata/派读) 2025 edition — 2024: cats 7 153, dogs 5 258;
2026 edition — 2025: cats 7 289, dogs 5 343 (urban, 10^4 animals).
"""
from pathlib import Path
import pandas as pd
OUT = Path(__file__).resolve().parents[1] / "results" / "extension"; OUT.mkdir(parents=True, exist_ok=True)
f = {"cat": {2024: 7297, 2026: 8386}, "dog": {2024: 5110, 2026: 5061}}
for k in f: f[k][2025] = (f[k][2024] * f[k][2026]) ** 0.5
obs = {"cat": {2023: 6980, 2024: 7153, 2025: 7289}, "dog": {2023: 5175, 2024: 5258, 2025: 5343}}
rows = [{"species": k, "year": y, "forecast_1e4": round(f[k][y], 1), "observed_1e4": obs[k][y],
         "error_pct": round(100 * (f[k][y] / obs[k][y] - 1), 2),
         "forecast_growth_pct": round(100 * (f[k][y] / (f[k][y - 1] if y - 1 in f[k] else obs[k][y - 1]) - 1), 2),
         "observed_growth_pct": round(100 * (obs[k][y] / obs[k][y - 1] - 1), 2)} for k in f for y in (2024, 2025)]
t = pd.DataFrame(rows); t.to_csv(OUT / "e1_forecast_check.csv", index=False); print(t.to_string(index=False))
