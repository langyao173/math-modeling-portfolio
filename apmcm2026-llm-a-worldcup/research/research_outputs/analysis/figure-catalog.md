# Figure Catalog

## Figure 1: problem1_total_goals_trend.png

- **Purpose:** Show tournament total-goal evolution.
- **Data source:** Supplied, SHA-256-verified World Cup CSV files
- **Observation:** GoalsScored increased by 1.070 per calendar year over the observed editions (Pearson r=0.830); no single breakpoint improved BIC by the pre-specified threshold.
- **Interpretation:** Total goals combine scoring intensity and tournament size.
- **Caveat:** Descriptive historical evidence through 2014; no post-2014 observations used.

## Figure 2: problem1_goals_per_match_trend.png

- **Purpose:** Separate scoring intensity from match count.
- **Data source:** Supplied, SHA-256-verified World Cup CSV files
- **Observation:** goals_per_match decreased by 0.027 per calendar year over the observed editions (Pearson r=-0.784); no single breakpoint improved BIC by the pre-specified threshold.
- **Interpretation:** The rate is the appropriate complement to the aggregate series.
- **Caveat:** Descriptive historical evidence through 2014; no post-2014 observations used.

## Figure 3: problem2_halftime_fulltime_scatter.png

- **Purpose:** Assess half-time information for full-time goals.
- **Data source:** Supplied, SHA-256-verified World Cup CSV files
- **Observation:** Pearson correlation is 0.696.
- **Interpretation:** Half-time score is predictive but does not determine the final total.
- **Caveat:** Descriptive historical evidence through 2014; no post-2014 observations used.
