# Model Card

- **Selected outcome system:** advanced_nested_calibrated
- **Training data cutoff:** 2014-07-13
- **Intended use:** reproducible educational/competition analysis of supplied World Cup data.
- **Not intended for:** betting, live 2026 operations, or claims using post-2014 team strength.
- **Information sets:** pre-match outcome model; half-time update goal model.
- **Promotion decision:** promoted_then_one_time_2014_audit_complete

## Frozen official metrics

### problem1_2018

| prediction_year | data_cutoff | qualified_teams | matches_played | attendance_strategy | forecast_attendance | predicted_total_goals | actual_total_goals | signed_error | absolute_error | empirical_loocv_interval_90_lower | empirical_loocv_interval_90_upper | model_hash | model_artifact | audit_timestamp_utc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2018 | 2014 | 32 | 64 | pre_2018_log_linear_forecast | 3.82841e+06 | 152.553 | 169 | -16.4472 | 16.4472 | 116.08 | 189.026 | 658c9d2e1ee080e40a3200aaad8e35f20799f1a6d7f7ab59d4e351b479d291bf | models/problem1_aggregate_ols.json | 2026-07-17T16:00:00+00:00 |

### problem2_2014

| mae | mse | rmse | r2 | n |
| --- | --- | --- | --- | --- |
| 0.974391 | 1.36608 | 1.16879 | 0.554182 | 64 |

### problem3_2014

| accuracy | macro_f1 | log_loss | multiclass_brier | ranked_probability_score | calibration_error | n | recall_home_win | recall_draw | recall_away_win |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.53125 | 0.388718 | 0.99264 | 0.59193 | 0.214902 | 0.0925802 | 64 | 0.75 | 0 | 0.565217 |

### problem3_static_joint_2014_audit

| accuracy | macro_f1 | log_loss | multiclass_brier | ranked_probability_score | calibration_error | n | recall_home_win | recall_draw | recall_away_win |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.53125 | 0.388992 | 0.988921 | 0.58901 | 0.212247 | 0.145092 | 64 | 0.75 | 0 | 0.565217 |

## Known limitations

The source ends in 2014; historical formats, national-team identities, and player identifiers are heterogeneous; one 2014 attendance value is missing; no official 2026 matchup/input table is supplied. Probabilities and intervals are therefore accompanied by cutoff and OOD metadata.