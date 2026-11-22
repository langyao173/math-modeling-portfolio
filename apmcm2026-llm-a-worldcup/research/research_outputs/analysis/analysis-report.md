# Strict Analysis Report

## Analysis question

Do leakage-safe probabilistic enhancements improve proper scoring rules across multiple expanding-origin World Cup folds without degrading mandatory baseline behavior?

## Comparison unit and evidence

- Unit of repeated comparison: held-out tournament edition (n=7).
- Compared model families: 5.
- The 2014 edition is excluded from candidate promotion and reported only as the frozen official test.
- The evaluated candidates are deterministic; research mode records an explicit repeat-fit probability check.

## Mean outer-fold metrics

| model | runtime_seconds | accuracy | macro_f1 | log_loss | multiclass_brier | ranked_probability_score | calibration_error | n | recall_home_win | recall_draw | recall_away_win | scoreline_log_loss | total_goal_crps | total_goal_mae | total_goal_rmse | pit_mean | pit_variance | pit_ks_statistic | pit_ks_p_value_diagnostic | support_overflow_rows | n_joint | interval_50_coverage | interval_50_mean_width | interval_80_coverage | interval_80_mean_width | interval_90_coverage | interval_90_mean_width | seed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| advanced_nested_calibrated | nan | 0.533826 | 0.396597 | 0.954824 | 0.564611 | 0.192091 | 0.111697 | 58.8571 | 0.784849 | 0.009524 | 0.57681 | 2.80443 | 0.888829 | 1.31744 | 1.60992 | 0.473019 | 0.081864 | 0.1272 | 0.383408 | 0 | nan | 0.653159 | 2.19797 | 0.893201 | 3.94145 | 0.968922 | 5.36693 | 31415 |
| advanced_nested_raw | nan | 0.531593 | 0.389735 | 0.954562 | 0.564833 | 0.191804 | 0.091923 | 58.8571 | 0.784849 | 0 | 0.57681 | 2.80417 | 0.889083 | 1.31627 | 1.61014 | 0.474216 | 0.081817 | 0.117569 | 0.479889 | 0 | nan | 0.650927 | 2.18578 | 0.893201 | 3.94145 | 0.968922 | 5.38152 | 31415 |
| independent_mirrored_poisson | 0.03 | 0.526442 | 0.388315 | 0.984454 | 0.58393 | 0.20022 | 0.112466 | 58.8571 | 0.76403 | 0 | 0.601101 | 2.82478 | 0.875567 | 1.3044 | 1.58597 | 0.452922 | 0.08207 | 0.138042 | 0.295829 | 0 | 58.8571 | 0.689045 | 2.4272 | 0.882555 | 4.0297 | 0.97785 | 5.62483 | nan |
| legacy_mirrored_dixon_coles | 0.03 | 0.529361 | 0.389992 | 0.975247 | 0.577634 | 0.19736 | 0.09414 | 58.8571 | 0.763393 | 0 | 0.601405 | 2.81238 | 0.877275 | 1.31027 | 1.58739 | 0.453708 | 0.082352 | 0.13166 | 0.342582 | 0 | 58.8571 | 0.685096 | 2.40144 | 0.885302 | 4.0364 | 0.980598 | 5.59203 | nan |
| official_outcome_logit | 0.01 | 0.531422 | 0.392232 | 0.991014 | 0.588748 | 0.200921 | 0.113007 | 58.8571 | 0.770006 | 0 | 0.606815 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |

## Frozen official test metrics

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

## Interpretation boundary

Observed fold differences support model-selection decisions only. They do not establish a causal mechanism, and seven or fewer historical tournament folds provide limited inferential power. The supplied data stop in 2014, so 2026 outputs are explicitly stale-cutoff scenarios.