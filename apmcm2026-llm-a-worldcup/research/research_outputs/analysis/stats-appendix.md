# Statistical Appendix

## Protocol

- Expanding-origin outer folds, never random K-fold.
- Paired Wilcoxon signed-rank comparison across tournament folds when n>=3.
- Signed-rank rank-biserial effect size and exact fold win count.
- No p-value is interpreted as evidence from the single 2014 holdout.

## Paired comparisons

| candidate | baseline | metric | clusters | replicates | seed | mean_difference | median_difference | ci_low | ci_high | sampling_unit | paired |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| advanced_nested_calibrated | legacy_mirrored_dixon_coles | log_loss | 7 | 2000 | 17290 | -0.0204227 | -0.0245166 | -0.0365239 | -0.00345106 | tournament_edition | True |
| advanced_nested_calibrated | legacy_mirrored_dixon_coles | multiclass_brier | 7 | 2000 | 17291 | -0.0130222 | -0.0154694 | -0.0237492 | -0.00266794 | tournament_edition | True |
| advanced_nested_calibrated | legacy_mirrored_dixon_coles | scoreline_log_loss | 7 | 2000 | 17292 | -0.00795182 | -0.00763874 | -0.0282272 | 0.010524 | tournament_edition | True |
| advanced_nested_calibrated | legacy_mirrored_dixon_coles | total_goal_crps | 7 | 2000 | 17293 | 0.0115536 | -0.0070141 | -0.00838316 | 0.0350726 | tournament_edition | True |

## Assumptions and limitations

World Cup editions are temporally ordered and heterogeneous. Random-fold estimates are therefore not used. Calibration is fit on pooled development OOF base predictions; its application to frozen 2014 is audit-only and is not used for selection. Development-fit diagnostics are not presented as held-out evidence.