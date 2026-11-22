# Private Emulator Paired Delta Report

Promotion rule:
- `{'invalid_count': 0, 'p95_compile_lt': 12.0, 'paired_mean_delta_gte': 0.003, 'negative_delta_fraction_lte': 0.2, 'worst_seed_delta_gte': -0.004, 'near_miss_false_accept': 0}`

## Final Choice Scoreboard

| Variant | Weighted delta | Negative frac | Worst seed | Invalid | Near-miss | p95 compile | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| v18_measurement_tail | 0.000114 | 0.000 | 0.000000 | 0 | 0 | 11.271 | fail |

## Suite Details

- v18_measurement_tail / public_replay: mean_delta=0.000000, std=0.000000, worst_seed=0.000000, negatives=0/50, invalid=0, p95=11.271s
- v18_measurement_tail / tracka_structural: mean_delta=0.000195, std=0.000862, worst_seed=0.000000, negatives=0/40, invalid=0, p95=2.997s
