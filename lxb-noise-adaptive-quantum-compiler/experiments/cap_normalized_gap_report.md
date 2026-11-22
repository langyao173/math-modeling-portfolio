# Cap-Normalized Gap Report

This ranks samples by `(legal_cap - current) / legal_cap`, weighted by their final-score sample weight.

- Current Final: `0.827636718750`
- Legal same-width Final ceiling: `0.915557547263`
- Sum weighted normalized cap gap: `0.100108732956`
- Sum raw weighted gap to legal cap: `0.087920828513`

## Ranking

| Rank | Sample | Track | Current | Legal cap | Efficiency | Normalized gap | Weighted cap gap | Raw weighted cap gap | CX | Gate-noise gap | Readout-remap lift |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `track_b_001` | track_b | 0.666992 | 0.858252 | 0.777152 | 0.222848 | 0.027856 | 0.023908 | 21 | 0.154541 | 0.048613 |
| 2 | `track_b_002` | track_b | 0.613281 | 0.762046 | 0.804782 | 0.195218 | 0.024402 | 0.018596 | 19 | 0.229248 | 0.013796 |
| 3 | `track_a_007` | track_a | 0.804688 | 0.951883 | 0.845363 | 0.154637 | 0.014497 | 0.013800 | 14 | 0.074707 | 0.010524 |
| 4 | `track_a_008` | track_a | 0.804688 | 0.942365 | 0.853902 | 0.146098 | 0.013697 | 0.012907 | 14 | 0.071289 | 0.014184 |
| 5 | `track_a_002` | track_a | 0.871094 | 0.951883 | 0.915126 | 0.084874 | 0.007957 | 0.007574 | 8 | 0.064941 | 0.010524 |
| 6 | `track_a_004` | track_a | 0.914062 | 0.951883 | 0.960267 | 0.039733 | 0.003725 | 0.003546 | 5 | 0.056152 | 0.000961 |
| 7 | `track_a_005` | track_a | 0.925781 | 0.951883 | 0.972578 | 0.027422 | 0.002571 | 0.002447 | 5 | 0.041504 | 0.004899 |
| 8 | `track_a_006` | track_a | 0.929688 | 0.951883 | 0.976682 | 0.023318 | 0.002186 | 0.002081 | 3 | 0.047852 | 0.000961 |
| 9 | `track_a_003` | track_a | 0.929688 | 0.951883 | 0.976682 | 0.023318 | 0.002186 | 0.002081 | 3 | 0.047363 | 0.000961 |
| 10 | `track_a_001` | track_a | 0.941406 | 0.951883 | 0.988993 | 0.011007 | 0.001032 | 0.000982 | 2 | 0.039062 | 0.000961 |

## Interpretation

- Normalized ranking still prioritizes B002/B001, but now as cap-efficiency deficits rather than raw readout-width headroom.
- A007/A008 remain the top Track A semantic targets after cap normalization.
- Best-readout remap lift is modest for A007/A008 and B002, so large jumps require gate/noise implementation changes, not only final measurement relocation.
- B001 has the largest readout remap lift among legal candidates, but its same-width cap is still only 0.858252.
- Top normalized residuals: track_b_001(0.027856), track_b_002(0.024402), track_a_007(0.014497), track_a_008(0.013697), track_a_002(0.007957)
