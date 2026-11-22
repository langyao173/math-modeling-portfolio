# Exact Readout Ceiling

This diagnostic computes the optimistic score if all gates were perfect and only mandatory final readout noise remained.

- Track A cap: `0.950693606049`
- Track B cap: `0.810149370905`
- Final cap: `0.915557547263`

## Current Headroom

- Track A: `0.061728762299`
- Track B: `0.200774370905`
- Final: `0.096490164450`

## Per-Sample Headroom

| sample | track | current | cap | headroom |
|---|---|---:|---:|---:|
| `track_b_001` | track_b | 0.634765625000 | 0.858252404404 | 0.223486779404 |
| `track_b_002` | track_b | 0.583984375000 | 0.762046337406 | 0.178061962406 |
| `track_a_008` | track_a | 0.795312500000 | 0.942364625770 | 0.147052125770 |
| `track_a_007` | track_a | 0.821093750000 | 0.951883460374 | 0.130789710374 |
| `track_a_002` | track_a | 0.860156250000 | 0.951883460374 | 0.091727210374 |
| `track_a_005` | track_a | 0.916406250000 | 0.951883460374 | 0.035477210374 |
| `track_a_006` | track_a | 0.919531250000 | 0.951883460374 | 0.032352210374 |
| `track_a_004` | track_a | 0.921093750000 | 0.951883460374 | 0.030789710374 |
| `track_a_003` | track_a | 0.935156250000 | 0.951883460374 | 0.016727210374 |
| `track_a_001` | track_a | 0.942968750000 | 0.951883460374 | 0.008914710374 |

## Samples

| sample | track | rule | width | cap | method |
|---|---|---|---:|---:|---|
| `track_a_001` | track_a | bitstring | 6 | 0.951883460374 | deterministic_bitstring_best_readout |
| `track_a_002` | track_a | bitstring | 6 | 0.951883460374 | deterministic_bitstring_best_readout |
| `track_a_003` | track_a | bitstring | 6 | 0.951883460374 | deterministic_bitstring_best_readout |
| `track_a_004` | track_a | bitstring | 6 | 0.951883460374 | deterministic_bitstring_best_readout |
| `track_a_005` | track_a | bitstring | 6 | 0.951883460374 | deterministic_bitstring_best_readout |
| `track_a_006` | track_a | bitstring | 6 | 0.951883460374 | deterministic_bitstring_best_readout |
| `track_a_007` | track_a | bitstring | 6 | 0.951883460374 | deterministic_bitstring_best_readout |
| `track_a_008` | track_a | bitstring | 7 | 0.942364625770 | deterministic_bitstring_best_readout |
| `track_b_001` | track_b | allowed_bitstrings | 16 | 0.858252404404 | allowed_bitstrings_best_readout |
| `track_b_002` | track_b | parity_checks | 24 | 0.762046337406 | parity_checks_readout_dominant_grouped_cap |
