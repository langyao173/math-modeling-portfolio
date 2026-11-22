# Exact Readout Ceiling

This diagnostic computes the optimistic score if all gates were perfect and only mandatory final readout noise remained.

- Track A cap: `0.950693606049`
- Track B cap: `0.810149370905`
- Final cap: `0.915557547263`

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
