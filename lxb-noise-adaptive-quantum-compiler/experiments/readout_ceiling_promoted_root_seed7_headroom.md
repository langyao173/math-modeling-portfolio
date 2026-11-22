# Exact Readout Ceiling

This diagnostic computes the optimistic score if all gates were perfect and only mandatory final readout noise remained.

- Track A cap: `0.950693606049`
- Track B cap: `0.810149370905`
- Final cap: `0.915557547263`

## Current Headroom

- Track A: `0.060556887299`
- Track B: `0.170012652155`
- Final: `0.087920828513`

## Per-Sample Headroom

| sample | track | current | cap | headroom |
|---|---|---:|---:|---:|
| `track_b_001` | track_b | 0.666992187500 | 0.858252404404 | 0.191260216904 |
| `track_b_002` | track_b | 0.613281250000 | 0.762046337406 | 0.148765087406 |
| `track_a_007` | track_a | 0.804687500000 | 0.951883460374 | 0.147195960374 |
| `track_a_008` | track_a | 0.804687500000 | 0.942364625770 | 0.137677125770 |
| `track_a_002` | track_a | 0.871093750000 | 0.951883460374 | 0.080789710374 |
| `track_a_004` | track_a | 0.914062500000 | 0.951883460374 | 0.037820960374 |
| `track_a_005` | track_a | 0.925781250000 | 0.951883460374 | 0.026102210374 |
| `track_a_006` | track_a | 0.929687500000 | 0.951883460374 | 0.022195960374 |
| `track_a_003` | track_a | 0.929687500000 | 0.951883460374 | 0.022195960374 |
| `track_a_001` | track_a | 0.941406250000 | 0.951883460374 | 0.010477210374 |

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
