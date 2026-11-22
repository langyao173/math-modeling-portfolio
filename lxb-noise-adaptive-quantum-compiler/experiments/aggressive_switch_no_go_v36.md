# Aggressive Switch No-Go V36

## Purpose

After promoting the stronger root, three additional legal switches were tested
as possible moves toward the readout-only score cap:

- `enable_readout_heavy_relabel=True`
- `enable_track_b_relabel=True`
- `enable_broad_exact_small_router=True`

Each was evaluated through a black-box compiler wrapper under the same public
seed7 settings as the promoted root.

## Public Seed7 Results

| variant | Track A | Track B | Final | invalid | mean compile |
|---|---:|---:|---:|---:|---:|
| promoted root | 0.89013671875 | 0.64013671875 | 0.82763671875 | 0 | 5.6187s |
| readout-heavy relabel | 0.89013671875 | 0.64013671875 | 0.82763671875 | 0 | 7.1506s |
| Track B relabel | 0.89013671875 | 0.64013671875 | 0.82763671875 | 0 | 5.8364s |
| broad exact router | 0.89013671875 | 0.64013671875 | 0.82763671875 | 0 | 5.4037s |

## Sample-Level Findings

- Track B relabel selected identical circuits on all public samples.
- Broad exact-router selected identical circuits on all public samples.
- Readout-heavy relabel only changed A002's estimated log reliability from
  `-0.1504296156` to `-0.1500515042`; sampled public seed7 success did not
  change.

## Decision

No additional runtime switch promotion.

The promoted root remains the best verified legal runtime configuration from
this set of aggressive toggles.
