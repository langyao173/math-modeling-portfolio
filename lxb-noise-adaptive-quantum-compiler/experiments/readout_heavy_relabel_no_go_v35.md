# Readout-Heavy Relabel No-Go V35

## Question

After promoting the stronger root, A007/A008 remained large Track A residuals.
The disabled `enable_readout_heavy_relabel` pass was tested as a low-risk way to
move final measurements onto better-readout physical qubits without changing CX
or depth.

## Focused Replay

2,048-shot five-seed focused Track A replay:

| sample | readout-heavy effect |
|---|---|
| A007 | no selected circuit change |
| A008 | no selected circuit change |
| A002 | same CX/depth, ELR improves from `-0.1504296156` to `-0.1500515042`, mean success changes from `0.87392578125` to `0.8748046875` |

## Public Seed7 Replay

Full public seed7 replay:

| variant | Track A | Track B | Final | invalid | mean compile |
|---|---:|---:|---:|---:|---:|
| promoted root | 0.89013671875 | 0.64013671875 | 0.82763671875 | 0 | 5.6187s |
| readout-heavy candidate | 0.89013671875 | 0.64013671875 | 0.82763671875 | 0 | 7.1506s |

## Decision

Do not promote `enable_readout_heavy_relabel=True`.

The pass adds compile time, does not affect A007/A008, and shows no full-public
seed7 score gain.
