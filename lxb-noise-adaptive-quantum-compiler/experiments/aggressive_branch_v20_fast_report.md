# Aggressive Branch V20 Fast Report

## Decision

The frozen submission root remains unchanged. `v20_aggressive_fast` is an opt-in aggressive package exposed only through `scripts/variant_compiler.py --variant v20_aggressive_fast`.

This branch is public-positive and invalid-clean, but synthetic Track B affine is slightly negative, so it should be treated as an aggressive submission candidate rather than an automatic replacement for the conservative V19 root.

## Implementation

- Added `measurement_channel_physical_placement_v2.py`.
- The pass only triggers on structural affine parity-star Clifford circuits detected from the input circuit:
  - single final `z` measurement,
  - Track B Clifford-only gate set,
  - affine parity-star channel detected by `detect_affine_parity_star`,
  - no benchmark name, path, manifest, public summary, expected output, or offline ruleset access.
- It preserves measurement-channel semantics and does not alter CX count or depth.
- It enumerates disjoint physical star placements using:
  - two-qubit log error,
  - terminal readout log cost,
  - hub/leaf sensitivity proxy,
  - deterministic beam search.
- Every emitted candidate must pass:
  - full validator,
  - measured affine-channel equivalence,
  - unchanged CX count,
  - non-increasing depth,
  - bounded compile time.

## Public Replay

Five-seed public replay, seeds `7/11/17/23/31`, shots A `256`, shots B `1024`, timeout `30s`.

| Variant | Mean Final | Track A | Track B | Invalid | p95 Compile | Max Compile |
|---|---:|---:|---:|---:|---:|---:|
| V19 frozen root | 0.8190673828125 | 0.88896484375 | 0.609375 | 0 | 8.908s | 9.540s |
| V20 aggressive fast | 0.8236083984375 | 0.88896484375 | 0.6275390625 | 0 | 10.631s | 11.864s |

Per-seed Final deltas:

| Seed | Delta Final | Delta Track B |
|---:|---:|---:|
| 7 | +0.005859375 | +0.0234375 |
| 11 | +0.0042724609375 | +0.01708984375 |
| 17 | +0.0052490234375 | +0.02099609375 |
| 23 | +0.0054931640625 | +0.02197265625 |
| 31 | +0.0018310546875 | +0.00732421875 |

Mean Final delta: `+0.004541015625`.

## Pressure Smoke

| Sample | V19 CX / Depth / ELR | V20 CX / Depth / ELR |
|---|---:|---:|
| A007 | 14 / 22 / -0.223123 | 14 / 22 / -0.223123 |
| A008 | 13 / 23 / -0.258184 | 13 / 23 / -0.258184 |
| B001 | 25 / 9 / -0.525124 | 25 / 9 / -0.525124 |
| B002 | 18 / 6 / -0.618166 | 18 / 6 / -0.569287 |

B002 terminal readout cost improves from `0.380269` to `0.322875`; Clifford sensitivity improves from `0.534160` to `0.493422`.

## Synthetic Holdout

| Suite | V19 | V20 Aggressive Fast | Invalid |
|---|---:|---:|---:|
| Track A structural | 0.76357421875 | 0.76357421875 | 0 / 0 |
| Track B affine | 0.5058430989583333 | 0.504541015625 | 1 / 1 |

The Track B affine invalid count is the existing holdout issue and is not introduced by V20. The slight Track B affine mean regression means this branch is not a no-brainer conservative root replacement.

## Verification

- `python -m pytest`: `89 passed`.
- Root CLI remains only `--circuit`, `--backend`, `--out`.
- Runtime scan found no benchmark names, manifests, public summaries, expected outputs, network calls, hidden/private data access, or offline mined-ruleset loading.
- `compiler.py` SHA256: `7C504D2CE0F7991AF6D8BB2C9EC505699072085714A477DF048553872DEECF05`.

## Recommendation

Keep V19 as the frozen conservative root. Use `v20_aggressive_fast` as the high-score aggressive package if the final submission strategy accepts a small synthetic Track B affine risk in exchange for a public five-seed mean gain of `+0.00454`, no seed regressions, invalid `0`, and p95 compile below `12s`.
