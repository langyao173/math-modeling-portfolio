# Track B Placement/Readout V2 Report

This is a development-only search. It does not modify root `compiler.py`.

- Seed: `7`
- Shots per candidate: `4096`
- Keep decision: `False`
- Track B delta from best per-sample candidates: `0.005005`
- Final delta estimate: `0.001251`

## track_b_001

- Candidates/simulated: `8` / `8`
- Root: success `0.648193`, CX `21`, depth `16`, ELR `-0.506463`, readout `0.211167`.
- Best: `b001_leaf_tree_015_copyout`, success `0.656006`, CX `26`, depth `12`, ELR `-0.545798`, readout `0.198327`.

| Candidate | Success | CX | Depth | ELR | Readout | Bad B001 Edges |
|---|---:|---:|---:|---:|---:|---:|
| `b001_leaf_tree_015_copyout` | 0.656006 | 26 | 12 | -0.545798 | 0.198327 | 0 |
| `root_current` | 0.648193 | 21 | 16 | -0.506463 | 0.211167 | 4 |
| `b001_leaf_tree_007_copyout` | 0.646973 | 25 | 10 | -0.531878 | 0.200329 | 0 |
| `b001_leaf_tree_013_copyout` | 0.636475 | 22 | 13 | -0.544725 | 0.210310 | 0 |
| `b001_leaf_tree_007` | 0.636230 | 24 | 10 | -0.548809 | 0.233186 | 0 |
| `b001_leaf_tree_005` | 0.632324 | 24 | 13 | -0.544574 | 0.229771 | 1 |
| `b001_leaf_tree_015` | 0.629639 | 25 | 12 | -0.562729 | 0.231184 | 0 |
| `b001_leaf_tree_013` | 0.625488 | 21 | 13 | -0.561656 | 0.243167 | 0 |

## track_b_002

- Candidates/simulated: `160` / `24`
- Root: success `0.637451`, CX `19`, depth `6`, ELR `-0.552356`, readout `0.290018`.
- Best: `b002_affine_star_025_copyout`, success `0.639648`, CX `19`, depth `6`, ELR `-0.552356`, readout `0.290018`.

| Candidate | Success | CX | Depth | ELR | Readout | Bad B001 Edges |
|---|---:|---:|---:|---:|---:|---:|
| `b002_affine_star_025_copyout` | 0.639648 | 19 | 6 | -0.552356 | 0.290018 | 2 |
| `b002_affine_star_013_copyout` | 0.637695 | 19 | 6 | -0.552356 | 0.290018 | 2 |
| `root_current` | 0.637451 | 19 | 6 | -0.552356 | 0.290018 | 2 |
| `b002_affine_star_022_copyout` | 0.637207 | 19 | 6 | -0.552356 | 0.290018 | 2 |
| `b002_affine_star_033_copyout` | 0.635498 | 19 | 6 | -0.552356 | 0.290018 | 2 |
| `b002_affine_star_038_copyout` | 0.634033 | 19 | 6 | -0.552356 | 0.290018 | 2 |
| `b002_affine_star_043_copyout` | 0.633545 | 19 | 6 | -0.552356 | 0.290018 | 2 |
| `b002_affine_star_031_copyout` | 0.632324 | 19 | 6 | -0.552356 | 0.290018 | 2 |
| `b002_affine_star_039_copyout` | 0.631836 | 19 | 6 | -0.552356 | 0.290018 | 2 |
| `b002_affine_star_042_copyout` | 0.630859 | 19 | 6 | -0.552356 | 0.290018 | 2 |

## Decision

Do not merge unless all keep-gate thresholds pass and compile tail is acceptable.