# Track A Truth Table And ANF Generator-Cracking Probe

This probe asks what A002/A007/A008 compute, rather than searching for local deletions.

- Decision: `No <=11/<=6 candidate found by this generator probe; keep root frozen unless a stronger global synthesizer is introduced.`

## Summary

| Sample | Family | Width | Permutation | Affine | Max degree | Likely classes | Candidate found |
|---|---|---:|---:|---:|---:|---|---:|
| `track_a_002` | `hs` | 6 | True | True | 1 | `affine_linear_reversible` | True |
| `track_a_007` | `adder` | 6 | True | False | 3 | `nonlinear_reversible, carry_or_comparator_candidate, ripple_carry_like_nested_dependencies, Peres_MAJ_UMA_local_candidate` | False |
| `track_a_008` | `structured_random` | 7 | True | False | 2 | `quadratic_reversible, controlled_increment_or_quadratic_boundary_candidate` | False |

## track_a_002

- Structure: `affine_linear_reversible`.
- Lower bound note: `affine truth table`.

| Output bit | Degree | Terms | Dependencies |
|---:|---:|---|---|
| y0 | 1 | `1 xor x1` | `[1]` |
| y1 | 1 | `x1 xor x0` | `[0, 1]` |
| y2 | 1 | `x3` | `[3]` |
| y3 | 1 | `1 xor x3 xor x2 xor x1 xor x0` | `[0, 1, 2, 3]` |
| y4 | 1 | `1 xor x5` | `[5]` |
| y5 | 1 | `x5 xor x4 xor x3 xor x2 xor x1 xor x0` | `[0, 1, 2, 3, 4, 5]` |

### Exact Synthesis Candidates

| Candidate | Kind | Found | Max CX | CX | Reason/Target |
|---|---|---:|---:|---:|---|
| `track_a_002_exact_affine_cnot_le_6` | `affine_cnot_exact_mitm` | False | 6 | None | `A002 <= 6 gate` |
| `track_a_002_exact_affine_cnot_le_8` | `affine_cnot_exact_mitm` | True | 8 | 8 | `diagnostic only; exact MITM above 8 CNOT is intentionally not run` |

## track_a_007

- Structure: `nonlinear_reversible, carry_or_comparator_candidate, ripple_carry_like_nested_dependencies, Peres_MAJ_UMA_local_candidate`.
- Lower bound note: `CNOT networks can only realize affine maps; nonlinear ANF terms require Toffoli/phase/H-boundary resources.`.

| Output bit | Degree | Terms | Dependencies |
|---:|---:|---|---|
| y0 | 1 | `x0` | `[0]` |
| y1 | 1 | `1 xor x1` | `[1]` |
| y2 | 1 | `1 xor x2 xor x1 xor x0` | `[0, 1, 2]` |
| y3 | 2 | `1 xor x3 xor x0 xor x0*x1` | `[0, 1, 3]` |
| y4 | 1 | `1 xor x4 xor x2 xor x1 xor x0` | `[0, 1, 2, 4]` |
| y5 | 3 | `1 xor x5 xor x3 xor x2 xor x2*x3 xor x1 xor x1*x3 xor x0 xor x0*x3 xor x0*x2 xor x0*x1*x2` | `[0, 1, 2, 3, 5]` |

### Exact Synthesis Candidates

| Candidate | Kind | Found | Max CX | CX | Reason/Target |
|---|---|---:|---:|---:|---|
| `track_a_007_cnot_only_impossible` | `lower_bound` | False | None | None | `non_affine_truth_table_requires_non_CNOT_semantics` |

## track_a_008

- Structure: `quadratic_reversible, controlled_increment_or_quadratic_boundary_candidate`.
- Lower bound note: `CNOT networks can only realize affine maps; nonlinear ANF terms require Toffoli/phase/H-boundary resources.`.

| Output bit | Degree | Terms | Dependencies |
|---:|---:|---|---|
| y0 | 1 | `x1 xor x0` | `[0, 1]` |
| y1 | 1 | `x1` | `[1]` |
| y2 | 1 | `1 xor x2` | `[2]` |
| y3 | 1 | `1 xor x5 xor x3` | `[3, 5]` |
| y4 | 1 | `x4` | `[4]` |
| y5 | 2 | `1 xor x4 xor x4*x5 xor x3 xor x3*x4 xor x2 xor x1 xor x1*x4 xor x0 xor x0*x4` | `[0, 1, 2, 3, 4, 5]` |
| y6 | 2 | `x6 xor x4 xor x4*x5 xor x3 xor x3*x4 xor x2` | `[2, 3, 4, 5, 6]` |

### Exact Synthesis Candidates

| Candidate | Kind | Found | Max CX | CX | Reason/Target |
|---|---|---:|---:|---:|---|
| `track_a_008_cnot_only_impossible` | `lower_bound` | False | None | None | `non_affine_truth_table_requires_non_CNOT_semantics` |
