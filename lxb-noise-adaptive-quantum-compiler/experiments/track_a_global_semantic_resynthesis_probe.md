# Track A Global Semantic Resynthesis Probe

This is a development-only global semantic probe. It does not modify root `compiler.py`.

- Seed: `7`
- Diagnostic shots: `4096`
- Public-style shots: `256`
- Keep decision: `False`
- Final seed7 delta estimate: `0.000000`

## Keep Gate

| Metric | Value | Pass |
|---|---:|---:|
| track_a_002 raw diagnostic delta | 0.000000 | False |
| track_a_007 raw diagnostic delta | 0.000000 | False |
| track_a_008 raw diagnostic delta | 0.000000 | False |
| Final seed7 delta estimate | 0.000000 | False |
| Keep | 0 | False |

## track_a_002

- Family: `hs`
- Truth classification: `affine_linear_reversible`, deterministic `True`, permutation `True`, affine `True`, ANF degrees `[1, 1, 1, 1, 1, 1]`.
- Source candidates/routed/valid/accepted: `1` / `1` / `1` / `0`.
- Root: diagnostic `0.878906`, public-style `0.871094`, CX `8`, depth `8`, ELR `-0.150430`.
- Best: `root_current`, diagnostic `0.878906`, public-style `0.871094`, CX `8`, depth `8`, ELR `-0.150430`.
- Rejections: `{'metric_guard_failed': 1}`

| Candidate | Method | Valid | Guard | Source CX | Routed CX | Depth | ELR | Diag Success | Public Success | Reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `global_affine_synthesis_r1` | `affine_truth_table` | True | False | 15 | 15 | 14 | -0.243039 | 0.817871 | 0.820312 | `metric_guard_failed` |

## track_a_007

- Family: `adder`
- Truth classification: `nonlinear_reversible`, deterministic `True`, permutation `True`, affine `False`, ANF degrees `[1, 1, 1, 2, 1, 3]`.
- Source candidates/routed/valid/accepted: `3` / `3` / `2` / `0`.
- Root: diagnostic `0.828369`, public-style `0.804688`, CX `14`, depth `22`, ELR `-0.223123`.
- Best: `root_current`, diagnostic `0.828369`, public-style `0.804688`, CX `14`, depth `22`, ELR `-0.223123`.
- Rejections: `{'metric_guard_failed': 2, 'physical_final_z_miter_failed': 1}`

| Candidate | Method | Valid | Guard | Source CX | Routed CX | Depth | ELR | Diag Success | Public Success | Reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `relative_phase_toffoli_0_1_3_2_3_5_r0` | `relative_phase_toffoli_global` | True | False | 9 | 14 | 22 | -0.223123 | 0.828369 | 0.804688 | `metric_guard_failed` |
| `relative_phase_toffoli_0_1_3_r4` | `relative_phase_toffoli_global` | True | False | 12 | 19 | 26 | -0.271010 | 0.795898 | 0.757812 | `metric_guard_failed` |
| `relative_phase_toffoli_2_3_5_r9` | `relative_phase_toffoli_global` | False | False | 12 | None | None | 0.000000 | 0.000000 | 0.000000 | `physical_final_z_miter_failed` |

## track_a_008

- Family: `structured_random`
- Truth classification: `quadratic_reversible`, deterministic `True`, permutation `True`, affine `False`, ANF degrees `[1, 1, 1, 1, 1, 2, 2]`.
- Source candidates/routed/valid/accepted: `3` / `3` / `0` / `0`.
- Root: diagnostic `0.817871`, public-style `0.804688`, CX `14`, depth `28`, ELR `-0.243048`.
- Best: `root_current`, diagnostic `0.817871`, public-style `0.804688`, CX `14`, depth `28`, ELR `-0.243048`.
- Rejections: `{'physical_final_z_miter_failed': 3}`

| Candidate | Method | Valid | Guard | Source CX | Routed CX | Depth | ELR | Diag Success | Public Success | Reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `relative_phase_toffoli_3_4_5_r9` | `relative_phase_toffoli_global` | False | False | 13 | None | None | 0.000000 | 0.000000 | 0.000000 | `physical_final_z_miter_failed` |
| `relative_phase_toffoli_3_4_5_0_4_5_r9` | `relative_phase_toffoli_global` | False | False | 10 | None | None | 0.000000 | 0.000000 | 0.000000 | `physical_final_z_miter_failed` |
| `relative_phase_toffoli_0_4_5_r9` | `relative_phase_toffoli_global` | False | False | 13 | None | None | 0.000000 | 0.000000 | 0.000000 | `physical_final_z_miter_failed` |

## Decision

If this probe does not clear the keep gate, stop algorithmic sprinting and keep the current root as the release candidate.