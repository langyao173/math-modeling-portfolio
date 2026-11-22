# Track A Semantic Mining Report

This is a development-only targeted semantic mining run. It does not modify root `compiler.py`.

- Targets: `track_a_002, track_a_007, track_a_008`
- Rules loaded: `97`
- Diagnostic shots: `4096`
- Public-style shots: `256`
- Keep decision: `False`

## track_a_002

- Family: `hs`
- Source variants/routed candidates: `0` / `0`
- Metric-guard accepted: `0`
- Root: diagnostic `0.878906`, public-style `0.871094`, CX `8`, depth `8`, ELR `-0.150430`.
- Best: `root_current`, diagnostic `0.878906`, public-style `0.871094`, CX `8`, depth `8`, ELR `-0.150430`.
- Rejections: `{}`

| Candidate | Kind | Valid | Guard | Diag Success | Public Success | CX | Depth | ELR | Reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|

## track_a_007

- Family: `adder`
- Source variants/routed candidates: `3` / `3`
- Metric-guard accepted: `0`
- Root: diagnostic `0.828369`, public-style `0.804688`, CX `14`, depth `22`, ELR `-0.223123`.
- Best: `root_current`, diagnostic `0.828369`, public-style `0.804688`, CX `14`, depth `22`, ELR `-0.223123`.
- Rejections: `{'metric_guard_failed': 3}`

| Candidate | Kind | Valid | Guard | Diag Success | Public Success | CX | Depth | ELR | Reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `relative_phase_toffoli_0_1_3_2_3_5` | `relative_phase_toffoli` | True | False | 0.828369 | 0.804688 | 14 | 22 | -0.223123 | `metric_guard_failed` |
| `relative_phase_toffoli_2_3_5` | `relative_phase_toffoli` | True | False | 0.827881 | 0.804688 | 14 | 21 | -0.223540 | `metric_guard_failed` |
| `relative_phase_toffoli_0_1_3` | `relative_phase_toffoli` | True | False | 0.815186 | 0.777344 | 17 | 26 | -0.245458 | `metric_guard_failed` |

## track_a_008

- Family: `structured_random`
- Source variants/routed candidates: `3` / `3`
- Metric-guard accepted: `0`
- Root: diagnostic `0.817871`, public-style `0.804688`, CX `14`, depth `28`, ELR `-0.243048`.
- Best: `root_current`, diagnostic `0.817871`, public-style `0.804688`, CX `14`, depth `28`, ELR `-0.243048`.
- Rejections: `{'metric_guard_failed': 3}`

| Candidate | Kind | Valid | Guard | Diag Success | Public Success | CX | Depth | ELR | Reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `relative_phase_toffoli_3_4_5_0_4_5` | `relative_phase_toffoli` | True | False | 0.817871 | 0.804688 | 14 | 28 | -0.243048 | `metric_guard_failed` |
| `relative_phase_toffoli_0_4_5` | `relative_phase_toffoli` | True | False | 0.796387 | 0.781250 | 19 | 28 | -0.280693 | `metric_guard_failed` |
| `relative_phase_toffoli_3_4_5` | `relative_phase_toffoli` | True | False | 0.795654 | 0.820312 | 18 | 32 | -0.274228 | `metric_guard_failed` |

## Decision

Do not merge unless a target clears the raw-success gate and the public-style final delta clears `+0.003` with zero invalid candidates.