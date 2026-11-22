# Contextual Row-Phase Window Miner Report

This is a development-only contextual miner. It does not modify root `compiler.py`.

- Seed: `7`
- Diagnostic shots: `4096`
- Public-style shots: `256`
- Rules loaded: `97`
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
- Windows: `29`, tested contextual candidates: `92`, miter count: `92`.
- Valid candidates: `0`, metric-guard accepted: `0`.
- Root: diagnostic `0.878906`, public-style `0.871094`, CX `8`, depth `8`, ELR `-0.150430`.
- Best: `root_current`, diagnostic `0.878906`, public-style `0.871094`, CX `8`, depth `8`, ELR `-0.150430`.
- Rejections: `{'contextual_final_z_miter_failed': 92, 'duplicate': 170}`

| Candidate | Kind | Valid | Guard | Diag Success | Public Success | CX | Depth | ELR | Reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `context_global_delete_cx_triple_9_10_11` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |
| `context_global_delete_cx_triple_8_9_11` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |
| `context_global_delete_cx_triple_8_9_10` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |
| `context_global_delete_cx_triple_8_10_11` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |
| `context_global_delete_cx_triple_7_9_11` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |
| `context_global_delete_cx_triple_7_9_10` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |
| `context_global_delete_cx_triple_7_8_9` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |
| `context_global_delete_cx_triple_7_8_11` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |
| `context_global_delete_cx_triple_7_8_10` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |
| `context_global_delete_cx_triple_7_10_11` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |
| `context_global_delete_cx_triple_6_9_11` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |
| `context_global_delete_cx_triple_6_9_10` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |
| `context_global_delete_cx_triple_6_8_9` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |
| `context_global_delete_cx_triple_6_8_11` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |

## track_a_007

- Family: `adder`
- Windows: `90`, tested contextual candidates: `281`, miter count: `281`.
- Valid candidates: `0`, metric-guard accepted: `0`.
- Root: diagnostic `0.828369`, public-style `0.804688`, CX `14`, depth `22`, ELR `-0.223123`.
- Best: `root_current`, diagnostic `0.828369`, public-style `0.804688`, CX `14`, depth `22`, ELR `-0.223123`.
- Rejections: `{'contextual_final_z_miter_failed': 281, 'duplicate': 1740}`

| Candidate | Kind | Valid | Guard | Diag Success | Public Success | CX | Depth | ELR | Reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `context_global_delete_cx_triple_8_9_25` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |
| `context_global_delete_cx_triple_8_9_23` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |
| `context_global_delete_cx_triple_8_9_21` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |
| `context_global_delete_cx_triple_8_9_18` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |
| `context_global_delete_cx_triple_8_9_17` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |
| `context_global_delete_cx_triple_8_9_15` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |
| `context_global_delete_cx_triple_8_9_14` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |
| `context_global_delete_cx_triple_8_9_13` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |
| `context_global_delete_cx_triple_8_9_11` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |
| `context_global_delete_cx_triple_8_9_10` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |
| `context_global_delete_cx_triple_8_14_17` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |
| `context_global_delete_cx_triple_8_13_25` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |
| `context_global_delete_cx_triple_8_13_23` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |
| `context_global_delete_cx_triple_8_13_21` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |

## track_a_008

- Family: `structured_random`
- Windows: `90`, tested contextual candidates: `269`, miter count: `269`.
- Valid candidates: `0`, metric-guard accepted: `0`.
- Root: diagnostic `0.817871`, public-style `0.804688`, CX `14`, depth `28`, ELR `-0.243048`.
- Best: `root_current`, diagnostic `0.817871`, public-style `0.804688`, CX `14`, depth `28`, ELR `-0.243048`.
- Rejections: `{'contextual_final_z_miter_failed': 269, 'duplicate': 1755}`

| Candidate | Kind | Valid | Guard | Diag Success | Public Success | CX | Depth | ELR | Reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `context_global_delete_cx_triple_7_9_29` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |
| `context_global_delete_cx_triple_7_9_27` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |
| `context_global_delete_cx_triple_7_9_25` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |
| `context_global_delete_cx_triple_7_9_24` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |
| `context_global_delete_cx_triple_7_9_21` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |
| `context_global_delete_cx_triple_7_9_18` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |
| `context_global_delete_cx_triple_7_9_16` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |
| `context_global_delete_cx_triple_7_9_14` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |
| `context_global_delete_cx_triple_7_9_11` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |
| `context_global_delete_cx_triple_7_16_21` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |
| `context_global_delete_cx_triple_7_16_18` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |
| `context_global_delete_cx_triple_7_14_29` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |
| `context_global_delete_cx_triple_7_14_27` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |
| `context_global_delete_cx_triple_7_14_25` | `global_delete_cx_triple` | False | False | 0.000000 | 0.000000 | None | None | 0.000000 | `contextual_final_z_miter_failed` |

## Decision

Do not merge unless A007/A008 reach `CX <= 12` with raw success `+0.04`, or A002 raw success improves by `+0.03`, and the public-style final delta is at least `+0.003`.