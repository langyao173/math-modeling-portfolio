# Release Audit Final

Status: release candidate frozen. No runtime compiler code was modified during final hardening.

## Final Decision

Freeze the current root compiler as the release candidate.

Do not merge:

- Track B placement/readout v2
- Track B noisy selector v3
- A source-level semantic mining
- contextual row-phase window miner
- global semantic resynthesis probe
- legal-cap gap closing branches
- additional mined row-phase rules

Reason: each branch failed its keep gate under validator/equivalence/noiseless/metric checks or produced only sub-threshold deltas.

## Hashes

| file | SHA256 |
|---|---|
| `compiler.py` | `040AF8D79B7F0131397808CDC015B07A15D0A717F00DD091A595C50D8B81DCE9` |
| `src/noise_adaptive_hackathon/submission/compiler.py` | `9E585BABF311B8F3C76607C2528F5BFF1F7FDC1B0054942EEBCB150DC839F5B5` |
| `report/report.md` | `E0E591EF23D2BDDF2E6FF4C919A8E41504C301E6814905097E58A0EB73DB0CEE` |
| `report/report.pdf` | `D5C0E787105D69ACB65906880B1E62D40FB19BACD7CEDC7C7F522CBCDA0DF210` |
| `scripts/build_gap_ledger.py` | `B2A507A0DEACA6D026930285BC123D15D82BD38702CB5649BDF997B06BB5AF3C` |
| `scripts/run_track_b_placement_v2.py` | `BF28CF9836F40AFFB288A9EF706A55B44A5E6E35F62F6B62A09618EC48DBF59F` |
| `scripts/run_tracka_semantic_mining.py` | `E0AA78CF829192A47EB6FCDA879A9FED9077949CE6EE1EAADF1B7DEAE7DEC070` |
| `scripts/contextual_row_phase_window_miner.py` | `6704F6B5EB483D1E53A3DDBFB0CC152A6D8E98C86602815D5F666707BBA37903` |
| `scripts/track_a_global_semantic_resynthesis_probe.py` | `AB65C3AF98B647324065991A445C84CF04D750960E248CB6327243AD05C5B00F` |
| `scripts/cap_normalized_gap_report.py` | `C69278774F49DA9C4CE275433E3DFED89A5FC07968CC4324DE1F138E6138673F` |
| `scripts/track_b_noisy_selector_v3.py` | `355DA6BCB298D7E87A9FF524AF566AC5BA51D75DCDB98B9E7DDDBB565C334549` |
| `scripts/track_a_exact_synthesis_hard_search.py` | `FB9293472C423E6ED5718973D502405F4BCE5E022F8F0D8A8C52B34552BA915B` |
| `scripts/package.py` | `783788873F0242902D4BBB01A75BA0D2782858E8C7453CD80D7F11AE79D08C61` |

Package archive hash is recorded after `submission_package.zip` is built.

## Multi-Seed Public Replay

See `experiments/final_multiseed_reproduction.md`.

Summary:

- seeds: 7, 11, 17, 23, 31, 37, 41
- mean Final: 0.832746233259
- mean Track A: 0.893763950893
- mean Track B: 0.649693080357
- total invalid: 0
- overall p95 compile sec: 13.982924
- overall max compile sec: 17.451201
- timeout sec: 30

## Validation

Current project `.venv`:

```text
93 passed in 146.37s
```

Clean temporary venv:

- venv creation succeeded.
- dependency installation timed out before packages were installed.
- clean-env pytest is therefore not claimed.

CLI smoke:

- `compiler.py --circuit --backend --out` completed successfully.
- Cirq JSON output and sibling `.layout.json` were generated.

## Runtime Audit

See `experiments/final_forbidden_access_audit.md`.
See also `experiments/final_package_audit.md` for the final package-level audit.

Summary:

- no runtime import of experimental scripts or miners
- no runtime access to benchmark manifests, public summaries, expected outputs, sample names, hidden/private paths, or scripts
- one benign docstring hit: "not benchmark-name based"
- root CLI still exposes only `--circuit`, `--backend`, and `--out`

## Residual And No-Go Evidence

See `experiments/final_residual_no_go_summary.md`.

Summary:

- B placement/readout v2 did not clear merge thresholds.
- B noisy selector v3 selected the root for both public Track B samples and produced zero seed7 delta.
- A semantic mining produced no accepted gain.
- contextual row-phase mining produced no valid full-circuit final-Z candidates.
- global semantic resynthesis did not beat current A002/A007/A008 implementations.
- legal-cap gap closing ranked residuals but found no same-width Track B selector gain and no A002/A007/A008 low-CX hard-search hit.

## Report Framing

The final report should be read as a method system, not a version log:

1. noise-aware candidate bank and Pareto selection
2. Track B final-Z measurement-channel resynthesis
3. Track A final-Z row-phase equivalence
4. residual analysis and no-go diagnostics
