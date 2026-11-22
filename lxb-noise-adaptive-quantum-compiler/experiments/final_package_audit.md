# Final Package Audit

Status: passed for the frozen release-candidate runtime. The root compiler was not modified.

## Required Hash

| file | SHA256 |
|---|---|
| `compiler.py` | `040AF8D79B7F0131397808CDC015B07A15D0A717F00DD091A595C50D8B81DCE9` |
| `src/noise_adaptive_hackathon/submission/compiler.py` | `9E585BABF311B8F3C76607C2528F5BFF1F7FDC1B0054942EEBCB150DC839F5B5` |
| `report/report.md` | `86CDD4AA097743E7DFB6212A563DE37ECBF51160A672519D06ADD1D0975AAA4D` |
| `report/report.pdf` | `5AEA172DD9DE134C10D9B4D6E8DA98657716CDD1EBAEDED1000564C3EAAB868E` |

The final archive hash is recorded outside the archive in `submission_package.zip.sha256`:

```text
A57216F96EE4F4868D578003C03D83D3AEB918849BFB98F84E9DFC919FD093A8  submission_package.zip
```

## Runtime Scope

Runtime entrypoint:

- `compiler.py`

Runtime package:

- `src/noise_adaptive_hackathon/submission/**/*.py`

Development-only scripts:

- `scripts/**/*.py`
- `experiments/**/*`

The final runtime compiler does not import development scripts. The final submission zip does not include the repository `scripts/` or `experiments/` directories.

## CLI Contract

The root parser exposes exactly the competition arguments:

- `--circuit`
- `--backend`
- `--out`

The runtime reads only:

- the circuit path supplied by `--circuit`
- the backend path supplied by `--backend`

The runtime writes:

- the Cirq JSON path supplied by `--out`
- the sibling `.layout.json` sidecar

CLI smoke command:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe compiler.py `
  --circuit benchmarks\track_a\public\track_a_001\circuit.json `
  --backend backends\small\small_snapshot_a.json `
  --out temp\final_cli_smoke_current\compiled.json
```

Generated files:

- `temp/final_cli_smoke_current/compiled.json`
- `temp/final_cli_smoke_current/compiled.layout.json`

Sidecar schema sample:

```json
{
  "schema_version": "1.0",
  "initial_layout": {
    "5": 1,
    "0": 0,
    "2": 2,
    "1": 5,
    "3": 4,
    "4": 3
  }
}
```

## Forbidden-Access Scan

Runtime scan scope:

- `compiler.py`
- `src/noise_adaptive_hackathon/submission/**/*.py`

Forbidden pattern set:

- `benchmark`
- `public_summary`
- `expected`
- `manifest`
- `sample_name`
- `hidden`
- `private`
- `benchmarks\`
- `benchmarks/`
- `scripts\`
- `scripts/`

Result:

- One benign documentation-string hit: `src/noise_adaptive_hackathon/submission/track_a_affine_collapse.py:37`, "This is deliberately semantic and structural, not benchmark-name based."
- No runtime file access to public metadata, answers, manifests, sample names, public summaries, hidden paths, private paths, or scripts was found.

## Experimental Import Scan

The runtime scan found no imports or references to:

- `contextual_row_phase_window_miner`
- `track_a_global_semantic_resynthesis_probe`
- `run_track_b_placement_v2`
- `run_tracka_semantic_mining`
- `row_phase_rule_miner`
- `offline_relative_phase_rule_miner`
- `cap_normalized_gap_report`
- `track_b_noisy_selector_v3`
- `track_a_exact_synthesis_hard_search`
- `import scripts`
- `from scripts`

Result: no hits.

## Format Compliance

Checked by tests, validator, and CLI smoke:

- Cirq JSON output is generated.
- `.layout.json` sidecar is generated.
- sidecar schema version is `1.0`.
- output is produced through the root CLI only.
- public replay invalid count is `0`.
- Track B-compatible outputs remain under the guarded Clifford-only runtime branch.

## Minimal Zip Content

The final zip contains only:

- `1_Report/report.md`
- `1_Report/report.pdf`
- `2_Code/compiler.py`
- `2_Code/pyproject.toml`
- `2_Code/README.md`
- `2_Code/AGENTS.md`
- runtime source packages under `2_Code/src/noise_adaptive_hackathon/`
- `3_Evaluation_Results/final_multiseed_reproduction.md`

The final zip does not contain:

- `.venv`
- `__pycache__`
- `.pytest_cache`
- `*.pyc`
- `*.pyo`
- `*.log`
- `*.egg-info`
- repository `scripts/`
- repository `experiments/`
- no-go reports
- contextual miner artifacts
- hard-search artifacts
- selector-v3 artifacts
- public summary JSON files

## Tests

Current project environment:

```text
93 passed in 146.37s
```

## Final Decision

The package audit supports freezing the current release candidate. No experimental branch is imported by runtime and no forbidden metadata access was found.
