# Final Forbidden-Access Audit

Status: passed for runtime compiler scope, with one benign documentation-string hit.

## Runtime Scope

The runtime compiler entrypoint is:

- `compiler.py`
- `src/noise_adaptive_hackathon/submission/**/*.py`

The development-only scope is:

- `scripts/**/*.py`
- `experiments/**/*`
- evaluator scripts and public replay tools

The development-only scope may read public metadata for analysis. It is not imported by the runtime compiler.

## CLI Contract

`compiler.py` defines exactly the required competition arguments:

- `--circuit`
- `--backend`
- `--out`

The runtime reads:

- the circuit path passed through `--circuit`
- the backend snapshot path passed through `--backend`

The runtime writes:

- the Cirq JSON file passed through `--out`
- the sibling `.layout.json` sidecar through `write_layout_sidecar`

CLI smoke was executed successfully:

```powershell
.\.venv\Scripts\python.exe compiler.py `
  --circuit benchmarks\track_a\public\track_a_001\circuit.json `
  --backend backends\small\small_snapshot_a.json `
  --out temp\final_cli_smoke\compiled.json
```

Generated files:

- `temp/final_cli_smoke/compiled.json`
- `temp/final_cli_smoke/compiled.layout.json`

## Forbidden Pattern Scan

Scan command:

```powershell
Select-String `
  -Path 'compiler.py','src\noise_adaptive_hackathon\submission\*.py','src\noise_adaptive_hackathon\submission\track_a_resynthesis\*.py' `
  -Pattern 'benchmark','public_summary','expected','manifest','sample_name','hidden','private','benchmarks\\','benchmarks/','scripts\\','scripts/' `
  -CaseSensitive:$false
```

Result:

- One benign docstring hit at `src/noise_adaptive_hackathon/submission/track_a_affine_collapse.py:37`: "not benchmark-name based."
- No runtime file access to benchmark manifests, public summaries, expected outputs, sample names, hidden/private paths, or scripts.

## Experimental Import Scan

Scan command:

```powershell
Select-String `
  -Path 'compiler.py','src\noise_adaptive_hackathon\submission\*.py','src\noise_adaptive_hackathon\submission\track_a_resynthesis\*.py' `
  -Pattern 'contextual_row_phase_window_miner','track_a_global_semantic_resynthesis_probe','run_track_b_placement_v2','run_tracka_semantic_mining','row_phase_rule_miner','offline_relative_phase_rule_miner','import scripts','from scripts' `
  -CaseSensitive:$false
```

Result:

- No hits.
- The runtime compiler does not import experimental miners, global probes, placement probes, or script modules.

## Test Status

Current project `.venv`:

```text
93 passed in 158.00s
```

Clean temporary venv check:

- `uv venv temp/final_clean_venv` succeeded.
- `uv pip install --python temp\final_clean_venv\Scripts\python.exe -e ".[dev]"` timed out twice before installing dependencies.
- Therefore the clean-env pytest result is not claimed. The reproducible evidence is the current project `.venv` pytest result above.

