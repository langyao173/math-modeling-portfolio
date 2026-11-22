# Final Clean-Room Dry Run

Status: passed.

The final `submission_package.zip` was expanded into `temp/cleanroom_zip_dry_run/`, and the compiler was run from the extracted `2_Code/` directory rather than from the repository root.

## Package Hash

```text
A57216F96EE4F4868D578003C03D83D3AEB918849BFB98F84E9DFC919FD093A8  submission_package.zip
```

## Dependency Import Smoke

Command:

```powershell
.\.venv\Scripts\python.exe -c "import cirq, networkx, numpy, pydantic, stim, stimcirq; print('deps_ok')"
```

Result:

```text
deps_ok
```

## CLI Dry Run

Working directory:

```text
temp/cleanroom_zip_dry_run/2_Code
```

Command:

```powershell
.\.venv\Scripts\python.exe compiler.py `
  --circuit ..\input\circuit.json `
  --backend ..\input\backend.json `
  --out ..\out\compiled.json
```

Result:

- `temp/cleanroom_zip_dry_run/out/compiled.json` generated.
- `temp/cleanroom_zip_dry_run/out/compiled.layout.json` generated.

Sidecar sample:

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

## CLI Argument Contract

The extracted `2_Code/compiler.py` exposes exactly:

- `--circuit`
- `--backend`
- `--out`

## Content Audit

Forbidden content scan found no files or directories matching:

- `.venv`
- `__pycache__`
- `.pytest_cache`
- `*.pyc`
- `*.pyo`
- `*.log`
- `*.egg-info`
- `scripts`
- `experiments`
- `no_go`
- `contextual`
- `hard_search`
- `selector_v3`
- `gap_report`
- `public_summary`

The final package contains 81 files.
