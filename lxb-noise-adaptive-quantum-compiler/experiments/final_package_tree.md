# Final Package Tree

The final archive is built from `submission_package/` into `submission_package.zip`.

## Top-Level Layout

```text
submission_package/
  1_Report/
    report.md
    report.pdf
  2_Code/
    compiler.py
    pyproject.toml
    README.md
    AGENTS.md
    src/
    scripts/
  3_Evaluation_Results/
    experiments/
```

## Runtime Entrypoint

Only `2_Code/compiler.py` is the black-box runtime entrypoint. It imports the package under `2_Code/src/noise_adaptive_hackathon/submission/`.

The `2_Code/scripts/` directory is included for reproducibility and audit only. Runtime compiler execution does not import scripts.

## Required Contents

### `1_Report`

| file | bytes |
|---|---:|
| `report.md` | 20846 |
| `report.pdf` | 42031 |

### `2_Code`

Top-level files:

| file | bytes |
|---|---:|
| `AGENTS.md` | 1695 |
| `compiler.py` | 2265 |
| `pyproject.toml` | 898 |
| `README.md` | 5714 |

### `3_Evaluation_Results`

The final submission package is intentionally minimal. It includes only:

- `final_multiseed_reproduction.md`

## Packaging Command

```powershell
$env:PYTHONPATH='scripts'
.\.venv\Scripts\python.exe scripts\package.py
Compress-Archive -Path submission_package\* -DestinationPath submission_package.zip -Force
```

## Archive

Archive SHA256 is recorded outside the archive in `submission_package.zip.sha256` after packaging.

Current archive SHA256:

```text
A57216F96EE4F4868D578003C03D83D3AEB918849BFB98F84E9DFC919FD093A8  submission_package.zip
```
