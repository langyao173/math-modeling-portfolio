# Final Multi-Seed Reproduction

Status: release-candidate verification only. No parameter tuning or root compiler edits were made during this run.

Compiler hash:

- `compiler.py`: `040AF8D79B7F0131397808CDC015B07A15D0A717F00DD091A595C50D8B81DCE9`

The seven public replay summaries below are the immutable RC replay artifacts under `experiments/final_public_summary_seed*.json`. The root compiler hash was rechecked during final hardening and is unchanged.

## Command

```powershell
$env:PYTHONPATH='src'
$seeds = @(7,11,17,23,31,37,41)
foreach ($seed in $seeds) {
  .\.venv\Scripts\python.exe scripts\evaluate.py `
    --seed $seed `
    --shots-a 256 `
    --shots-b 1024 `
    --timeout-sec 30 `
    --variant final_rc_multiseed `
    --decision frozen_rc_verify `
    --out ("experiments/final_public_summary_seed{0}.json" -f $seed)
}
```

## Public Replay Matrix

| seed | Final | Track A | Track B | invalid | p95 compile sec | max compile sec |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.827636718750 | 0.890136718750 | 0.640136718750 | 0 | 17.451201 | 17.451201 |
| 11 | 0.834106445312 | 0.892089843750 | 0.660156250000 | 0 | 16.639128 | 16.639128 |
| 17 | 0.820922851562 | 0.882812500000 | 0.635253906250 | 0 | 16.443113 | 16.443113 |
| 23 | 0.833007812500 | 0.891113281250 | 0.658691406250 | 0 | 13.964210 | 13.964210 |
| 31 | 0.837402343750 | 0.894531250000 | 0.666015625000 | 0 | 13.982924 | 13.982924 |
| 37 | 0.836547851562 | 0.902343750000 | 0.639160156250 | 0 | 13.305827 | 13.305827 |
| 41 | 0.839599609375 | 0.903320312500 | 0.648437500000 | 0 | 13.376242 | 13.376242 |

## Aggregate

| metric | value |
|---|---:|
| mean Final | 0.832746233259 |
| mean Track A | 0.893763950893 |
| mean Track B | 0.649693080357 |
| total invalid | 0 |
| overall p95 compile sec, 70 sample compiles | 13.982924 |
| overall max compile sec | 17.451201 |
| timeout sec | 30 |

Note: with ten samples per seed, the per-seed p95 is the slowest sample for that seed. The overall p95 is computed over all 70 sample compile times. It is above the older 12 second ideal target, but remains comfortably below the 30 second evaluator timeout.

## Final Verification Status

- Public replay seeds: `7, 11, 17, 23, 31, 37, 41`.
- Total invalid samples: `0`.
- Overall p95 compile: `13.982924` seconds.
- Overall max compile: `17.451201` seconds.
- Root compiler was not modified after this replay; later work only generated diagnostic scripts and audit documents.
