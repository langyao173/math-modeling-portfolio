# RC Hardening Addendum After Global Semantic Probe

Date: 2026-06-21

## Decision

The current root compiler remains the release candidate. No experimental branch
is promoted.

## Recent No-Go Evidence

| Probe | Scope | Result | Keep |
|---|---|---:|---:|
| Track B placement/readout v2 | B001/B002 physical placement and copyout only | Final seed7 estimate +0.001251 | False |
| Track A semantic mining | A002/A007/A008 source-level row-phase and relative-phase rewrites | Final seed7 estimate +0.000000 | False |
| Contextual row-phase window miner | A002/A007/A008 compiled-circuit contextual window replacement | 0 valid candidates after final-Z miter | False |
| Track A global semantic resynthesis probe | A002/A007/A008 truth-table classification and from-scratch synthesis | Final seed7 estimate +0.000000 | False |

## Global Probe Findings

- A002 is deterministic affine reversible, but the from-scratch affine synthesis routed to 15 CX versus the root's 8 CX and failed the metric guard.
- A007 is deterministic nonlinear reversible with ANF degrees `[1, 1, 1, 2, 1, 3]`. The best global relative-phase candidate routes back to the same 14-CX incumbent; other candidates regress or fail physical final-Z miter.
- A008 is deterministic quadratic reversible with ANF degrees `[1, 1, 1, 1, 1, 2, 2]`. The tested global relative-phase candidates fail the physical final-Z miter after routing.

## Verification

- `python -m pytest`: 93 passed.
- Runtime forbidden-access scan over `compiler.py` and `src/noise_adaptive_hackathon/submission/*.py`: clean.
- Root CLI contract unchanged.

## Operational Rule

Stop algorithmic sprinting unless a future probe can demonstrate a full keep-gate pass before touching root:

- A007/A008 reaches CX <= 11 or raw success +0.05, or A002 reaches CX <= 6 or raw success +0.03.
- Final seed7 delta >= +0.004.
- invalid = 0.
- pytest passes.
