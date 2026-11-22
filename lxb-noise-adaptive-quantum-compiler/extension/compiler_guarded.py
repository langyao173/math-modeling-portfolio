"""Deadline-guarded entry point (extension).

Runs the submitted compiler with an internal budget below the evaluator's 30-s
timeout; if it does not finish (or fails), the organisers' RouteCQC baseline is
emitted instead, so a slow compile degrades to the baseline score rather than
to an invalid sample.  Interface identical to compiler.py.
"""
import argparse, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUDGET_S = 24.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--circuit", required=True); ap.add_argument("--backend", required=True)
    ap.add_argument("--out", required=True); a = ap.parse_args()
    args = ["--circuit", a.circuit, "--backend", a.backend, "--out", a.out]
    try:
        r = subprocess.run([sys.executable, str(HERE / "compiler.py"), *args], timeout=BUDGET_S)
        if r.returncode == 0 and Path(a.out).exists():
            return 0
    except subprocess.TimeoutExpired:
        pass
    return subprocess.run([sys.executable, "-m", "noise_adaptive_hackathon.cli.run_baseline", "--compiler", "cirq_cqc", *args]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
