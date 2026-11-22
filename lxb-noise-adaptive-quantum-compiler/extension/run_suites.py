"""Score the submitted compiler and the organisers' RouteCQC baseline on S1/S2."""
from __future__ import annotations
import json, sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from noise_adaptive_hackathon.evaluator.submission import evaluate_submission_command

REPO, SUITES, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
SEEDS = [int(s) for s in sys.argv[4].split(",")] if len(sys.argv) > 4 else [7, 11]
OUT.mkdir(parents=True, exist_ok=True)
ALL = {"submission": [sys.executable, str(REPO / "compiler.py")],
       "baseline_cirq_cqc": [sys.executable, "-m", "noise_adaptive_hackathon.cli.run_baseline", "--compiler", "cirq_cqc"],
       "guarded_submission": [sys.executable, str(REPO / "compiler_guarded.py")]}
WHICH = sys.argv[5].split(",") if len(sys.argv) > 5 else ["submission", "baseline_cirq_cqc"]
TIMEOUT = int(sys.argv[6]) if len(sys.argv) > 6 else 60
CMDS = {k: ALL[k] for k in WHICH}


def to_dict(o):
    if is_dataclass(o): return asdict(o)
    if hasattr(o, "model_dump"): return o.model_dump()
    if hasattr(o, "__dict__"): return {k: to_dict(v) for k, v in vars(o).items()}
    if isinstance(o, (list, tuple)): return [to_dict(v) for v in o]
    return o


for suite in sorted(p for p in SUITES.iterdir() if p.is_dir()):
    for name, cmd in CMDS.items():
        for seed in SEEDS:
            f = OUT / f"{suite.name}__{name}__seed{seed}.json"
            if f.exists():
                continue
            s = evaluate_submission_command(root=suite, command=cmd, shots_a=256, shots_b=1024, seed=seed, timeout_sec=TIMEOUT)
            f.write_text(json.dumps(to_dict(s), indent=1, default=str))
            print(suite.name, name, seed, getattr(s, "track_a_score", None), flush=True)
