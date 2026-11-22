#!/usr/bin/env python3
"""Rebuild every source-backed figure and its independent audit, without fitting.

Use --final for a publication export. It refuses incomplete registered modern
predictions, matched-mask stress experiments or sparse-bound computations.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import subprocess
import sys

HERE = Path(__file__).resolve().parent
UP = HERE.parent


def main(args):
    project = args.project_dir
    if project is None:
        candidate = UP / "project/MathorCup_A"
        project = candidate if (candidate / "data/A/1.csv").is_file() else next(
            (p for p in HERE.parents if (p / "data/A/1.csv").is_file()), None)
    if project is None:
        raise FileNotFoundError("Supply --project-dir with the original competition project.")
    project = project.resolve()
    checks = {}
    if args.final:
        specifications = [
            (UP / "experiments/modern_prediction/missingness/status.json",
             {"complete": True, "completed_folds": 5}),
            (UP / "experiments/sparse_transfer/final_summary/COMPLETE.json",
             {"complete": True, "all_perspective": True, "sdp_finished": True}),
        ]
        failures = []
        for path, required in specifications:
            state = json.loads(path.read_text()) if path.is_file() else {}
            matches = {key: state.get(key) == value for key, value in required.items()}
            checks[str(path)] = dict(required=required, observed={key: state.get(key) for key in required},
                                     passed=all(matches.values()))
            failures += [f"{path}: {key} must be {required[key]!r}; got {state.get(key)!r}"
                         for key, passed in matches.items() if not passed]
        if failures:
            raise SystemExit("Final figure export blocked by incomplete source records:\n" + "\n".join(failures))

    commands = []

    def run(script, *parameters):
        command = [sys.executable, str(HERE / script), *map(str, parameters)]
        commands.append(command)
        subprocess.run(command, cwd=HERE, check=True)

    # Recompute numerical evidence before a final figure can be emitted.
    modern_options = ["--project-dir", project]
    if args.final:
        modern_options.append("--require-complete")
    run("audit_modern_prediction.py", *modern_options)
    run("audit_missingness.py", *modern_options)
    run("audit_new_evidence.py")
    run("build_mechanism_figures.py", "--project-dir", project)
    run("build_quantitative_figures.py", "--project-dir", project,
        "--paired-modern-model", args.paired_modern_model)
    run("audit_visuals.py")
    files = [*sorted((HERE / "figures").glob("*")), *sorted((HERE / "quantitative").glob("*"))]
    hashes = {str(path.relative_to(HERE)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in files if path.is_file() and path.suffix in [".svg", ".pdf", ".png", ".drawio"]}
    report = dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  publication_mode=args.final, project=str(project), completion_checks=checks,
                  commands=commands, artifact_sha256=hashes,
                  interpretation="Successful source and vector checks. Final manuscript pages still require visual inspection after embedding.")
    output = HERE / "qa/rebuild_manifest.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(dict(report=str(output), publication_mode=args.final,
                          hashed_artifact_count=len(hashes)), ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--paired-modern-model", default="raw231_y16_no_residual")
    main(parser.parse_args())
