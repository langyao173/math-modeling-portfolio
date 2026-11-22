"""Command-line entry point for the immutable pre-2014 research freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from worldcup_forecasting.validation.pre2014_freeze import build_freeze, git_output

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--evidence-root", type=Path, default=ROOT / "temp" / "advanced_evidence"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "freeze")
    parser.add_argument("--source-commit", default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    source_commit = args.source_commit or git_output(root, "rev-parse", "HEAD")
    manifest = build_freeze(
        root,
        args.evidence_root,
        args.output,
        source_commit=source_commit,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "freeze_id": manifest["freeze_id"],
                "source_commit": manifest["source_commit"],
                "artifact_count": len(manifest["artifact_records"]),
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
