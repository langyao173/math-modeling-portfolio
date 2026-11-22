from __future__ import annotations

import argparse
from pathlib import Path

from common import DEFAULT_MANIFEST, best_manifest_entry, load_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare a summary JSON against the best logged manifest entry.")
    parser.add_argument("summary", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = load_json(args.summary)
    best = best_manifest_entry(args.manifest)
    current = float(summary.get("final_score", 0.0))
    if best is None:
        print(f"current={current:.6f}; no manifest baseline found")
        return 0
    delta = current - float(best.get("final_score", 0.0))
    print(
        "current={:.6f}; best={:.6f} variant={}; delta={:+.6f}".format(
            current,
            float(best.get("final_score", 0.0)),
            best.get("variant", "unknown"),
            delta,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
