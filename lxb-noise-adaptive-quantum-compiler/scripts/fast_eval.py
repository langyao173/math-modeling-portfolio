from __future__ import annotations

from pathlib import Path
import sys

from evaluate import main


if __name__ == "__main__":
    default_args = [
        "--shots-a",
        "64",
        "--shots-b",
        "128",
        "--timeout-sec",
        "30",
        "--out",
        str(Path("experiments") / "fast_public_summary.json"),
        "--variant",
        "fast_full",
    ]
    raise SystemExit(main(default_args + sys.argv[1:]))
