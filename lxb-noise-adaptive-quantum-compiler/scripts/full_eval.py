from __future__ import annotations

from pathlib import Path
import sys

from evaluate import main


if __name__ == "__main__":
    default_args = [
        "--shots-a",
        "256",
        "--shots-b",
        "1024",
        "--timeout-sec",
        "30",
        "--out",
        str(Path("experiments") / "full_public_summary.json"),
        "--variant",
        "full",
    ]
    raise SystemExit(main(default_args + sys.argv[1:]))
