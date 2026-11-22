"""Extension E1, step 1 — replay Problem 3 policies on 300 fresh seeds per split.

Uses the submission's own command `code/p3_run.py custom` unchanged.  Seeds
2026130000-2026130299 were never used before this extension (the submission used
2026111700-29 for validation and 2026112400-29 for test).  Each policy is replayed on
300 seeds x 3 cases (S1-S3) x 5 duration mappings = 4 500 trials per split.

Requires the official task folder and interim/frames.npz, which step s02 of the pipeline
builds from the raw archives (see SUPPORT_PACKAGE.md).  About 20 minutes on two cores.

    python extension/e1_replay_large.py [--procs 2]

Outputs: results/extension/e1_trials_<split>.csv.gz and e1_runtime_<split>.json.
The process logs and per-case summaries that `p3_run.py` writes as by-products are
removed again, so that results/ and logs/ keep the submitted files only.
"""
import argparse, gzip, json, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES, LOGS, OUT = ROOT / "results", ROOT / "logs", ROOT / "results" / "extension"
SEEDS = [2026130000 + i for i in range(300)]
POLICIES = ["FROZEN", "EDD-median", "RO-feat-static", "RO-feat-learn", "RO-bma-static", "CLAIRVOYANT"]
sys.path.insert(0, str(ROOT / "code"))


def preflight():
    """Problem 3 replays need the official task folder and interim/frames.npz (built by step s02 from the raw archives)."""
    from common import TASK_DIR, INTERIM
    miss = [p for p in (Path(TASK_DIR) / "common_conditions.json", Path(INTERIM) / "frames.npz") if not p.exists()]
    if miss:
        sys.exit("missing %s - see SUPPORT_PACKAGE.md, step 1, then run `python code/run_all.py --only s02_build_dataset`"
                 % ", ".join(map(str, miss)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--n-seeds", type=int, default=len(SEEDS), help="first n of the 300 seeds (for a quick check)")
    a = ap.parse_args(); preflight(); OUT.mkdir(parents=True, exist_ok=True)
    for split in ("validation", "test"):
        tag = f"e1_{split}"
        subprocess.run([sys.executable, str(ROOT / "code" / "p3_run.py"), "custom", "--split", split,
                        "--seeds", ",".join(map(str, SEEDS[:a.n_seeds])), "--policies", ",".join(POLICIES),
                        "--tag", tag, "--procs", str(a.procs)], check=True, cwd=ROOT)
        with open(RES / f"p3_trials_{tag}.csv", "rb") as fi, gzip.open(OUT / f"e1_trials_{split}.csv.gz", "wb") as fo:
            shutil.copyfileobj(fi, fo)
        shutil.move(RES / f"p3_runtime_{tag}.json", OUT / f"e1_runtime_{split}.json")
        for f in [RES / f"p3_trials_{tag}.csv", RES / f"p3_summary_{tag}.csv", *LOGS.glob(f"p3_process_log_*_{tag}.jsonl.gz")]:
            if f.exists():
                f.unlink()


if __name__ == "__main__":
    main()
