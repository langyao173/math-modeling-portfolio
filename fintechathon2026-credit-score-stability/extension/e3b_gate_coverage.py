"""Supplement to E2 (e3_stability_frontier.py): how much of each unchanged-risk cohort the stability gate covers.

For every phase and cohort (gate-stable, similar repayment state, similar payment structure) this reports the
number of out-of-fold score transitions and the share that pass the `seven_05` stability gate. A cohort that the
gate covers only partly keeps the tail of its ungated transitions whatever the clipping radius.

Run:  python extension/e3b_gate_coverage.py <path-to-043-package> results/extension
(row-level inputs are not redistributed; the output is an aggregate table)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

COHORTS = ["stable_05", "repayment_state", "flow_structure"]


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    edges = pd.read_parquet(root / "evidence/current/joint/edges.parquet")
    rows = []
    for phase in (0, 1):
        for cohort in COHORTS:
            g = edges[edges.phase_after.eq(phase) & edges.fold.ge(0) & edges[cohort]]
            rows.append({"phase": phase, "cohort": cohort, "transitions": int(len(g)),
                         "gated_share": float(g.seven_05.mean())})
    (out / "e3_gate_coverage.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
