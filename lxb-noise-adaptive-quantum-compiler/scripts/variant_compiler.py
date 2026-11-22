from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cirq

from noise_adaptive_hackathon.evaluator.equivalence import write_layout_sidecar
from noise_adaptive_hackathon.formats.backend import CompetitionBackend
from noise_adaptive_hackathon.submission.variants import ABLATION_VARIANTS, VariantCompiler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compile with a fixed ablation variant.")
    parser.add_argument("--variant", required=True, choices=ABLATION_VARIANTS)
    parser.add_argument("--circuit", required=True, type=Path)
    parser.add_argument("--backend", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    circuit = cirq.read_json(args.circuit)
    backend = CompetitionBackend.parse_raw(args.backend.read_text(encoding="utf-8"))
    compiler = VariantCompiler(args.variant)
    compiled = compiler.compile(circuit, backend, seed=7)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cirq.to_json(compiled, args.out)
    write_layout_sidecar(args.out, compiler.last_initial_layout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
