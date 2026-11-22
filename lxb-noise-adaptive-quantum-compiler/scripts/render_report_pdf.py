from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from common import ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the Markdown report to a simple PDF.")
    parser.add_argument("--source", type=Path, default=ROOT / "report" / "report.md")
    parser.add_argument("--out", type=Path, default=ROOT / "report" / "report.pdf")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    text = args.source.read_text(encoding="utf-8")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for raw_line in text.splitlines():
        if not raw_line:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(raw_line, width=96, replace_whitespace=False) or [""])

    with PdfPages(args.out) as pdf:
        page_lines: list[str] = []
        for line in lines:
            page_lines.append(line)
            if len(page_lines) >= 52:
                _write_page(pdf, page_lines)
                page_lines = []
        if page_lines:
            _write_page(pdf, page_lines)
    return 0


def _write_page(pdf: PdfPages, lines: list[str]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.06, 0.96, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=8)
    pdf.savefig(fig)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
