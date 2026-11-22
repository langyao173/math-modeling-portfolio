"""Extract FIFA 2026 Annexe C into a deterministic, validated CSV artifact.

This is an offline provenance tool, not a runtime dependency.  It calls the
``pdftotext`` executable from Poppler/MiKTeX because that keeps PDF parsing out
of the production package.  The source PDF is accepted only when its SHA-256
matches the official-regulations snapshot pinned below.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

SOURCE_URL = (
    "https://digitalhub.fifa.com/m/636f5c9c6f29771f/original/"
    "FWC2026_regulations_EN.pdf"
)
SOURCE_SHA256 = "8e4e9f868cb49f2dd8852622c290450da8adaf8cf262a81f08d16c7e5a598f96"
ANNEX_PDF_PAGES = (41, 49)
WINNER_COLUMNS = ("1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L")
GROUPS = tuple("ABCDEFGHIJKL")
ALLOWED_THIRD_GROUPS = {
    "1A": frozenset("CEFHI"),
    "1B": frozenset("EFGIJ"),
    "1D": frozenset("BEFIJ"),
    "1E": frozenset("ABCDF"),
    "1G": frozenset("AEHIJ"),
    "1I": frozenset("CDFGH"),
    "1K": frozenset("DEIJL"),
    "1L": frozenset("EHIJK"),
}
CELL_PATTERN = re.compile(r"3([A-L])")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bbox_words(pdf_path: Path) -> list[tuple[int, float, float, str]]:
    executable = shutil.which("pdftotext")
    if executable is None:
        raise RuntimeError("pdftotext is required to regenerate the Annexe C table")
    command = [
        executable,
        "-f",
        str(ANNEX_PDF_PAGES[0]),
        "-l",
        str(ANNEX_PDF_PAGES[1]),
        "-bbox-layout",
        str(pdf_path),
        "-",
    ]
    completed = subprocess.run(command, check=True, capture_output=True)
    root = ET.fromstring(completed.stdout)
    words: list[tuple[int, float, float, str]] = []
    pages = [element for element in root.iter() if element.tag.endswith("page")]
    for page_index, page in enumerate(pages):
        for word in page.iter():
            if not word.tag.endswith("word"):
                continue
            text = "".join(word.itertext()).strip()
            words.append(
                (
                    page_index,
                    float(word.attrib["yMin"]),
                    float(word.attrib["xMin"]),
                    text,
                )
            )
    return words


def _extract_rows(pdf_path: Path) -> list[dict[str, str | int]]:
    lines: dict[tuple[int, float], list[tuple[float, str]]] = defaultdict(list)
    for page, y_min, x_min, text in _bbox_words(pdf_path):
        # Table cells on a printed row share a baseline.  Rounding at 0.1 pt is
        # much tighter than the 16.5 pt row spacing and absorbs PDF float noise.
        lines[(page, round(y_min, 1))].append((x_min, text))

    extracted: dict[int, tuple[str, ...]] = {}
    for cells in lines.values():
        ordered = [text for _, text in sorted(cells)]
        for position, token in enumerate(ordered):
            if not token.isdigit():
                continue
            option = int(token)
            assignments = tuple(ordered[position + 1 : position + 9])
            if not 1 <= option <= 495 or len(assignments) != 8:
                continue
            if not all(CELL_PATTERN.fullmatch(value) for value in assignments):
                continue
            if option in extracted:
                raise RuntimeError(f"Duplicate Annexe C option extracted: {option}")
            extracted[option] = assignments

    if set(extracted) != set(range(1, 496)):
        missing = sorted(set(range(1, 496)).difference(extracted))
        raise RuntimeError(f"Annexe C extraction is incomplete; missing options: {missing}")

    records: list[dict[str, str | int]] = []
    for option in range(1, 496):
        assignments = extracted[option]
        groups = tuple(value[1] for value in assignments)
        record: dict[str, str | int] = {
            "option": option,
            "qualifying_groups": "".join(sorted(groups)),
        }
        record.update(dict(zip(WINNER_COLUMNS, assignments, strict=True)))
        records.append(record)
    _validate_rows(records)
    return records


def _validate_rows(rows: list[dict[str, str | int]]) -> None:
    if len(rows) != 495:
        raise RuntimeError("Annexe C must contain exactly 495 options")
    expected_sets = {"".join(values) for values in itertools.combinations(GROUPS, 8)}
    observed_sets: set[str] = set()
    for expected_option, row in enumerate(rows, start=1):
        if row["option"] != expected_option:
            raise RuntimeError("Annexe C options must be sequential from 1 to 495")
        qualifying = str(row["qualifying_groups"])
        assignments = tuple(str(row[column]) for column in WINNER_COLUMNS)
        assigned_groups = tuple(value[1] for value in assignments)
        if qualifying != "".join(sorted(set(assigned_groups))) or len(assigned_groups) != 8:
            raise RuntimeError(f"Option {expected_option} has inconsistent third-place groups")
        if any(not CELL_PATTERN.fullmatch(value) for value in assignments):
            raise RuntimeError(f"Option {expected_option} contains an invalid assignment")
        if any(
            winner[1] == third[1]
            for winner, third in zip(WINNER_COLUMNS, assignments, strict=True)
        ):
            raise RuntimeError(f"Option {expected_option} creates a same-group round-of-32 tie")
        if any(
            third[1] not in ALLOWED_THIRD_GROUPS[winner]
            for winner, third in zip(WINNER_COLUMNS, assignments, strict=True)
        ):
            raise RuntimeError(f"Option {expected_option} violates a variable fixture slot")
        observed_sets.add(qualifying)
    if observed_sets != expected_sets:
        raise RuntimeError("Annexe C does not cover every choose(12, 8) group combination once")


def _canonical_csv(rows: list[dict[str, str | int]]) -> bytes:
    stream = io.StringIO(newline="")
    columns = ("option", "qualifying_groups", *WINNER_COLUMNS)
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def generate(pdf_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Validate the source PDF and write canonical Annexe C data artifacts."""
    pdf_path = pdf_path.resolve()
    if _sha256(pdf_path) != SOURCE_SHA256:
        raise RuntimeError("Source PDF SHA-256 does not match the pinned official regulations")
    rows = _extract_rows(pdf_path)
    csv_bytes = _canonical_csv(rows)
    csv_sha256 = hashlib.sha256(csv_bytes).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "fifa_2026_annex_c.csv"
    metadata_path = output_dir / "fifa_2026_annex_c.metadata.json"
    csv_path.write_bytes(csv_bytes)
    metadata = {
        "schema_version": 1,
        "source_url": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
        "source_pdf_pages": list(ANNEX_PDF_PAGES),
        "row_count": 495,
        "winner_columns": list(WINNER_COLUMNS),
        "combination_count": 495,
        "csv_sha256": csv_sha256,
        "validation": {
            "all_choose_12_8_combinations_once": True,
            "assignments_match_qualifying_groups": True,
            "same_group_round_of_32_matches": 0,
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return csv_path, metadata_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Pinned official FIFA regulations PDF")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/worldcup_forecasting/simulation/data"),
    )
    args = parser.parse_args()
    csv_path, metadata_path = generate(args.pdf, args.output_dir)
    print(csv_path)
    print(metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
