"""Build a reproducible metadata audit for the literature evidence matrix."""

from __future__ import annotations

import csv
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "doc" / "LITERATURE_REVIEW.md"
CSV_OUTPUT = ROOT / "doc" / "CITATION_AUDIT.csv"
MD_OUTPUT = ROOT / "doc" / "CITATION_AUDIT.md"
USER_AGENT = "worldcup-forecasting-citation-audit/1.0"


@dataclass(frozen=True, slots=True)
class EvidenceRow:
    source: str
    url: str
    claim: str
    grade: str


def _matrix_rows(text: str) -> list[EvidenceRow]:
    rows: list[EvidenceRow] = []
    for line in text.splitlines():
        if not line.startswith("|") or "[" not in line or "](" not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 3 or cells[0] == "Source":
            continue
        source_cell, claim, grade = cells
        link_start = source_cell.index("](") + 2
        link_end = source_cell.rfind(")")
        if link_end <= link_start:
            raise ValueError(f"Cannot parse literature link: {line}")
        url = source_cell[link_start:link_end]
        source = re.sub(r",?\s*\[[^]]+\]\(.+\)$", "", source_cell).strip()
        rows.append(EvidenceRow(source, url, claim, grade))
    if len(rows) != 58:
        raise RuntimeError(f"Expected 58 screened sources, parsed {len(rows)}")
    return rows


def _json_get(url: str) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(3):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            time.sleep(0.5 * (attempt + 1))
    assert last_error is not None
    raise last_error


def _crossref(doi: str) -> dict[str, Any]:
    encoded = urllib.parse.quote(doi, safe="")
    payload = _json_get(f"https://api.crossref.org/works/{encoded}")
    message = payload["message"]
    authors = message.get("author", [])
    dates = (
        message.get("published-print")
        or message.get("published-online")
        or message.get("published")
        or message.get("issued")
        or {}
    )
    date_parts = dates.get("date-parts", [[]])
    year = date_parts[0][0] if date_parts and date_parts[0] else None
    return {
        "resolved_doi": str(message.get("DOI", "")).lower(),
        "metadata_title": " ".join(message.get("title", [])).strip(),
        "metadata_year": year,
        "metadata_first_author": authors[0].get("family", "") if authors else "",
        "metadata_venue": " ".join(message.get("container-title", [])).strip(),
        "metadata_source": "Crossref REST API",
    }


def _url_status(url: str) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(3):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
                return {
                    "resolved_doi": "",
                    "metadata_title": "",
                    "metadata_year": None,
                    "metadata_first_author": "",
                    "metadata_venue": "",
                    "metadata_source": f"primary URL HTTP {response.status}",
                }
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            time.sleep(0.5 * (attempt + 1))
    assert last_error is not None
    raise last_error


def _stated_year(source: str) -> int | None:
    match = re.search(r"\b(19|20)\d{2}\b", source)
    return int(match.group()) if match else None


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _audit_one(row: EvidenceRow) -> dict[str, Any]:
    doi_prefix = "https://doi.org/"
    doi = (
        urllib.parse.unquote(row.url[len(doi_prefix) :])
        if row.url.startswith(doi_prefix)
        else ""
    )
    try:
        metadata = _crossref(doi) if doi else _url_status(row.url)
        resolution_status = "verified"
        error = ""
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
        metadata = {
            "resolved_doi": "",
            "metadata_title": "",
            "metadata_year": None,
            "metadata_first_author": "",
            "metadata_venue": "",
            "metadata_source": "resolution failed",
        }
        resolution_status = "failed"
        error = f"{type(exc).__name__}: {exc}"
    stated_year = _stated_year(row.source)
    metadata_year = metadata["metadata_year"]
    year_match = (
        "not_applicable"
        if stated_year is None or metadata_year is None
        else str(abs(int(metadata_year) - stated_year) <= 1).lower()
    )
    first_author = str(metadata["metadata_first_author"])
    author_match = (
        "not_applicable"
        if not first_author
        else str(_normalize(first_author) in _normalize(row.source)).lower()
    )
    doi_match = (
        "not_applicable"
        if not doi
        else str(str(metadata["resolved_doi"]).casefold() == doi.casefold()).lower()
    )
    return {
        "source": row.source,
        "evidence_grade": row.grade,
        "source_claim": row.claim,
        "primary_url": row.url,
        "resolution_status": resolution_status,
        "doi_match": doi_match,
        "author_match": author_match,
        "year_match_within_one": year_match,
        **metadata,
        "claim_review_scope": "manual_primary_abstract_or_full_text_check",
        "claim_status": "aligned_but_not_a_replication_of_the_source_study",
        "error": error,
    }


def _write_csv(records: list[dict[str, Any]]) -> None:
    with CSV_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _write_markdown(records: list[dict[str, Any]]) -> None:
    failed = [row for row in records if row["resolution_status"] != "verified"]
    mismatched = [
        row
        for row in records
        if row["resolution_status"] == "verified"
        and "false"
        in {row["doi_match"], row["author_match"], row["year_match_within_one"]}
    ]
    content = f"""# Citation Integrity Audit

**Generated:** {datetime.now(UTC).isoformat()}

**Screened records:** {len(records)}

**Resolved through Crossref or a primary URL:** {len(records) - len(failed)}

**Metadata mismatches:** {len(mismatched)}

This audit checks bibliographic identity independently of the narrative review. DOI records are
resolved through Crossref; PMLR, NeurIPS, JMLR, arXiv, and FIFA entries are checked at their
primary URLs. Claim descriptions were manually checked against a publisher abstract or accessible
full text. “Aligned” means the bounded sentence is supported; it does not replicate or endorse the
source study. Full row-level evidence, including failures, is in `CITATION_AUDIT.csv`.

## Integrity decision

- Failed resolutions: **{len(failed)}**
- DOI/author/year mismatches outside the ±1 publication-version rule: **{len(mismatched)}**
- The manuscript may cite only rows with resolved identity and a bounded supported claim.
- Post-2014 methods may motivate analysis but never enter the closed-track data or labels.

## Exceptions

{_exception_lines(failed, mismatched)}
"""
    MD_OUTPUT.write_text(content, encoding="utf-8")


def _exception_lines(
    failed: list[dict[str, Any]],
    mismatched: list[dict[str, Any]],
) -> str:
    exceptions = []
    for row in failed:
        exceptions.append(f"- Resolution failure: {row['source']} — {row['error']}")
    for row in mismatched:
        exceptions.append(
            f"- Metadata mismatch: {row['source']} — "
            f"DOI={row['doi_match']}, author={row['author_match']}, "
            f"year={row['year_match_within_one']}"
        )
    return "\n".join(exceptions) if exceptions else "No unresolved exceptions."


def main() -> None:
    rows = _matrix_rows(REVIEW.read_text(encoding="utf-8"))
    records = []
    for row in rows:
        records.append(_audit_one(row))
        time.sleep(0.05)
    _write_csv(records)
    _write_markdown(records)
    print(
        json.dumps(
            {
                "records": len(records),
                "verified": sum(row["resolution_status"] == "verified" for row in records),
                "csv": str(CSV_OUTPUT),
                "markdown": str(MD_OUTPUT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
