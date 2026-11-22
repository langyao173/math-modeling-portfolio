"""Build the strict, deterministic single ZIP uploaded to the official portal.

The script intentionally packages only explicit allowlists.  It never copies raw
download trees, caches, credentials, or arbitrary files from the repository.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import mmap
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Sequence
from urllib.parse import urlsplit, urlunsplit


ROOT = Path(__file__).resolve().parents[2]
DELIVERABLES_DIR = ROOT / "deliverables"
OUTPUT_DIR = DELIVERABLES_DIR / "submission_staging"
TRACK_NAME = "能源调度"
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
PORTAL_MAX_BYTES = 50_000_000
MAX_MP4_TABLE_ENTRIES = 100_000
MAX_MP4_SAMPLES = 2_000_000
REGISTRATION_ID_RE = re.compile(r"YX26\d{5}\Z")
WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]")
POSIX_ABSOLUTE_PATH_RE = re.compile(r"(?:^|[\s=:'\"])/(?!/)")
UNC_ABSOLUTE_PATH_RE = re.compile(r"(?:^|[\s=:'\"])(?:\\\\|//)[^\s\\/]+[\\/]")
SECRET_KEY_RE = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|access[_-]?key|secret|token|password|passwd|"
    r"credential|authorization|cookie|session)(?:$|[_-])",
    re.IGNORECASE,
)
QUERY_KEY_RE = re.compile(
    r"^(?:query|queries|query[_-]?parameters?|query[_-]?params?|params?|"
    r"search[_-]?params?)$",
    re.IGNORECASE,
)
WINDOWS_INVALID_NAME_RE = re.compile(r"[<>:\"/\\|?*\x00-\x1f]")
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


CODE_REQUIRED_FILES = (
    "NOTICE.md",
    "README.md",
    "instances/config.yaml",
    "src/01_download/README.md",
    "src/01_download/build_manifest.py",
    "src/01_download/download_api_samples.py",
    "src/01_download/source_registry.json",
    "src/01_download/validate_raw_data.py",
    "src/02_clean/README.md",
    "src/02_clean/clean_mvp_data.py",
    "src/02_clean/validate_cleaned_data.py",
    "src/07_solve_isling/README.md",
    "src/07_solve_isling/build_competition_report.py",
    "src/07_solve_isling/build_exact_controlled_baseline.py",
    "src/07_solve_isling/build_final_deliverable.py",
    "src/07_solve_isling/build_industrial_ising_evidence.py",
    "src/07_solve_isling/build_submission_bundle.py",
    "src/07_solve_isling/promote_cloud_candidate.py",
    "src/07_solve_isling/solve_qubo_platform.py",
    "tests/test_submission_security.py",
    "tests/test_portal_submission_zip.py",
    "tests/test_exact_controlled_baseline.py",
)

CODE_OPTIONAL_FILES = (
    "pyproject.toml",
    "requirements.txt",
    "requirements-submission.txt",
    "src/07_solve_isling/verify_submission.py",
    "uv.lock",
)

DATA_REQUIRED_FILES = (
    "DATA_PROVENANCE.md",
    "NOTICE.md",
    "data/interim/grid_case.json",
    "data/interim/jobs_manifest.parquet",
    "data/interim/raw_long_jobs.parquet",
    "data/interim/site_timeseries.parquet",
    "data/processed/actions.parquet",
    "deliverables/final_solution_package/validation_data/actions_for_validation.csv",
    "deliverables/final_solution_package/validation_data/site_timeseries_for_validation.csv",
)

# The final package is already bound to its own strict MANIFEST.json and is
# independently checked by verify_submission.py.  The staging builder reads
# that manifest instead of duplicating a second, drift-prone hard-coded list.

COMPETITION_REPORT_FILES = frozenset(
    {
        "COMPETITION_REPORT.md",
        "tables/01_final_metrics.csv",
        "tables/02_cloud_repair_polish_progression.csv",
        "tables/03_structural_vs_true_recourse.csv",
        "tables/04_solver_baselines.csv",
        "tables/05_sensitivity.csv",
        "tables/06_dummy_jobs.csv",
        "tables/07_tail_scenarios.csv",
        "tables/08_cloud_staged_check.csv",
        "tables/09_industrial_lns_portfolio.csv",
        "tables/10_same_qubo_oracle_replacement.csv",
        "tables/11_industrial_metrics.csv",
        "tables/12_cloud_contribution.csv",
        "tables/13_qubo_energy_true_objective.csv",
        "tables/14_penalty_sweep.csv",
        "tables/15_exact_controlled_baseline.csv",
        "tables/15_exact_controlled_baseline.json",
        "tables/required_01_final_metrics.csv",
        "tables/required_02_optimization_process.csv",
        "tables/required_03_baseline_comparison.csv",
        "tables/required_04_dummy_jobs.csv",
        "tables/required_05_cleanroom_audit.csv",
    }
)

RESULT_SUPPORT_FILES: tuple[str, ...] = ()

RESULT_OPTIONAL_FILES = (
    "deliverables/SHA256SUMS.txt",
    "deliverables/SUBMISSION_VERIFICATION.json",
    "deliverables/SUBMISSION_VERIFICATION.md",
)

DATA_PACKAGE_README_BYTES = (
    "# Submission data package\n\n"
    "This archive contains only the processed/intermediate inputs and validation CSVs "
    "required to reproduce the submitted result. `data/download_manifest.json` is a "
    "sanitized provenance manifest: secret-named fields, query data, URL queries, "
    "credentials, and absolute local paths are removed. Raw external datasets are not "
    "redistributed.\n"
).encode("utf-8")

_DROP = object()


class SubmissionError(RuntimeError):
    """Raised when a strict submission gate fails."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one deterministic portal ZIP under deliverables/submission_staging."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--registration-id",
        required=True,
        help="Official ID in the exact form YX26 followed by five digits.",
    )
    parser.add_argument(
        "--work-name",
        required=True,
        help="Official work name (2-60 characters; no path or underscore characters).",
    )
    parser.add_argument(
        "--paper",
        required=True,
        type=Path,
        help="Submission-ready PDF paper.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="Optional MP4 cloud-platform demonstration video.",
    )
    parser.add_argument(
        "--require-video",
        action="store_true",
        help="Fail unless --video is supplied and valid.",
    )
    return parser.parse_args(argv)


def validate_registration_id(value: str) -> str:
    if not REGISTRATION_ID_RE.fullmatch(value):
        raise SubmissionError(
            "--registration-id must match YX26 followed by exactly five digits "
            "(example: YX2600001)."
        )
    return value


def validate_work_name(value: str) -> str:
    if value != value.strip():
        raise SubmissionError("--work-name must not have leading or trailing whitespace.")
    if not 2 <= len(value) <= 60:
        raise SubmissionError("--work-name must contain 2-60 characters.")
    if "_" in value:
        raise SubmissionError("--work-name must not contain '_' because '_' separates official filename fields.")
    if WINDOWS_INVALID_NAME_RE.search(value):
        raise SubmissionError("--work-name contains a control, path, or Windows-invalid filename character.")
    if value.endswith((".", " ")) or value in {".", ".."}:
        raise SubmissionError("--work-name has an unsafe trailing character.")
    return value


def resolve_regular_file(path: Path, *, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise SubmissionError(f"{label} does not exist: {path}") from exc
    if not resolved.is_file():
        raise SubmissionError(f"{label} is not a regular file: {resolved}")
    if resolved.stat().st_size <= 0:
        raise SubmissionError(f"{label} is empty: {resolved}")
    return resolved


def is_link_like(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", lambda: False)
    if path.is_symlink() or is_junction():
        return True
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError:
        return False
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def validate_repository_path_components(path: Path, *, relative: str, label: str) -> None:
    """Reject linked ancestors before resolving a repository-owned path."""
    root_absolute = ROOT.absolute()
    try:
        parts = path.absolute().relative_to(root_absolute).parts
    except ValueError as exc:
        raise SubmissionError(f"{label} source is outside the repository: {relative}") from exc
    cursor = ROOT
    for part in parts:
        cursor = cursor / part
        if is_link_like(cursor):
            raise SubmissionError(
                f"Refusing {label} path with a linked ancestor: {relative} ({cursor})"
            )


def reject_active_pdf_object_graph(root: Any) -> None:
    """Reject active/attached content anywhere reachable from the PDF catalog."""
    unsafe_keys = {
        "/AA",
        "/AF",
        "/EF",
        "/EmbeddedFiles",
        "/JavaScript",
        "/JS",
        "/XFA",
    }
    unsafe_actions = {
        "/GoToR",
        "/ImportData",
        "/JavaScript",
        "/Launch",
        "/Movie",
        "/Rendition",
        "/RichMediaExecute",
        "/Sound",
        "/SubmitForm",
    }
    unsafe_types = {"/EmbeddedFile", "/Filespec"}
    unsafe_subtypes = {"/FileAttachment", "/Movie", "/RichMedia", "/Sound"}
    stack = [root]
    seen_indirect: set[tuple[int, int]] = set()
    seen_direct: set[int] = set()
    visited = 0
    while stack:
        raw = stack.pop()
        if hasattr(raw, "idnum") and hasattr(raw, "generation"):
            indirect_key = (int(raw.idnum), int(raw.generation))
            if indirect_key in seen_indirect:
                continue
            seen_indirect.add(indirect_key)
            try:
                value = raw.get_object()
            except Exception as exc:
                raise SubmissionError("--paper contains an unreadable indirect object.") from exc
        else:
            value = raw
            if isinstance(value, (dict, list, tuple)):
                direct_key = id(value)
                if direct_key in seen_direct:
                    continue
                seen_direct.add(direct_key)
        visited += 1
        if visited > 200_000:
            raise SubmissionError("--paper object graph exceeds the safe inspection budget.")
        if isinstance(value, dict):
            keys = {str(key) for key in value}
            if keys & unsafe_keys:
                raise SubmissionError("--paper contains active or associated-file PDF objects.")
            if str(value.get("/S", "")) in unsafe_actions:
                raise SubmissionError("--paper contains an unsafe chained action.")
            if str(value.get("/Type", "")) in unsafe_types:
                raise SubmissionError("--paper contains an embedded-file object.")
            if str(value.get("/Subtype", "")) in unsafe_subtypes:
                raise SubmissionError("--paper contains an active-media or attachment object.")
            stack.extend(value.values())
        elif isinstance(value, (list, tuple)):
            stack.extend(value)


def validate_paper(path: Path) -> Path:
    resolved = resolve_regular_file(path, label="Paper")
    if resolved.suffix.lower() != ".pdf":
        raise SubmissionError("--paper must point to a .pdf file.")
    if resolved.stat().st_size < 1024:
        raise SubmissionError("--paper is too small to be a complete submission PDF.")
    with resolved.open("rb") as handle:
        if handle.read(8)[:5] != b"%PDF-":
            raise SubmissionError("--paper does not have a valid PDF header.")
        handle.seek(max(0, resolved.stat().st_size - 4096))
        if b"%%EOF" not in handle.read():
            raise SubmissionError("--paper is truncated or missing its PDF EOF marker.")
    try:
        try:
            from pypdf import PdfReader  # type: ignore[import-not-found]
        except ImportError:
            from PyPDF2 import PdfReader  # type: ignore[import-not-found,no-redef]
    except ImportError as exc:
        raise SubmissionError(
            "A real PDF parser is required for the paper gate; install the declared pypdf dependency."
        ) from exc
    try:
        reader = PdfReader(str(resolved), strict=True)
        if bool(reader.is_encrypted):
            raise SubmissionError("--paper must not be encrypted.")
        page_count = len(reader.pages)
        root = reader.trailer.get("/Root")
        if hasattr(root, "get_object"):
            root = root.get_object()
        if not isinstance(root, dict):
            raise SubmissionError("--paper has no readable PDF document catalog.")
        names = root.get("/Names")
        if hasattr(names, "get_object"):
            names = names.get_object()
        if isinstance(names, dict) and any(
            key in names for key in ("/EmbeddedFiles", "/JavaScript")
        ):
            raise SubmissionError("--paper must not contain embedded files or JavaScript.")
        acroform = root.get("/AcroForm")
        if hasattr(acroform, "get_object"):
            acroform = acroform.get_object()
        if isinstance(acroform, dict):
            fields = acroform.get("/Fields", [])
            if hasattr(fields, "get_object"):
                fields = fields.get_object()
            if "/XFA" in acroform or "/AA" in acroform or "/JS" in acroform:
                raise SubmissionError("--paper must not contain active AcroForm or XFA content.")
            if isinstance(fields, (list, tuple)) and len(fields) > 0:
                raise SubmissionError("--paper must be a static document without interactive form fields.")
        if "/AF" in root:
            raise SubmissionError("--paper must not contain associated files.")
        reject_active_pdf_object_graph(root)
        if "/AA" in root:
            raise SubmissionError("--paper must not contain document-level additional actions.")
        open_action = root.get("/OpenAction")
        if hasattr(open_action, "get_object"):
            open_action = open_action.get_object()
        if isinstance(open_action, dict) and str(open_action.get("/S", "")) != "/GoTo":
            raise SubmissionError("--paper contains an unsafe document-open action.")
        for page in reader.pages:
            if "/AA" in page:
                raise SubmissionError("--paper pages must not contain additional actions.")
            annotations = page.get("/Annots", [])
            if hasattr(annotations, "get_object"):
                annotations = annotations.get_object()
            for annotation in annotations or []:
                if hasattr(annotation, "get_object"):
                    annotation = annotation.get_object()
                if not isinstance(annotation, dict):
                    continue
                subtype = str(annotation.get("/Subtype", ""))
                if subtype in {"/FileAttachment", "/RichMedia", "/Movie", "/Sound"}:
                    raise SubmissionError("--paper contains an embedded or active-media annotation.")
                if "/FS" in annotation or "/AA" in annotation:
                    raise SubmissionError("--paper contains a file specification or additional action.")
                action = annotation.get("/A")
                if hasattr(action, "get_object"):
                    action = action.get_object()
                if isinstance(action, dict) and str(action.get("/S", "")) in {
                    "/JavaScript",
                    "/Launch",
                    "/SubmitForm",
                    "/ImportData",
                }:
                    raise SubmissionError("--paper contains an unsafe annotation action.")
    except SubmissionError:
        raise
    except Exception as exc:
        raise SubmissionError(f"--paper cannot be parsed as a complete PDF: {exc}") from exc
    if page_count < 1:
        raise SubmissionError("--paper contains no readable pages.")
    return resolved


def _iter_mp4_boxes(data: mmap.mmap, start: int, end: int) -> Iterable[tuple[bytes, int, int]]:
    cursor = start
    while cursor < end:
        if end - cursor < 8:
            raise SubmissionError("--video has a truncated MP4 box header.")
        size32, box_type = struct.unpack_from(">I4s", data, cursor)
        header_size = 8
        if size32 == 1:
            if end - cursor < 16:
                raise SubmissionError("--video has a truncated extended MP4 box header.")
            box_size = struct.unpack_from(">Q", data, cursor + 8)[0]
            header_size = 16
        elif size32 == 0:
            box_size = end - cursor
        else:
            box_size = size32
        if box_size < header_size or cursor + box_size > end:
            raise SubmissionError("--video contains an invalid or out-of-bounds MP4 box.")
        content_start = cursor + header_size
        box_end = cursor + box_size
        yield box_type, content_start, box_end
        cursor = box_end


def _mp4_duration_seconds(data: mmap.mmap, start: int, end: int) -> float:
    if end - start < 20:
        raise SubmissionError("--video has a truncated MP4 movie header.")
    version = data[start]
    if version == 0:
        timescale_offset, duration_offset, duration_size = start + 12, start + 16, 4
    elif version == 1:
        if end - start < 32:
            raise SubmissionError("--video has a truncated version-1 MP4 movie header.")
        timescale_offset, duration_offset, duration_size = start + 20, start + 24, 8
    else:
        raise SubmissionError(f"--video has an unsupported MP4 movie-header version: {version}")
    timescale = struct.unpack_from(">I", data, timescale_offset)[0]
    duration = struct.unpack_from(">I" if duration_size == 4 else ">Q", data, duration_offset)[0]
    if timescale <= 0 or duration <= 0:
        raise SubmissionError("--video must declare a positive duration.")
    return duration / timescale


def _mp4_fullbox_entry_count(data: mmap.mmap, start: int, end: int, *, label: str) -> int:
    if end - start < 8:
        raise SubmissionError(f"--video has a truncated {label} box.")
    count = struct.unpack_from(">I", data, start + 4)[0]
    if count <= 0:
        raise SubmissionError(f"--video has no entries in its {label} box.")
    if count > MAX_MP4_TABLE_ENTRIES:
        raise SubmissionError(f"--video has too many entries in its {label} box.")
    return count


def _mp4_sample_table_summary(
    data: mmap.mmap,
    start: int,
    end: int,
    *,
    media_ranges: Sequence[tuple[int, int]],
) -> tuple[int, int, bytes]:
    boxes = list(_iter_mp4_boxes(data, start, end))
    by_kind: dict[bytes, list[tuple[int, int]]] = {}
    for kind, child_start, child_end in boxes:
        by_kind.setdefault(kind, []).append((child_start, child_end))
    for required in (b"stsd", b"stsz", b"stsc", b"stts"):
        if len(by_kind.get(required, [])) != 1:
            raise SubmissionError(f"--video must contain exactly one {required.decode()} box.")
    chunk_boxes = by_kind.get(b"stco", []) + by_kind.get(b"co64", [])
    if len(chunk_boxes) != 1:
        raise SubmissionError("--video must contain exactly one chunk-offset table.")

    stsd_start, stsd_end = by_kind[b"stsd"][0]
    entry_count = _mp4_fullbox_entry_count(data, stsd_start, stsd_end, label="sample-description")
    allowed_codecs = {b"avc1", b"avc3", b"hev1", b"hvc1", b"vp09", b"av01", b"mp4v"}
    description_codecs: list[bytes] = []
    description_cursor = stsd_start + 8
    for _ in range(entry_count):
        if description_cursor + 8 > stsd_end:
            raise SubmissionError("--video has a truncated sample-description entry.")
        entry_size, codec = struct.unpack_from(">I4s", data, description_cursor)
        if entry_size < 86 or description_cursor + entry_size > stsd_end:
            raise SubmissionError("--video has an invalid sample-description entry.")
        if codec not in allowed_codecs:
            raise SubmissionError(f"--video uses an unsupported video codec entry: {codec!r}.")
        description_codecs.append(codec)
        description_cursor += entry_size
    if description_cursor != stsd_end:
        raise SubmissionError("--video sample-description table has trailing or unparsed bytes.")

    stsz_start, stsz_end = by_kind[b"stsz"][0]
    if stsz_end - stsz_start < 12:
        raise SubmissionError("--video has a truncated sample-size table.")
    default_sample_size, sample_count = struct.unpack_from(">II", data, stsz_start + 4)
    if sample_count <= 0:
        raise SubmissionError("--video declares no media samples.")
    if sample_count > MAX_MP4_SAMPLES:
        raise SubmissionError("--video declares too many media samples for safe validation.")
    if default_sample_size:
        if stsz_start + 12 != stsz_end:
            raise SubmissionError("--video fixed sample-size table has trailing bytes.")
        total_sample_bytes = default_sample_size * sample_count

        def sample_size_at(index: int) -> int:
            return default_sample_size
    else:
        expected_end = stsz_start + 12 + 4 * sample_count
        if expected_end != stsz_end:
            raise SubmissionError("--video has a truncated or padded variable sample-size table.")

        def sample_size_at(index: int) -> int:
            return struct.unpack_from(">I", data, stsz_start + 12 + 4 * index)[0]

        total_sample_bytes = 0
        for index in range(sample_count):
            size = sample_size_at(index)
            if size <= 0:
                raise SubmissionError("--video sample table contains a zero-byte sample.")
            total_sample_bytes += size

    stsc_start, stsc_end = by_kind[b"stsc"][0]
    stsc_entries = _mp4_fullbox_entry_count(data, stsc_start, stsc_end, label="sample-to-chunk")
    if stsc_start + 8 + stsc_entries * 12 != stsc_end:
        raise SubmissionError("--video has a truncated or padded sample-to-chunk table.")
    sample_to_chunk = [
        struct.unpack_from(">III", data, stsc_start + 8 + 12 * index)
        for index in range(stsc_entries)
    ]
    if sample_to_chunk[0][0] != 1:
        raise SubmissionError("--video sample-to-chunk mapping must start at chunk 1.")
    previous_first_chunk = 0
    for first_chunk, samples_per_chunk, description_index in sample_to_chunk:
        if (
            first_chunk <= previous_first_chunk
            or samples_per_chunk <= 0
            or description_index <= 0
            or description_index > len(description_codecs)
        ):
            raise SubmissionError("--video has an invalid sample-to-chunk entry.")
        previous_first_chunk = first_chunk
    stts_start, stts_end = by_kind[b"stts"][0]
    timing_entries = _mp4_fullbox_entry_count(data, stts_start, stts_end, label="time-to-sample")
    if stts_start + 8 + timing_entries * 8 != stts_end:
        raise SubmissionError("--video has a truncated or padded time-to-sample table.")
    timed_samples = 0
    for index in range(timing_entries):
        count, delta = struct.unpack_from(">II", data, stts_start + 8 + 8 * index)
        if count <= 0 or delta <= 0:
            raise SubmissionError("--video time-to-sample entries must be strictly positive.")
        timed_samples += count
    if timed_samples != sample_count:
        raise SubmissionError("--video timing table does not match the declared sample count.")

    chunk_start, chunk_end = chunk_boxes[0]
    chunk_count = _mp4_fullbox_entry_count(data, chunk_start, chunk_end, label="chunk-offset")
    is_64_bit = (chunk_start, chunk_end) in by_kind.get(b"co64", [])
    width = 8 if is_64_bit else 4
    if chunk_start + 8 + chunk_count * width != chunk_end:
        raise SubmissionError("--video has a truncated or padded chunk-offset table.")
    format_code = ">Q" if is_64_bit else ">I"
    if any(first_chunk > chunk_count for first_chunk, _, _ in sample_to_chunk):
        raise SubmissionError("--video sample-to-chunk mapping references a missing chunk.")
    mapped_samples = 0
    for index, (first_chunk, samples_per_chunk, _) in enumerate(sample_to_chunk):
        next_first_chunk = (
            sample_to_chunk[index + 1][0]
            if index + 1 < len(sample_to_chunk)
            else chunk_count + 1
        )
        mapped_samples += (next_first_chunk - first_chunk) * samples_per_chunk
    if mapped_samples != sample_count:
        raise SubmissionError("--video chunk mapping does not match the declared sample count.")
    sample_cursor = 0
    mapping_index = 0
    for chunk_number in range(1, chunk_count + 1):
        offset = struct.unpack_from(
            format_code,
            data,
            chunk_start + 8 + width * (chunk_number - 1),
        )[0]
        if not any(
            media_start <= offset < media_end for media_start, media_end in media_ranges
        ):
            raise SubmissionError("--video has a chunk offset outside every media-data box.")
        while (
            mapping_index + 1 < len(sample_to_chunk)
            and sample_to_chunk[mapping_index + 1][0] <= chunk_number
        ):
            mapping_index += 1
        samples_per_chunk = sample_to_chunk[mapping_index][1]
        chunk_sample_end = sample_cursor + samples_per_chunk
        if chunk_sample_end > sample_count:
            raise SubmissionError("--video chunk mapping consumes too many samples.")
        chunk_bytes = sum(sample_size_at(index) for index in range(sample_cursor, chunk_sample_end))
        byte_end = offset + chunk_bytes
        if not any(
            media_start <= offset and byte_end <= media_end
            for media_start, media_end in media_ranges
        ):
            raise SubmissionError("--video sample byte range escapes its media-data box.")
        sample_cursor = chunk_sample_end
    if sample_cursor != sample_count:
        raise SubmissionError("--video chunk mapping does not consume every declared sample.")
    first_description_index = sample_to_chunk[0][2] - 1
    return sample_count, total_sample_bytes, description_codecs[first_description_index]


def _validate_mp4_structure(path: Path) -> None:
    if path.stat().st_size < 1024:
        raise SubmissionError("--video is too small to contain a real demonstration stream.")
    with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
        top_level = list(_iter_mp4_boxes(data, 0, len(data)))
        if not top_level or top_level[0][0] != b"ftyp":
            raise SubmissionError("--video does not start with an MP4 file-type box.")
        moov_boxes = [(start, end) for kind, start, end in top_level if kind == b"moov"]
        media_ranges = [(start, end) for kind, start, end in top_level if kind == b"mdat"]
        if len(moov_boxes) != 1 or not media_ranges:
            raise SubmissionError("--video must contain exactly one movie box and at least one media-data box.")
        media_payload_bytes = sum(end - start for start, end in media_ranges)
        if media_payload_bytes < 1024:
            raise SubmissionError("--video media-data boxes contain too little payload for a real recording.")
        moov_start, moov_end = moov_boxes[0]
        movie_boxes = list(_iter_mp4_boxes(data, moov_start, moov_end))
        movie_headers = [(start, end) for kind, start, end in movie_boxes if kind == b"mvhd"]
        if len(movie_headers) != 1:
            raise SubmissionError("--video must contain exactly one MP4 movie header.")
        duration_seconds = _mp4_duration_seconds(data, *movie_headers[0])
        validated_video_tracks = 0
        total_video_samples = 0
        total_video_sample_bytes = 0
        for kind, track_start, track_end in movie_boxes:
            if kind != b"trak":
                continue
            for child_kind, child_start, child_end in _iter_mp4_boxes(data, track_start, track_end):
                if child_kind != b"mdia":
                    continue
                media_boxes = list(_iter_mp4_boxes(data, child_start, child_end))
                is_video = any(
                    media_kind == b"hdlr"
                    and media_end - media_start >= 12
                    and data[media_start + 8 : media_start + 12] == b"vide"
                    for media_kind, media_start, media_end in media_boxes
                )
                if not is_video:
                    continue
                sample_table: tuple[int, int] | None = None
                for media_kind, media_start, media_end in media_boxes:
                    if media_kind == b"minf":
                        for minf_kind, minf_start, minf_end in _iter_mp4_boxes(data, media_start, media_end):
                            if minf_kind != b"stbl":
                                continue
                            sample_count, sample_bytes, _ = _mp4_sample_table_summary(
                                data,
                                minf_start,
                                minf_end,
                                media_ranges=media_ranges,
                            )
                            sample_table = (sample_count, sample_bytes)
                if is_video and sample_table is not None:
                    validated_video_tracks += 1
                    total_video_samples += sample_table[0]
                    total_video_sample_bytes += sample_table[1]
        if validated_video_tracks < 1 or total_video_samples < 1:
            raise SubmissionError("--video contains no complete, sampled video track.")
        if total_video_sample_bytes > media_payload_bytes:
            raise SubmissionError("--video sample sizes exceed the available media-data payload.")
        if duration_seconds <= 0:
            raise SubmissionError("--video has no positive playable duration.")


def validate_video(path: Path | None, *, required: bool) -> Path | None:
    if path is None:
        if required:
            raise SubmissionError("--require-video was set but --video was not supplied.")
        return None
    resolved = resolve_regular_file(path, label="Video")
    if resolved.suffix.lower() != ".mp4":
        raise SubmissionError("--video must be an .mp4 file for portal compatibility.")
    _validate_mp4_structure(resolved)
    return resolved


def json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sensitive_key(key: str) -> bool:
    collapsed = re.sub(r"[^a-z0-9]", "", key.casefold())
    sensitive_fragments = (
        "apikey",
        "accesskey",
        "clientsecret",
        "credential",
        "authorization",
        "password",
        "passwd",
        "token",
        "cookie",
        "session",
    )
    query_names = {"query", "queries", "queryparameter", "queryparameters", "queryparam", "queryparams"}
    return bool(
        SECRET_KEY_RE.search(key)
        or QUERY_KEY_RE.fullmatch(key)
        or any(fragment in collapsed for fragment in sensitive_fragments)
        or collapsed in query_names
    )


def _sanitize_string(value: str) -> str | object:
    stripped = value.strip()
    parsed = urlsplit(stripped)
    if parsed.scheme and parsed.netloc:
        if parsed.username is not None or parsed.password is not None:
            return _DROP
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    if (
        WINDOWS_ABSOLUTE_PATH_RE.search(value)
        or POSIX_ABSOLUTE_PATH_RE.search(value)
        or UNC_ABSOLUTE_PATH_RE.search(value)
        or PureWindowsPath(stripped).is_absolute()
        or PurePosixPath(stripped).is_absolute()
    ):
        return _DROP
    return value


def sanitize_manifest_node(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for raw_key in sorted(value, key=str):
            key = str(raw_key)
            if _is_sensitive_key(key):
                continue
            child = sanitize_manifest_node(value[raw_key])
            if child is not _DROP:
                cleaned[key] = child
        return cleaned
    if isinstance(value, list):
        cleaned_list = []
        for item in value:
            child = sanitize_manifest_node(item)
            if child is not _DROP:
                cleaned_list.append(child)
        return cleaned_list
    if isinstance(value, str):
        return _sanitize_string(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise SubmissionError(f"Unsupported value in download_manifest.json: {type(value).__name__}")


def validate_sanitized_manifest(value: Any, path: str = "$" ) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if _is_sensitive_key(key):
                raise SubmissionError(f"Sanitized manifest retained sensitive/query key at {path}.{key}")
            validate_sanitized_manifest(child, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            validate_sanitized_manifest(child, f"{path}[{index}]")
        return
    if isinstance(value, str):
        parsed = urlsplit(value.strip())
        if parsed.scheme and parsed.netloc:
            if parsed.query or parsed.fragment or parsed.username is not None or parsed.password is not None:
                raise SubmissionError(f"Sanitized manifest retained URL query/credential data at {path}")
            return
        if (
            WINDOWS_ABSOLUTE_PATH_RE.search(value)
            or POSIX_ABSOLUTE_PATH_RE.search(value)
            or UNC_ABSOLUTE_PATH_RE.search(value)
            or PureWindowsPath(value.strip()).is_absolute()
            or PurePosixPath(value.strip()).is_absolute()
        ):
            raise SubmissionError(f"Sanitized manifest retained an absolute path at {path}")


def sanitized_download_manifest() -> bytes:
    source = ROOT / "data" / "download_manifest.json"
    validate_repository_source(
        source,
        relative="data/download_manifest.json",
        label="data manifest",
    )
    if not source.is_file():
        raise SubmissionError(f"Required data manifest is missing: {source}")
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SubmissionError(f"Cannot read download_manifest.json: {exc}") from exc
    cleaned = sanitize_manifest_node(raw)
    if cleaned is _DROP or not isinstance(cleaned, dict):
        raise SubmissionError("download_manifest.json did not sanitize to a JSON object.")
    validate_sanitized_manifest(cleaned)
    return json_bytes(cleaned)


def validate_repository_source(path: Path, *, relative: str, label: str) -> None:
    validate_repository_path_components(path, relative=relative, label=label)
    if not path.is_file():
        raise SubmissionError(f"Missing {label} allowlist file: {relative}")
    try:
        path.resolve(strict=True).relative_to(ROOT.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise SubmissionError(f"{label} source escapes the repository: {relative}") from exc


def require_files(relative_paths: Iterable[str], *, label: str) -> list[tuple[str, Path]]:
    selected: list[tuple[str, Path]] = []
    for relative in sorted(set(relative_paths)):
        path = ROOT / PurePosixPath(relative)
        validate_repository_source(path, relative=relative, label=label)
        selected.append((relative, path))
    return selected


def collect_strict_tree(base: Path, allowed: frozenset[str], *, label: str) -> list[tuple[str, Path]]:
    validate_repository_path_components(base, relative=str(base), label=label)
    if not base.is_dir():
        raise SubmissionError(f"Missing {label} directory: {base}")
    actual: set[str] = set()
    for path in base.rglob("*"):
        relative_path = path.relative_to(base).as_posix()
        if is_link_like(path):
            raise SubmissionError(f"Refusing linked {label} entry: {relative_path}")
        if (
            path.is_file()
            and "__pycache__" not in path.relative_to(base).parts
            and path.suffix.lower() not in {".pyc", ".pyo"}
        ):
            actual.add(relative_path)
    missing = sorted(allowed - actual)
    unexpected = sorted(actual - allowed)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing=" + ", ".join(missing))
        if unexpected:
            details.append("unexpected=" + ", ".join(unexpected))
        raise SubmissionError(f"{label} strict allowlist mismatch: {'; '.join(details)}")
    selected = []
    for relative in sorted(allowed):
        path = base / PurePosixPath(relative)
        validate_repository_source(path, relative=relative, label=label)
        selected.append((relative, path))
    return selected


def _validate_archive_name(name: str) -> str:
    normalized = PurePosixPath(name).as_posix()
    parts = PurePosixPath(normalized).parts
    if (
        not normalized
        or normalized != name
        or normalized.startswith("/")
        or "\\" in name
        or ".." in parts
        or len(normalized) > 240
    ):
        raise SubmissionError(f"Unsafe archive member name: {name!r}")
    for part in parts:
        if (
            not part
            or part.endswith((".", " "))
            or ":" in part
            or WINDOWS_INVALID_NAME_RE.search(part)
            or part.split(".", 1)[0].upper() in WINDOWS_RESERVED_NAMES
        ):
            raise SubmissionError(f"Windows-unsafe archive member name: {name!r}")
    return normalized


def collect_manifest_bound_tree(base: Path, *, label: str) -> list[tuple[str, Path]]:
    """Collect a package only when every file is bound by its own manifest."""
    manifest_path = base / "MANIFEST.json"
    validate_repository_source(manifest_path, relative="MANIFEST.json", label=label)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SubmissionError(f"Cannot parse {label} MANIFEST.json: {exc}") from exc
    records = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(records, list) or manifest.get("file_count") != len(records):
        raise SubmissionError(f"{label} manifest has an invalid file list or count.")

    expected = {"MANIFEST.json"}
    selected: list[tuple[str, Path]] = [("MANIFEST.json", manifest_path)]
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise SubmissionError(f"{label} manifest record {index} is not an object.")
        relative = record.get("path")
        declared_size = record.get("bytes")
        declared_hash = record.get("sha256")
        if not isinstance(relative, str) or _validate_archive_name(relative) != relative:
            raise SubmissionError(f"{label} manifest record {index} has an unsafe path.")
        if relative == "MANIFEST.json" or relative in expected:
            raise SubmissionError(f"{label} manifest has a duplicate/reserved path: {relative}")
        if not isinstance(declared_size, int) or declared_size < 0:
            raise SubmissionError(f"{label} manifest has an invalid byte count for {relative}.")
        if not isinstance(declared_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", declared_hash):
            raise SubmissionError(f"{label} manifest has an invalid SHA-256 for {relative}.")
        path = base / PurePosixPath(relative)
        validate_repository_source(path, relative=relative, label=label)
        if path.stat().st_size != declared_size or sha256_path(path) != declared_hash:
            raise SubmissionError(f"{label} payload no longer matches its manifest: {relative}")
        expected.add(relative)
        selected.append((relative, path))

    actual = {
        path.relative_to(base).as_posix()
        for path in base.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise SubmissionError(
            f"{label} manifest membership mismatch: missing={missing}; unexpected={unexpected}"
        )
    return sorted(selected, key=lambda item: item[0])


def write_deterministic_zip(
    destination: Path,
    file_entries: Iterable[tuple[str, Path]],
    byte_entries: Iterable[tuple[str, bytes]] = (),
) -> list[str]:
    entries: dict[str, bytes] = {}
    for archive_name, source in file_entries:
        name = _validate_archive_name(archive_name)
        if name in entries:
            raise SubmissionError(f"Duplicate archive member: {name}")
        entries[name] = source.read_bytes()
    for archive_name, payload in byte_entries:
        name = _validate_archive_name(archive_name)
        if name in entries:
            raise SubmissionError(f"Duplicate archive member: {name}")
        entries[name] = payload

    with zipfile.ZipFile(
        destination,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for name in sorted(entries):
            info = zipfile.ZipInfo(filename=name, date_time=FIXED_ZIP_TIME)
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100644 & 0xFFFF) << 16
            archive.writestr(
                info,
                entries[name],
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    verify_zip(destination, expected_names=sorted(entries))
    return sorted(entries)


def validate_portal_archive_byte_count(byte_count: int) -> None:
    """Enforce the portal's strict, decimal 50 MB upper bound."""
    if not isinstance(byte_count, int) or byte_count < 0:
        raise SubmissionError("Portal archive byte count must be a non-negative integer.")
    if byte_count >= PORTAL_MAX_BYTES:
        raise SubmissionError(
            "Portal archive must be strictly smaller than 50,000,000 bytes; "
            f"actual={byte_count}."
        )


def write_portal_zip(
    destination: Path,
    component_dir: Path,
    member_names: Sequence[str],
) -> list[str]:
    """Write the deterministic, flat ZIP that is the only portal upload."""
    if not component_dir.is_dir() or is_link_like(component_dir):
        raise SubmissionError("Portal component directory must be a real directory.")
    normalized_names = [_validate_archive_name(name) for name in member_names]
    if any(PurePosixPath(name).name != name for name in normalized_names):
        raise SubmissionError("Portal archive members must be files at the ZIP root.")
    if normalized_names != sorted(normalized_names):
        raise SubmissionError("Portal archive member names must be supplied in sorted order.")
    if len(normalized_names) != len(set(normalized_names)):
        raise SubmissionError("Portal archive member list contains duplicates.")
    if len({name.casefold() for name in normalized_names}) != len(normalized_names):
        raise SubmissionError("Portal archive member list contains a case-insensitive collision.")
    files: list[tuple[str, Path]] = []
    for name in normalized_names:
        source = component_dir / name
        if not source.is_file() or source.is_symlink():
            raise SubmissionError(f"Portal component is missing or unsafe: {name}")
        files.append((name, source))
    written = write_deterministic_zip(destination, files)
    validate_portal_archive_byte_count(destination.stat().st_size)
    return written


def verify_portal_publish_directory(directory: Path, archive_name: str) -> None:
    """Require the published staging directory to contain exactly one ZIP."""
    expected = _validate_archive_name(archive_name)
    if PurePosixPath(expected).name != expected or not expected.endswith("_提交材料.zip"):
        raise SubmissionError("Portal archive filename is not a safe submission-package name.")
    if not directory.is_dir() or is_link_like(directory):
        raise SubmissionError("Portal publish path must be a real directory.")
    entries = list(directory.iterdir())
    if len(entries) != 1 or entries[0].name != expected or not entries[0].is_file():
        raise SubmissionError(
            "Final submission_staging must contain only the single portal ZIP."
        )
    validate_portal_archive_byte_count(entries[0].stat().st_size)


def verify_zip(path: Path, *, expected_names: Sequence[str]) -> None:
    with zipfile.ZipFile(path, "r") as archive:
        if archive.testzip() is not None:
            raise SubmissionError(f"CRC verification failed for archive: {path.name}")
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if names != list(expected_names):
            raise SubmissionError(f"Archive order/content mismatch: {path.name}")
        if len(names) != len(set(names)):
            raise SubmissionError(f"Archive has duplicate members: {path.name}")
        for info in infos:
            _validate_archive_name(info.filename)
            if info.date_time != FIXED_ZIP_TIME:
                raise SubmissionError(f"Archive timestamp is not deterministic: {path.name}:{info.filename}")


def verifier_help(verifier: Path) -> str:
    completed = subprocess.run(
        [sys.executable, str(verifier), "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise SubmissionError(
            f"Submission verifier exists but its --help failed ({completed.returncode}):\n"
            f"{completed.stdout.strip()}"
        )
    return completed.stdout


def run_verifier(command_args: Sequence[str], *, phase: str) -> None:
    verifier = ROOT / "src" / "07_solve_isling" / "verify_submission.py"
    completed = subprocess.run(
        [sys.executable, str(verifier), *command_args],
        cwd=ROOT,
        check=False,
    )
    if completed.returncode != 0:
        raise SubmissionError(f"Submission verifier failed during {phase} with exit code {completed.returncode}.")


def run_prebuild_verifier_if_available() -> tuple[Path | None, str]:
    verifier = ROOT / "src" / "07_solve_isling" / "verify_submission.py"
    if not verifier.is_file():
        raise SubmissionError(
            "Strict verifier is required; refusing to build portal attachments without it."
        )
    help_text = verifier_help(verifier)
    package_dir = DELIVERABLES_DIR / "final_solution_package"
    zip_path = DELIVERABLES_DIR / "TAOLIN_DC_SIM_final_solution_package.zip"
    output_flags = ("--output-json", "--output-markdown", "--output-sha256")
    supports_isolated_outputs = all(flag in help_text for flag in output_flags)
    with tempfile.TemporaryDirectory(
        prefix=".submission-preverify-",
        dir=DELIVERABLES_DIR,
    ) as temporary_output:
        temporary_output_dir = Path(temporary_output)
        output_args: list[str] = []
        if supports_isolated_outputs:
            output_args = [
                "--output-json",
                str(temporary_output_dir / "SUBMISSION_VERIFICATION.json"),
                "--output-markdown",
                str(temporary_output_dir / "SUBMISSION_VERIFICATION.md"),
                "--output-sha256",
                str(temporary_output_dir / "SHA256SUMS.txt"),
            ]
        else:
            raise SubmissionError(
                "Verifier does not support isolated output paths; refusing to overwrite "
                "the frozen canonical verification reports during attachment staging."
            )

        if "--package-dir" in help_text and "--zip-path" in help_text:
            run_verifier(
                [
                    "--package-dir",
                    str(package_dir),
                    "--zip-path",
                    str(zip_path),
                    *output_args,
                ],
                phase="canonical-package verification",
            )
        else:
            raise SubmissionError(
                "Verifier does not advertise --package-dir/--zip-path; refusing an "
                "ambiguous prebuild verification run."
            )
    return verifier, help_text


def run_staging_verifier_if_supported(
    verifier: Path | None,
    help_text: str,
    staging_dir: Path,
    *,
    require_video: bool,
) -> None:
    if verifier is None:
        raise SubmissionError("Strict staging verifier is required but unavailable.")
    if "--staging-dir" not in help_text:
        raise SubmissionError(
            "Verifier does not advertise --staging-dir; refusing an unverified attachment build."
        )
    args = ["--staging-dir", str(staging_dir)]
    if require_video:
        if "--require-video" not in help_text:
            raise SubmissionError(
                "Verifier supports staging checks but not --require-video; cannot enforce the requested video gate."
            )
        args.append("--require-video")
    run_verifier(args, phase="staging verification")


def run_portal_verifier_if_supported(
    verifier: Path | None,
    help_text: str,
    portal_zip: Path,
    *,
    require_video: bool,
) -> None:
    if verifier is None:
        raise SubmissionError("Strict portal ZIP verifier is required but unavailable.")
    if "--portal-zip" not in help_text:
        raise SubmissionError(
            "Verifier does not advertise --portal-zip; refusing an unverified portal archive."
        )
    args = ["--portal-zip", str(portal_zip)]
    if require_video:
        if "--require-video" not in help_text:
            raise SubmissionError(
                "Verifier cannot enforce the requested video gate for the portal archive."
            )
        args.append("--require-video")
    run_verifier(args, phase="single portal ZIP verification")


def run_with_lock_retry(operation: Any, *, label: str, attempts: int = 10) -> None:
    """Retry transient Windows sharing violations without hiding persistent failures."""
    if attempts <= 0:
        raise ValueError("attempts must be positive")
    for attempt in range(attempts):
        try:
            operation()
            return
        except PermissionError as exc:
            if attempt + 1 == attempts:
                raise SubmissionError(
                    f"{label} remained locked after {attempts} attempts: {exc}"
                ) from exc
            gc.collect()
            time.sleep(min(0.1 * (2**attempt), 1.0))


def replace_with_retry(source: Path, destination: Path) -> None:
    run_with_lock_retry(
        lambda: os.replace(source, destination),
        label=f"Cannot replace {destination.name}",
    )


def safe_remove_tree(path: Path, *, parent: Path, prefix: str) -> None:
    resolved_parent = parent.resolve(strict=True)
    if is_link_like(path):
        raise SubmissionError(f"Refusing to recursively remove a linked path: {path}")
    resolved_path = path.resolve(strict=True)
    if resolved_path.parent != resolved_parent or not resolved_path.name.startswith(prefix):
        raise SubmissionError(f"Refusing to remove unsafe temporary path: {resolved_path}")
    run_with_lock_retry(
        lambda: shutil.rmtree(resolved_path),
        label=f"Cannot remove temporary directory {resolved_path.name}",
    )


def publish_atomic(staging_temp: Path) -> None:
    backup: Path | None = None
    validate_repository_path_components(
        OUTPUT_DIR,
        relative=OUTPUT_DIR.relative_to(ROOT).as_posix(),
        label="submission output",
    )
    if OUTPUT_DIR.exists():
        if not OUTPUT_DIR.is_dir():
            raise SubmissionError(f"Output path exists but is not a directory: {OUTPUT_DIR}")
        backup = DELIVERABLES_DIR / f".{OUTPUT_DIR.name}.backup-{uuid.uuid4().hex}"
        replace_with_retry(OUTPUT_DIR, backup)
    try:
        replace_with_retry(staging_temp, OUTPUT_DIR)
    except BaseException:
        if backup is not None and backup.exists() and not OUTPUT_DIR.exists():
            replace_with_retry(backup, OUTPUT_DIR)
        raise
    if backup is not None and backup.exists():
        safe_remove_tree(backup, parent=DELIVERABLES_DIR, prefix=f".{OUTPUT_DIR.name}.backup-")


def attachment_record(path: Path, *, role: str, zip_entries: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "bytes": path.stat().st_size,
        "file": path.name,
        "role": role,
        "sha256": sha256_path(path),
    }
    if zip_entries is not None:
        record["zip_entries"] = zip_entries
    return record


def verify_staged_attachments(
    staging_dir: Path,
    manifest_name: str,
    manifest_payload: dict[str, Any],
) -> None:
    expected_names = {
        str(record["file"])
        for record in manifest_payload["attachments"]
    } | {manifest_name}
    actual_names = {path.name for path in staging_dir.iterdir() if path.is_file()}
    non_files = [path.name for path in staging_dir.iterdir() if not path.is_file()]
    if non_files or actual_names != expected_names:
        raise SubmissionError(
            "Staging output mismatch: "
            f"expected={sorted(expected_names)}; actual={sorted(actual_names)}; "
            f"non_files={sorted(non_files)}"
        )
    manifest_path = staging_dir / manifest_name
    if manifest_path.read_bytes() != json_bytes(manifest_payload):
        raise SubmissionError("Attachment manifest changed after it was written.")
    for record in manifest_payload["attachments"]:
        attachment = staging_dir / str(record["file"])
        if attachment.stat().st_size != int(record["bytes"]):
            raise SubmissionError(f"Attachment size no longer matches manifest: {attachment.name}")
        if sha256_path(attachment) != str(record["sha256"]):
            raise SubmissionError(f"Attachment SHA256 no longer matches manifest: {attachment.name}")


def build_bundle(args: argparse.Namespace) -> Path:
    registration_id = validate_registration_id(args.registration_id)
    work_name = validate_work_name(args.work_name)
    paper = validate_paper(args.paper)
    video = validate_video(args.video, required=args.require_video)
    verifier, verifier_help_text = run_prebuild_verifier_if_available()

    validate_repository_path_components(
        DELIVERABLES_DIR,
        relative="deliverables",
        label="submission output",
    )

    code_entries = require_files(CODE_REQUIRED_FILES, label="code")
    for relative in CODE_OPTIONAL_FILES:
        path = ROOT / PurePosixPath(relative)
        if path.exists():
            validate_repository_source(path, relative=relative, label="optional code")
            code_entries.append((relative, path))
    code_entries.sort(key=lambda item: item[0])

    data_entries = require_files(DATA_REQUIRED_FILES, label="data")
    sanitized_manifest = sanitized_download_manifest()

    final_entries = collect_manifest_bound_tree(
        DELIVERABLES_DIR / "final_solution_package",
        label="final solution package",
    )
    report_entries = collect_strict_tree(
        DELIVERABLES_DIR / "competition_report",
        COMPETITION_REPORT_FILES,
        label="competition report",
    )
    support_entries = require_files(RESULT_SUPPORT_FILES, label="result support")
    for relative in RESULT_OPTIONAL_FILES:
        path = ROOT / PurePosixPath(relative)
        if path.exists():
            validate_repository_source(path, relative=relative, label="optional result support")
            support_entries.append((relative, path))
    support_entries.sort(key=lambda item: item[0])

    DELIVERABLES_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = Path(tempfile.mkdtemp(prefix=".submission-components-", dir=DELIVERABLES_DIR))
    portal_temp_path: Path | None = None
    published = False
    try:
        prefix = f"{TRACK_NAME}_{registration_id}_{work_name}"
        paper_name = f"{prefix}_论文.pdf"
        code_name = f"{prefix}_代码.zip"
        data_name = f"{prefix}_数据集.zip"
        result_name = f"{prefix}_补充材料.zip"
        video_name = f"{prefix}_演示视频.mp4"
        manifest_name = f"{prefix}_附件清单.json"
        portal_name = f"{prefix}_提交材料.zip"
        for filename in (
            paper_name,
            code_name,
            data_name,
            result_name,
            video_name,
            manifest_name,
            portal_name,
        ):
            if len(filename) > 180:
                raise SubmissionError(f"Official filename is too long ({len(filename)} characters): {filename}")

        staged_paper = temp_path / paper_name
        shutil.copyfile(paper, staged_paper)

        staged_code = temp_path / code_name
        code_names = write_deterministic_zip(staged_code, code_entries)

        staged_data = temp_path / data_name
        data_names = write_deterministic_zip(
            staged_data,
            data_entries,
            byte_entries=(
                ("DATA_PACKAGE_README.md", DATA_PACKAGE_README_BYTES),
                ("data/download_manifest.json", sanitized_manifest),
            ),
        )

        result_file_entries = [
            (f"final_solution_package/{relative}", path)
            for relative, path in final_entries
        ]
        result_file_entries.extend(
            (f"competition_report/{relative}", path)
            for relative, path in report_entries
        )
        result_file_entries.extend(
            (PurePosixPath(relative).name, path)
            for relative, path in support_entries
        )
        staged_result = temp_path / result_name
        result_names = write_deterministic_zip(staged_result, result_file_entries)

        records = [
            attachment_record(staged_paper, role="paper"),
            attachment_record(staged_code, role="code", zip_entries=len(code_names)),
            attachment_record(staged_data, role="data", zip_entries=len(data_names)),
            attachment_record(staged_result, role="calculation_results", zip_entries=len(result_names)),
        ]
        if video is not None:
            staged_video = temp_path / video_name
            shutil.copyfile(video, staged_video)
            records.append(attachment_record(staged_video, role="demonstration_video"))

        manifest_payload = {
            "attachments": records,
            "filename_pattern": "选题方向_报名编号_作品名称_文件类型",
            "manifest_self_hash_excluded": True,
            "registration_id": registration_id,
            "schema_version": 1,
            "track": TRACK_NAME,
            "video_required": bool(args.require_video),
            "work_name": work_name,
        }
        staged_manifest = temp_path / manifest_name
        staged_manifest.write_bytes(json_bytes(manifest_payload))

        verify_staged_attachments(temp_path, manifest_name, manifest_payload)
        run_staging_verifier_if_supported(
            verifier,
            verifier_help_text,
            temp_path,
            require_video=bool(args.require_video),
        )
        verify_staged_attachments(temp_path, manifest_name, manifest_payload)

        component_names = sorted(
            [str(record["file"]) for record in records] + [manifest_name]
        )
        portal_temp_path = Path(
            tempfile.mkdtemp(prefix=".submission_staging-", dir=DELIVERABLES_DIR)
        )
        staged_portal_zip = portal_temp_path / portal_name
        portal_members = write_portal_zip(
            staged_portal_zip,
            temp_path,
            component_names,
        )
        verify_portal_publish_directory(portal_temp_path, portal_name)
        run_portal_verifier_if_supported(
            verifier,
            verifier_help_text,
            staged_portal_zip,
            require_video=bool(args.require_video),
        )
        verify_portal_publish_directory(portal_temp_path, portal_name)
        publish_atomic(portal_temp_path)
        published = True
    finally:
        if temp_path.exists():
            safe_remove_tree(temp_path, parent=DELIVERABLES_DIR, prefix=".submission-components-")
        if not published and portal_temp_path is not None and portal_temp_path.exists():
            safe_remove_tree(
                portal_temp_path,
                parent=DELIVERABLES_DIR,
                prefix=".submission_staging-",
            )

    final_portal_zip = OUTPUT_DIR / portal_name
    verify_portal_publish_directory(OUTPUT_DIR, portal_name)
    print(f"Portal upload ZIP: {final_portal_zip}")
    print(f"Portal ZIP bytes: {final_portal_zip.stat().st_size}")
    print(f"Portal ZIP SHA256: {sha256_path(final_portal_zip)}")
    print(f"Portal ZIP members: {len(portal_members)}")
    print(f"Portal ZIP video present: {video is not None}")
    return final_portal_zip


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        build_bundle(args)
    except SubmissionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
