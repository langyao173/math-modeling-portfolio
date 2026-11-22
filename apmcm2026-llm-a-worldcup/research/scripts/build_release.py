"""Build a deterministic, self-contained competition release ZIP."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from worldcup_forecasting.validation.audit_orchestration import verify_cached_audit
from worldcup_forecasting.verify import verify_manifest

ROOT_FILES = (
    "README.md",
    "main.py",
    "requirements.txt",
    "verify.py",
    "pyproject.toml",
    "uv.lock",
)
ROOT_DIRECTORIES = (
    "artifacts",
    "configs",
    "data/raw",
    "doc",
    "matlab",
    "outputs",
    "plan",
    "research_outputs",
    "scripts",
    "src",
    "tests",
    "temp/advanced_evidence",
)
EXCLUDED_PARTS = {
    ".coverage",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
EXCLUDED_RELATIVE_PATHS = {
    "doc/PAPER.aux",
    "doc/PAPER.blg",
    "doc/PAPER.fdb_latexmk",
    "doc/PAPER.fls",
    "doc/PAPER.log",
    "doc/PAPER.out",
    "doc/PAPER.xdv",
}
REVIEW_ALIASES = {
    "PAPER.pdf": "doc/PAPER.pdf",
    "MODEL_CARD.md": "doc/MODEL_CARD.md",
    "REPRODUCIBILITY.md": "doc/README.md",
    "AUDIT_REPORT.md": "doc/AUDIT.md",
}
ZIP_TIMESTAMP = (2026, 7, 18, 0, 0, 0)


def build_release(project_root: Path, release_dir: Path | None = None) -> dict[str, object]:
    """Verify retained outputs and build a byte-deterministic source/evidence package."""
    root = project_root.expanduser().resolve()
    if not (root / "pyproject.toml").is_file() or not (root / "data" / "raw").is_dir():
        raise FileNotFoundError(f"Not a World Cup forecasting project root: {root}")
    final_verification = _portable_verification(
        verify_manifest(root / "outputs" / "manifest.json")
    )
    research_verification = _portable_verification(
        verify_manifest(root / "research_outputs" / "manifest.json")
    )
    paper_verification = _verify_paper_manifest(root)
    audit_verification = verify_cached_audit(root / "temp" / "advanced_evidence")
    destination = (release_dir or root / "release").expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    zip_path = destination / "worldcup_forecasting_apmcm2026.zip"
    files = _collect_files(root)
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in files
    ]
    aliases = [
        {
            "path": alias,
            "source": source,
            "bytes": (root / source).stat().st_size,
            "sha256": _file_sha256(root / source),
        }
        for alias, source in sorted(REVIEW_ALIASES.items())
    ]
    package_manifest = {
        "package_format_version": 1,
        "root_directory": "worldcup_forecasting",
        "source_file_count": len(records),
        "review_alias_count": len(aliases),
        "final_manifest_verification": final_verification,
        "research_manifest_verification": research_verification,
        "paper_manifest_verification": paper_verification,
        "cached_audit_verification": audit_verification,
        "files": records,
        "review_aliases": aliases,
    }
    manifest_bytes = (
        json.dumps(package_manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    checksum_lines = [f"{record['sha256']}  {record['path']}" for record in records]
    checksum_lines.extend(f"{record['sha256']}  {record['path']}" for record in aliases)
    checksums_bytes = ("\n".join(checksum_lines) + "\n").encode("utf-8")
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = path.relative_to(root).as_posix()
            _write_member(archive, f"worldcup_forecasting/{relative}", path.read_bytes())
        for alias, source in sorted(REVIEW_ALIASES.items()):
            _write_member(
                archive,
                f"worldcup_forecasting/{alias}",
                (root / source).read_bytes(),
            )
        _write_member(archive, "worldcup_forecasting/checksums.sha256", checksums_bytes)
        _write_member(archive, "worldcup_forecasting/MANIFEST.json", manifest_bytes)
        _write_member(
            archive,
            "worldcup_forecasting/PACKAGE_MANIFEST.json",
            manifest_bytes,
        )
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP CRC verification failed at {bad}")
    package_hash = _file_sha256(zip_path)
    checksum_path = destination / "worldcup_forecasting_apmcm2026.zip.sha256"
    checksum_path.write_text(
        f"{package_hash}  {zip_path.name}\n",
        encoding="ascii",
    )
    report = {
        "status": "complete",
        "zip_path": str(zip_path),
        "zip_sha256": package_hash,
        "zip_bytes": zip_path.stat().st_size,
        "source_file_count": len(records),
        "zip_member_count": len(records) + len(aliases) + 3,
        "checksum_path": str(checksum_path),
    }
    (destination / "release_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def _collect_files(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for relative in ROOT_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required release file is missing: {path}")
        candidates.append(path)
    for relative in ROOT_DIRECTORIES:
        directory = root / relative
        if not directory.is_dir():
            raise FileNotFoundError(f"Required release directory is missing: {directory}")
        for path in directory.rglob("*"):
            relative_path = path.relative_to(root)
            if path.is_symlink():
                raise ValueError(f"Release source cannot contain a symlink: {relative_path}")
            if not path.is_file() or _excluded(relative_path):
                continue
            candidates.append(path)
    unique = {path.resolve(): path for path in candidates}
    return sorted(unique.values(), key=lambda path: path.relative_to(root).as_posix())


def _excluded(path: Path) -> bool:
    return (
        bool(set(path.parts).intersection(EXCLUDED_PARTS))
        or path.suffix in EXCLUDED_SUFFIXES
        or path.as_posix() in EXCLUDED_RELATIVE_PATHS
    )


def _verify_paper_manifest(root: Path) -> dict[str, object]:
    manifest_path = root / "doc" / "artifact_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = payload.get("artifacts")
    source_hashes = payload.get("source_hashes")
    boundary = payload.get("evidence_boundary")
    if not isinstance(artifacts, list) or not isinstance(source_hashes, dict):
        raise ValueError("Paper artifact manifest is malformed")
    if boundary != {
        "advanced_audit_opened": True,
        "audit_target_reads": 1,
        "maximum_selection_year": 2010,
        "simulation_2026_status": "promoted_replacement_format_scenario",
    }:
        raise ValueError("Paper evidence boundary changed")
    for record in artifacts:
        _verify_file_record(root / "doc", record)
    for record in source_hashes.values():
        _verify_file_record(root, record, require_bytes=False)
    return {
        "status": "verified",
        "artifact_count": len(artifacts),
        "source_count": len(source_hashes),
        "advanced_audit_opened": True,
        "audit_target_reads": 1,
        "simulation_2026_status": "promoted_replacement_format_scenario",
    }


def _verify_file_record(
    base: Path,
    record: object,
    *,
    require_bytes: bool = True,
) -> None:
    if not isinstance(record, dict):
        raise ValueError("Manifest file record is malformed")
    relative = record.get("path")
    expected_hash = record.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected_hash, str):
        raise ValueError("Manifest file record has invalid fields")
    path = (base / relative).resolve()
    try:
        path.relative_to(base.resolve())
    except ValueError as error:
        raise ValueError(f"Manifest path escapes its root: {relative}") from error
    if not path.is_file() or _file_sha256(path) != expected_hash:
        raise ValueError(f"Manifest-bound file changed: {relative}")
    if require_bytes and path.stat().st_size != record.get("bytes"):
        raise ValueError(f"Manifest-bound file size changed: {relative}")


def _write_member(archive: ZipFile, name: str, content: bytes) -> None:
    info = ZipInfo(name, date_time=ZIP_TIMESTAMP)
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    archive.writestr(info, content, compress_type=ZIP_DEFLATED, compresslevel=9)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_verification(report: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in report.items() if key != "manifest_path"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--release-dir", type=Path, default=None)
    arguments = parser.parse_args()
    report = build_release(arguments.project_root, arguments.release_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
