"""Artifact hashing, manifests, and run-level verification."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    """Hash a file without loading it entirely into memory."""
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(
    output_dir: Path,
    run_metadata: dict[str, Any],
    required_paths: set[str] | None = None,
) -> Path:
    """Write the manifest last after verifying required outputs."""
    output_dir = output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    files = [path for path in output_dir.rglob("*") if path.is_file() and path != manifest_path]
    relative_paths = sorted(path.relative_to(output_dir).as_posix() for path in files)
    if required_paths:
        missing = sorted(required_paths.difference(relative_paths))
        if missing:
            raise RuntimeError(f"Cannot complete manifest; required outputs missing: {missing}")
    records = [
        {
            "path": path.relative_to(output_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(files)
    ]
    payload = {
        "status": "complete",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "required_paths": sorted(required_paths or set()),
        "relative_paths": relative_paths,
        "artifacts": records,
        **run_metadata,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path
