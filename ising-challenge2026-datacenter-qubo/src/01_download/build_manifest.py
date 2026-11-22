"""Build a compact manifest and data card for downloaded raw data."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "src" / "01_download" / "source_registry.json"
MANIFEST = ROOT / "data" / "download_manifest.json"
DATA_CARD = ROOT / "reports" / "data_card.md"
EVENT_LOG = ROOT / "reports" / "download_events.jsonl"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def iter_files(path: Path):
    for item in sorted(path.rglob("*")):
        if item.is_file() and ".git" not in item.parts:
            yield item


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"files": 0, "bytes": 0, "sha256": None}
    digest = hashlib.sha256()
    files = 0
    total = 0
    for file_path in iter_files(path):
        rel = file_path.relative_to(path).as_posix()
        file_hash = sha256_file(file_path)
        size = file_path.stat().st_size
        files += 1
        total += size
        digest.update(rel.encode("utf-8"))
        digest.update(str(size).encode("ascii"))
        digest.update(file_hash.encode("ascii"))
    return {
        "files": files,
        "bytes": total,
        "sha256": digest.hexdigest() if files else None,
    }


def git_head(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_events() -> list[dict[str, Any]]:
    if not EVENT_LOG.exists():
        return []
    events_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    with EVENT_LOG.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                event = json.loads(line)
                local_path = event.get("local_path", "")
                try:
                    path = Path(local_path)
                    if path.is_absolute():
                        event["local_path"] = path.relative_to(ROOT).as_posix()
                except ValueError:
                    pass
                key = (event.get("source_name", ""), event.get("local_path", ""))
                events_by_key[key] = event
    return list(events_by_key.values())


def archive_entries() -> list[dict[str, Any]]:
    base = ROOT / "data" / "raw" / "workload" / "alibaba_gpu" / "cluster-trace-gpu-v2020" / "data"
    urls = {
        "pai_group_tag_table.tar.gz": "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_group_tag_table.tar.gz",
        "pai_instance_table.tar.gz": "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_instance_table.tar.gz",
        "pai_job_table.tar.gz": "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_job_table.tar.gz",
        "pai_machine_metric.tar.gz": "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_machine_metric.tar.gz",
        "pai_machine_spec.tar.gz": "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_machine_spec.tar.gz",
        "pai_sensor_table.tar.gz": "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_sensor_table.tar.gz",
        "pai_task_table.tar.gz": "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_task_table.tar.gz",
    }
    entries = []
    for name, url in urls.items():
        path = base / name
        if path.exists():
            entries.append(
                {
                    "source_name": "alibaba_gpu_v2020_archive",
                    "source_url": url,
                    "download_date": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
                    "api_endpoint": None,
                    "query_parameters": {},
                    "local_path": path.relative_to(ROOT).as_posix(),
                    "rows_files_size": {"rows": None, "files": 1, "bytes": path.stat().st_size},
                    "sha256": sha256_file(path),
                    "license": "See Alibaba clusterdata upstream release terms",
                    "time_zone": "source-defined",
                    "unit": "GPU trace tar.gz archive",
                    "status": "downloaded",
                }
            )
        else:
            entries.append(
                {
                    "source_name": "alibaba_gpu_v2020_archive",
                    "source_url": url,
                    "download_date": now_utc(),
                    "api_endpoint": None,
                    "query_parameters": {},
                    "local_path": path.relative_to(ROOT).as_posix(),
                    "rows_files_size": {"rows": None, "files": 0, "bytes": 0},
                    "sha256": None,
                    "license": "See Alibaba clusterdata upstream release terms",
                    "time_zone": "source-defined",
                    "unit": "GPU trace tar.gz archive",
                    "status": "missing",
                }
            )
    return entries


def v2026_capacity_entry() -> dict[str, Any]:
    return {
        "source_name": "alibaba_gpu_v2026_bulk_archives",
        "source_url": "https://tre-clusterdata.oss-cn-hangzhou.aliyuncs.com/cluster-trace-gpu-v2026/data/",
        "download_date": now_utc(),
        "api_endpoint": None,
        "query_parameters": {},
        "local_path": "data/raw/workload/alibaba_gpu/cluster-trace-gpu-v2026/data",
        "rows_files_size": {
            "rows": None,
            "files": 0,
            "bytes": 356275833963,
            "required_bytes_from_upstream_docs": 356275833963,
        },
        "sha256": None,
        "license": "See Alibaba clusterdata upstream release terms",
        "time_zone": "source-defined",
        "unit": "GPU trace ZIP archives",
        "status": "not_downloaded_capacity",
        "notes": "Upstream v2026 docs list approximately 356.3 GB compressed; local drive had about 28.8 GiB free before downloads.",
    }


def build_manifest() -> dict[str, Any]:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    generated_at = now_utc()
    entries: list[dict[str, Any]] = []

    for source in registry:
        local_path = ROOT / source["local_path"]
        summary = directory_summary(local_path)
        entry = {
            "source_name": source["source_name"],
            "source_url": source["source_url"],
            "download_date": generated_at,
            "api_endpoint": source.get("api_endpoint"),
            "query_parameters": {},
            "local_path": source["local_path"],
            "rows_files_size": {
                "rows": None,
                "files": summary["files"],
                "bytes": summary["bytes"],
            },
            "sha256": summary["sha256"],
            "git_head": git_head(local_path),
            "license": source["license"],
            "time_zone": source["time_zone"],
            "unit": source["unit"],
            "status": "present" if summary["files"] else "empty",
            "notes": source.get("notes", ""),
        }
        entries.append(entry)

    entries.extend(archive_entries())
    entries.append(v2026_capacity_entry())
    entries.extend(load_events())

    manifest = {
        "generated_at": generated_at,
        "root": str(ROOT),
        "entries": entries,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def fmt_bytes(value: int | None) -> str:
    if not value:
        return "0 B"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{value} B"


def write_data_card(manifest: dict[str, Any]) -> None:
    lines = [
        "# Data Card",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "## Download Status",
        "",
        "| Source | Status | Files | Size | Local path | Commit/hash | Notes |",
        "| --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for entry in manifest["entries"]:
        files = entry["rows_files_size"].get("files")
        size = fmt_bytes(entry["rows_files_size"].get("bytes"))
        commit_hash = entry.get("git_head") or entry.get("sha256") or ""
        if commit_hash:
            commit_hash = commit_hash[:12]
        notes = entry.get("notes", "")
        lines.append(
            "| {source} | {status} | {files} | {size} | `{path}` | `{hash}` | {notes} |".format(
                source=entry["source_name"],
                status=entry["status"],
                files=files,
                size=size,
                path=entry["local_path"],
                hash=commit_hash,
                notes=notes.replace("|", "/"),
            )
        )

    lines.extend(
        [
            "",
            "## Required Raw Fields",
            "",
            "- `jobs_manifest.parquet`: `job_id`, `workload`, `preemptible`, `migratable`, GPU utilization, power, SLA, migration cost, feasible mode set.",
            "- `actions.parquet`: `action_id`, `job_id`, `site`, `start_slot`, `end_slot`, `mode`, GPU/power vectors, action cost, dummy flag, conflict keys.",
            "- `site_timeseries.parquet`: `site`, `timestamp_utc`, renewable capacity factors and MW, LMP, carbon intensity, demand/flexibility signals.",
            "- `grid_case.json`: bus, branch, generator, cost, base load, and renewable/data-center bus mappings.",
            "",
            "## Caveats",
            "",
            "- Alibaba v2026 bulk archives were not downloaded because the upstream compressed payload is about 356.3 GB.",
            "- CAISO OASIS may fail on networks whose DNS/firewall cannot reach `oasis.caiso.com`; rerun `download_api_samples.py --only caiso` on a reachable network.",
            "- API sample files use one MVP site and fixed historical dates; production experiments should expand the site catalog and date windows.",
        ]
    )
    DATA_CARD.parent.mkdir(parents=True, exist_ok=True)
    DATA_CARD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    manifest = build_manifest()
    write_data_card(manifest)
    print(json.dumps({"manifest": str(MANIFEST), "data_card": str(DATA_CARD)}, indent=2))


if __name__ == "__main__":
    main()
