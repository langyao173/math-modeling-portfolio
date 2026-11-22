"""Runtime paths and input discovery."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


SRC_DIR = Path(__file__).resolve().parent
PACKAGED_DATA_DIR = SRC_DIR / "data"
OFFICIAL_FILES = (
    "WorldCups_preview.csv",
    "WorldCupMatches.csv",
    "WorldCupPlayers.csv",
)


def _candidate_test_dirs(
    explicit: str | os.PathLike[str] | None = None,
) -> Iterable[Path]:
    seen: set[str] = set()
    raw_candidates = [
        explicit,
        os.environ.get("TESTDATA_DIR"),
        "/testdata",
        "C:/testdata",
        Path.cwd() / "testdata",
        SRC_DIR / "testdata",
    ]
    for raw in raw_candidates:
        if not raw:
            continue
        path = Path(raw).expanduser()
        key = str(path.resolve()) if path.exists() else str(path.absolute())
        if key not in seen:
            seen.add(key)
            yield path


def find_testdata_dir(
    explicit: str | os.PathLike[str] | None = None,
) -> Path | None:
    for path in _candidate_test_dirs(explicit):
        if path.is_dir():
            return path
    return None


def has_official_training_files(directory: Path | None) -> bool:
    return bool(
        directory
        and all((directory / name).is_file() for name in OFFICIAL_FILES)
    )


def resolve_training_data_dir(
    explicit_testdata: str | os.PathLike[str] | None = None,
) -> tuple[Path, Path | None]:
    test_dir = find_testdata_dir(explicit_testdata)
    if has_official_training_files(test_dir):
        return test_dir, test_dir
    return PACKAGED_DATA_DIR, test_dir


def resolve_output_root() -> Path:
    candidate = SRC_DIR
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / ".world_cup_write_probe"
        probe.touch(exist_ok=True)
        probe.unlink(missing_ok=True)
        return candidate.resolve()
    except OSError as exc:
        raise OSError(
            f"提交文件中的 src 目录“{candidate}”不可写；"
            "请先把压缩包解压到可写目录。"
        ) from exc


def ensure_output_dirs(output_root: Path) -> dict[str, Path]:
    base = output_root / "outputs"
    result = {
        "base": base,
        "tables": base / "tables",
        "figures": base / "figures",
        "logs": base / "logs",
    }
    for path in result.values():
        path.mkdir(parents=True, exist_ok=True)
    return result
