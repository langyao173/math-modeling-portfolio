from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def raw_dir(project_root: Path) -> Path:
    return project_root / "data" / "raw"
