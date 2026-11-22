import importlib.util
import json
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import pytest

from worldcup_forecasting.verify import verify_manifest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_release.py"
_SPEC = importlib.util.spec_from_file_location("worldcup_release_builder", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
build_release = _MODULE.build_release


@pytest.mark.integration
def test_release_builder_is_deterministic_and_crc_clean(
    project_root: Path,
    tmp_path: Path,
) -> None:
    first = build_release(project_root, tmp_path / "first")
    second = build_release(project_root, tmp_path / "second")

    assert first["zip_sha256"] == second["zip_sha256"]
    assert first["source_file_count"] == second["source_file_count"]
    with ZipFile(Path(str(first["zip_path"]))) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert "worldcup_forecasting/PACKAGE_MANIFEST.json" in names
        assert "worldcup_forecasting/MANIFEST.json" in names
        assert "worldcup_forecasting/checksums.sha256" in names
        assert "worldcup_forecasting/requirements.txt" in names
        assert "worldcup_forecasting/PAPER.pdf" in names
        assert "worldcup_forecasting/MODEL_CARD.md" in names
        assert "worldcup_forecasting/REPRODUCIBILITY.md" in names
        assert "worldcup_forecasting/AUDIT_REPORT.md" in names
        assert any("temp/advanced_evidence/audit/" in name for name in names)
        assert "worldcup_forecasting/.venv" not in names
        assert not any("__pycache__" in name for name in names)
        assert not any(name.endswith(".aux") for name in names)
        archive.extractall(tmp_path / "extracted")

    extracted = tmp_path / "extracted" / "worldcup_forecasting"
    verify_manifest(extracted / "outputs" / "manifest.json")
    verify_manifest(extracted / "research_outputs" / "manifest.json")
    package_manifest = json.loads((extracted / "MANIFEST.json").read_text(encoding="utf-8"))
    assert package_manifest["paper_manifest_verification"]["artifact_count"] == 33
    assert package_manifest["cached_audit_verification"]["target_reads"] == 1
    checksums = (extracted / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    for line in checksums:
        expected, relative = line.split("  ", maxsplit=1)
        assert sha256((extracted / relative).read_bytes()).hexdigest() == expected
