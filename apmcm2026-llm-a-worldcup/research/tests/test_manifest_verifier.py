import json
from hashlib import sha256
from pathlib import Path

import pytest

from worldcup_forecasting.uncertainty.certificates import (
    build_prediction_certificate,
    write_certificate,
)
from worldcup_forecasting.verify import ManifestVerificationError, verify_manifest


def test_manifest_verifier_accepts_exact_inventory(tmp_path: Path) -> None:
    artifact = tmp_path / "tables" / "metrics.csv"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("metric,value\nmae,1.2\n", encoding="utf-8")
    payload = {
        "status": "complete",
        "required_paths": ["tables/metrics.csv"],
        "relative_paths": ["tables/metrics.csv"],
        "artifacts": [
            {
                "path": "tables/metrics.csv",
                "bytes": artifact.stat().st_size,
                "sha256": sha256(artifact.read_bytes()).hexdigest(),
            }
        ],
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    summary = verify_manifest(manifest, required_paths={"tables/metrics.csv"})

    assert summary["verified_artifacts"] == 1
    assert summary["total_bytes"] == artifact.stat().st_size


def test_manifest_verifier_rejects_tampered_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "result.txt"
    artifact.write_text("original", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "artifact_profile": "official",
                "run": {"config_name": "official", "track": "closed"},
                "required_paths": ["result.txt"],
                "relative_paths": ["result.txt"],
                "artifacts": [
                    {
                        "path": "result.txt",
                        "bytes": artifact.stat().st_size,
                        "sha256": sha256(artifact.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    artifact.write_text("tampered", encoding="utf-8")

    with pytest.raises(ManifestVerificationError, match="mismatch"):
        verify_manifest(manifest, required_paths={"result.txt"})


def test_manifest_verifier_rejects_weakened_required_contract(tmp_path: Path) -> None:
    artifact = tmp_path / "result.txt"
    artifact.write_text("result", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "artifact_profile": "official",
                "run": {"config_name": "official", "track": "closed"},
                "required_paths": ["result.txt"],
                "relative_paths": ["result.txt"],
                "artifacts": [
                    {
                        "path": "result.txt",
                        "bytes": artifact.stat().st_size,
                        "sha256": sha256(artifact.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ManifestVerificationError, match="required-artifact contract"):
        verify_manifest(manifest)


@pytest.mark.parametrize(
    ("profile", "config_name", "track", "message"),
    [
        (None, "official", "closed", "artifact_profile"),
        ("official", "research", "closed_research", "disagrees"),
        ("research", "final", "final_frozen_closed", "disagrees"),
    ],
)
def test_manifest_verifier_rejects_missing_or_mismatched_profile(
    tmp_path: Path,
    profile: str | None,
    config_name: str,
    track: str,
    message: str,
) -> None:
    artifact = tmp_path / "result.txt"
    artifact.write_text("result", encoding="utf-8")
    payload = {
        "status": "complete",
        "run": {"config_name": config_name, "track": track},
        "required_paths": ["result.txt"],
        "relative_paths": ["result.txt"],
        "artifacts": [
            {
                "path": "result.txt",
                "bytes": artifact.stat().st_size,
                "sha256": sha256(artifact.read_bytes()).hexdigest(),
            }
        ],
    }
    if profile is not None:
        payload["artifact_profile"] = profile
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestVerificationError, match=message):
        verify_manifest(manifest)


def test_manifest_verifier_rejects_certificate_model_hash_mismatch(tmp_path: Path) -> None:
    model = tmp_path / "models" / "model.json"
    model.parent.mkdir(parents=True)
    model.write_text('{"coefficient":1}\n', encoding="utf-8")
    certificate = build_prediction_certificate(
        information_set="pre_match",
        data_cutoff="2014-07-13",
        raw_data_hashes={"matches": "a" * 64},
        feature_schema=["x"],
        model_hash="0" * 64,
        model_artifact="models/model.json",
        config_hash="b" * 64,
        random_seed=1729,
        software_versions={"python": "3.12"},
        calibration_method="none",
        calibration_artifact=None,
        prediction={"home_win": 0.5, "draw": 0.25, "away_win": 0.25},
        goal_interval_90=None,
        out_of_distribution_score=0.0,
        invariant_status={"probability_simplex": True},
        scenario_label="test",
    )
    certificate_path = tmp_path / "certificates" / "prediction.json"
    write_certificate(certificate, certificate_path)
    artifacts = [model, certificate_path]
    records = [
        {
            "path": artifact.relative_to(tmp_path).as_posix(),
            "bytes": artifact.stat().st_size,
            "sha256": sha256(artifact.read_bytes()).hexdigest(),
        }
        for artifact in artifacts
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "resolved_config_sha256": "b" * 64,
                "required_paths": [record["path"] for record in records],
                "relative_paths": [record["path"] for record in records],
                "artifacts": records,
            }
        ),
        encoding="utf-8",
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["resolved_config_sha256"] = "c" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ManifestVerificationError, match="config hash mismatch"):
        verify_manifest(
            manifest,
            required_paths={record["path"] for record in records},
        )

    payload["resolved_config_sha256"] = "b" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ManifestVerificationError, match="model hash mismatch"):
        verify_manifest(
            manifest,
            required_paths={record["path"] for record in records},
        )
