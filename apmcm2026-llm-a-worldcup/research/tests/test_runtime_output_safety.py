import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from worldcup_forecasting.application.runtime import (
    OUTPUT_OWNERSHIP_FILE,
    OutputSafetyError,
    staged_output_directory,
    validate_output_target,
    write_csv,
)
from worldcup_forecasting.pipeline import run_pipeline
from worldcup_forecasting.validation.artifact_integrity import build_stage_manifest
from worldcup_forecasting.validation.runtime_reproducibility import (
    drop_live_runtime_columns,
    replace_runtime_with_reference,
)


def test_gzip_csv_is_byte_deterministic_across_paths(tmp_path: Path) -> None:
    frame = pd.DataFrame({"team": ["A", "B"], "probability": [0.25, 0.75]})
    first = write_csv(frame, tmp_path / "first.csv.gz")
    second = write_csv(frame, tmp_path / "second.csv.gz")

    assert first.read_bytes() == second.read_bytes()
    assert (
        hashlib.sha256(first.read_bytes()).hexdigest()
        == hashlib.sha256(second.read_bytes()).hexdigest()
    )
    pd.testing.assert_frame_equal(pd.read_csv(first), frame)


def test_live_wall_clock_cannot_change_scientific_table_or_manifest(tmp_path: Path) -> None:
    charges = {"candidate": 0.4}
    manifests = []
    sealed_frames = []
    for run, live_time in enumerate((0.017, 9.812)):
        observed = pd.DataFrame(
            {
                "candidate_id": ["candidate"],
                "runtime_seconds": [live_time],
                "measured_runtime_seconds": [live_time],
                "outer_fit_runtime_seconds": [live_time / 2],
                "log_loss": [0.9],
            }
        )
        sealed = replace_runtime_with_reference(
            drop_live_runtime_columns(observed),
            identity_column="candidate_id",
            charges=charges,
        )
        stage = tmp_path / f"run-{run}"
        write_csv(sealed, stage / "scientific.csv")
        manifests.append(build_stage_manifest(stage, ("scientific.csv",)))
        sealed_frames.append(sealed)

    pd.testing.assert_frame_equal(sealed_frames[0], sealed_frames[1])
    assert manifests[0] == manifests[1]


@pytest.fixture
def safety_root(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    for relative in (
        "data/raw",
        "src",
        "configs",
        "doc",
        "tests",
        "matlab",
        "plan",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    return root


@pytest.mark.parametrize(
    "relative",
    [
        ".",
        "data",
        "data/raw",
        "src",
        "configs",
        "doc",
        "tests",
        "matlab",
        "plan",
    ],
)
def test_protected_project_targets_are_rejected_before_mutation(
    safety_root: Path,
    relative: str,
) -> None:
    target = (safety_root / relative).resolve()
    marker = target / "safety-marker.txt" if target.is_dir() else None
    if marker is not None:
        marker.write_text("must survive", encoding="utf-8")

    with pytest.raises(OutputSafetyError):
        validate_output_target(safety_root, target, safety_root / "outputs")

    if marker is not None:
        assert marker.read_text(encoding="utf-8") == "must survive"
        marker.unlink()


def test_project_ancestor_is_rejected(safety_root: Path) -> None:
    with pytest.raises(OutputSafetyError):
        validate_output_target(
            safety_root,
            safety_root.parent,
            safety_root / "outputs",
        )


@pytest.mark.parametrize("target", [Path.home(), Path.home().parent, Path(Path.home().anchor)])
def test_broad_user_and_drive_targets_are_rejected(
    safety_root: Path,
    target: Path,
) -> None:
    with pytest.raises(OutputSafetyError):
        validate_output_target(safety_root, target, safety_root / "outputs")


def test_nonempty_custom_output_requires_matching_ownership_sentinel(
    safety_root: Path,
    tmp_path: Path,
) -> None:
    target = tmp_path / "unowned"
    target.mkdir()
    marker = target / "personal.txt"
    marker.write_text("do not delete", encoding="utf-8")

    with pytest.raises(OutputSafetyError, match="ownership sentinel"):
        with staged_output_directory(
            safety_root,
            target,
            safety_root / "outputs",
            clean=True,
        ):
            pass

    assert marker.read_text(encoding="utf-8") == "do not delete"


def test_caller_cannot_declare_an_arbitrary_legacy_default(
    safety_root: Path,
    tmp_path: Path,
) -> None:
    target = tmp_path / "fake-default"
    target.mkdir()
    marker = target / "personal.txt"
    marker.write_text("do not delete", encoding="utf-8")

    with pytest.raises(OutputSafetyError, match="sole legacy output exception"):
        validate_output_target(safety_root, target, target)

    assert marker.read_text(encoding="utf-8") == "do not delete"


def test_external_configured_data_paths_are_protected(
    safety_root: Path,
    tmp_path: Path,
) -> None:
    target = tmp_path / "external-output"
    external_raw = target / "raw"

    with pytest.raises(OutputSafetyError, match="protected project path"):
        validate_output_target(
            safety_root,
            target,
            safety_root / "outputs",
            protected_paths=[external_raw],
        )


def test_empty_custom_output_can_be_claimed_and_safely_recleaned(
    safety_root: Path,
    tmp_path: Path,
) -> None:
    target = tmp_path / "claimed"
    target.mkdir()

    with staged_output_directory(
        safety_root,
        target,
        safety_root / "outputs",
        clean=True,
    ) as workspace:
        (workspace.staging_dir / "first.txt").write_text("first", encoding="utf-8")
        workspace.promote()

    sentinel = json.loads((target / OUTPUT_OWNERSHIP_FILE).read_text(encoding="utf-8"))
    assert sentinel["owner"] == "worldcup_forecasting"
    assert (target / "first.txt").is_file()

    with staged_output_directory(
        safety_root,
        target,
        safety_root / "outputs",
        clean=True,
    ) as workspace:
        assert not (workspace.staging_dir / "first.txt").exists()
        (workspace.staging_dir / "second.txt").write_text("second", encoding="utf-8")
        workspace.promote()

    assert not (target / "first.txt").exists()
    assert (target / "second.txt").read_text(encoding="utf-8") == "second"


def test_tampered_sentinel_never_authorizes_cleanup(
    safety_root: Path,
    tmp_path: Path,
) -> None:
    target = tmp_path / "tampered"
    target.mkdir()
    (target / OUTPUT_OWNERSHIP_FILE).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "owner": "worldcup_forecasting",
                "project_root_sha256": "wrong-project",
            }
        ),
        encoding="utf-8",
    )
    marker = target / "personal.txt"
    marker.write_text("survives", encoding="utf-8")

    with pytest.raises(OutputSafetyError, match="invalid ownership sentinel"):
        validate_output_target(safety_root, target, safety_root / "outputs")

    assert marker.read_text(encoding="utf-8") == "survives"


def test_sentinel_is_bound_to_the_exact_output_path(
    safety_root: Path,
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    with staged_output_directory(
        safety_root,
        first,
        safety_root / "outputs",
        clean=True,
    ) as workspace:
        workspace.promote()

    second = tmp_path / "second"
    second.mkdir()
    (second / OUTPUT_OWNERSHIP_FILE).write_bytes((first / OUTPUT_OWNERSHIP_FILE).read_bytes())
    (second / "personal.txt").write_text("survives", encoding="utf-8")

    with pytest.raises(OutputSafetyError, match="invalid ownership sentinel"):
        validate_output_target(safety_root, second, safety_root / "outputs")

    assert (second / "personal.txt").read_text(encoding="utf-8") == "survives"


def test_legacy_default_output_preserves_matlab_only_during_clean_promotion(
    safety_root: Path,
) -> None:
    default_output = safety_root / "outputs"
    matlab = default_output / "matlab"
    matlab.mkdir(parents=True)
    (matlab / "candidate_metrics.csv").write_text("metric,value\nmae,1\n", encoding="utf-8")
    (default_output / "stale.txt").write_text("stale", encoding="utf-8")

    with staged_output_directory(
        safety_root,
        default_output,
        default_output,
        clean=True,
    ) as workspace:
        assert (workspace.staging_dir / "matlab" / "candidate_metrics.csv").is_file()
        assert not (workspace.staging_dir / "stale.txt").exists()
        (workspace.staging_dir / "fresh.txt").write_text("fresh", encoding="utf-8")
        workspace.promote()

    assert (default_output / "matlab" / "candidate_metrics.csv").is_file()
    assert not (default_output / "stale.txt").exists()
    assert (default_output / "fresh.txt").is_file()


def test_relocated_project_accepts_regular_stale_sentinel_only_at_default(
    safety_root: Path,
    tmp_path: Path,
) -> None:
    original_output = safety_root / "outputs"
    with staged_output_directory(
        safety_root,
        original_output,
        original_output,
        clean=True,
    ) as workspace:
        workspace.promote()

    relocated_root = tmp_path / "relocated-project"
    relocated_output = relocated_root / "outputs"
    relocated_output.mkdir(parents=True)
    (relocated_output / OUTPUT_OWNERSHIP_FILE).write_bytes(
        (original_output / OUTPUT_OWNERSHIP_FILE).read_bytes()
    )
    (relocated_output / "packaged-result.txt").write_text("portable", encoding="utf-8")

    assert (
        validate_output_target(relocated_root, relocated_output, relocated_output)
        == relocated_output.resolve()
    )


def test_preserved_matlab_link_is_rejected_without_following_it(
    safety_root: Path,
    tmp_path: Path,
) -> None:
    target = tmp_path / "owned"
    with staged_output_directory(
        safety_root,
        target,
        safety_root / "outputs",
        clean=True,
    ) as workspace:
        workspace.promote()

    external = tmp_path / "external-matlab"
    external.mkdir()
    evidence = external / "evidence.txt"
    evidence.write_text("external", encoding="utf-8")
    matlab_link = target / "matlab"
    try:
        matlab_link.symlink_to(external, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Directory symlinks are unavailable: {exc}")

    with pytest.raises(OutputSafetyError, match="reparse-point output"):
        with staged_output_directory(
            safety_root,
            target,
            safety_root / "outputs",
            clean=True,
        ):
            pass

    assert evidence.read_text(encoding="utf-8") == "external"


def test_generated_link_blocks_promotion_and_leaves_no_final_output(
    safety_root: Path,
    tmp_path: Path,
) -> None:
    target = tmp_path / "generated-link"
    external = tmp_path / "external.txt"
    external.write_text("external", encoding="utf-8")

    with pytest.raises(OutputSafetyError, match="reparse-point output"):
        with staged_output_directory(
            safety_root,
            target,
            safety_root / "outputs",
            clean=True,
        ) as workspace:
            link = workspace.staging_dir / "predictions" / "external.txt"
            try:
                link.symlink_to(external)
            except OSError as exc:
                pytest.skip(f"File symlinks are unavailable: {exc}")
            workspace.promote()

    assert not target.exists()
    assert external.read_text(encoding="utf-8") == "external"


def test_failed_staged_run_leaves_previous_output_untouched(
    safety_root: Path,
    tmp_path: Path,
) -> None:
    target = tmp_path / "transactional"

    with pytest.raises(RuntimeError, match="simulated failure"):
        with staged_output_directory(
            safety_root,
            target,
            safety_root / "outputs",
            clean=True,
        ) as workspace:
            (workspace.staging_dir / "manifest.json").write_text(
                '{"status":"complete"}',
                encoding="utf-8",
            )
            raise RuntimeError("simulated failure")

    assert not target.exists()
    assert not list(tmp_path.glob(".transactional.staging-*"))


def test_default_output_matlab_subtree_cannot_be_selected_as_override(safety_root: Path) -> None:
    with pytest.raises(OutputSafetyError):
        validate_output_target(
            safety_root,
            safety_root / "outputs" / "matlab",
            safety_root / "outputs",
        )


def test_pipeline_rejects_protected_override_before_touching_source(project_root: Path) -> None:
    source_file = project_root / "src" / "worldcup_forecasting" / "pipeline.py"
    before = source_file.read_bytes()

    with pytest.raises(OutputSafetyError):
        run_pipeline(
            project_root=project_root,
            config_name="official",
            output_override=project_root / "src",
        )

    assert source_file.read_bytes() == before


def test_configured_output_does_not_gain_legacy_status(
    project_root: Path,
    tmp_path: Path,
) -> None:
    target = tmp_path / "configured-custom"
    target.mkdir()
    marker = target / "personal.txt"
    marker.write_text("survives", encoding="utf-8")

    with pytest.raises(OutputSafetyError, match="ownership sentinel"):
        run_pipeline(
            project_root=project_root,
            config_name="official",
            overrides=[f"paths.output_dir={target.as_posix()}"],
        )

    assert marker.read_text(encoding="utf-8") == "survives"


def test_pipeline_rejects_output_overlap_with_external_raw_path(
    project_root: Path,
    tmp_path: Path,
) -> None:
    target = tmp_path / "external-data-container"
    external_raw = target / "raw"

    with pytest.raises(OutputSafetyError, match="protected project path"):
        run_pipeline(
            project_root=project_root,
            config_name="official",
            output_override=target,
            overrides=[f"paths.raw_dir={external_raw.as_posix()}"],
        )

    assert not target.exists()


def test_pipeline_failure_rolls_back_owned_output(
    project_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "owned-output"
    default_output = project_root / "outputs"
    with staged_output_directory(
        project_root,
        target,
        default_output,
        clean=True,
    ) as workspace:
        (workspace.staging_dir / "previous.txt").write_text("stable", encoding="utf-8")
        workspace.promote()

    def fail_after_staging(_: Path) -> None:
        raise RuntimeError("simulated canonical-data failure")

    monkeypatch.setattr(
        "worldcup_forecasting.pipeline.build_canonical_data",
        fail_after_staging,
    )
    with pytest.raises(RuntimeError, match="simulated canonical-data failure"):
        run_pipeline(
            project_root=project_root,
            config_name="official",
            output_override=target,
        )

    assert (target / "previous.txt").read_text(encoding="utf-8") == "stable"
    assert not (target / "manifest.json").exists()
    assert not list(tmp_path.glob(".owned-output.staging-*"))
