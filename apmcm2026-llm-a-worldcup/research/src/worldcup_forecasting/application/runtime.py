"""Deterministic runtime helpers with ownership-gated output transactions."""

from __future__ import annotations

import gzip
import io
import json
import os
import shutil
import stat
import tempfile
import warnings
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

REQUIRED_ARTIFACTS = {
    "tables/problem1_descriptive_statistics.csv",
    "tables/problem1_2018_prediction.csv",
    "tables/problem2_stage_statistics.csv",
    "tables/problem2_stage_goal_statistics.csv",
    "tables/problem2_metrics.csv",
    "tables/problem2_metrics_metadata.json",
    "tables/problem2_half_time_interval_diagnostics.json",
    "tables/problem3_metrics.csv",
    "tables/problem3_high_correlation_summary.json",
    "figures/problem1_total_goals_trend.png",
    "figures/problem1_goals_per_match_trend.png",
    "figures/problem2_halftime_fulltime_scatter.png",
    "figures/problem3_feature_correlation_heatmap.png",
    "figures/problem3_confusion_matrix.png",
    "predictions/2026_half_time_demo.json",
    "predictions/2026_outcome_demo.json",
    "predictions/2026_joint_goal_demo.json",
    "predictions/problem1_2018_total_goals.json",
    "predictions/problem2_2026_scenario.json",
    "predictions/problem3_2026_scenario.json",
    "certificates/problem1_2018_certificate.json",
    "certificates/problem2_2026_certificate.json",
    "certificates/problem3_2026_certificate.json",
    "certificates/2026_half_time_demo.json",
    "certificates/2026_outcome_demo.json",
    "certificates/2026_joint_goal_demo.json",
    "models/problem1_aggregate_ols.json",
    "models/problem1_loocv_interval_calibration.json",
    "models/problem2_match_goal_ols_2014.json",
    "models/problem2_half_time_increment_2014.json",
    "models/problem2_half_time_oof_calibration.json",
    "models/problem3_outcome_logit_2014.json",
    "models/problem3_joint_poisson_static_2014.json",
    "audit/schema_report.json",
    "audit/data_lineage.json",
    "audit/missingness.csv",
    "audit/duplicate_report.csv",
    "audit/team_name_registry.csv",
    "audit/feature_availability_registry.csv",
    "audit/feature_availability_report.json",
    "logs/pipeline.log",
    "logs/problem1_regression.txt",
    "logs/run.jsonl",
}

RESEARCH_REQUIRED_ARTIFACTS = {
    "evidence/advanced/ablation/artifact_manifest.json",
    "evidence/advanced/audit/audit_metrics.csv",
    "evidence/advanced/audit/audit_result_certificate.json",
    "evidence/advanced/audit/audit_state.json",
    "evidence/advanced/evaluation/artifact_manifest.json",
    "evidence/advanced/nested/artifact_manifest.json",
    "evidence/advanced/promotion/promotion_certificate.json",
    "evidence/advanced/simulation/artifact_manifest.json",
    "evidence/advanced/stress/artifact_manifest.json",
    "evidence/advanced/uncertainty/decomposition_artifact_manifest.json",
    "evidence/advanced/uncertainty/parameter_artifact_manifest.json",
    "tables/research_advanced_summary.json",
    "tables/research_ablation_matrix.csv",
    "tables/research_calibration.json",
    "tables/research_cv_metrics.csv",
    "tables/research_frozen_2014.csv",
    "tables/research_paired_comparisons.csv",
    "tables/research_promotion_gate.csv",
    "tables/research_promotion_gate.json",
    "tables/research_stress_tests.json",
}

OUTPUT_OWNERSHIP_FILE = ".worldcup_forecasting-output.json"
_OUTPUT_OWNER = "worldcup_forecasting"
_OUTPUT_SCHEMA_VERSION = 1
_OUTPUT_SUBDIRECTORIES = (
    "analysis",
    "audit",
    "certificates",
    "figures",
    "logs",
    "models",
    "predictions",
    "reports",
    "tables",
)
_PROTECTED_PROJECT_CHILDREN = (
    ".git",
    ".venv",
    "config",
    "configs",
    "data",
    "doc",
    "matlab",
    "plan",
    "src",
    "tests",
)


class OutputSafetyError(ValueError):
    """Raised before an unsafe or unowned output target can be mutated."""


@dataclass
class OutputWorkspace:
    """A sibling staging directory for rollback-capable two-phase promotion."""

    project_root: Path
    final_dir: Path
    default_output_dir: Path
    staging_dir: Path
    configured_protected_paths: tuple[Path, ...] = ()
    promoted: bool = False

    def promote(self) -> Path:
        """Replace the authorized final directory only after generation succeeds."""
        if self.promoted:
            raise RuntimeError("Output workspace has already been promoted")
        assert_output_tree_has_no_links(self.staging_dir)
        _assert_owned_directory(self.project_root, self.staging_dir, self.final_dir)
        validate_output_target(
            self.project_root,
            self.final_dir,
            self.default_output_dir,
            protected_paths=self.configured_protected_paths,
        )

        backup: Path | None = None
        if self.final_dir.exists():
            backup = self.final_dir.parent / (f".{self.final_dir.name}.backup-{uuid4().hex}")
            self.final_dir.rename(backup)
        try:
            self.staging_dir.rename(self.final_dir)
        except BaseException:
            if backup is not None and backup.exists() and not self.final_dir.exists():
                backup.rename(self.final_dir)
            raise

        self.promoted = True
        if backup is not None and backup.exists():
            try:
                _remove_path(backup)
            except OSError as exc:  # pragma: no cover - platform/open-handle dependent
                warnings.warn(
                    f"Promoted output but could not remove prior backup {backup}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return self.final_dir


def validate_output_target(
    project_root: Path,
    output_dir: Path,
    default_output_dir: Path,
    *,
    protected_paths: Iterable[Path] = (),
) -> Path:
    """Validate path scope and ownership without modifying the filesystem.

    A new or empty directory is claimable. A nonempty custom directory must
    contain the exact project-bound ownership sentinel. The canonical default
    ``project/outputs`` directory is the sole legacy migration exception.
    """
    root = project_root.expanduser().resolve()
    output = output_dir.expanduser().resolve()
    default = (root / "outputs").resolve()
    supplied_default = default_output_dir.expanduser().resolve()
    if supplied_default != default:
        raise OutputSafetyError(
            "The sole legacy output exception is the canonical project/outputs directory; "
            f"received {supplied_default}"
        )

    _reject_broad_or_protected_target(root, output, default, protected_paths)
    if output.exists() and not output.is_dir():
        raise OutputSafetyError(f"Output target exists but is not a directory: {output}")
    if not output.exists():
        return output

    entries = list(output.iterdir())
    if not entries:
        return output

    sentinel_path = output / OUTPUT_OWNERSHIP_FILE
    if sentinel_path.exists():
        if output == default:
            if _is_link_or_reparse(sentinel_path) or not sentinel_path.is_file():
                raise OutputSafetyError(f"Refusing unsafe ownership sentinel: {sentinel_path}")
            # The canonical in-project output is intentionally relocatable with a
            # packaged project. Its old regular sentinel may name the extraction
            # source, while the fixed root/outputs boundary remains authorized.
            return output
        if (
            _is_link_or_reparse(sentinel_path)
            or not sentinel_path.is_file()
            or not _sentinel_matches(root, output, sentinel_path)
        ):
            raise OutputSafetyError(f"Refusing invalid ownership sentinel: {sentinel_path}")
        return output

    if output == default:
        return output
    raise OutputSafetyError(
        "Refusing to clean a nonempty custom output directory without the matching "
        f"ownership sentinel {OUTPUT_OWNERSHIP_FILE}: {output}"
    )


@contextmanager
def staged_output_directory(
    project_root: Path,
    output_dir: Path,
    default_output_dir: Path,
    *,
    clean: bool,
    preserve_matlab: bool = True,
    protected_paths: Iterable[Path] = (),
) -> Iterator[OutputWorkspace]:
    """Generate under a sibling staging tree and promote only on request.

    The caller must explicitly call :meth:`OutputWorkspace.promote` after all
    artifact gates and the complete manifest succeed. Exceptions remove the
    owned staging tree and leave the previous final directory untouched.
    """
    root = project_root.expanduser().resolve()
    configured_protected = tuple(path.expanduser().resolve() for path in protected_paths)
    final = validate_output_target(
        root,
        output_dir,
        default_output_dir,
        protected_paths=configured_protected,
    )
    if final.exists():
        assert_output_tree_has_no_links(final)
    default = (root / "outputs").resolve()
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{final.name}.staging-", dir=final.parent)).resolve()
    workspace = OutputWorkspace(
        root,
        final,
        default,
        staging,
        configured_protected,
    )
    try:
        if final.exists():
            _copy_previous_output(
                final,
                staging,
                clean=clean,
                preserve_matlab=preserve_matlab,
            )
        old_manifest = staging / "manifest.json"
        if old_manifest.exists() or old_manifest.is_symlink():
            _remove_path(old_manifest)
        _write_ownership_sentinel(root, staging, final)
        _create_output_subdirectories(staging)
        yield workspace
    finally:
        if not workspace.promoted and staging.exists():
            _remove_path(staging)


def prepare_output_directory(
    output_dir: Path,
    *,
    project_root: Path,
    default_output_dir: Path | None = None,
    preserve_matlab: bool = True,
    protected_paths: Iterable[Path] = (),
) -> Path:
    """Directly clean an authorized exact directory.

    The one-click pipeline uses :func:`staged_output_directory` instead. This
    lower-level helper remains for callers that explicitly accept non-atomic
    cleanup, but it cannot be called without the project safety boundary.
    """
    root = project_root.expanduser().resolve()
    default = (default_output_dir or root / "outputs").expanduser().resolve()
    resolved = validate_output_target(
        root,
        output_dir,
        default,
        protected_paths=protected_paths,
    )
    if resolved.exists():
        assert_output_tree_has_no_links(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    _write_ownership_sentinel(root, resolved, resolved)
    for child in list(resolved.iterdir()):
        if child.name == OUTPUT_OWNERSHIP_FILE:
            continue
        if preserve_matlab and child.name == "matlab":
            _assert_regular_tree(child)
            continue
        _remove_path(child)
    _create_output_subdirectories(resolved)
    return resolved


def write_csv(frame: pd.DataFrame, path: Path) -> Path:
    """Write one stable UTF-8 CSV with no implicit index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.casefold() == ".gz":
        # pandas/gzip otherwise embeds the destination basename in the header.
        # Keep both the name and timestamp fields fixed so the same table is
        # byte-identical across clean-room extraction paths.
        with path.open("wb") as raw_handle:
            with gzip.GzipFile(
                filename="",
                fileobj=raw_handle,
                mode="wb",
                compresslevel=9,
                mtime=0,
            ) as compressed_handle:
                with io.TextIOWrapper(
                    compressed_handle,
                    encoding="utf-8",
                    newline="",
                ) as text_handle:
                    frame.to_csv(
                        text_handle,
                        index=False,
                        lineterminator="\n",
                    )
    else:
        frame.to_csv(
            path,
            index=False,
            lineterminator="\n",
            encoding="utf-8",
        )
    return path


def write_json(payload: Any, path: Path) -> Path:
    """Write JSON with NumPy/pandas scalars normalized."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return path


def assert_output_tree_has_no_links(directory: Path) -> None:
    """Reject links, junctions, reparse points, and non-regular artifacts."""
    resolved = directory.resolve()
    if not resolved.is_dir():
        raise OutputSafetyError(f"Output workspace is not a directory: {resolved}")
    for child in resolved.iterdir():
        _assert_regular_tree(child)


def _reject_broad_or_protected_target(
    root: Path,
    output: Path,
    default: Path,
    configured_protected_paths: Iterable[Path],
) -> None:
    drive_root = Path(output.anchor).resolve()
    home = Path.home().resolve()
    if output == drive_root or output == home or _is_ancestor(output, home):
        raise OutputSafetyError(f"Refusing broad output target: {output}")
    if output == root or _is_ancestor(output, root):
        raise OutputSafetyError(f"Output target overlaps the project root: {output}")

    protected = [root / child for child in _PROTECTED_PROJECT_CHILDREN]
    protected.extend(path.expanduser().resolve() for path in configured_protected_paths)
    for path in protected:
        if _paths_overlap(output, path.resolve()):
            raise OutputSafetyError(
                f"Output target overlaps protected project path {path}: {output}"
            )

    preserved_matlab = (default / "matlab").resolve()
    if output != default and _paths_overlap(output, preserved_matlab):
        raise OutputSafetyError(
            f"Output target overlaps the preserved MATLAB evidence path: {output}"
        )


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or _is_ancestor(left, right) or _is_ancestor(right, left)


def _is_ancestor(candidate: Path, descendant: Path) -> bool:
    return candidate != descendant and descendant.is_relative_to(candidate)


def _normalized_path_hash(path: Path) -> str:
    normalized = os.path.normcase(str(path.resolve())).replace("\\", "/")
    return sha256(normalized.encode("utf-8")).hexdigest()


def _sentinel_payload(project_root: Path, owned_output_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": _OUTPUT_SCHEMA_VERSION,
        "owner": _OUTPUT_OWNER,
        "project_root_sha256": _normalized_path_hash(project_root),
        "owned_output_sha256": _normalized_path_hash(owned_output_dir),
    }


def _sentinel_matches(project_root: Path, owned_output_dir: Path, path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return payload == _sentinel_payload(project_root, owned_output_dir)


def _write_ownership_sentinel(
    project_root: Path,
    directory: Path,
    owned_output_dir: Path,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / OUTPUT_OWNERSHIP_FILE
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{OUTPUT_OWNERSHIP_FILE}.",
        suffix=".tmp",
        dir=directory,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                _sentinel_payload(project_root, owned_output_dir),
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
            )
            stream.write("\n")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _assert_owned_directory(
    project_root: Path,
    directory: Path,
    owned_output_dir: Path,
) -> None:
    sentinel = directory / OUTPUT_OWNERSHIP_FILE
    if (
        _is_link_or_reparse(sentinel)
        or not sentinel.is_file()
        or not _sentinel_matches(project_root, owned_output_dir, sentinel)
    ):
        raise OutputSafetyError(f"Staging directory lost its ownership sentinel: {directory}")


def _copy_previous_output(
    source: Path,
    target: Path,
    *,
    clean: bool,
    preserve_matlab: bool,
) -> None:
    for child in source.iterdir():
        if child.name in {OUTPUT_OWNERSHIP_FILE, "manifest.json"}:
            continue
        if clean and not (preserve_matlab and child.name == "matlab"):
            continue
        destination = target / child.name
        _assert_regular_tree(child)
        if child.is_dir():
            shutil.copytree(child, destination, symlinks=False)
        else:
            shutil.copy2(child, destination)


def _create_output_subdirectories(directory: Path) -> None:
    for name in _OUTPUT_SUBDIRECTORIES:
        (directory / name).mkdir(parents=True, exist_ok=True)


def _remove_path(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
    elif _is_junction(path) or _is_reparse_point(path):
        if path.is_dir():
            path.rmdir()
        else:
            path.unlink()
    elif path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _assert_regular_tree(path: Path) -> None:
    if _is_link_or_reparse(path):
        raise OutputSafetyError(f"Linked or reparse-point output is forbidden: {path}")
    if path.is_dir():
        for child in path.iterdir():
            _assert_regular_tree(child)
    elif not path.is_file():
        raise OutputSafetyError(f"Non-regular output artifact is forbidden: {path}")


def _is_link_or_reparse(path: Path) -> bool:
    return path.is_symlink() or _is_junction(path) or _is_reparse_point(path)


def _is_junction(path: Path) -> bool:
    predicate = getattr(path, "is_junction", None)
    return bool(predicate and predicate())


def _is_reparse_point(path: Path) -> bool:
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError:
        return False
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(flag and attributes & flag)


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set | tuple):
        return list(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")
