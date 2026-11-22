import logging
from pathlib import Path

import pytest

from worldcup_forecasting.data import build_canonical_data
from worldcup_forecasting.experiments.matlab_bridge import (
    verify_advanced_matlab_artifacts,
    verify_matlab_parity,
)
from worldcup_forecasting.pipeline import _validate_optional_matlab


@pytest.mark.integration
def test_retained_matlab_experiment_matches_python_oracles(
    project_root: Path,
    raw_dir: Path,
) -> None:
    matlab_dir = project_root / "outputs" / "matlab"
    if not (matlab_dir / "parity_summary.json").is_file():
        pytest.skip("Optional MATLAB R2026a experiment is not retained in this package")
    bundle = build_canonical_data(raw_dir)

    report = verify_matlab_parity(bundle, matlab_dir)

    assert report["status"] == "passed"
    assert all(check["passed"] for check in report["checks"])
    assert report["maximum_absolute_numeric_error"] <= 1e-10


def test_partial_retained_matlab_evidence_fails_closed(tmp_path: Path) -> None:
    matlab = tmp_path / "matlab"
    matlab.mkdir()
    (matlab / "candidate_metrics.csv").write_text("metric,value\nmae,1\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="incomplete"):
        _validate_optional_matlab(object(), tmp_path, logging.getLogger("matlab-test"))


def test_advanced_matlab_prototype_is_complete_and_independent(project_root: Path) -> None:
    report = verify_advanced_matlab_artifacts(project_root / "artifacts" / "matlab")

    assert report["status"] == "passed"
    assert report["outer_years"] == [1986, 1990, 1994, 1998, 2002, 2006, 2010]
    assert report["production_dependency"] is False
    assert report["maximum_invariant_error"] <= 1e-10
    assert report["rate_product_range"] > 0
    assert report["claim_scope"] == "independent prototype; not production evidence"


def test_advanced_matlab_prototype_rejects_claim_or_audit_contamination(tmp_path: Path) -> None:
    matlab = tmp_path / "matlab"
    matlab.mkdir()
    (matlab / "advanced_prototype_metrics.csv").write_text(
        "test_year,log_loss,multiclass_brier,accuracy,n\n2014,1,0.6,0.5,64\n",
        encoding="utf-8",
    )
    (matlab / "advanced_theory_invariants.json").write_text(
        '{"matlab_release":"2026a","swap_error":0,"rate_product_range":1,'
        '"kl_margin_error":0,"kl_swap_error":0,"all_passed":true,'
        '"claim_scope":"production evidence"}\n',
        encoding="utf-8",
    )
    (matlab / "advanced_experiment_card.json").write_text(
        '{"experiment_id":"x","runtime_seconds":1,"production_dependency":true}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="2014|production|claim|year"):
        verify_advanced_matlab_artifacts(matlab)
