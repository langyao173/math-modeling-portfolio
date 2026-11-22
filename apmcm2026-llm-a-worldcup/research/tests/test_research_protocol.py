from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from worldcup_forecasting.application.research import (
    _development_point_in_time_ok,
    _joint_model,
    evaluate_promotion_gate,
)
from worldcup_forecasting.config import load_config
from worldcup_forecasting.pipeline import (
    _validate_deployment_contract,
    _validate_profile_isolation,
)
from worldcup_forecasting.validation.bootstrap import paired_fold_comparison


def _gate_config() -> SimpleNamespace:
    return SimpleNamespace(
        primary_metric="log_loss",
        minimum_fold_win_fraction=0.5,
        accuracy_non_degradation_tolerance=0.02,
        calibration_non_degradation_tolerance=0.02,
        max_runtime_seconds_per_fold=10.0,
        max_probability_invariant_error=1e-10,
        max_determinism_error=1e-12,
    )


def _gate_table(*, include_2014: bool = False) -> pd.DataFrame:
    years = [1986, 1990, 1994, 1998, 2002, 2006, 2010]
    if include_2014:
        years[-1] = 2014
    candidate_scores = [0.9, 0.9, 0.9, 0.9, 1.1, 1.1, 1.1]
    rows = []
    for year, candidate_score in zip(years, candidate_scores, strict=True):
        rows.extend(
            [
                {
                    "test_year": year,
                    "model": "candidate",
                    "log_loss": candidate_score,
                    "accuracy": 0.55,
                    "calibration_error": 0.08,
                    "runtime_seconds": 0.2,
                },
                {
                    "test_year": year,
                    "model": "baseline",
                    "log_loss": 1.0,
                    "accuracy": 0.55,
                    "calibration_error": 0.08,
                    "runtime_seconds": 0.1,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_final_profile_does_not_inherit_research(project_root: Path) -> None:
    final = load_config(project_root, "final")
    research = load_config(project_root, "research")

    assert "research" not in final
    assert final.project.track == "final_frozen_closed"
    assert {
        "enable_dynamic_features",
        "enable_player_history",
        "enable_dixon_coles",
        "promotion_gate",
    }.issubset(set(research.research))
    _validate_profile_isolation(final)

    contaminated = load_config(
        project_root,
        "final",
        ["+research.enable_dynamic_features=true"],
    )
    with pytest.raises(ValueError, match="must not reference research"):
        _validate_profile_isolation(contaminated)


def test_final_deployment_metadata_cannot_disagree_with_executable_model(
    project_root: Path,
) -> None:
    final = load_config(project_root, "final")
    _validate_deployment_contract(final)

    inconsistent = load_config(
        project_root,
        "final",
        ["deployment.selected_model=outcome_logit"],
    )
    with pytest.raises(ValueError, match="not executable"):
        _validate_deployment_contract(inconsistent)


def test_promotion_gate_requires_majority_and_excludes_frozen_2014() -> None:
    passed = evaluate_promotion_gate(
        _gate_table(),
        _gate_config(),
        candidate="candidate",
        baseline="baseline",
        simplex_error=0.0,
        symmetry_error_value=0.0,
        determinism_error=0.0,
        point_in_time_ok=True,
        maximum_selection_year=2010,
    )
    assert passed["passed"] is True
    assert passed["checks"]["strict_majority_of_folds_improved"] is True
    assert passed["checks"]["frozen_2014_excluded_from_selection"] is True

    contaminated = evaluate_promotion_gate(
        _gate_table(include_2014=True),
        _gate_config(),
        candidate="candidate",
        baseline="baseline",
        simplex_error=0.0,
        symmetry_error_value=0.0,
        determinism_error=0.0,
        point_in_time_ok=True,
        maximum_selection_year=2010,
    )
    assert contaminated["passed"] is False
    assert contaminated["checks"]["frozen_2014_excluded_from_selection"] is False


def test_joint_candidate_toggles_change_model_contract(project_root: Path) -> None:
    enabled = load_config(project_root, "research")
    disabled = load_config(
        project_root,
        "research",
        ["research.enable_dynamic_features=false", "research.enable_dixon_coles=false"],
    )

    enabled_model = _joint_model(enabled)
    disabled_model = _joint_model(disabled)
    assert enabled_model.config.optional_diff_features == (
        "pre_match_elo_diff",
        "attack_state_diff",
        "defense_state_diff",
    )
    assert enabled_model.config.rho is None
    assert disabled_model.config.optional_diff_features == ()
    assert disabled_model.config.rho == 0.0


def test_frozen_2014_invariant_cannot_change_development_gate() -> None:
    frame = pd.DataFrame(
        {
            "Year": [2006, 2010, 2014],
            "history_cutoff_year": [2002, 2006, 2014],
        }
    )
    assert _development_point_in_time_ok(frame, maximum_year=2010) is True

    frame.loc[frame["Year"].eq(2010), "history_cutoff_year"] = 2010
    assert _development_point_in_time_ok(frame, maximum_year=2010) is False


def test_rank_biserial_uses_signed_ranks_not_sign_counts() -> None:
    rows = []
    for year, candidate, baseline in zip(
        [1998, 2002, 2006], [0.0, 0.0, 11.0], [1.0, 2.0, 1.0], strict=True
    ):
        rows.extend(
            [
                {"test_year": year, "model": "candidate", "score": candidate},
                {"test_year": year, "model": "baseline", "score": baseline},
            ]
        )
    result = paired_fold_comparison(
        pd.DataFrame(rows), "candidate", "baseline", "score", lower_is_better=True
    )

    assert result["folds_improved"] == 2
    assert result["rank_biserial_directional"] == pytest.approx(0.0)
