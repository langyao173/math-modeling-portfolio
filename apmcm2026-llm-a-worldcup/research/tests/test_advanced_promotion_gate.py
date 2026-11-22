import pandas as pd
import pytest

from worldcup_forecasting.validation.promotion import evaluate_advanced_promotion


def _folds() -> pd.DataFrame:
    rows = []
    for index, year in enumerate([1986, 1990, 1994, 1998, 2002, 2006, 2010]):
        rows.extend(
            [
                {
                    "test_year": year,
                    "model": "advanced",
                    "log_loss": 0.90 if index < 4 else 1.03,
                    "multiclass_brier": 0.56 if index < 4 else 0.61,
                },
                {
                    "test_year": year,
                    "model": "legacy",
                    "log_loss": 1.00,
                    "multiclass_brier": 0.60,
                },
            ]
        )
    return pd.DataFrame(rows)


def _decision() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "test_year": [1986, 1990, 1994, 1998, 2002, 2006, 2010],
            "positive_draw_predictions": [1, 2, 1, 1, 0, 0, 0],
            "test_match_count": [52, 52, 52, 64, 64, 64, 64],
            "macro_f1_delta": [0.01, 0.02, 0.01, 0.01, -0.01, -0.01, -0.01],
            "draw_recall_delta": [0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0],
        }
    )


def test_advanced_promotion_requires_empirical_and_structural_gates() -> None:
    result = evaluate_advanced_promotion(
        _folds(),
        _decision(),
        candidate="advanced",
        baseline="legacy",
        model_identity_hash="a" * 64,
        artifact_identity_hash="a" * 64,
        structural_checks={"simplex": True, "symmetry": True, "leakage": True},
    )

    assert result["probability_candidate_passed"] is True
    assert result["decision_layer_passed"] is True
    assert result["system_promoted"] is True
    assert result["metric_evidence"]["log_loss"]["fold_wins"] == 4
    assert result["checks"]["frozen_2014_excluded"] is True


def test_advanced_promotion_fails_on_identity_mismatch_or_2014() -> None:
    contaminated = _folds()
    contaminated.loc[contaminated["test_year"].eq(2010), "test_year"] = 2014
    with pytest.raises(ValueError, match="exact integer editions"):
        evaluate_advanced_promotion(
            contaminated,
            _decision(),
            candidate="advanced",
            baseline="legacy",
            model_identity_hash="a" * 64,
            artifact_identity_hash="b" * 64,
            structural_checks={"simplex": True, "symmetry": True, "leakage": True},
        )

    result = evaluate_advanced_promotion(
        _folds(),
        _decision(),
        candidate="advanced",
        baseline="legacy",
        model_identity_hash="a" * 64,
        artifact_identity_hash="b" * 64,
        structural_checks={"simplex": True, "symmetry": True, "leakage": True},
    )

    assert result["system_promoted"] is False
    assert result["checks"]["exact_model_identity"] is False
    assert result["checks"]["frozen_2014_excluded"] is True


def test_advanced_promotion_rejects_missing_pairs() -> None:
    with pytest.raises(ValueError, match="seven paired"):
        evaluate_advanced_promotion(
            _folds().loc[lambda frame: frame["test_year"].ne(1986)],
            _decision(),
            candidate="advanced",
            baseline="legacy",
            model_identity_hash="a" * 64,
            artifact_identity_hash="a" * 64,
            structural_checks={"simplex": True},
        )


def test_advanced_promotion_requires_all_frozen_structural_checks() -> None:
    with pytest.raises(ValueError, match="structural.*symmetry|missing.*symmetry"):
        evaluate_advanced_promotion(
            _folds(),
            _decision(),
            candidate="advanced",
            baseline="legacy",
            model_identity_hash="a" * 64,
            artifact_identity_hash="a" * 64,
            structural_checks={"simplex": True, "leakage": True},
        )


def test_advanced_promotion_rejects_impossible_metric_values() -> None:
    negative_loss = _folds()
    negative_loss.loc[negative_loss.index[0], "log_loss"] = -0.1
    with pytest.raises(ValueError, match="nonnegative|log_loss"):
        evaluate_advanced_promotion(
            negative_loss,
            _decision(),
            candidate="advanced",
            baseline="legacy",
            model_identity_hash="a" * 64,
            artifact_identity_hash="a" * 64,
            structural_checks={"simplex": True, "symmetry": True, "leakage": True},
        )

    fractional_count = _decision().astype({"positive_draw_predictions": float})
    fractional_count.loc[0, "positive_draw_predictions"] = 0.5
    with pytest.raises(ValueError, match="integer|count"):
        evaluate_advanced_promotion(
            _folds(),
            fractional_count,
            candidate="advanced",
            baseline="legacy",
            model_identity_hash="a" * 64,
            artifact_identity_hash="a" * 64,
            structural_checks={"simplex": True, "symmetry": True, "leakage": True},
        )

    impossible_delta = _decision()
    impossible_delta.loc[0, "macro_f1_delta"] = 2.0
    with pytest.raises(ValueError, match=r"\[-1, 1\]|delta"):
        evaluate_advanced_promotion(
            _folds(),
            impossible_delta,
            candidate="advanced",
            baseline="legacy",
            model_identity_hash="a" * 64,
            artifact_identity_hash="a" * 64,
            structural_checks={"simplex": True, "symmetry": True, "leakage": True},
        )


def test_advanced_promotion_rejects_fractional_fold_years_and_malformed_hashes() -> None:
    fractional = _folds().astype({"test_year": float})
    fractional.loc[fractional["test_year"].eq(1986), "test_year"] = 1986.5
    with pytest.raises(ValueError, match="integer editions"):
        evaluate_advanced_promotion(
            fractional,
            _decision(),
            candidate="advanced",
            baseline="legacy",
            model_identity_hash="a" * 64,
            artifact_identity_hash="a" * 64,
            structural_checks={"simplex": True, "symmetry": True, "leakage": True},
        )

    result = evaluate_advanced_promotion(
        _folds(),
        _decision(),
        candidate="advanced",
        baseline="legacy",
        model_identity_hash="not-a-digest",
        artifact_identity_hash="not-a-digest",
        structural_checks={"simplex": True, "symmetry": True, "leakage": True},
    )
    assert result["checks"]["exact_model_identity"] is False
    assert result["system_promoted"] is False


def test_advanced_promotion_rejects_universal_draw_decision_collapse() -> None:
    collapsed = _decision()
    collapsed["positive_draw_predictions"] = collapsed["test_match_count"]
    result = evaluate_advanced_promotion(
        _folds(),
        collapsed,
        candidate="advanced",
        baseline="legacy",
        model_identity_hash="a" * 64,
        artifact_identity_hash="a" * 64,
        structural_checks={"simplex": True, "symmetry": True, "leakage": True},
    )
    assert result["checks"]["no_universal_draw_collapse"] is False
    assert result["decision_layer_passed"] is False
    assert result["system_promoted"] is False
