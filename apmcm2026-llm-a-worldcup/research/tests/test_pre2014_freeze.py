from __future__ import annotations

import pandas as pd
import pytest

from worldcup_forecasting.validation.freeze_contracts import (
    OUTER_YEARS,
    candidate_registry,
    validate_oof,
)


def _oof_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": range(1, 8),
            "prediction_year": list(OUTER_YEARS),
            "train_end_year": [year - 4 for year in OUTER_YEARS],
            "outer_fold_id": [f"outer_{year}" for year in OUTER_YEARS],
            "candidate_id": ["a" * 64] * 7,
            "feature_schema_hash": ["b" * 64] * 7,
            "home_win": [0.4] * 7,
            "draw": [0.3] * 7,
            "away_win": [0.3] * 7,
            "is_oof": [True] * 7,
        }
    )


def test_freeze_oof_contract_is_pre2014_and_probability_coherent() -> None:
    validated = validate_oof(_oof_frame())
    assert validated["prediction_year"].max() == 2010
    contaminated = _oof_frame()
    contaminated.loc[6, "prediction_year"] = 2014
    with pytest.raises(ValueError, match="seven-fold|pre-2014"):
        validate_oof(contaminated)


def test_candidate_registry_requires_exact_frozen_budget() -> None:
    rows = []
    for candidate_order in range(36):
        for fold in range(7):
            rows.append(
                {
                    "candidate_id": f"{candidate_order:064d}",
                    "candidate_order": candidate_order,
                    "config_hash": f"{candidate_order + 100:064d}",
                    "alpha": 0.1,
                    "half_life_years": None,
                    "shared_score": False,
                    "residual": False,
                    "eligible": True,
                    "measured_runtime_seconds": fold / 10,
                    "failure_reason": "",
                }
            )
    candidate = pd.DataFrame(rows)
    selected = candidate.iloc[[0]][["candidate_id"]]
    registry = candidate_registry(candidate, selected)
    assert len(registry) == 36
    assert registry["eligible_all_folds"].all()
    with pytest.raises(ValueError, match="exactly 36"):
        candidate_registry(candidate.loc[candidate["candidate_order"] < 35], selected)
