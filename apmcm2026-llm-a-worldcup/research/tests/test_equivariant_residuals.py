"""TDD contracts for the bounded equivariant residual correction layer."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd
import pytest


def _symbols() -> tuple[type[Any], type[Any]]:
    module_name = "worldcup_forecasting.models.residuals"
    try:
        module = importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.fail(
            f"RED residual-correction contract is absent: {module_name} ({exc})",
            pytrace=False,
        )
    return module.EquivariantResidualConfig, module.EquivariantResidualCorrector


def _training_fixture(
    seed: int = 1701,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    count = 160
    pace = rng.normal(size=count)
    era = rng.normal(size=count)
    strength = rng.normal(size=count)
    host = rng.choice([-1.0, 0.0, 1.0], size=count)
    frame = pd.DataFrame(
        {
            "pace": pace,
            "era": era,
            "strength_diff": strength,
            "host_diff": host,
        }
    )
    base_home = np.exp(0.15 + 0.08 * pace + 0.12 * strength)
    base_away = np.exp(0.15 + 0.08 * pace - 0.12 * strength)
    base_rates = np.column_stack([base_home, base_away])
    common_residual = 0.22 * pace - 0.08 * era
    contrast_residual = 0.20 * strength + 0.10 * host
    corrected = base_rates * np.exp(
        np.column_stack(
            [common_residual + contrast_residual, common_residual - contrast_residual]
        )
    )
    observed = rng.poisson(corrected).astype(float)
    prediction_year = np.resize(np.asarray([1998, 2002, 2006, 2010]), count)
    provenance = pd.DataFrame(
        {
            "match_id": np.arange(count),
            "is_oof": True,
            "prediction_role": "structural_oof",
            "train_end_year": prediction_year - 4,
            "prediction_year": prediction_year,
        }
    )
    return frame, base_rates, observed, provenance


def _fit(config: Any | None = None) -> Any:
    config_type, corrector_type = _symbols()
    frame, rates, observed, provenance = _training_fixture()
    resolved = config if config is not None else config_type(max_abs_log_correction=0.30)
    return corrector_type(resolved).fit_oof(
        frame,
        rates,
        observed,
        provenance,
        even_features=("pace", "era"),
        odd_features=("strength_diff", "host_diff"),
    )


def test_disabled_corrector_is_identity_and_enabled_corrector_requires_fit() -> None:
    config_type, corrector_type = _symbols()
    frame = pd.DataFrame({"pace": [1.0], "strength_diff": [0.5]})
    rates = np.asarray([[1.4, 0.9]])

    disabled = corrector_type(config_type(enabled=False))
    np.testing.assert_array_equal(disabled.transform_rates(frame, rates), rates)
    assert disabled.metadata_["enabled"] is False
    assert disabled.metadata_["status"] == "disabled_identity"

    enabled = corrector_type(config_type(enabled=True))
    with pytest.raises(RuntimeError, match="fit|fitted"):
        enabled.transform_rates(frame, rates)


def test_fit_oof_rejects_in_sample_nonchronological_and_nonstructural_rows() -> None:
    config_type, corrector_type = _symbols()
    frame, rates, observed, provenance = _training_fixture()

    in_sample = provenance.copy()
    in_sample.loc[0, "is_oof"] = False
    with pytest.raises(ValueError, match="OOF|out-of-fold"):
        corrector_type(config_type()).fit_oof(
            frame,
            rates,
            observed,
            in_sample,
            even_features=("pace", "era"),
            odd_features=("strength_diff", "host_diff"),
        )

    nonchronological = provenance.copy()
    nonchronological.loc[0, "train_end_year"] = nonchronological.loc[0, "prediction_year"]
    with pytest.raises(ValueError, match="earlier|chronolog|train_end"):
        corrector_type(config_type()).fit_oof(
            frame,
            rates,
            observed,
            nonchronological,
            even_features=("pace", "era"),
            odd_features=("strength_diff", "host_diff"),
        )

    nonstructural = provenance.copy()
    nonstructural.loc[0, "prediction_role"] = "residual_in_sample"
    with pytest.raises(ValueError, match="structural"):
        corrector_type(config_type()).fit_oof(
            frame,
            rates,
            observed,
            nonstructural,
            even_features=("pace", "era"),
            odd_features=("strength_diff", "host_diff"),
        )


def test_fit_and_transform_validate_shapes_positivity_and_feature_schema() -> None:
    config_type, corrector_type = _symbols()
    frame, rates, observed, provenance = _training_fixture()
    model = corrector_type(config_type())
    with pytest.raises(ValueError, match="positive|rate"):
        model.fit_oof(
            frame,
            np.where(np.arange(rates.size).reshape(rates.shape) == 0, 0.0, rates),
            observed,
            provenance,
            even_features=("pace", "era"),
            odd_features=("strength_diff", "host_diff"),
        )
    with pytest.raises(ValueError, match="shape|two"):
        model.fit_oof(
            frame,
            rates,
            observed[:, 0],
            provenance,
            even_features=("pace", "era"),
            odd_features=("strength_diff", "host_diff"),
        )

    fitted = _fit()
    with pytest.raises(ValueError, match="missing|feature|host_diff"):
        fitted.transform_rates(frame.drop(columns="host_diff"), rates)
    invalid_rates = rates.copy()
    invalid_rates[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite|rate"):
        fitted.transform_rates(frame, invalid_rates)


def test_corrections_are_positive_nontrivial_and_bounded_in_log_rate_space() -> None:
    config_type, _ = _symbols()
    bound = 0.12
    model = _fit(config_type(max_abs_log_correction=bound, alpha_even=1.0, alpha_odd=1.0))
    probe = pd.DataFrame(
        {
            "pace": [-100.0, -1.0, 1.0, 100.0],
            "era": [50.0, 0.0, 0.0, -50.0],
            "strength_diff": [-100.0, -1.0, 1.0, 100.0],
            "host_diff": [-1.0, 0.0, 0.0, 1.0],
        }
    )
    base = np.asarray([[1.3, 0.8], [1.0, 1.0], [0.9, 1.4], [2.0, 0.4]])
    corrected = model.transform_rates(probe, base)
    log_correction = np.log(corrected / base)

    assert np.isfinite(corrected).all() and np.all(corrected > 0)
    assert np.max(np.abs(log_correction)) <= bound + 1e-12
    assert np.max(np.abs(log_correction)) > 1e-4
    assert model.metadata_["fit_scope"] == "chronological_structural_oof_only"
    assert model.metadata_["training_rows"] == 160
    assert model.metadata_["max_abs_log_correction"] == bound
    assert model.metadata_["even_features"] == ["pace", "era"]
    assert model.metadata_["odd_features"] == ["strength_diff", "host_diff"]
    assert model.metadata_["deterministic"] is True


def test_antisymmetrized_odd_correction_makes_team_swap_exact() -> None:
    model = _fit()
    frame = pd.DataFrame(
        {
            "pace": [-0.7, 0.4, 1.2],
            "era": [0.1, -0.2, 0.5],
            "strength_diff": [-2.0, 0.3, 1.5],
            "host_diff": [-1.0, 0.0, 1.0],
        }
    )
    base = np.asarray([[0.8, 1.5], [1.2, 1.0], [2.1, 0.6]])
    swapped = frame.copy()
    swapped[["strength_diff", "host_diff"]] *= -1.0

    direct = model.transform_rates(frame, base)
    reverse = model.transform_rates(swapped, base[:, ::-1])
    np.testing.assert_allclose(direct[:, 0], reverse[:, 1], atol=1e-12, rtol=0)
    np.testing.assert_allclose(direct[:, 1], reverse[:, 0], atol=1e-12, rtol=0)


def test_repeated_fit_is_numerically_deterministic() -> None:
    first = _fit()
    second = _fit()
    frame, rates, _, _ = _training_fixture()
    first_rates = first.transform_rates(frame, rates)
    second_rates = second.transform_rates(frame, rates)

    np.testing.assert_allclose(first_rates, second_rates, atol=0.0, rtol=0.0)
    assert first.metadata_ == second.metadata_


def test_explicit_imputation_is_audited_and_preserves_swap_equivariance() -> None:
    config_type, corrector_type = _symbols()
    frame, rates, observed, provenance = _training_fixture()
    frame.loc[0, "pace"] = np.nan
    frame.loc[1, "strength_diff"] = np.nan
    model = corrector_type(config_type(missing_policy="impute")).fit_oof(
        frame,
        rates,
        observed,
        provenance,
        even_features=("pace", "era"),
        odd_features=("strength_diff", "host_diff"),
    )
    probe = pd.DataFrame(
        {
            "pace": [np.nan],
            "era": [0.2],
            "strength_diff": [np.nan],
            "host_diff": [1.0],
        }
    )
    swapped = probe.copy()
    swapped["host_diff"] *= -1.0
    base = np.asarray([[1.5, 0.7]])
    direct = model.transform_rates(probe, base)
    reverse = model.transform_rates(swapped, base[:, ::-1])

    np.testing.assert_allclose(direct[:, 0], reverse[:, 1], atol=1e-12, rtol=0)
    np.testing.assert_allclose(direct[:, 1], reverse[:, 0], atol=1e-12, rtol=0)
    assert model.metadata_["missing_policy"] == "impute"
    assert model.metadata_["odd_missing_imputation"] == "symmetry_neutral_zero"
    assert set(model.metadata_["even_imputation_values"]) == {"pace", "era"}


def test_default_missing_policy_fails_closed() -> None:
    model = _fit()
    frame = pd.DataFrame(
        {
            "pace": [np.nan],
            "era": [0.0],
            "strength_diff": [0.0],
            "host_diff": [0.0],
        }
    )
    with pytest.raises(ValueError, match="missing|finite"):
        model.transform_rates(frame, np.asarray([[1.0, 1.0]]))
