"""Private fitting primitives for the paired A0--A8 experiment."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from worldcup_forecasting.models._advanced_count_utils import host_sign
from worldcup_forecasting.models.advanced_count import (
    EquivariantCountConfig,
    EquivariantCountModel,
    joint_batch_from_rates,
)
from worldcup_forecasting.models.residuals import (
    EquivariantResidualConfig,
    EquivariantResidualCorrector,
)

from ._asymmetric_count import AsymmetricSidePoisson

Clock = Callable[[], float]


@dataclass(frozen=True, slots=True)
class VariantPlan:
    variant_id: str
    removed_component: str
    include_dynamic: bool = True
    include_personnel: bool = True
    include_era_format: bool = True
    shared_score: bool = True
    residual: bool = True
    calibration: bool = True
    enforce_shrinkage: bool = True
    asymmetric: bool = False


@dataclass(frozen=True)
class StructuralRecord:
    validation_year: int
    train_end_year: int
    indices: np.ndarray
    rates: np.ndarray


@dataclass(frozen=True)
class VariantBaseFit:
    plan: VariantPlan
    config: EquivariantCountConfig
    outer_joint: np.ndarray
    swapped_outer_joint: np.ndarray
    outer_rates: np.ndarray
    inner_joint: np.ndarray
    inner_goals: np.ndarray
    inner_provenance: pd.DataFrame
    runtime_seconds: float
    train_end_year: int
    residual_fit_max_year: int | None
    fit_index_hash: str
    config_hash: str


def variant_plans(shared_score: bool, residual: bool) -> tuple[VariantPlan, ...]:
    """Return the immutable A0--A8 intervention definitions."""
    common = {"shared_score": shared_score, "residual": residual}
    return (
        VariantPlan("A0", "none_full_model", **common),
        VariantPlan("A1", "dynamic_ratings", include_dynamic=False, **common),
        VariantPlan("A2", "player_features", include_personnel=False, **common),
        VariantPlan("A3", "era_effects", include_era_format=False, **common),
        VariantPlan("A4", "correlated_count_model", shared_score=False, residual=residual),
        VariantPlan("A5", "machine_learning_residual", shared_score=shared_score, residual=False),
        VariantPlan(
            "A6", "probability_calibration", calibration=False, **common
        ),
        VariantPlan(
            "A7", "hierarchical_shrinkage", enforce_shrinkage=False, **common
        ),
        VariantPlan("A8", "symmetry_constraint", asymmetric=True, **common),
    )


def fit_variant_base(
    frame: pd.DataFrame,
    *,
    outer_year: int,
    plan: VariantPlan,
    alpha: float,
    half_life_years: float | None,
    count_config_factory: Callable[..., EquivariantCountConfig],
    latest_inner_folds: int,
    maximum_development_year: int,
    runtime_clock: Clock,
) -> VariantBaseFit:
    """Fit one intervention with strict expanding-origin residual evidence."""
    config = count_config_factory(
        alpha=alpha,
        half_life_years=half_life_years,
        shared_score=plan.shared_score,
        include_dynamic=plan.include_dynamic,
        include_personnel=plan.include_personnel,
        include_era_format=plan.include_era_format,
        enforce_shrinkage=plan.enforce_shrinkage,
    )
    start = float(runtime_clock())
    years = pd.to_numeric(frame[config.year_column], errors="raise").to_numpy(int)
    eligible_inner_years = sorted(set(years[years < outer_year].tolist()))[1:]
    if len(eligible_inner_years) < latest_inner_folds:
        raise ValueError("Ablation fold has too few earlier inner editions")
    # Match the frozen selection window instead of silently expanding the
    # residual/calibration evidence to weak, very early editions.
    inner_years = eligible_inner_years[-latest_inner_folds:]
    design, even, odd = residual_design(frame, config)
    records: list[StructuralRecord] = []
    calibration_joints: list[np.ndarray] = []
    calibration_goals: list[np.ndarray] = []
    provenance: list[pd.DataFrame] = []
    retained_years = set(inner_years)
    model_type = AsymmetricSidePoisson if plan.asymmetric else EquivariantCountModel
    for validation_year in inner_years:
        train_index = np.flatnonzero(years < validation_year)
        valid_index = np.flatnonzero(years == validation_year)
        model = model_type(config).fit(frame.iloc[train_index])
        base_rates = model.predict_rates(frame.iloc[valid_index])
        shared_fraction = float(model.shared_fraction_)
        final_rates = base_rates
        if plan.residual and records:
            corrector, _ = fit_residual(
                frame, design, records, even, odd, maximum_development_year
            )
            final_rates = corrector.transform_rates(design.iloc[valid_index], base_rates)
        records.append(
            StructuralRecord(
                validation_year,
                int(years[train_index].max()),
                valid_index,
                base_rates,
            )
        )
        if validation_year in retained_years:
            calibration_joints.append(
                joint_batch_from_rates(
                    final_rates,
                    shared_fraction=shared_fraction,
                    max_score=config.max_score,
                    min_retained_mass=config.min_retained_mass,
                    max_support=config.max_support,
                ).probability
            )
            calibration_goals.append(goals(frame.iloc[valid_index], config))
            provenance.append(
                pd.DataFrame(
                    {
                        "is_oof": True,
                        "prediction_year": validation_year,
                        "train_end_year": int(years[train_index].max()),
                    },
                    index=range(len(valid_index)),
                )
            )

    train_index = np.flatnonzero(years < outer_year)
    valid_index = np.flatnonzero(years == outer_year)
    final_model = model_type(config).fit(frame.iloc[train_index])
    base_outer = final_model.predict_rates(frame.iloc[valid_index])
    swapped = swap_frame(frame.iloc[valid_index], config)
    base_swapped = final_model.predict_rates(swapped)
    residual_max: int | None = None
    if plan.residual and records:
        corrector, residual_max = fit_residual(
            frame, design, records, even, odd, maximum_development_year
        )
        base_outer = corrector.transform_rates(design.iloc[valid_index], base_outer)
        swapped_design, _, _ = residual_design(swapped, config)
        base_swapped = corrector.transform_rates(swapped_design, base_swapped)
    shared_fraction = float(final_model.shared_fraction_)
    outer_joint = _joint(base_outer, shared_fraction, config)
    swapped_joint = _joint(base_swapped, shared_fraction, config)
    elapsed = _elapsed(runtime_clock, start)
    ids = match_ids(frame.iloc[train_index]).tolist()
    residual_ids = (
        match_ids(frame.iloc[np.concatenate([record.indices for record in records])]).tolist()
        if plan.residual
        else []
    )
    config_identity = hashlib.sha256(
        json.dumps(
            {"model": asdict(config), "plan": asdict(plan)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    fit_hash = hashlib.sha256(
        json.dumps(
            {
                "training_match_ids": [str(value) for value in ids],
                "inner_oof_years": inner_years,
                "residual_oof_match_ids": [str(value) for value in residual_ids],
                "config": config_identity,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return VariantBaseFit(
        plan=plan,
        config=config,
        outer_joint=outer_joint,
        swapped_outer_joint=swapped_joint,
        outer_rates=base_outer,
        inner_joint=pad_joint(calibration_joints),
        inner_goals=np.vstack(calibration_goals).astype(int),
        inner_provenance=pd.concat(provenance, ignore_index=True),
        runtime_seconds=elapsed,
        train_end_year=int(years[train_index].max()),
        residual_fit_max_year=residual_max,
        fit_index_hash=fit_hash,
        config_hash=config_identity,
    )


def residual_design(
    frame: pd.DataFrame, config: EquivariantCountConfig
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    design = pd.DataFrame(index=frame.index)
    even: list[str] = []
    odd: list[str] = []
    for name in (*config.common_pace_features, *config.even_numeric_features):
        if name in frame and name not in even:
            design[name] = pd.to_numeric(frame[name], errors="coerce")
            even.append(name)
    for name in config.odd_numeric_features:
        if name in frame and name not in odd:
            design[name] = pd.to_numeric(frame[name], errors="coerce")
            odd.append(name)
    for position, (home, away) in enumerate(config.paired_numeric_features):
        left = pd.to_numeric(frame[home], errors="coerce")
        right = pd.to_numeric(frame[away], errors="coerce")
        even_name, odd_name = f"pair_even_{position}", f"pair_odd_{position}"
        design[even_name], design[odd_name] = left + right, left - right
        even.append(even_name)
        odd.append(odd_name)
    if config.host_side_column and config.host_side_column in frame:
        design["host_side_signed"] = frame[config.host_side_column].map(host_sign)
        odd.append("host_side_signed")
    if not even:
        design["even_origin"] = 0.0
        even.append("even_origin")
    if not odd:
        design["odd_origin"] = 0.0
        odd.append("odd_origin")
    return design, tuple(even), tuple(odd)


def fit_residual(
    frame: pd.DataFrame,
    design: pd.DataFrame,
    records: Sequence[StructuralRecord],
    even: tuple[str, ...],
    odd: tuple[str, ...],
    maximum_year: int,
) -> tuple[EquivariantResidualCorrector, int]:
    indices = np.concatenate([record.indices for record in records])
    training_design = design.iloc[indices].reset_index(drop=True)
    usable_even = tuple(name for name in even if not training_design[name].isna().all())
    if not usable_even:
        training_design["even_fit_origin"] = 0.0
        design["even_fit_origin"] = 0.0
        usable_even = ("even_fit_origin",)
    provenance = pd.concat(
        [
            pd.DataFrame(
                {
                    "match_id": match_ids(frame.iloc[record.indices]),
                    "is_oof": True,
                    "prediction_role": "structural_oof",
                    "train_end_year": record.train_end_year,
                    "prediction_year": record.validation_year,
                }
            )
            for record in records
        ],
        ignore_index=True,
    )
    corrector = EquivariantResidualCorrector(
        EquivariantResidualConfig(
            missing_policy="impute", maximum_fit_prediction_year=maximum_year
        )
    ).fit_oof(
        training_design,
        np.vstack([record.rates for record in records]),
        goals(frame.iloc[indices]),
        provenance,
        usable_even,
        odd,
    )
    return corrector, max(record.validation_year for record in records)


def swap_frame(frame: pd.DataFrame, config: EquivariantCountConfig) -> pd.DataFrame:
    swapped = frame.copy(deep=True)
    home, away = config.home_team_column, config.away_team_column
    swapped[[home, away]] = frame[[away, home]].to_numpy()
    for name in config.odd_numeric_features:
        swapped[name] = -pd.to_numeric(frame[name], errors="raise")
    for left, right in config.paired_numeric_features:
        swapped[[left, right]] = frame[[right, left]].to_numpy()
    if config.host_side_column and config.host_side_column in frame:
        swapped[config.host_side_column] = frame[config.host_side_column].map(
            _swapped_host_side
        )
    return swapped


def _swapped_host_side(value: object) -> str:
    sign = host_sign(value)
    if sign > 0:
        return "away"
    if sign < 0:
        return "home"
    return "neutral"


def goals(frame: pd.DataFrame, config: EquivariantCountConfig | None = None) -> np.ndarray:
    candidates = []
    if config is not None:
        candidates.append(
            (
                config.home_goals_column or config.target_home_column,
                config.away_goals_column or config.target_away_column,
            )
        )
    candidates.extend(
        [
            ("target_home_goals", "target_away_goals"),
            ("Home Team Goals", "Away Team Goals"),
            ("home_goals", "away_goals"),
        ]
    )
    for home, away in candidates:
        if home in frame and away in frame:
            values = frame[[home, away]].apply(pd.to_numeric, errors="raise").to_numpy(float)
            valid = (
                np.isfinite(values).all()
                and np.all(values >= 0)
                and np.allclose(values, np.rint(values), atol=1e-12)
            )
            if valid:
                return values
    raise ValueError(
        "Ablation frame has no complete nonnegative integer goal targets"
    )


def match_ids(frame: pd.DataFrame) -> np.ndarray:
    for name in ("MatchID", "match_id"):
        if name in frame:
            return frame[name].to_numpy()
    raise ValueError("Ablation frame requires MatchID or match_id")


def pad_joint(batches: Sequence[np.ndarray]) -> np.ndarray:
    support = max(batch.shape[1] for batch in batches)
    result = np.zeros((sum(len(batch) for batch in batches), support, support))
    offset = 0
    for batch in batches:
        result[offset : offset + len(batch), : batch.shape[1], : batch.shape[2]] = batch
        offset += len(batch)
    return result


def _joint(
    rates: np.ndarray, shared_fraction: float, config: EquivariantCountConfig
) -> np.ndarray:
    return joint_batch_from_rates(
        rates,
        shared_fraction=shared_fraction,
        max_score=config.max_score,
        min_retained_mass=config.min_retained_mass,
        max_support=config.max_support,
    ).probability


def _elapsed(clock: Clock, start: float) -> float:
    value = float(clock()) - start
    if not np.isfinite(value) or value < 0:
        raise ValueError("runtime_clock must be finite and nondecreasing")
    return value


__all__ = [
    "VariantBaseFit",
    "VariantPlan",
    "fit_variant_base",
    "goals",
    "match_ids",
    "variant_plans",
]
