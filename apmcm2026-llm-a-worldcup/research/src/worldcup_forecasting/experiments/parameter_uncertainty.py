"""Tournament-cluster refits for bounded parameter-sensitivity evidence."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import numpy as np
import pandas as pd

RefitCallback = Callable[[pd.DataFrame, int], Any]
_OUTCOMES = ("home_win", "draw", "away_win")


@dataclass(frozen=True, slots=True)
class TournamentBootstrapConfig:
    """Frozen cluster-bootstrap controls."""

    replicates: int = 200
    seed: int = 1729
    year_column: str = "Year"
    match_id_column: str = "MatchID"
    maximum_training_year: int = 2010
    minimum_success_fraction: float = 0.9


@dataclass(frozen=True)
class TournamentBootstrapResult:
    """Prediction draws, resampling ledger, failures, and interpretation."""

    predictions: pd.DataFrame
    draw_ledger: pd.DataFrame
    failures: pd.DataFrame
    metadata: dict[str, Any]


def run_tournament_cluster_refits(
    training_frame: pd.DataFrame,
    probe_frame: pd.DataFrame,
    refit_callback: RefitCallback,
    *,
    config: TournamentBootstrapConfig | None = None,
) -> TournamentBootstrapResult:
    """Refit on whole-edition bootstrap samples and predict a fixed probe.

    This estimates sensitivity to the finite set of historical tournaments. It
    is not an IID match bootstrap and is not a future coverage guarantee.
    """
    cfg = config or TournamentBootstrapConfig()
    frame, probe, editions = _validate_inputs(
        training_frame, probe_frame, refit_callback, cfg
    )
    rng = np.random.default_rng(int(cfg.seed))
    prediction_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for draw_id in range(int(cfg.replicates)):
        sampled_years = rng.choice(editions, size=len(editions), replace=True)
        sampled = _sample_editions(frame, sampled_years, cfg.year_column)
        refit_seed = int(cfg.seed) + draw_id
        sample_hash = _sample_hash(sampled_years)
        try:
            model = refit_callback(sampled, refit_seed)
            probability = _predict(model, probe)
        except Exception as exc:  # candidate-local failure is retained in the ledger
            failure_rows.append(
                {
                    "bootstrap_draw_id": draw_id,
                    "refit_seed": refit_seed,
                    "sample_hash": sample_hash,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            continue
        ledger_rows.append(
            {
                "bootstrap_draw_id": draw_id,
                "refit_seed": refit_seed,
                "sample_hash": sample_hash,
                "sampled_editions": "|".join(str(int(year)) for year in sampled_years),
                "sampled_row_count": int(len(sampled)),
            }
        )
        for position, match_id in enumerate(probe[cfg.match_id_column]):
            prediction_rows.append(
                {
                    "bootstrap_draw_id": draw_id,
                    "match_id": match_id,
                    **{
                        outcome: float(probability[position, column])
                        for column, outcome in enumerate(_OUTCOMES)
                    },
                }
            )
    successes = len(ledger_rows)
    success_fraction = successes / int(cfg.replicates)
    if success_fraction < float(cfg.minimum_success_fraction):
        raise RuntimeError(
            "Tournament-cluster refit success fraction "
            f"{success_fraction:.3f} is below {cfg.minimum_success_fraction:.3f}"
        )
    metadata: dict[str, Any] = {
        "sampling_unit": "tournament_edition",
        "replicates_requested": int(cfg.replicates),
        "replicates_succeeded": successes,
        "replicates_failed": len(failure_rows),
        "success_fraction": success_fraction,
        "seed": int(cfg.seed),
        "maximum_training_year": int(cfg.maximum_training_year),
        "post_2010_rows_used": False,
        "interpretation": "historical_cluster_parameter_sensitivity_not_coverage_guarantee",
    }
    return TournamentBootstrapResult(
        predictions=pd.DataFrame(prediction_rows),
        draw_ledger=pd.DataFrame(ledger_rows),
        failures=pd.DataFrame(
            failure_rows,
            columns=(
                "bootstrap_draw_id",
                "refit_seed",
                "sample_hash",
                "error_type",
                "error_message",
            ),
        ),
        metadata=metadata,
    )


def _validate_inputs(
    training_frame: pd.DataFrame,
    probe_frame: pd.DataFrame,
    callback: RefitCallback,
    config: TournamentBootstrapConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    if not isinstance(training_frame, pd.DataFrame) or training_frame.empty:
        raise ValueError("training_frame must be a nonempty DataFrame")
    if not isinstance(probe_frame, pd.DataFrame) or probe_frame.empty:
        raise ValueError("probe_frame must be a nonempty DataFrame")
    if not callable(callback):
        raise TypeError("refit_callback must be callable")
    if isinstance(config.replicates, bool) or int(config.replicates) < 4:
        raise ValueError("At least four tournament bootstrap replicates are required")
    invalid_cutoff = (
        isinstance(config.maximum_training_year, (bool, np.bool_))
        or not isinstance(config.maximum_training_year, (int, np.integer))
        or config.maximum_training_year <= 0
        or config.maximum_training_year > 2010
    )
    if invalid_cutoff:
        raise ValueError("Tournament bootstrap training is frozen through 2010")
    if not 0 < float(config.minimum_success_fraction) <= 1:
        raise ValueError("minimum_success_fraction must lie in (0, 1]")
    required = {config.year_column, config.match_id_column}
    missing = sorted(required.difference(training_frame.columns))
    if missing:
        raise ValueError(f"training_frame is missing columns: {missing}")
    if config.match_id_column not in probe_frame:
        raise ValueError("probe_frame is missing its match identifier")
    training_ids = training_frame[config.match_id_column]
    if training_ids.isna().any() or training_ids.duplicated().any():
        raise ValueError("training match identifiers must be complete and unique")
    probe_ids = probe_frame[config.match_id_column]
    if probe_ids.isna().any() or probe_ids.duplicated().any():
        raise ValueError("probe match identifiers must be complete and unique")
    years = pd.to_numeric(training_frame[config.year_column], errors="coerce")
    if (
        years.isna().any()
        or not np.isfinite(years.to_numpy(float)).all()
        or not np.equal(years, np.floor(years)).all()
    ):
        raise ValueError("training edition years must be finite integers")
    if years.max() > int(config.maximum_training_year):
        raise ValueError("Tournament bootstrap training is frozen through 2010")
    editions = np.sort(years.astype(int).unique())
    if len(editions) < 3:
        raise ValueError("At least three historical tournament clusters are required")
    return training_frame.copy(), probe_frame.copy(), editions


def _sample_editions(
    frame: pd.DataFrame,
    sampled_years: np.ndarray,
    year_column: str,
) -> pd.DataFrame:
    parts = []
    for occurrence, year in enumerate(sampled_years):
        block = frame.loc[pd.to_numeric(frame[year_column]).eq(int(year))].copy()
        block["bootstrap_cluster_occurrence"] = occurrence
        parts.append(block)
    return pd.concat(parts, ignore_index=True)


def _predict(model: Any, probe: pd.DataFrame) -> np.ndarray:
    if not callable(getattr(model, "predict_proba", None)):
        raise TypeError("refit_callback must return a model exposing predict_proba")
    probability = np.asarray(model.predict_proba(probe), dtype=float)
    if probability.shape != (len(probe), 3):
        raise ValueError("Bootstrap model returned an invalid WDL shape")
    if not np.isfinite(probability).all() or np.any(probability < 0):
        raise ValueError("Bootstrap model returned invalid WDL probabilities")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("Bootstrap WDL probabilities must sum to one")
    return probability


def _sample_hash(sampled_years: np.ndarray) -> str:
    payload = "|".join(str(int(year)) for year in sampled_years).encode()
    return sha256(payload).hexdigest()


__all__ = [
    "TournamentBootstrapConfig",
    "TournamentBootstrapResult",
    "run_tournament_cluster_refits",
]
