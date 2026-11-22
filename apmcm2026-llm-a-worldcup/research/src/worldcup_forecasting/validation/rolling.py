"""Chronological validation folds for sparse tournament data."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TournamentFold:
    """One expanding-origin tournament split."""

    test_year: int
    train_index: np.ndarray
    test_index: np.ndarray


def expanding_origin_folds(
    frame: pd.DataFrame,
    test_years: Iterable[int],
    year_column: str = "Year",
) -> list[TournamentFold]:
    """Create folds where every training observation predates the test edition."""
    if year_column not in frame:
        raise ValueError(f"Missing year column: {year_column}")
    years = frame[year_column].to_numpy(dtype=int)
    folds: list[TournamentFold] = []
    for test_year in test_years:
        train_index = np.flatnonzero(years < int(test_year))
        test_index = np.flatnonzero(years == int(test_year))
        if train_index.size == 0 or test_index.size == 0:
            raise ValueError(f"Invalid expanding-origin fold for {test_year}")
        if years[train_index].max() >= test_year:
            raise AssertionError("Training data are not strictly earlier than test data")
        folds.append(TournamentFold(int(test_year), train_index, test_index))
    return folds


def official_2014_split(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return the fixed 1930-2010 training and 2014 test indices."""
    years = frame["Year"].to_numpy(dtype=int)
    train_index = np.flatnonzero(years <= 2010)
    test_index = np.flatnonzero(years == 2014)
    if train_index.size != 772 or test_index.size != 64:
        raise AssertionError(
            f"Official split must be 772/64 after deduplication, got "
            f"{train_index.size}/{test_index.size}"
        )
    return train_index, test_index
