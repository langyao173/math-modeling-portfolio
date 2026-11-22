"""Tournament-clustered bootstrap summaries and paired comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


def clustered_mean_interval(
    values: np.ndarray,
    clusters: np.ndarray,
    replicates: int = 2_000,
    seed: int = 1_729,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Bootstrap clusters, retaining within-tournament dependence."""
    observations = np.asarray(values, dtype=float)
    group_ids = np.asarray(clusters)
    if observations.shape[0] != group_ids.shape[0] or observations.size == 0:
        raise ValueError("Values and clusters must be nonempty and aligned")
    unique = np.unique(group_ids)
    rng = np.random.default_rng(seed)
    sampled_means = np.empty(replicates, dtype=float)
    group_values = {group: observations[group_ids == group] for group in unique}
    for index in range(replicates):
        sample = rng.choice(unique, size=unique.size, replace=True)
        sampled_means[index] = np.concatenate([group_values[group] for group in sample]).mean()
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(observations.mean()),
        "ci_low": float(np.quantile(sampled_means, alpha)),
        "ci_high": float(np.quantile(sampled_means, 1.0 - alpha)),
        "clusters": int(unique.size),
        "replicates": int(replicates),
    }


def paired_fold_comparison(
    table: pd.DataFrame,
    candidate: str,
    baseline: str,
    metric: str,
    lower_is_better: bool = True,
) -> dict[str, float | int | str]:
    """Compare two methods on identical tournament folds."""
    subset = table.loc[table["model"].isin([candidate, baseline])]
    pivot = subset.pivot(index="test_year", columns="model", values=metric).dropna()
    if candidate not in pivot or baseline not in pivot or pivot.shape[0] < 3:
        raise ValueError("At least three paired tournament folds are required")
    difference = pivot[candidate].to_numpy() - pivot[baseline].to_numpy()
    oriented = -difference if lower_is_better else difference
    try:
        statistic, p_value = wilcoxon(oriented, alternative="greater")
    except ValueError:
        statistic, p_value = 0.0, 1.0
    nonzero = oriented[oriented != 0]
    if nonzero.size == 0:
        rank_biserial = 0.0
    else:
        ranks = rankdata(np.abs(nonzero), method="average")
        positive_rank_sum = float(ranks[nonzero > 0].sum())
        negative_rank_sum = float(ranks[nonzero < 0].sum())
        rank_biserial = (positive_rank_sum - negative_rank_sum) / float(ranks.sum())
    return {
        "candidate": candidate,
        "baseline": baseline,
        "metric": metric,
        "folds": int(pivot.shape[0]),
        "mean_candidate_minus_baseline": float(difference.mean()),
        "median_candidate_minus_baseline": float(np.median(difference)),
        "folds_improved": int(np.sum(oriented > 0)),
        "wilcoxon_statistic": float(statistic),
        "p_value_one_sided": float(p_value),
        "rank_biserial_directional": rank_biserial,
    }
