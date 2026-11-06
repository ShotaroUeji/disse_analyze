"""Dependency-aware summaries and source-level bootstrap intervals."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def bootstrap_pair_mean(
    pair_frame: pd.DataFrame,
    metric: str,
    *,
    n_sources: int,
    seed: int = 0,
    n_bootstrap: int = 1000,
) -> dict:
    """Bootstrap a pair metric by resampling source IDs, not 45 pairs independently."""
    source_ids = sorted(set(pair_frame.source_a) | set(pair_frame.source_b))
    if len(source_ids) != n_sources:
        raise ValueError(f"Expected {n_sources} sources, found {len(source_ids)}")
    values = {
        (row.source_a, row.source_b): float(getattr(row, metric))
        for row in pair_frame.itertuples()
    }
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(source_ids, size=n_sources, replace=True)
        selected = []
        for i, j in combinations(range(n_sources), 2):
            if sampled[i] == sampled[j]:
                continue
            key = (sampled[i], sampled[j])
            reverse = (sampled[j], sampled[i])
            selected.append(values[key] if key in values else values[reverse])
        if selected:
            bootstrap.append(float(np.mean(selected)))
    bootstrap_array = np.asarray(bootstrap, dtype=np.float64)
    return {
        "mean": float(pair_frame[metric].mean()),
        "std": float(pair_frame[metric].std(ddof=1)),
        "median": float(pair_frame[metric].median()),
        "bootstrap_mean": float(bootstrap_array.mean()),
        "bootstrap_ci_low": float(np.percentile(bootstrap_array, 2.5)),
        "bootstrap_ci_high": float(np.percentile(bootstrap_array, 97.5)),
        "n_bootstrap": int(len(bootstrap_array)),
    }


def paired_model_difference(
    pair_frames: dict[str, pd.DataFrame], metric: str
) -> dict:
    """Compare two models on the same source pairs using paired differences."""
    models = sorted(pair_frames)
    if len(models) != 2:
        raise ValueError("paired_model_difference expects exactly two models")
    left, right = models
    a = pair_frames[left].set_index(["source_a", "source_b"])[metric]
    b = pair_frames[right].set_index(["source_a", "source_b"])[metric]
    joined = pd.concat([a.rename(left), b.rename(right)], axis=1).dropna()
    difference = joined[right] - joined[left]
    return {
        "metric": metric,
        "model_left": left,
        "model_right": right,
        "n_pairs": int(len(difference)),
        "difference_right_minus_left_mean": float(difference.mean()),
        "difference_std": float(difference.std(ddof=1)),
        "difference_median": float(difference.median()),
    }
