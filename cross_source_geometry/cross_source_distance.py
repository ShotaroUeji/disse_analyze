"""Cross-source cosine-distance analysis."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .load_embeddings import EmbeddingRecord


@dataclass
class PairDistance:
    source_a: str
    source_b: str
    distance_matrix: np.ndarray
    same_angle_mean: float
    different_angle_mean: float
    separation: float
    delta_means: dict[float, float]


def cosine_distance_matrix(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    a = torch.nn.functional.normalize(a.float(), dim=-1)
    b = torch.nn.functional.normalize(b.float(), dim=-1)
    matrix = (1.0 - a @ b.T).detach().cpu().numpy().astype(np.float64)
    return np.clip(matrix, 0.0, 2.0)


def angular_difference_deg(a: np.ndarray, b: np.ndarray, axis: str) -> np.ndarray:
    diff = np.abs(a[:, None] - b[None, :])
    if axis == "azimuth":
        diff = np.minimum(diff, 360.0 - diff)
    return diff


def pair_distance(record_a: EmbeddingRecord, record_b: EmbeddingRecord) -> PairDistance:
    if record_a.model != record_b.model or record_a.axis != record_b.axis:
        raise ValueError("Pair records must have the same model and axis")
    if not np.allclose(record_a.angles_deg, record_b.angles_deg, atol=1e-6):
        raise ValueError("Pair angle grids do not match")
    matrix = cosine_distance_matrix(record_a.embeddings, record_b.embeddings)
    n = matrix.shape[0]
    same = np.diag(matrix)
    offdiag = ~np.eye(n, dtype=bool)
    different = matrix[offdiag]
    delta = angular_difference_deg(record_a.angles_deg, record_b.angles_deg, record_a.axis)
    delta_means = {
        float(value): float(matrix[np.isclose(delta, value, atol=1e-6)].mean())
        for value in np.unique(delta)
    }
    same_mean = float(same.mean())
    different_mean = float(different.mean())
    return PairDistance(
        source_a=record_a.source_id,
        source_b=record_b.source_id,
        distance_matrix=matrix,
        same_angle_mean=same_mean,
        different_angle_mean=different_mean,
        separation=different_mean - same_mean,
        delta_means=delta_means,
    )


def analyze_records(
    records: list[EmbeddingRecord],
    output_dir: Path,
) -> dict:
    """Analyze all 45 unordered source pairs and write machine-readable data."""
    if len(records) < 2:
        raise ValueError("At least two sources are required")
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_dir = output_dir / "pair_distance_matrices"
    pair_dir.mkdir(exist_ok=True)
    pairs: list[PairDistance] = []
    for record_a, record_b in combinations(records, 2):
        result = pair_distance(record_a, record_b)
        pairs.append(result)
        # Symmetrizing makes the unordered pair representation independent of
        # which source happened to be listed first.
        np.save(pair_dir / f"{record_a.source_id}__{record_b.source_id}.npy", result.distance_matrix)

    average_matrix = np.mean([0.5 * (pair.distance_matrix + pair.distance_matrix.T) for pair in pairs], axis=0)
    np.save(output_dir / "average_cross_source_distance_matrix.npy", average_matrix)

    pair_rows = []
    delta_rows = []
    value_rows = []
    for pair in pairs:
        pair_rows.append({
            "source_a": pair.source_a,
            "source_b": pair.source_b,
            "same_angle_mean": pair.same_angle_mean,
            "different_angle_mean": pair.different_angle_mean,
            "separation": pair.separation,
        })
        for delta, value in pair.delta_means.items():
            delta_rows.append({
                "source_a": pair.source_a,
                "source_b": pair.source_b,
                "angular_difference_deg": delta,
                "mean_cross_source_distance": value,
            })
        for i, angle_i in enumerate(records[0].angles_deg):
            for j, angle_j in enumerate(records[0].angles_deg):
                value_rows.append({
                    "source_a": pair.source_a,
                    "source_b": pair.source_b,
                    "angle_a_deg": float(angle_i),
                    "angle_b_deg": float(angle_j),
                    "angular_difference_deg": float(
                        angular_difference_deg(
                            np.asarray([angle_i]), np.asarray([angle_j]), records[0].axis
                        )[0, 0]
                    ),
                    "cosine_distance": float(pair.distance_matrix[i, j]),
                    "same_angle": bool(i == j),
                })
    pd.DataFrame(pair_rows).to_csv(output_dir / "pair_distance_metrics.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(output_dir / "cross_source_distance_by_delta.csv", index=False)
    pd.DataFrame(value_rows).to_csv(output_dir / "cross_source_distance_values.csv", index=False)

    delta_df = pd.DataFrame(delta_rows)
    delta_summary = (
        delta_df.groupby("angular_difference_deg")["mean_cross_source_distance"]
        .agg(["mean", "std", "median"])
        .reset_index()
        .rename(columns={"mean": "distance_mean", "std": "distance_std", "median": "distance_median"})
    )
    delta_summary.to_csv(output_dir / "cross_source_distance_by_delta_summary.csv", index=False)

    return {
        "n_sources": len(records),
        "n_pairs": len(pairs),
        "n_points": records[0].n_points,
        "embedding_dim": records[0].embedding_dim,
        "average_matrix": average_matrix,
        "pair_rows": pair_rows,
        "delta_summary": delta_summary,
    }
