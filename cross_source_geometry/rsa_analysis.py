"""Representational Similarity Analysis over within-source distance matrices."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .cross_source_distance import cosine_distance_matrix
from .load_embeddings import EmbeddingRecord


def within_source_distance(record: EmbeddingRecord) -> np.ndarray:
    return cosine_distance_matrix(record.embeddings, record.embeddings)


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    indices = np.triu_indices(matrix.shape[0], k=1)
    return matrix[indices]


def rsa_pair(record_a: EmbeddingRecord, record_b: EmbeddingRecord) -> float:
    a = upper_triangle(within_source_distance(record_a))
    b = upper_triangle(within_source_distance(record_b))
    result = spearmanr(a, b)
    value = float(result.statistic)
    return value if np.isfinite(value) else 0.0


def analyze_records(records: list[EmbeddingRecord], output_dir: Path) -> dict:
    if len(records) < 2:
        raise ValueError("At least two sources are required for RSA")
    output_dir.mkdir(parents=True, exist_ok=True)
    matrices = {record.source_id: within_source_distance(record) for record in records}
    np.savez(output_dir / "within_source_distance_matrices.npz", **matrices)
    rows = []
    for record_a, record_b in combinations(sorted(records, key=lambda item: item.source_id), 2):
        rows.append({
            "source_a": record_a.source_id,
            "source_b": record_b.source_id,
            "rsa_spearman": rsa_pair(record_a, record_b),
        })
    pair_frame = pd.DataFrame(rows)
    pair_frame.to_csv(output_dir / "rsa_pairwise.csv", index=False)
    source_ids = sorted(record.source_id for record in records)
    rsa_matrix = np.eye(len(source_ids), dtype=np.float64)
    values = {(row.source_a, row.source_b): float(row.rsa_spearman) for row in pair_frame.itertuples()}
    for i, source_a in enumerate(source_ids):
        for j, source_b in enumerate(source_ids):
            if i < j:
                rsa_matrix[i, j] = rsa_matrix[j, i] = values[(source_a, source_b)]
    pd.DataFrame(rsa_matrix, index=source_ids, columns=source_ids).to_csv(output_dir / "rsa_matrix.csv")
    return {
        "pair_frame": pair_frame,
        "rsa_matrix": rsa_matrix,
        "mean_rsa": float(pair_frame.rsa_spearman.mean()),
        "median_rsa": float(pair_frame.rsa_spearman.median()),
    }
