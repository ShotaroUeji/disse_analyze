"""Optional orthogonal Procrustes alignment between source embeddings."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .load_embeddings import EmbeddingRecord


def orthogonal_map(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return R minimizing ||source @ R - target||_F with R.T @ R=I."""
    cross = source.T @ target
    u, _, vh = np.linalg.svd(cross, full_matrices=False)
    return u @ vh


def mean_cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return float(np.mean(1.0 - np.sum(a * b, axis=1)))


def pair_procrustes(record_a: EmbeddingRecord, record_b: EmbeddingRecord, seed: int = 0) -> dict:
    if record_a.axis != record_b.axis or not np.allclose(record_a.angles_deg, record_b.angles_deg, atol=1e-6):
        raise ValueError("Procrustes pairs must have the same axis and angle grid")
    a = record_a.embeddings.numpy().astype(np.float64)
    b = record_b.embeddings.numpy().astype(np.float64)
    rng = np.random.default_rng(seed)
    train = np.arange(len(a))[::2]
    test = np.setdiff1d(np.arange(len(a)), train)
    rotation = orthogonal_map(a[train], b[train])
    before = mean_cosine_distance(a[test], b[test])
    after = mean_cosine_distance(a[test] @ rotation, b[test])
    shuffled = b[test].copy()
    rng.shuffle(shuffled, axis=0)
    shuffled_after = mean_cosine_distance(a[test] @ rotation, shuffled)
    return {
        "source_a": record_a.source_id,
        "source_b": record_b.source_id,
        "n_train_angles": len(train),
        "n_test_angles": len(test),
        "test_distance_before": before,
        "test_distance_after": after,
        "improvement": before - after,
        "shuffled_test_distance_after": shuffled_after,
    }


def analyze_records(records: list[EmbeddingRecord], output_dir: Path, seed: int = 0) -> dict:
    if len(records) < 2:
        raise ValueError("At least two sources are required for Procrustes")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [pair_procrustes(a, b, seed=seed) for a, b in combinations(sorted(records, key=lambda item: item.source_id), 2)]
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "procrustes_pairwise.csv", index=False)
    summary = frame.agg({
        "test_distance_before": ["mean", "median", "std"],
        "test_distance_after": ["mean", "median", "std"],
        "improvement": ["mean", "median", "std"],
        "shuffled_test_distance_after": ["mean", "median", "std"],
    })
    summary.to_csv(output_dir / "procrustes_summary.csv")
    return {"pair_frame": frame, "summary": summary}
