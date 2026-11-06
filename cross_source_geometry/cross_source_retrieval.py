"""Cross-source spatial retrieval evaluation."""

from __future__ import annotations

from itertools import combinations
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .cross_source_distance import cosine_distance_matrix
from .load_embeddings import EmbeddingRecord


def circular_error_deg(predicted: float, target: float, axis: str) -> float:
    diff = abs(float(predicted) - float(target))
    return float(min(diff, 360.0 - diff) if axis == "azimuth" else diff)


def chance_hit_rate(n_gallery: int, n_positive: int, k: int) -> float:
    if n_positive <= 0 or n_gallery <= 0:
        return 0.0
    k = min(k, n_gallery)
    if n_gallery - n_positive < k:
        return 1.0
    return 1.0 - math.comb(n_gallery - n_positive, k) / math.comb(n_gallery, k)


def _query_result(
    distances: np.ndarray,
    gallery_angles: np.ndarray,
    positive_mask: np.ndarray,
    target_angle: float,
    axis: str,
    *,
    model: str,
    task: str,
    query_source: str,
    gallery_source: str,
    query_angle: float,
) -> dict:
    order = np.argsort(distances, kind="stable")
    positive_positions = np.flatnonzero(positive_mask[order])
    if len(positive_positions) == 0:
        raise ValueError("A retrieval query has no positive gallery item")
    first_position = int(positive_positions[0])
    top1 = int(order[0])
    top1_angle = float(gallery_angles[top1])
    return {
        "model": model,
        "task": task,
        "query_source": query_source,
        "gallery_source": gallery_source,
        "query_angle_deg": float(query_angle),
        "n_gallery": int(len(distances)),
        "n_positive": int(positive_mask.sum()),
        "rank_first_positive": first_position + 1,
        "hit_r1": bool(first_position < 1),
        "hit_r5": bool(first_position < 5),
        "top1_angle_deg": top1_angle,
        "top1_circular_mae_deg": circular_error_deg(top1_angle, target_angle, axis),
        "top1_cosine_distance": float(distances[top1]),
        "chance_r1": chance_hit_rate(len(distances), int(positive_mask.sum()), 1),
        "chance_r5": chance_hit_rate(len(distances), int(positive_mask.sum()), 5),
    }


def analyze_records(records: list[EmbeddingRecord], output_dir: Path) -> dict:
    """Run pairwise, leave-one-source-out, and within-source-neighbor tasks."""
    if len(records) < 2:
        raise ValueError("At least two sources are required")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    model = records[0].model
    axis = records[0].axis
    records = sorted(records, key=lambda record: record.source_id)
    angles = records[0].angles_deg

    # Pairwise A->B and B->A exact-angle retrieval.
    for record_a, record_b in combinations(records, 2):
        matrix = cosine_distance_matrix(record_a.embeddings, record_b.embeddings)
        for query_record, gallery_record, distance_matrix in (
            (record_a, record_b, matrix),
            (record_b, record_a, matrix.T),
        ):
            for query_index, query_angle in enumerate(angles):
                positive = np.isclose(gallery_record.angles_deg, query_angle, atol=1e-6)
                rows.append(_query_result(
                    distance_matrix[query_index],
                    gallery_record.angles_deg,
                    positive,
                    float(query_angle),
                    axis,
                    model=model,
                    task="pairwise_cross_source",
                    query_source=query_record.source_id,
                    gallery_source=gallery_record.source_id,
                    query_angle=float(query_angle),
                ))

    # Leave-one-source-out: nine positive instances at the query angle.
    for query_record in records:
        gallery_records = [record for record in records if record.source_id != query_record.source_id]
        gallery_embeddings = torch.cat([record.embeddings for record in gallery_records], dim=0)
        gallery_angles = np.concatenate([record.angles_deg for record in gallery_records])
        matrix = cosine_distance_matrix(query_record.embeddings, gallery_embeddings)
        for query_index, query_angle in enumerate(angles):
            positive = np.isclose(gallery_angles, query_angle, atol=1e-6)
            result = _query_result(
                matrix[query_index], gallery_angles, positive, float(query_angle), axis,
                model=model, task="leave_one_source_out", query_source=query_record.source_id,
                gallery_source="other_sources", query_angle=float(query_angle),
            )
            rows.append(result)

    # Same-source comparison: exclude self and treat physical nearest angles as positives.
    for record in records:
        matrix = cosine_distance_matrix(record.embeddings, record.embeddings)
        physical = np.abs(record.angles_deg[:, None] - record.angles_deg[None, :])
        if axis == "azimuth":
            physical = np.minimum(physical, 360.0 - physical)
        np.fill_diagonal(physical, np.inf)
        for query_index, query_angle in enumerate(angles):
            gallery_mask = np.ones(record.n_points, dtype=bool)
            gallery_mask[query_index] = False
            gallery_angles = record.angles_deg[gallery_mask]
            distances = matrix[query_index, gallery_mask]
            min_physical = float(np.min(physical[query_index]))
            positive = np.isclose(physical[query_index, gallery_mask], min_physical, atol=1e-6)
            rows.append(_query_result(
                distances, gallery_angles, positive, float(query_angle), axis,
                model=model, task="within_source_physical_neighbor",
                query_source=record.source_id, gallery_source=record.source_id,
                query_angle=float(query_angle),
            ))

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "retrieval_queries.csv", index=False)
    summary = (
        frame.groupby(["model", "task"], as_index=False)
        .agg(
            n_queries=("hit_r1", "size"),
            recall_at_1=("hit_r1", "mean"),
            recall_at_5=("hit_r5", "mean"),
            median_rank=("rank_first_positive", "median"),
            circular_mae_deg=("top1_circular_mae_deg", "mean"),
            median_circular_mae_deg=("top1_circular_mae_deg", "median"),
            chance_r1=("chance_r1", "mean"),
            chance_r5=("chance_r5", "mean"),
        )
    )
    summary.to_csv(output_dir / "retrieval_summary.csv", index=False)
    pair_summary = (
        frame[frame.task == "pairwise_cross_source"]
        .groupby(["query_source", "gallery_source"], as_index=False)
        .agg(recall_at_1=("hit_r1", "mean"), recall_at_5=("hit_r5", "mean"), median_rank=("rank_first_positive", "median"), circular_mae_deg=("top1_circular_mae_deg", "mean"))
    )
    pair_summary.to_csv(output_dir / "pairwise_retrieval_by_source.csv", index=False)
    source_summary = (
        frame[frame.task == "leave_one_source_out"]
        .groupby("query_source", as_index=False)
        .agg(recall_at_1=("hit_r1", "mean"), recall_at_5=("hit_r5", "mean"), median_rank=("rank_first_positive", "median"), circular_mae_deg=("top1_circular_mae_deg", "mean"))
    )
    source_summary.to_csv(output_dir / "leave_one_source_out_by_source.csv", index=False)
    return {"queries": frame, "summary": summary}
