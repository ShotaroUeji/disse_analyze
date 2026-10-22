#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute DISSE QNX(2) and Sadj from saved continuous-angle embeddings.

Example::

    python evaluate_disse_continuity.py \
      --emb-pt results/azimuth/no_reverb_continuous_azimuth_ep20_src107280_dry/continuous_azimuth_embeddings.pt \
      --axis azimuth

The default output is ``continuity_metrics_<key>.json`` beside the input .pt.
Only NumPy and PyTorch are required. No checkpoint inference or LKH run is needed.
Metric definitions and stable nearest-neighbor tie handling match
``evaluate_elsa_continuity.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_embeddings(
    path: Path, key: str, axis: str
) -> tuple[torch.Tensor, np.ndarray]:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    sweep_name = obj.get("sweep_name")
    if sweep_name in ("azimuth", "elevation") and sweep_name != axis:
        raise ValueError(f"Input sweep is {sweep_name!r}, but --axis is {axis!r}.")

    angle_keys = (
        ("elevations_deg", "angles_deg")
        if axis == "elevation"
        else ("azimuth_deg", "angles_deg")
    )
    angle_key = next((name for name in angle_keys if name in obj), None)
    if angle_key is None:
        raise ValueError(f"Input must contain one of these angle keys: {angle_keys}.")
    values = obj[angle_key]
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 3 or not np.isfinite(values).all():
        raise ValueError("Angles must be a finite one-dimensional array with N >= 3.")
    if axis == "azimuth":
        values = values % 360.0
    if len(np.unique(values)) != len(values):
        raise ValueError("Sweep angles must be unique (modulo 360 for azimuth).")

    available = obj.get("embeddings", {})
    if key not in available:
        raise ValueError(f"Embedding key {key!r} not found; available: {list(available)}")
    embeddings = available[key]
    if not torch.is_tensor(embeddings):
        raise ValueError(f"embeddings[{key!r}] must be a tensor.")
    embeddings = embeddings.detach().float().cpu()
    if (
        embeddings.ndim != 2
        or embeddings.shape[0] != len(values)
        or embeddings.shape[1] < 1
    ):
        raise ValueError("Embeddings must have shape [N, D] matching the angles, D >= 1.")
    if not torch.isfinite(embeddings).all():
        raise ValueError("Embeddings must contain only finite values.")
    if torch.any(torch.linalg.vector_norm(embeddings, dim=1) == 0):
        raise ValueError("Cosine distance is undefined for zero embedding vectors.")
    return embeddings, values


def continuity_metrics(
    embeddings: torch.Tensor, values_deg: np.ndarray, axis: str
) -> dict:
    """Use embedding cosine distance; azimuth is a cycle, elevation is a path."""
    embeddings = F.normalize(embeddings.float(), dim=-1)
    cosine = 1.0 - embeddings @ embeddings.T
    distance = np.clip(cosine.cpu().numpy().astype(np.float64), 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)

    physical = np.abs(values_deg[:, None] - values_deg[None, :])
    if axis == "azimuth":
        physical = np.minimum(physical, 360.0 - physical)

    # Exclude each point itself; stable sorting matches the ELSA evaluator.
    embedding_neighbors = distance.copy()
    physical_neighbors = physical.copy()
    np.fill_diagonal(embedding_neighbors, np.inf)
    np.fill_diagonal(physical_neighbors, np.inf)
    embedding_neighbors = np.argsort(embedding_neighbors, axis=1, kind="stable")[:, :2]
    physical_neighbors = np.argsort(physical_neighbors, axis=1, kind="stable")[:, :2]
    overlaps = [
        len(set(emb.tolist()) & set(phys.tolist())) / 2.0
        for emb, phys in zip(embedding_neighbors, physical_neighbors)
    ]

    # Physical adjacency is angular order, independently of any WordTour route.
    order = np.argsort(values_deg, kind="stable").tolist()
    edges = list(zip(order[:-1], order[1:]))
    if axis == "azimuth":
        edges.append((order[-1], order[0]))
    adjacent = np.asarray([distance[i, j] for i, j in edges], dtype=np.float64)
    denominator = float(np.sum(adjacent ** 2))
    # Collapsed embeddings have undefined Sadj; serialize it as JSON null.
    sadj = (
        float(np.sum((adjacent - adjacent.mean()) ** 2)) / denominator
        if denominator > np.finfo(np.float64).eps
        else None
    )
    return {
        "QNX(2)": float(np.mean(overlaps)),
        "Sadj": sadj,
        "definitions": {
            "QNX(2)": (
                "Mean overlap between each point's two nearest physical neighbors "
                "and two nearest embedding neighbors; higher is better."
            ),
            "Sadj": (
                "sum((d_adj-mean(d_adj))^2)/sum(d_adj^2), where d_adj is "
                "1-cosine for physically adjacent points; lower is better. "
                "Azimuth includes the wraparound edge; elevation excludes it. "
                "null means the denominator is numerically zero."
            ),
        },
        "diagnostics": {
            "qnx2_per_point": overlaps,
            "n_adjacent_edges": len(edges),
            "adjacent_cosine_distance_mean": float(adjacent.mean()),
            "adjacent_cosine_distance_std": float(adjacent.std()),
            "physical_adjacent_distances": [
                {
                    "from_index": i,
                    "to_index": j,
                    "from_deg": float(values_deg[i]),
                    "to_deg": float(values_deg[j]),
                    "cosine_distance": float(d),
                }
                for (i, j), d in zip(edges, adjacent)
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emb-pt", "--emb_pt", type=Path, required=True)
    parser.add_argument("--axis", choices=("azimuth", "elevation"), required=True)
    parser.add_argument("--key", default="audio_space", help="Embedding key (default: audio_space).")
    parser.add_argument(
        "--out-json", "--out_json", type=Path,
        help="Default: continuity_metrics_<key>.json beside the input .pt.",
    )
    args = parser.parse_args()
    try:
        embeddings, values_deg = load_embeddings(args.emb_pt, args.key, args.axis)
    except (OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))
    scores = continuity_metrics(embeddings, values_deg, args.axis)
    result = {
        "model": "DISSE",
        "emb_pt": str(args.emb_pt.resolve()),
        "axis": args.axis,
        "embedding_key": args.key,
        "n_points": len(values_deg),
        "embedding_dim": embeddings.shape[1],
        "values_deg": values_deg.tolist(),
        "scores": scores,
    }
    output = args.out_json or args.emb_pt.parent / f"continuity_metrics_{args.key}.json"
    serialized = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8")
    print(f"DISSE | axis={args.axis} | key={args.key} | N={len(values_deg)}")
    print(f"QNX(2): {scores['QNX(2)']:.9f}")
    print(f"Sadj  : {scores['Sadj']:.9f}" if scores["Sadj"] is not None else "Sadj  : undefined")
    print(f"[saved] {output}")


if __name__ == "__main__":
    main()
