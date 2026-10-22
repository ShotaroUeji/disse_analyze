#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the standard sequential cosine-distance circle for all saved runs.

The input layout is the output layout of ``run_continuity_experiment.py``::

    <input-root>/<source>/<model>/<axis>/<condition>/*embeddings.pt

One PNG is written per embedding file.  A single manifest CSV records the
source, model, axis, room/reverberation condition, input path, output path,
number of points, and the circumference scale used by the plot.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from plot_embedding_cosine_polar import (
    _reference_index,
    adjacent_cosine_distances,
    cosine_distances_from_reference,
    cumulative_circle_positions,
    load_embedding_file,
    make_distance_circle_plot,
)


def _discover(input_root: Path, model: str | None, axis: str | None):
    patterns = []
    models = (model,) if model else ("disse", "elsa")
    axes = (axis,) if axis else ("azimuth", "elevation")
    for model_name in models:
        for axis_name in axes:
            filename = (
                f"continuous_{axis_name}_embeddings.pt"
                if model_name == "disse"
                else "elsa_embeddings.pt"
            )
            patterns.append(input_root.glob(f"*/{model_name}/{axis_name}/*/{filename}"))
    paths = [path for iterator in patterns for path in iterator]
    return sorted(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("results/continuity_additional_3x3_10sources"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/visualizations/cosine_distance_circle_10sources"),
    )
    parser.add_argument("--model", choices=("disse", "elsa"))
    parser.add_argument("--axis", choices=("azimuth", "elevation"))
    parser.add_argument("--source", help="Only process one source ID.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--label-step", type=int, default=12)
    args = parser.parse_args()
    if args.label_step < 1:
        parser.error("--label-step must be at least 1")

    paths = _discover(args.input_root, args.model, args.axis)
    if args.source:
        paths = [path for path in paths if path.parts[-5] == args.source]
    if not paths:
        parser.error(f"No embedding files found below {args.input_root}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.csv"
    rows = []
    for index, emb_path in enumerate(paths, start=1):
        source_id = emb_path.parts[-5]
        model = emb_path.parts[-4]
        axis = emb_path.parts[-3]
        condition = emb_path.parts[-2]
        key = "audio_space" if model == "disse" else "audio_elsa"
        output_path = args.output_root / source_id / model / axis / condition / "distance_circle.png"
        if output_path.exists() and not args.overwrite:
            rows.append({
                "source_id": source_id,
                "model": model,
                "axis": axis,
                "condition": condition,
                "embedding_key": key,
                "embedding_path": str(emb_path.resolve()),
                "plot_path": str(output_path.resolve()),
                "n_points": "skipped_existing",
                "distance_circle_scale": "",
            })
            continue

        embeddings, values_deg, key = load_embedding_file(emb_path, axis, key)
        reference_index = _reference_index(values_deg, axis)
        reference_distances = cosine_distances_from_reference(embeddings, reference_index)
        edge_endpoints, edge_distances = adjacent_cosine_distances(embeddings, values_deg, axis)
        theta, _, scale = cumulative_circle_positions(
            values_deg, edge_endpoints, edge_distances, axis
        )
        make_distance_circle_plot(
            values_deg,
            theta,
            edge_endpoints,
            edge_distances,
            axis,
            title=f"{model.upper()} | source {source_id} | {axis} | {condition}",
            output_path=output_path,
            label_step=args.label_step,
        )
        rows.append({
            "source_id": source_id,
            "model": model,
            "axis": axis,
            "condition": condition,
            "embedding_key": key,
            "embedding_path": str(emb_path.resolve()),
            "plot_path": str(output_path.resolve()),
            "n_points": str(len(values_deg)),
            "distance_circle_scale": f"{scale:.12g}",
        })
        print(f"[{index}/{len(paths)}] {output_path}")

    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "source_id",
            "model",
            "axis",
            "condition",
            "embedding_key",
            "embedding_path",
            "plot_path",
            "n_points",
            "distance_circle_scale",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {manifest_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
