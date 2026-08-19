#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate_lkh_dummy_path_output.py

Evaluate a Hamiltonian path obtained from an LKH tour with a dummy node.

Input:
  - labels.tsv from make_lkh_path_tsp_from_elevation_embeddings.py
  - LKH .out tour file

Process:
  1. Read LKH tour.
  2. Find dummy node.
  3. Cut the cycle at dummy node.
  4. Remove dummy.
  5. Evaluate whether the resulting path is monotonic with physical elevation.

Output:
  - summary_*.json
  - path_*.csv
  - path_sequence_*.png
  - path_jumps_*.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_labels(path: Path):
    id_to_label = {}
    id_to_value = {}
    dummy_id = None

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            node_id = int(row["node_id"])
            label = row["label"]
            is_dummy = int(row["is_dummy"]) == 1

            id_to_label[node_id] = label

            if is_dummy:
                dummy_id = node_id
                id_to_value[node_id] = float("nan")
            else:
                id_to_value[node_id] = float(row["elevation_deg"])

    if dummy_id is None:
        raise ValueError("No dummy node found in labels file.")

    return id_to_label, id_to_value, dummy_id


def read_lkh_tour(path: Path):
    ids = []
    in_section = False

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s == "TOUR_SECTION":
                in_section = True
                continue
            if not in_section:
                continue
            if s == "-1" or s == "EOF":
                break
            if s:
                ids.append(int(s))

    if not ids:
        raise ValueError("No TOUR_SECTION ids found.")

    return ids


def cut_tour_at_dummy(tour_ids, dummy_id):
    """
    LKH returns a cycle.
    Rotate it so dummy is first, then remove dummy.

    Example:
      [3, 4, dummy, 1, 2] -> [1, 2, 3, 4]
    """
    if dummy_id not in tour_ids:
        raise ValueError(f"dummy_id={dummy_id} not found in tour.")

    k = tour_ids.index(dummy_id)
    rotated = tour_ids[k + 1:] + tour_ids[:k]
    return rotated


def rankdata_simple(x):
    """
    Simple rankdata without scipy.
    Handles ties by average rank.
    """
    x = np.asarray(x)
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=float)

    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j)
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1

    return ranks


def spearman_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rx = rankdata_simple(x)
    ry = rankdata_simple(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt(np.sum(rx * rx) * np.sum(ry * ry))
    if denom == 0:
        return float("nan")
    return float(np.sum(rx * ry) / denom)


def evaluate_path(elevations, tol_deg=2.0):
    elevations = np.asarray(elevations, dtype=float)
    n = len(elevations)
    steps = np.arange(n, dtype=float)

    jumps = np.abs(np.diff(elevations))
    if n > 1:
        ideal_step = (float(np.max(elevations)) - float(np.min(elevations))) / (n - 1)
    else:
        ideal_step = float("nan")

    rho = spearman_corr(steps, elevations)
    abs_rho = abs(rho)

    if len(jumps) > 0:
        edge_match_ratio = float(np.mean(np.abs(jumps - ideal_step) <= tol_deg))
    else:
        edge_match_ratio = float("nan")

    return {
        "n": int(n),
        "min_elevation_deg": float(np.min(elevations)),
        "max_elevation_deg": float(np.max(elevations)),
        "ideal_step_deg": float(ideal_step),
        "spearman_step_vs_elevation": float(rho),
        "abs_spearman_step_vs_elevation": float(abs_rho),
        "mean_abs_jump_deg": float(np.mean(jumps)) if len(jumps) else float("nan"),
        "median_abs_jump_deg": float(np.median(jumps)) if len(jumps) else float("nan"),
        "max_abs_jump_deg": float(np.max(jumps)) if len(jumps) else float("nan"),
        "edge_match_ratio_within_tol": edge_match_ratio,
        "tol_deg": float(tol_deg),
        "path_elevations_deg": [float(x) for x in elevations.tolist()],
        "path_jumps_deg": [float(x) for x in jumps.tolist()],
    }


def choose_orientation(path_ids, id_to_value):
    """
    Path direction is arbitrary.
    Choose orientation with positive Spearman correlation.
    """
    vals = np.array([id_to_value[i] for i in path_ids], dtype=float)
    rho = spearman_corr(np.arange(len(vals)), vals)

    if rho < 0:
        path_ids = list(reversed(path_ids))
        vals = vals[::-1]

    return path_ids, vals


def plot_path_sequence(elevations, out_path: Path, title: str):
    plt.figure(figsize=(9, 4.8))
    plt.plot(np.arange(len(elevations)), elevations, marker="o", linewidth=1.5)
    plt.xlabel("Path step")
    plt.ylabel("Elevation [deg]")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_path_jumps(elevations, out_path: Path, title: str):
    jumps = np.abs(np.diff(elevations))

    plt.figure(figsize=(9, 4.8))
    plt.bar(np.arange(len(jumps)), jumps)
    plt.xlabel("Path edge index")
    plt.ylabel("|Δ elevation| [deg]")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True)
    p.add_argument("--tour_out", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--tol_deg", type=float, default=2.0)
    args = p.parse_args()

    labels_path = Path(args.labels)
    tour_path = Path(args.tour_out)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    id_to_label, id_to_value, dummy_id = read_labels(labels_path)
    tour_ids = read_lkh_tour(tour_path)

    path_ids = cut_tour_at_dummy(tour_ids, dummy_id)
    path_ids, elevations = choose_orientation(path_ids, id_to_value)

    metrics = evaluate_path(elevations, tol_deg=args.tol_deg)

    labels = [id_to_label[i] for i in path_ids]

    summary = {
        "name": args.name,
        "labels_file": str(labels_path),
        "tour_out": str(tour_path),
        "dummy_id": int(dummy_id),
        **metrics,
        "path_node_ids": [int(i) for i in path_ids],
        "path_labels": labels,
    }

    with (out_dir / f"summary_{args.name}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with (out_dir / f"path_{args.name}.csv").open("w", encoding="utf-8") as f:
        f.write("path_step,node_id,label,elevation_deg,jump_to_next_deg\n")
        jumps = np.abs(np.diff(elevations))
        for i, node_id in enumerate(path_ids):
            jump = "" if i == len(path_ids) - 1 else f"{float(jumps[i]):.8f}"
            f.write(f"{i},{node_id},{id_to_label[node_id]},{float(elevations[i]):.8f},{jump}\n")

    plot_path_sequence(
        elevations,
        out_dir / f"path_sequence_{args.name}.png",
        f"{args.name}: elevation sequence along LKH dummy path",
    )

    plot_path_jumps(
        elevations,
        out_dir / f"path_jumps_{args.name}.png",
        f"{args.name}: elevation jumps along LKH dummy path",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[✓] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
