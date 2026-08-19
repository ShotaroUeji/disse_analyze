#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_azimuth_tsp.py

Use embeddings from continuous_azimuth_heatmap.py
and analyze whether azimuth embeddings form a circular 1D order
via approximate TSP (nearest-neighbor + 2-opt).

Input:
  continuous_azimuth_embeddings.pt

Output:
  - summary.json
  - tour_audio_space.csv / tour_audio_source.csv / ...
  - circle plots
  - tour-step plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def cosine_distance_matrix(Z: torch.Tensor) -> np.ndarray:
    Z = F.normalize(Z.float(), dim=-1)
    D = 1.0 - (Z @ Z.T)
    D = D.detach().cpu().numpy()
    return np.clip(D, 0.0, 2.0)


def tour_length(D: np.ndarray, tour: list[int]) -> float:
    n = len(tour)
    return float(sum(D[tour[i], tour[(i + 1) % n]] for i in range(n)))


def nearest_neighbor_tour(D: np.ndarray, start: int) -> list[int]:
    n = D.shape[0]
    unvisited = set(range(n))
    unvisited.remove(start)
    tour = [start]
    current = start

    while unvisited:
        nxt = min(unvisited, key=lambda j: D[current, j])
        tour.append(nxt)
        unvisited.remove(nxt)
        current = nxt

    return tour


def two_opt(D: np.ndarray, tour: list[int], max_iter: int = 2000) -> list[int]:
    n = len(tour)
    best = tour[:]
    best_len = tour_length(D, best)
    improved = True
    it = 0

    while improved and it < max_iter:
        improved = False
        it += 1
        for i in range(1, n - 2):
            for k in range(i + 1, n - 1):
                new_tour = best[:i] + best[i:k + 1][::-1] + best[k + 1:]
                new_len = tour_length(D, new_tour)
                if new_len + 1e-12 < best_len:
                    best = new_tour
                    best_len = new_len
                    improved = True
        # loop again if improved
    return best


def solve_tsp_multistart(D: np.ndarray) -> tuple[list[int], float]:
    n = D.shape[0]
    best_tour = None
    best_len = float("inf")

    for start in range(n):
        t = nearest_neighbor_tour(D, start)
        t = two_opt(D, t)
        L = tour_length(D, t)
        if L < best_len:
            best_len = L
            best_tour = t

    return best_tour, best_len


def circular_dist_deg(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def signed_forward_diff_deg(a_next: float, a_now: float) -> float:
    """Return forward modular difference in [0, 360)."""
    return (a_next - a_now) % 360.0


def infer_step_deg(angles_deg: np.ndarray) -> float:
    vals = np.sort(np.unique(np.round(angles_deg, 8)))
    diffs = np.diff(np.r_[vals, vals[0] + 360.0])
    diffs = diffs[diffs > 1e-8]
    return float(np.min(diffs))


def analyze_tour_angles(angles_deg: np.ndarray, tour: list[int], tol: float = 1.0) -> dict:
    seq = angles_deg[tour]
    n = len(seq)
    step_deg = infer_step_deg(angles_deg)

    circular_jumps = np.array([
        circular_dist_deg(seq[(i + 1) % n], seq[i]) for i in range(n)
    ], dtype=float)

    forward_deltas = np.array([
        signed_forward_diff_deg(seq[(i + 1) % n], seq[i]) for i in range(n)
    ], dtype=float)

    forward_ratio = np.mean(np.abs(forward_deltas - step_deg) <= tol)
    reverse_ratio = np.mean(np.abs(forward_deltas - (360.0 - step_deg)) <= tol)
    best_ratio = max(forward_ratio, reverse_ratio)

    return {
        "step_deg": step_deg,
        "mean_circular_jump_deg": float(np.mean(circular_jumps)),
        "median_circular_jump_deg": float(np.median(circular_jumps)),
        "max_circular_jump_deg": float(np.max(circular_jumps)),
        "min_circular_jump_deg": float(np.min(circular_jumps)),
        "ratio_edges_matching_forward_step": float(forward_ratio),
        "ratio_edges_matching_reverse_step": float(reverse_ratio),
        "best_circular_order_ratio": float(best_ratio),
        "tour_angles_deg": [float(x) for x in seq.tolist()],
        "tour_circular_jumps_deg": [float(x) for x in circular_jumps.tolist()],
        "tour_forward_deltas_deg": [float(x) for x in forward_deltas.tolist()],
    }


def plot_tour_on_circle(
    angles_deg: np.ndarray,
    tour: list[int],
    title: str,
    out_path: Path,
):
    theta = np.deg2rad(angles_deg)
    x = np.cos(theta)
    y = np.sin(theta)

    plt.figure(figsize=(7, 7))
    ax = plt.gca()

    # unit circle
    t = np.linspace(0, 2*np.pi, 512)
    ax.plot(np.cos(t), np.sin(t), linewidth=1)

    # all nodes
    ax.scatter(x, y, s=30)

    # edges in TSP tour
    for i in range(len(tour)):
        a = tour[i]
        b = tour[(i + 1) % len(tour)]
        ax.plot([x[a], x[b]], [y[a], y[b]], linewidth=1.2)

    # label every 45 deg
    for deg in range(0, 360, 45):
        th = np.deg2rad(deg)
        ax.text(1.08*np.cos(th), 1.08*np.sin(th), f"{deg}°",
                ha="center", va="center", fontsize=10)

    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("cos(azimuth)")
    ax.set_ylabel("sin(azimuth)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_tour_sequence(
    angles_deg: np.ndarray,
    tour: list[int],
    title: str,
    out_path: Path,
):
    seq = angles_deg[tour]
    n = len(seq)

    plt.figure(figsize=(9, 4.5))
    plt.plot(np.arange(n), seq, marker="o", linewidth=1.5)
    plt.xlabel("Tour step")
    plt.ylabel("Azimuth [deg]")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_tour_csv(angles_deg: np.ndarray, tour: list[int], out_path: Path):
    seq = angles_deg[tour]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("tour_step,node_index,azimuth_deg\n")
        for i, idx in enumerate(tour):
            f.write(f"{i},{idx},{float(seq[i])}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_pt", required=True,
                        help="continuous_azimuth_embeddings.pt from continuous_azimuth_heatmap.py")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--keys", nargs="+",
                        default=["audio_space", "audio_source", "text_space", "text_source"])
    parser.add_argument("--tol_deg", type=float, default=1.0,
                        help="tolerance for matching ideal +step or -step edges")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    obj = torch.load(args.emb_pt, map_location="cpu")
    angles_deg = obj["angles_deg"].cpu().numpy().astype(float)
    emb_dict = obj["embeddings"]

    summary = {
        "emb_pt": str(args.emb_pt),
        "n_angles": int(len(angles_deg)),
        "angles_deg": [float(x) for x in angles_deg.tolist()],
        "results": {}
    }

    for key in args.keys:
        if key not in emb_dict:
            print(f"[skip] {key} not found in embeddings.")
            continue

        print(f"[info] solving TSP for {key}")
        Z = emb_dict[key]
        D = cosine_distance_matrix(Z)

        tour, L = solve_tsp_multistart(D)
        metrics = analyze_tour_angles(angles_deg, tour, tol=args.tol_deg)
        metrics["tour_length"] = float(L)

        summary["results"][key] = metrics

        save_tour_csv(angles_deg, tour, out_dir / f"tour_{key}.csv")
        np.save(out_dir / f"distance_matrix_{key}.npy", D)

        plot_tour_on_circle(
            angles_deg,
            tour,
            title=f"{key}: TSP tour on azimuth circle",
            out_path=out_dir / f"tour_circle_{key}.png",
        )

        plot_tour_sequence(
            angles_deg,
            tour,
            title=f"{key}: azimuth sequence along TSP tour",
            out_path=out_dir / f"tour_sequence_{key}.png",
        )

        print(
            f"  tour_length={L:.4f} | "
            f"best_circular_order_ratio={metrics['best_circular_order_ratio']:.3f} | "
            f"mean_jump={metrics['mean_circular_jump_deg']:.3f}"
        )

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[✓] wrote outputs to: {out_dir.resolve()}")
    print("Recommended files:")
    print(f"  - {out_dir / 'summary.json'}")
    print(f"  - {out_dir / 'tour_circle_audio_space.png'}")
    print(f"  - {out_dir / 'tour_sequence_audio_space.png'}")
    print(f"  - {out_dir / 'tour_circle_audio_source.png'}")
    print(f"  - {out_dir / 'tour_sequence_audio_source.png'}")


if __name__ == "__main__":
    main()