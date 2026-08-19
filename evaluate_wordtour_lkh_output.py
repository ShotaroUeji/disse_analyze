#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate LKH tour output produced through WordTour pipeline.

Inputs:
  - labels file from export_disse_emb_to_wordtour.py
  - LKH .out file

The LKH node ids are 1-based.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_labels(path):
    id_to_label = {}
    id_to_angle = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            idx, label, angle = line.strip().split("\t")
            idx = int(idx)
            id_to_label[idx] = label
            id_to_angle[idx] = float(angle)
    return id_to_label, id_to_angle


def read_lkh_tour(path):
    ids = []
    in_section = False
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
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
    return ids


def circular_dist_deg(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def signed_forward_diff_deg(a_next, a_now):
    return (a_next - a_now) % 360.0


def infer_step_deg(angles):
    vals = np.sort(np.unique(np.round(angles, 8)))
    diffs = np.diff(np.r_[vals, vals[0] + 360.0])
    diffs = diffs[diffs > 1e-8]
    return float(np.min(diffs))


def plot_tour_circle(angles, out_path, title):
    theta = np.deg2rad(angles)
    x = np.cos(theta)
    y = np.sin(theta)

    plt.figure(figsize=(7, 7))
    ax = plt.gca()

    # Reference physical azimuth circle
    t = np.linspace(0, 2*np.pi, 512)
    ax.plot(
        np.cos(t),
        np.sin(t),
        color="lightgray",
        linewidth=1.0,
        zorder=1,
    )

    # Azimuth nodes
    ax.scatter(
        x,
        y,
        s=30,
        color="black",
        zorder=3,
        label="Azimuth samples",
    )

    # WordTour/LKH tour edges
    for i in range(len(angles)):
        a = i
        b = (i + 1) % len(angles)

        ax.plot(
            [x[a], x[b]],
            [y[a], y[b]],
            color="tab:blue",
            linewidth=1.3,
            alpha=0.75,
            zorder=2,
        )

    # Angle labels
    for deg in range(0, 360, 45):
        th = np.deg2rad(deg)
        ax.text(
            1.08*np.cos(th),
            1.08*np.sin(th),
            f"{deg}°",
            ha="center",
            va="center",
            fontsize=10,
        )

    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("cos(azimuth)")
    ax.set_ylabel("sin(azimuth)")

    # Small explanation box
    ax.text(
        -1.15,
        -1.18,
        "Black dots: physical azimuth samples\n"
        "Blue lines: WordTour/LKH adjacent pairs\n"
        "Gray circle: physical azimuth circle",
        fontsize=9,
        va="bottom",
        ha="left",
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_tour_sequence(angles, out_path, title):
    plt.figure(figsize=(9, 4.5))
    plt.plot(np.arange(len(angles)), angles, marker="o", linewidth=1.5)
    plt.xlabel("Tour step")
    plt.ylabel("Azimuth [deg]")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True)
    p.add_argument("--tour_out", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--name", default="audio_space")
    p.add_argument("--tol_deg", type=float, default=1.0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    id_to_label, id_to_angle = read_labels(args.labels)
    tour_ids = read_lkh_tour(args.tour_out)

    angles = np.array([id_to_angle[i] for i in tour_ids], dtype=float)
    labels = [id_to_label[i] for i in tour_ids]

    n = len(angles)
    step_deg = infer_step_deg(angles)

    circular_jumps = np.array([
        circular_dist_deg(angles[(i+1) % n], angles[i]) for i in range(n)
    ])

    forward_deltas = np.array([
        signed_forward_diff_deg(angles[(i+1) % n], angles[i]) for i in range(n)
    ])

    forward_ratio = float(np.mean(np.abs(forward_deltas - step_deg) <= args.tol_deg))
    reverse_ratio = float(np.mean(np.abs(forward_deltas - (360.0 - step_deg)) <= args.tol_deg))
    best_ratio = max(forward_ratio, reverse_ratio)

    summary = {
        "name": args.name,
        "n": int(n),
        "step_deg": float(step_deg),
        "mean_circular_jump_deg": float(np.mean(circular_jumps)),
        "median_circular_jump_deg": float(np.median(circular_jumps)),
        "max_circular_jump_deg": float(np.max(circular_jumps)),
        "ratio_edges_matching_forward_step": forward_ratio,
        "ratio_edges_matching_reverse_step": reverse_ratio,
        "best_circular_order_ratio": best_ratio,
        "tour_labels": labels,
        "tour_angles_deg": [float(x) for x in angles.tolist()],
        "tour_circular_jumps_deg": [float(x) for x in circular_jumps.tolist()],
    }

    with open(out_dir / f"summary_{args.name}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(out_dir / f"tour_{args.name}.csv", "w", encoding="utf-8") as f:
        f.write("step,label,azimuth_deg,circular_jump_to_next_deg\n")
        for i, (lab, az) in enumerate(zip(labels, angles)):
            f.write(f"{i},{lab},{az},{circular_jumps[i]}\n")

    plot_tour_circle(
        angles,
        out_dir / f"tour_circle_{args.name}.png",
        f"{args.name}: WordTour/LKH tour on azimuth circle",
    )

    plot_tour_sequence(
        angles,
        out_dir / f"tour_sequence_{args.name}.png",
        f"{args.name}: azimuth sequence along WordTour/LKH tour",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[✓] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()