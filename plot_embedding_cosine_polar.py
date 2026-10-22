#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot spatial embeddings as cosine-distance polar diagrams.

The embedding at 0 degrees is used as the reference.  For every angle ``i``
the plotted radius is

    radius[i] = (1 - cos(z_0, z_i)) / 2

The division by two maps the cosine distance range [0, 2] to the unit-circle
range [0, 1].  The unscaled cosine distance is also written to a CSV file.

Azimuth uses the physical angle around the full circle (0 degrees is at the
top and angles increase clockwise).  Elevation uses its physical degree value
as the polar angle, so -45 to +45 degrees appears as an arc around 0 degrees.

``--plot-mode unit`` additionally creates a unit-circle view.  In that view
all points have radius 1; point colour shows distance from 0 degrees and edge
colour shows the cosine distance between adjacent physical angles.  This keeps
the angular layout visible without pretending that arbitrary embedding
distances can all be represented as radii on the same circle.

``--plot-mode distance-circle`` creates the sequential distance view: the
first point is placed at 0 degrees, and the next point is separated by an arc
proportional to the cosine distance of the adjacent embeddings.  The complete
sequence is scaled to the circumference of the unit circle; the raw distances
and scale factor are reported in the CSV.

Examples
--------
DISSE azimuth::

    ./.venv/bin/python plot_embedding_cosine_polar.py \
      --emb-pt results/continuity_additional_3x3_10sources/107280/disse/azimuth/small_t30_0p3s/continuous_azimuth_embeddings.pt \
      --axis azimuth --key audio_space

ELSA elevation::

    ./.venv/bin/python plot_embedding_cosine_polar.py \
      --emb-pt results/continuity_additional_3x3_10sources/107280/elsa/elevation/small_t30_0p3s/elsa_embeddings.pt \
      --axis elevation --key audio_elsa
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def load_embedding_file(
    path: Path, axis: str, key: str | None
) -> tuple[torch.Tensor, np.ndarray, str]:
    """Load and validate one saved DISSE or ELSA embedding file."""
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(obj, dict):
        raise ValueError("The .pt file must contain a dictionary.")

    saved_axis = obj.get("axis") or obj.get("sweep_name")
    if saved_axis in ("azimuth", "elevation") and saved_axis != axis:
        raise ValueError(f"The file contains axis={saved_axis!r}, not {axis!r}.")

    angle_names = ("azimuth_deg", "angles_deg") if axis == "azimuth" else (
        "elevations_deg", "angles_deg"
    )
    angle_name = next((name for name in angle_names if name in obj and obj[name] is not None), None)
    if angle_name is None:
        raise ValueError(f"No angle array found; expected one of {angle_names}.")
    values_deg = _as_numpy(obj[angle_name]).astype(np.float64).reshape(-1)

    available = obj.get("embeddings")
    if not isinstance(available, dict) or not available:
        raise ValueError("The .pt file must contain a non-empty 'embeddings' dictionary.")
    if key is None:
        preferred = "audio_space" if "audio_space" in available else "audio_elsa"
        key = preferred if preferred in available else next(iter(available))
    if key not in available:
        raise ValueError(f"Embedding key {key!r} not found; available: {list(available)}")
    embeddings = available[key]
    if not torch.is_tensor(embeddings):
        raise ValueError(f"embeddings[{key!r}] must be a tensor.")
    embeddings = embeddings.detach().float().cpu()

    if values_deg.ndim != 1 or len(values_deg) != embeddings.shape[0] or len(values_deg) < 2:
        raise ValueError("The angle array and embeddings must have the same length (at least 2).")
    if not np.isfinite(values_deg).all() or not torch.isfinite(embeddings).all():
        raise ValueError("Angles and embeddings must contain only finite values.")
    if embeddings.ndim != 2 or embeddings.shape[1] < 1:
        raise ValueError("Embeddings must have shape [N, D], D >= 1.")
    if torch.any(torch.linalg.vector_norm(embeddings, dim=1) == 0):
        raise ValueError("Cosine distance is undefined for a zero embedding vector.")
    if axis == "azimuth":
        values_deg = values_deg % 360.0
    return embeddings, values_deg, key


def cosine_distances_from_reference(embeddings: torch.Tensor, reference_index: int) -> np.ndarray:
    normalized = F.normalize(embeddings.float(), dim=-1)
    reference = normalized[reference_index]
    distances = 1.0 - normalized @ reference
    return distances.detach().cpu().numpy().astype(np.float64).clip(0.0, 2.0)


def _reference_index(values_deg: np.ndarray, axis: str) -> int:
    target = 0.0
    distances = np.abs(values_deg - target)
    if axis == "azimuth":
        distances = np.minimum(distances, 360.0 - distances)
    index = int(np.argmin(distances))
    if not np.isclose(distances[index], 0.0):
        raise ValueError("A 0-degree embedding is required as the cosine-distance reference.")
    return index


def adjacent_cosine_distances(
    embeddings: torch.Tensor, values_deg: np.ndarray, axis: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return physical-order edge endpoints and their cosine distances."""
    normalized = F.normalize(embeddings.float(), dim=-1)
    matrix = (1.0 - normalized @ normalized.T).detach().cpu().numpy().astype(np.float64)
    matrix = np.clip(matrix, 0.0, 2.0)
    order = np.argsort(values_deg, kind="stable")
    edges = list(zip(order[:-1], order[1:]))
    if axis == "azimuth":
        edges.append((int(order[-1]), int(order[0])))
    endpoints = np.asarray(edges, dtype=np.int64)
    distances = np.asarray([matrix[i, j] for i, j in edges], dtype=np.float64)
    return endpoints, distances


def _selected_labels(values_deg: np.ndarray, axis: str, label_step: int) -> set[int]:
    order = np.argsort(values_deg, kind="stable")
    selected = set(order[:: max(1, label_step)].tolist())
    selected.add(_reference_index(values_deg, axis))
    if axis == "elevation":
        selected.update({int(order[0]), int(order[-1])})
    return selected


def make_radial_plot(
    values_deg: np.ndarray,
    distances: np.ndarray,
    axis: str,
    title: str,
    output_path: Path,
    label_step: int,
) -> None:
    # Polar convention: zero at the top and clockwise direction.
    theta = np.deg2rad(values_deg)
    radius = distances / 2.0
    order = np.argsort(values_deg, kind="stable")

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0.0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.5", "1.0", "1.5", "2.0"])
    ax.set_ylabel("cosine distance (raw)", labelpad=25)
    ax.set_title(title, pad=20)

    # Connect in physical angular order. Azimuth is a closed cycle; elevation
    # is an open path from -45 to +45 degrees.
    line_order = order.tolist()
    if axis == "azimuth":
        line_order.append(line_order[0])
    ax.plot(theta[line_order], radius[line_order], color="0.45", linewidth=1.0, alpha=0.8)
    points = ax.scatter(theta, radius, c=values_deg, cmap="twilight" if axis == "azimuth" else "coolwarm",
                        s=38, edgecolors="black", linewidths=0.35, zorder=3)
    colorbar = fig.colorbar(points, ax=ax, pad=0.10, shrink=0.82)
    colorbar.set_label(f"{axis} [deg]")

    # Label every selected point, always including the 0-degree reference and endpoints.
    for index in sorted(_selected_labels(values_deg, axis, label_step)):
        ax.annotate(
            f"{values_deg[index]:g}°",
            (theta[index], radius[index]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )

    fig.text(
        0.02,
        0.02,
        "radius = cosine distance / 2; reference = 0° embedding",
        fontsize=8,
        color="0.35",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_unit_plot(
    values_deg: np.ndarray,
    reference_distances: np.ndarray,
    edge_endpoints: np.ndarray,
    edge_distances: np.ndarray,
    axis: str,
    title: str,
    output_path: Path,
    label_step: int,
) -> None:
    """Plot all physical-angle points on the unit circle.

    Point colour encodes distance from the 0-degree embedding.  Each physical
    adjacency edge is coloured by its embedding cosine distance.  For azimuth,
    the final-to-first edge uses the shortest wraparound arc.
    """
    theta = np.deg2rad(values_deg)
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0.0, 1.12)
    ax.set_yticks([0.5, 1.0])
    ax.set_yticklabels(["", "unit circle"])
    ax.set_title(title, pad=20)

    edge_norm = matplotlib.colors.Normalize(vmin=0.0, vmax=2.0)
    edge_cmap = plt.get_cmap("plasma")
    for (i, j), distance in zip(edge_endpoints, edge_distances):
        t0, t1 = float(theta[i]), float(theta[j])
        if axis == "azimuth" and abs(t1 - t0) > np.pi:
            if t1 < t0:
                t1 += 2.0 * np.pi
            else:
                t0 += 2.0 * np.pi
        ax.plot([t0, t1], [1.0, 1.0], color=edge_cmap(edge_norm(distance)), linewidth=2.0, alpha=0.85)

    point_norm = matplotlib.colors.Normalize(vmin=0.0, vmax=2.0)
    point_cmap = plt.get_cmap("viridis")
    points = ax.scatter(
        theta,
        np.ones_like(theta),
        c=reference_distances,
        cmap=point_cmap,
        norm=point_norm,
        s=52,
        edgecolors="black",
        linewidths=0.45,
        zorder=3,
    )
    point_bar = fig.colorbar(points, ax=ax, pad=0.10, shrink=0.82)
    point_bar.set_label("distance from 0° embedding")
    edge_bar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=edge_norm, cmap=edge_cmap),
        ax=ax,
        pad=0.16,
        shrink=0.82,
    )
    edge_bar.set_label("adjacent cosine distance")

    for index in sorted(_selected_labels(values_deg, axis, label_step)):
        ax.annotate(
            f"{values_deg[index]:g}°",
            (theta[index], 1.0),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    fig.text(
        0.02,
        0.02,
        "all points are on the unit circle; point/edge colours encode cosine distances",
        fontsize=8,
        color="0.35",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def cumulative_circle_positions(
    values_deg: np.ndarray,
    edge_endpoints: np.ndarray,
    edge_distances: np.ndarray,
    axis: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Map adjacent cosine distances to cumulative unit-circle positions.

    Azimuth is a closed cycle, so all edge distances are scaled to exactly
    2*pi.  Elevation is an open path, so the cumulative path is scaled to the
    physical elevation span (normally 90 degrees from -45 to +45), rather
    than being expanded to a full circle.
    """
    order = np.argsort(values_deg, kind="stable")
    total_distance = float(np.sum(edge_distances))
    if total_distance <= np.finfo(np.float64).eps:
        raise ValueError("All adjacent cosine distances are zero; no distance path can be plotted.")
    if axis == "azimuth":
        start_theta = 0.0
        target_span = 2.0 * np.pi
    else:
        ordered_values = values_deg[order]
        start_theta = float(np.deg2rad(ordered_values[0]))
        target_span = float(np.deg2rad(ordered_values[-1] - ordered_values[0]))
        if target_span <= np.finfo(np.float64).eps:
            raise ValueError("Elevation span must be greater than zero.")
    scale = target_span / total_distance

    theta_by_index = np.zeros(len(values_deg), dtype=np.float64)
    theta = start_theta
    for edge_index, point_index in enumerate(order):
        theta_by_index[int(point_index)] = theta
        if edge_index < len(edge_distances):
            theta += float(edge_distances[edge_index]) * scale
    return theta_by_index, edge_distances * scale, scale


def make_distance_circle_plot(
    values_deg: np.ndarray,
    theta: np.ndarray,
    edge_endpoints: np.ndarray,
    edge_distances: np.ndarray,
    axis: str,
    title: str,
    output_path: Path,
    label_step: int,
) -> None:
    """Plot points on a unit circle with spacing determined by adjacent distance."""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0.0, 1.12)
    if axis == "elevation":
        # Elevation is a bounded -45..+45 degree path, not a 360 degree cycle.
        ax.set_thetamin(-45.0)
        ax.set_thetamax(45.0)
    ax.set_yticks([1.0])
    ax.set_yticklabels(["unit circle"])
    ax.set_title(title, pad=20)

    edge_norm = matplotlib.colors.Normalize(vmin=0.0, vmax=2.0)
    edge_cmap = plt.get_cmap("plasma")
    for (i, j), distance in zip(edge_endpoints, edge_distances):
        t0, t1 = float(theta[i]), float(theta[j])
        if axis == "azimuth" and abs(t1 - t0) > np.pi:
            if t1 < t0:
                t1 += 2.0 * np.pi
            else:
                t0 += 2.0 * np.pi
        ax.plot(
            [t0, t1],
            [1.0, 1.0],
            color=edge_cmap(edge_norm(distance)),
            linewidth=3.0,
            alpha=0.9,
        )

    points = ax.scatter(
        theta,
        np.ones_like(theta),
        c=values_deg,
        cmap="twilight" if axis == "azimuth" else "coolwarm",
        s=52,
        edgecolors="black",
        linewidths=0.45,
        zorder=3,
    )
    point_bar = fig.colorbar(points, ax=ax, pad=0.10, shrink=0.82)
    point_bar.set_label(f"original {axis} [deg]")
    edge_bar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=edge_norm, cmap=edge_cmap),
        ax=ax,
        pad=0.16,
        shrink=0.82,
    )
    edge_bar.set_label("raw adjacent cosine distance")

    for index in sorted(_selected_labels(values_deg, axis, label_step)):
        ax.annotate(
            f"{values_deg[index]:g}°",
            (theta[index], 1.0),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    fig.text(
        0.02,
        0.02,
        "arc spacing is proportional to adjacent embedding cosine distance",
        fontsize=8,
        color="0.35",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_csv(
    path: Path,
    values_deg: np.ndarray,
    distances: np.ndarray,
    axis: str,
    edge_endpoints: np.ndarray,
    edge_distances: np.ndarray,
    distance_circle_theta: np.ndarray | None = None,
    distance_circle_steps: np.ndarray | None = None,
    distance_circle_scale: float | None = None,
) -> None:
    theta = np.deg2rad(values_deg)
    radius = distances / 2.0
    adjacent_to_next = {int(i): float(distance) for (i, _), distance in zip(edge_endpoints, edge_distances)}
    distance_circle_step_by_from = (
        {int(i): float(step) for (i, _), step in zip(edge_endpoints, distance_circle_steps)}
        if distance_circle_steps is not None
        else {}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            axis + "_deg",
            "cosine_distance_from_0deg_raw",
            "radius_unit_circle",
            "x",
            "y",
            "adjacent_cosine_distance_to_next_raw",
            "distance_circle_theta_deg",
            "distance_circle_arc_step",
        ])
        for index, (value, distance, r, t) in enumerate(zip(values_deg, distances, radius, theta)):
            # With zero at top and clockwise rotation: x=r*sin(theta), y=r*cos(theta).
            next_distance = adjacent_to_next.get(index)
            next_circle_step = distance_circle_step_by_from.get(index)
            writer.writerow([
                f"{value:g}",
                f"{distance:.10g}",
                f"{r:.10g}",
                f"{r * np.sin(t):.10g}",
                f"{r * np.cos(t):.10g}",
                "" if next_distance is None else f"{next_distance:.10g}",
                "" if distance_circle_theta is None else f"{np.rad2deg(distance_circle_theta[index]):.10g}",
                "" if next_circle_step is None else f"{next_circle_step:.10g}",
            ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emb-pt", "--emb_pt", type=Path, required=True)
    parser.add_argument("--axis", choices=("azimuth", "elevation"), required=True)
    parser.add_argument("--key", default=None, help="Embedding key; auto-detects audio_space/audio_elsa when omitted.")
    parser.add_argument("--out", type=Path, help="PNG path; default is beside the input .pt.")
    parser.add_argument("--csv", type=Path, help="CSV path; default is beside the output PNG.")
    parser.add_argument("--label-step", type=int, default=12, help="Label every Nth point (default: 12).")
    parser.add_argument(
        "--plot-mode",
        choices=("radial", "unit", "distance-circle", "both", "all"),
        default="distance-circle",
        help=(
            "radial: distance as radius; unit: fixed physical positions on unit circle; "
            "distance-circle: cumulative adjacent-distance positions (default); "
            "both: radial+unit; all: all three."
        ),
    )
    args = parser.parse_args()
    if args.label_step < 1:
        parser.error("--label-step must be at least 1")

    try:
        embeddings, values_deg, key = load_embedding_file(args.emb_pt, args.axis, args.key)
        reference_index = _reference_index(values_deg, args.axis)
        distances = cosine_distances_from_reference(embeddings, reference_index)
        edge_endpoints, edge_distances = adjacent_cosine_distances(embeddings, values_deg, args.axis)
        distance_circle_theta, distance_circle_steps, distance_circle_scale = cumulative_circle_positions(
            values_deg, edge_endpoints, edge_distances, args.axis
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    default_stem = f"embedding_cosine_polar_{args.axis}_{key}"
    output_path = args.out or args.emb_pt.parent / f"{default_stem}.png"
    csv_path = args.csv or output_path.with_suffix(".csv")
    if args.plot_mode in ("radial", "both", "all"):
        make_radial_plot(
            values_deg,
            distances,
            args.axis,
            title=f"{args.axis.capitalize()} embedding cosine distance ({key})",
            output_path=output_path,
            label_step=args.label_step,
        )
    if args.plot_mode in ("unit", "both", "all"):
        unit_output = output_path.with_name(output_path.stem + "_unit" + output_path.suffix)
        make_unit_plot(
            values_deg,
            distances,
            edge_endpoints,
            edge_distances,
            args.axis,
            title=f"{args.axis.capitalize()} embedding on the unit circle ({key})",
            output_path=unit_output,
            label_step=args.label_step,
        )
        print(f"[saved] {unit_output}")
    if args.plot_mode in ("distance-circle", "all"):
        distance_output = output_path.with_name(output_path.stem + "_distance_circle" + output_path.suffix)
        make_distance_circle_plot(
            values_deg,
            distance_circle_theta,
            edge_endpoints,
            edge_distances,
            args.axis,
            title=f"{args.axis.capitalize()} sequential cosine-distance circle ({key})",
            output_path=distance_output,
            label_step=args.label_step,
        )
        print(f"[saved] {distance_output}")
    write_csv(
        csv_path,
        values_deg,
        distances,
        args.axis,
        edge_endpoints,
        edge_distances,
        distance_circle_theta,
        distance_circle_steps,
        distance_circle_scale,
    )
    if args.plot_mode in ("radial", "both", "all"):
        print(f"[saved] {output_path}")
    print(f"[saved] {csv_path}")
    print(f"axis={args.axis} key={key} n={len(values_deg)} reference_angle={values_deg[reference_index]:g}°")
    print(f"cosine_distance_range={distances.min():.6f}..{distances.max():.6f}")
    print(f"distance_circle_scale={distance_circle_scale:.6f} rad per raw cosine distance")


if __name__ == "__main__":
    main()
