"""Figures for cross-source geometry analyses."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_average_heatmap(matrix: np.ndarray, angles_deg: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, origin="lower", cmap="viridis", vmin=0.0, vmax=2.0, aspect="equal")
    ticks = np.arange(0, len(angles_deg), max(1, len(angles_deg) // 12))
    ax.set_xticks(ticks, [f"{angles_deg[i]:g}" for i in ticks], rotation=45, ha="right")
    ax.set_yticks(ticks, [f"{angles_deg[i]:g}" for i in ticks])
    ax.set_xlabel("source A angle [deg]")
    ax.set_ylabel("source B angle [deg]")
    ax.set_title(title)
    bar = fig.colorbar(image, ax=ax)
    bar.set_label("cross-source cosine distance")
    _save(fig, path)


def plot_distance_distributions(pair_frame: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(
        [pair_frame["same_angle_mean"], pair_frame["different_angle_mean"], pair_frame["separation"]],
        labels=["same angle", "different angle", "Δ diff−same"],
        showmeans=True,
    )
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_ylabel("cosine distance")
    ax.set_title(title)
    _save(fig, path)


def plot_delta_curve(delta_frame: pd.DataFrame, title: str, path: Path) -> None:
    summary = (
        delta_frame.groupby("angular_difference_deg")["mean_cross_source_distance"]
        .agg(["mean", "std"])
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    x = summary["angular_difference_deg"].to_numpy()
    mean = summary["mean"].to_numpy()
    std = summary["std"].fillna(0.0).to_numpy()
    ax.plot(x, mean, marker="o", linewidth=1.8)
    ax.fill_between(x, mean - std, mean + std, alpha=0.2)
    ax.set_xlim(0, 180 if x.max() >= 180 else float(x.max()))
    ax.set_ylim(0, 2)
    ax.set_xlabel("angular difference δ [deg]")
    ax.set_ylabel("mean cross-source cosine distance")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    _save(fig, path)


def plot_retrieval_summary(summary: pd.DataFrame, title: str, path: Path) -> None:
    tasks = list(summary["task"])
    x = np.arange(len(tasks))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, summary["recall_at_1"], width, label="R@1")
    ax.bar(x + width / 2, summary["recall_at_5"], width, label="R@5")
    ax.set_xticks(x, tasks, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("recall")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _save(fig, path)


def plot_rsa_matrix(matrix: np.ndarray, source_ids: list[str], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
    ticks = np.arange(len(source_ids))
    ax.set_xticks(ticks, source_ids, rotation=45, ha="right")
    ax.set_yticks(ticks, source_ids)
    ax.set_title(title)
    bar = fig.colorbar(image, ax=ax)
    bar.set_label("Spearman RSA")
    _save(fig, path)


def plot_metric_comparison(frame: pd.DataFrame, metric: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    models = list(frame["model"])
    x = np.arange(len(models))
    ax.bar(x, frame[metric])
    ax.set_xticks(x, models)
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, path)


def plot_scatter(
    frame: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(frame[x], frame[y], alpha=0.8, edgecolors="black", linewidths=0.35)
    if len(frame) >= 2:
        coefficient = np.polyfit(frame[x].to_numpy(dtype=float), frame[y].to_numpy(dtype=float), 1)
        grid = np.linspace(float(frame[x].min()), float(frame[x].max()), 100)
        ax.plot(grid, coefficient[0] * grid + coefficient[1], color="tab:red", linewidth=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    _save(fig, path)


def plot_model_comparison(
    frame: pd.DataFrame,
    metrics: list[tuple[str, str]],
    title: str,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), squeeze=False)
    for axis, (column, label) in zip(axes[0], metrics):
        grouped = frame.groupby("model", as_index=False)[column].mean()
        x = np.arange(len(grouped))
        axis.bar(x, grouped[column])
        axis.set_xticks(x, grouped.model.tolist())
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, path)
