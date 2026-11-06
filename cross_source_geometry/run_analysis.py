#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run cross-source distance, retrieval, RSA, and optional Procrustes analyses.

Example (the requested first experiment)::

    ./.venv/bin/python -m cross_source_geometry.run_analysis \
        --input-root results/continuity_additional_3x3_10sources \
        --axis azimuth --room medium --t30 0.3 \
        --models elsa disse \
        --experiments distance retrieval rsa \
        --output-root results/cross_source_geometry

Use ``--room all --t30 all --axis both`` after the pilot to process every
existing acoustic condition without regenerating audio or running inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .cross_source_distance import analyze_records as analyze_distance
from .cross_source_retrieval import analyze_records as analyze_retrieval
from .load_embeddings import discover_embedding_paths, group_records, load_collection
from .plotting import (
    plot_average_heatmap,
    plot_delta_curve,
    plot_distance_distributions,
    plot_metric_comparison,
    plot_model_comparison,
    plot_retrieval_summary,
    plot_rsa_matrix,
    plot_scatter,
)
from .procrustes_analysis import analyze_records as analyze_procrustes
from .rsa_analysis import analyze_records as analyze_rsa
from .statistics import bootstrap_pair_mean, paired_model_difference


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(type(value).__name__)


def _condition_key(room: str, t30: float) -> str:
    return f"{room}_t30_{str(float(t30)).replace('.', 'p')}s"


def _parse_t30(value: str) -> list[float] | None:
    if value == "all":
        return None
    return [float(value)]


def _safe_model_condition_key(group_key: tuple[str, str, str, str]) -> str:
    return f"{group_key[2]}_t30_{group_key[3].split('_t30_', 1)[1]}"


def _source_text(value: Any) -> str:
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _write_metadata(output_root: Path, records: list) -> None:
    rows = []
    for record in records:
        rows.append({
            "source_id": record.source_id,
            "model": record.model,
            "axis": record.axis,
            "room": record.room,
            "t30_s": record.t30_s,
            "condition": record.condition,
            "embedding_key": record.embedding_key,
            "embedding_path": str(record.path),
            "n_points": record.n_points,
            "embedding_dim": record.embedding_dim,
            "norm_min": record.source_norm_min,
            "norm_max": record.source_norm_max,
        })
    pd.DataFrame(rows).sort_values(["axis", "room", "t30_s", "model", "source_id"]).to_csv(
        output_root / "input_manifest.csv", index=False
    )


def _plot_rsa_pairs(pair_frame: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(pair_frame["rsa_spearman"], bins=12, alpha=0.8, edgecolor="black")
    ax.set_xlim(-1, 1)
    ax.set_xlabel("RSA Spearman correlation")
    ax.set_ylabel("number of source pairs")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_report(output_root: Path, run_rows: list[dict], command: str) -> None:
    lines = [
        "# Cross-source geometry report",
        "",
        "## 実行概要",
        "",
        f"実行コマンド: `{command}`",
        "",
        "保存済みのDISSE/ELSA埋め込みだけを読み込み、音声生成・RIR生成・モデル推論は行っていない。",
        "方位角は周期的、仰角は非周期的な変数として扱った。45音源ペアは独立標本とみなさず、必要な要約では音源ID単位のbootstrapを使用する。",
        "",
        "## 条件別の主要結果",
        "",
        "| axis | room | T30 [s] | model | sources | dim | D_same | D_diff | Δ | Retrieval R@1 (LOSO) | RSA mean |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in run_rows:
        lines.append(
            f"| {row['axis']} | {row['room']} | {row['t30_s']:.1f} | {row['model']} | "
            f"{row['n_sources']} | {row['embedding_dim']} | {row.get('same_angle_mean', float('nan')):.4f} | "
            f"{row.get('different_angle_mean', float('nan')):.4f} | {row.get('separation', float('nan')):.4f} | "
            f"{row.get('loso_recall_at_1', float('nan')):.4f} | {row.get('rsa_mean', float('nan')):.4f} |"
        )
    lines.extend([
        "",
        "## 解釈上の注意",
        "",
        "- D_same が小さいことは、同じ角度の音源間埋め込みが近いことを示す。",
        "- RSA は音源内の距離構造の一致度であり、埋め込み座標の位置合わせそのものを証明しない。",
        "- 高いRSAと大きいCross-source距離が同時に得られた場合、位置・向きのずれの可能性は示唆されるが、回転を断定しない。Procrustesは追加解析として扱う。",
        "- Retrievalの正例は同一物理角度の別音源候補であり、chance levelを各CSVに併記した。",
        "",
        "## 出力",
        "",
        "条件別ディレクトリのCSVとPNG/PDF、`input_manifest.csv`、`aggregate_results.csv`を参照する。",
    ])
    (output_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--axis", choices=("azimuth", "elevation", "both"), default="azimuth")
    parser.add_argument("--room", choices=("small", "medium", "large", "all"), default="medium")
    parser.add_argument("--t30", default="0.3", help="Target T30 in seconds or 'all'.")
    parser.add_argument("--models", nargs="+", choices=("disse", "elsa"), default=["elsa", "disse"])
    parser.add_argument("--experiments", nargs="+", choices=("distance", "retrieval", "rsa", "procrustes"), default=["distance", "retrieval", "rsa"])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("--bootstrap should be at least 100")

    axes = ("azimuth", "elevation") if args.axis == "both" else (args.axis,)
    rooms = ("small", "medium", "large") if args.room == "all" else (args.room,)
    t30_values = _parse_t30(args.t30)
    paths = discover_embedding_paths(args.input_root, models=args.models, axes=axes, rooms=rooms, t30_s=t30_values)
    if not paths:
        parser.error("No matching embedding files were found")
    records = load_collection(paths)
    groups = group_records(records)
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_metadata(args.output_root, records)

    run_rows: list[dict] = []
    model_distance_pairs: dict[tuple[str, str, str, str], dict[str, pd.DataFrame]] = {}
    for group_key, group in sorted(groups.items()):
        model, axis, room, condition = group_key
        t30_s = group[0].t30_s
        condition_dir = args.output_root / axis / condition / model
        row = {
            "model": model,
            "axis": axis,
            "room": room,
            "t30_s": t30_s,
            "condition": condition,
            "n_sources": len(group),
            "embedding_dim": group[0].embedding_dim,
        }
        if "distance" in args.experiments:
            result = analyze_distance(group, condition_dir / "distance")
            pair_frame = pd.DataFrame(result["pair_rows"])
            model_distance_pairs.setdefault((axis, room, condition), {})[model] = pair_frame
            for metric in ("same_angle_mean", "different_angle_mean", "separation"):
                stats = bootstrap_pair_mean(pair_frame, metric, n_sources=len(group), seed=args.seed, n_bootstrap=args.bootstrap)
                _write_json(condition_dir / "distance" / f"bootstrap_{metric}.json", stats)
                row[metric] = stats["mean"]
            plot_average_heatmap(result["average_matrix"], group[0].angles_deg, f"{model.upper()} {axis} {condition}: average cross-source distance", condition_dir / "distance" / "average_cross_source_distance_heatmap.png")
            plot_distance_distributions(pair_frame, f"{model.upper()} {axis} {condition}: pair distances", condition_dir / "distance" / "same_vs_different_distance.png")
            plot_delta_curve(pd.read_csv(condition_dir / "distance" / "cross_source_distance_by_delta.csv"), f"{model.upper()} {axis} {condition}: distance by angular difference", condition_dir / "distance" / "distance_by_delta.png")
        if "retrieval" in args.experiments:
            result = analyze_retrieval(group, condition_dir / "retrieval")
            summary = result["summary"]
            retrieval_pair_frame = pd.read_csv(condition_dir / "retrieval" / "pairwise_retrieval_by_source.csv")
            loso = summary[summary.task == "leave_one_source_out"]
            if not loso.empty:
                row["loso_recall_at_1"] = float(loso.recall_at_1.iloc[0])
            plot_retrieval_summary(summary, f"{model.upper()} {axis} {condition}: retrieval", condition_dir / "retrieval" / "retrieval_summary.png")
        if "rsa" in args.experiments:
            result = analyze_rsa(group, condition_dir / "rsa")
            rsa_pair_frame = result["pair_frame"].copy()
            row["rsa_mean"] = result["mean_rsa"]
            plot_rsa_matrix(result["rsa_matrix"], sorted(record.source_id for record in group), f"{model.upper()} {axis} {condition}: RSA", condition_dir / "rsa" / "rsa_matrix.png")
            _plot_rsa_pairs(result["pair_frame"], f"{model.upper()} {axis} {condition}: RSA distribution", condition_dir / "rsa" / "rsa_distribution.png")
        if "distance" in args.experiments and "rsa" in args.experiments:
            distance_pair_frame = pd.read_csv(condition_dir / "distance" / "pair_distance_metrics.csv")
            distance_pair_frame["pair_key"] = distance_pair_frame.apply(lambda item: "__".join(sorted((_source_text(item.source_a), _source_text(item.source_b)))), axis=1)
            rsa_pair_frame["pair_key"] = rsa_pair_frame.apply(lambda item: "__".join(sorted((_source_text(item.source_a), _source_text(item.source_b)))), axis=1)
            relation = distance_pair_frame.merge(rsa_pair_frame[["pair_key", "rsa_spearman"]], on="pair_key")
            relation.to_csv(condition_dir / "rsa_distance_relations.csv", index=False)
            plot_scatter(
                relation,
                "rsa_spearman",
                "same_angle_mean",
                f"{model.upper()} {axis} {condition}: RSA vs same-angle distance",
                "RSA Spearman correlation",
                "same-angle cross-source distance",
                condition_dir / "rsa_distance_relations.png",
            )
        if "retrieval" in args.experiments and "rsa" in args.experiments:
            retrieval_pair_frame["pair_key"] = retrieval_pair_frame.apply(lambda item: "__".join(sorted((_source_text(item.query_source), _source_text(item.gallery_source)))), axis=1)
            retrieval_pair_frame = retrieval_pair_frame.groupby("pair_key", as_index=False).agg(recall_at_1=("recall_at_1", "mean"), recall_at_5=("recall_at_5", "mean"))
            rsa_pair_frame["pair_key"] = rsa_pair_frame.apply(lambda item: "__".join(sorted((_source_text(item.source_a), _source_text(item.source_b)))), axis=1)
            relation = rsa_pair_frame[["pair_key", "rsa_spearman"]].merge(retrieval_pair_frame, on="pair_key")
            relation.to_csv(condition_dir / "rsa_retrieval_relations.csv", index=False)
            plot_scatter(
                relation,
                "rsa_spearman",
                "recall_at_1",
                f"{model.upper()} {axis} {condition}: RSA vs cross-source R@1",
                "RSA Spearman correlation",
                "pairwise cross-source R@1",
                condition_dir / "rsa_retrieval_relations.png",
            )
        if "procrustes" in args.experiments:
            analyze_procrustes(group, condition_dir / "procrustes", seed=args.seed)
        _write_json(condition_dir / "run_metadata.json", row)
        run_rows.append(row)

    aggregate = pd.DataFrame(run_rows)
    aggregate.to_csv(args.output_root / "aggregate_results.csv", index=False)
    metric_columns = [
        "same_angle_mean",
        "different_angle_mean",
        "separation",
        "loso_recall_at_1",
        "rsa_mean",
    ]
    available_metrics = [column for column in metric_columns if column in aggregate.columns]
    if available_metrics:
        overall = (
            aggregate.groupby(["model", "axis"], as_index=False)[available_metrics]
            .mean()
            .assign(n_conditions=aggregate.groupby(["model", "axis"]).size().to_numpy())
        )
        overall.to_csv(args.output_root / "overall_summary_macro.csv", index=False)
        by_room = (
            aggregate.groupby(["model", "axis", "room"], as_index=False)[available_metrics]
            .mean()
            .assign(n_conditions=aggregate.groupby(["model", "axis", "room"]).size().to_numpy())
        )
        by_room.to_csv(args.output_root / "summary_by_room.csv", index=False)
        aggregate = aggregate.copy()
        aggregate["reverb_group"] = np.where(aggregate["t30_s"] == 0.0, "dry", "reverberant")
        by_reverb = (
            aggregate.groupby(["model", "axis", "reverb_group"], as_index=False)[available_metrics]
            .mean()
            .assign(n_conditions=aggregate.groupby(["model", "axis", "reverb_group"]).size().to_numpy())
        )
        by_reverb.to_csv(args.output_root / "summary_dry_vs_reverberant.csv", index=False)
        by_t30 = (
            aggregate.groupby(["model", "axis", "t30_s"], as_index=False)[available_metrics]
            .mean()
            .assign(n_conditions=aggregate.groupby(["model", "axis", "t30_s"]).size().to_numpy())
        )
        by_t30.to_csv(args.output_root / "summary_by_t30.csv", index=False)
    if len(set(aggregate.model)) > 1:
        plot_model_comparison(
            aggregate,
            [
                ("same_angle_mean", "D_same (lower is better)"),
                ("separation", "Δ = D_diff − D_same"),
                ("loso_recall_at_1", "LOSO R@1"),
                ("rsa_mean", "RSA mean"),
            ],
            "DISSE / ELSA cross-source geometry comparison",
            args.output_root / "model_comparison.png",
        )
        for axis_name in sorted(set(aggregate.axis)):
            plot_model_comparison(
                aggregate[aggregate.axis == axis_name],
                [
                    ("same_angle_mean", "D_same (lower is better)"),
                    ("separation", "Δ = D_diff − D_same"),
                    ("loso_recall_at_1", "LOSO R@1"),
                    ("rsa_mean", "RSA mean"),
                ],
                f"DISSE / ELSA {axis_name} cross-source comparison",
                args.output_root / f"model_comparison_{axis_name}.png",
            )

    # Paired model comparisons within each acoustic condition.
    comparison_rows = []
    for condition_key, frames in model_distance_pairs.items():
        if len(frames) == 2:
            for metric in ("same_angle_mean", "different_angle_mean", "separation"):
                comparison_rows.append({**dict(zip(("axis", "room", "condition"), condition_key)), **paired_model_difference(frames, metric)})
    if comparison_rows:
        pd.DataFrame(comparison_rows).to_csv(args.output_root / "paired_model_distance_comparison.csv", index=False)
    _write_report(args.output_root, run_rows, " ".join(str(item) for item in __import__("sys").argv))
    print(f"[done] analyzed {len(groups)} model/axis/condition groups")
    print(f"[saved] {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
