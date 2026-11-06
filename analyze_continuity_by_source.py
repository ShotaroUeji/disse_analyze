#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze per-source QNX(2) and Sadj from an existing continuity summary.

This script does not rerun inference. It reads the per-source rows already
written by ``run_continuity_experiment.py`` and creates source-level summaries,
model comparisons, T30 trends, and compact plots.

Example
-------
    ./.venv/bin/python analyze_continuity_by_source.py \
        --summary results/continuity_additional_3x3_10sources/summary.csv \
        --output-dir results/continuity_additional_3x3_10sources/by_source_analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "source",
    "model",
    "axis",
    "room",
    "target_t30_s",
    "QNX(2)",
    "Sadj",
}


def _rank_within_group(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["QNX(2)_rank_desc"] = frame.groupby(["model", "axis"])["qnx_mean"].rank(
        ascending=False, method="min"
    ).astype(int)
    frame["Sadj_rank_asc"] = frame.groupby(["model", "axis"])["sadj_mean"].rank(
        ascending=True, method="min"
    ).astype(int)
    return frame


def _slope(group: pd.DataFrame, column: str) -> float:
    by_t30 = group.groupby("target_t30_s", as_index=False)[column].mean().sort_values("target_t30_s")
    if len(by_t30) < 2:
        return float("nan")
    return float(np.polyfit(by_t30["target_t30_s"], by_t30[column], deg=1)[0])


def _make_source_summary(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        frame.groupby(["source", "model", "axis"], as_index=False)
        .agg(
            n_conditions=("QNX(2)", "size"),
            qnx_mean=("QNX(2)", "mean"),
            qnx_std=("QNX(2)", "std"),
            qnx_min=("QNX(2)", "min"),
            qnx_max=("QNX(2)", "max"),
            sadj_mean=("Sadj", "mean"),
            sadj_std=("Sadj", "std"),
            sadj_min=("Sadj", "min"),
            sadj_max=("Sadj", "max"),
        )
    )
    trends = []
    for (source, model, axis), group in frame.groupby(["source", "model", "axis"]):
        t30 = group.groupby("target_t30_s", as_index=False)[["QNX(2)", "Sadj"]].mean().sort_values("target_t30_s")
        qnx_dry = float(t30.loc[t30.target_t30_s == 0.0, "QNX(2)"].iloc[0])
        sadj_dry = float(t30.loc[t30.target_t30_s == 0.0, "Sadj"].iloc[0])
        qnx_reverb = float(t30.loc[t30.target_t30_s == 0.6, "QNX(2)"].iloc[0])
        sadj_reverb = float(t30.loc[t30.target_t30_s == 0.6, "Sadj"].iloc[0])
        trends.append(
            {
                "source": source,
                "model": model,
                "axis": axis,
                "qnx_t30_slope": _slope(group, "QNX(2)"),
                "sadj_t30_slope": _slope(group, "Sadj"),
                "qnx_0p6_minus_0p0": qnx_reverb - qnx_dry,
                "sadj_0p6_minus_0p0": sadj_reverb - sadj_dry,
            }
        )
    result = grouped.merge(pd.DataFrame(trends), on=["source", "model", "axis"], how="left")
    return _rank_within_group(result)


def _make_model_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    pivot = frame.pivot_table(
        index=["source", "axis", "room", "target_t30_s"],
        columns="model",
        values=["QNX(2)", "Sadj"],
        aggfunc="mean",
    )
    pivot.columns = [f"{metric}_{model}" for metric, model in pivot.columns]
    pivot = pivot.reset_index()
    required = ["QNX(2)_disse", "QNX(2)_elsa", "Sadj_disse", "Sadj_elsa"]
    missing = [column for column in required if column not in pivot.columns]
    if missing:
        raise ValueError(f"Both models are required for model comparison; missing {missing}")
    pivot["qnx_disse_minus_elsa"] = pivot["QNX(2)_disse"] - pivot["QNX(2)_elsa"]
    pivot["sadj_disse_minus_elsa"] = pivot["Sadj_disse"] - pivot["Sadj_elsa"]
    pivot["disse_better_qnx"] = pivot["qnx_disse_minus_elsa"] > 0
    pivot["disse_better_sadj"] = pivot["sadj_disse_minus_elsa"] < 0
    result = (
        pivot.groupby(["source", "axis"], as_index=False)
        .agg(
            n_conditions=("qnx_disse_minus_elsa", "size"),
            qnx_disse_minus_elsa_mean=("qnx_disse_minus_elsa", "mean"),
            sadj_disse_minus_elsa_mean=("sadj_disse_minus_elsa", "mean"),
            qnx_disse_better_fraction=("disse_better_qnx", "mean"),
            sadj_disse_better_fraction=("disse_better_sadj", "mean"),
        )
    )
    return result.sort_values(["axis", "source"])


def _heatmap(data: pd.DataFrame, value_columns: list[str], title: str, output: Path, *, cmap: str) -> None:
    sources = sorted(data["source"].unique())
    values = data.set_index("source").reindex(sources)[value_columns].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    im = ax.imshow(values, aspect="auto", cmap=cmap)
    ax.set_yticks(np.arange(len(sources)), sources)
    ax.set_xticks(np.arange(len(value_columns)), value_columns, rotation=30, ha="right")
    ax.set_title(title)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="value")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_scatter(summary: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    styles = {("disse", "azimuth"): ("tab:blue", "o"), ("disse", "elevation"): ("tab:cyan", "s"),
              ("elsa", "azimuth"): ("tab:red", "o"), ("elsa", "elevation"): ("tab:orange", "s")}
    for (model, axis), group in summary.groupby(["model", "axis"]):
        color, marker = styles[(model, axis)]
        ax.scatter(group["qnx_mean"], group["sadj_mean"], label=f"{model.upper()} {axis}", color=color, marker=marker, s=55)
        for row in group.itertuples():
            ax.annotate(str(row.source), (row.qnx_mean, row.sadj_mean), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("mean QNX(2) (higher is better)")
    ax.set_ylabel("mean Sadj (lower is better)")
    ax.set_title("Per-source QNX(2) vs Sadj")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_report(
    output: Path,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    input_path: Path,
    captions: pd.DataFrame | None,
) -> None:
    lines = [
        "# 音源ごとの QNX(2)・Sadj 解析",
        "",
        f"入力: `{input_path}`",
        "",
        "この解析は既存の `summary.csv` を読み込むだけで、モデル推論や音声生成は行っていない。各音源IDについて、9音響条件（3部屋×3 T30）の平均・標準偏差・最小値・最大値を求めた。",
        "",
        "## 指標の読み方",
        "",
        "- QNX(2): 高いほど、物理近傍が埋め込み近傍として保存される。",
        "- Sadj: 低いほど、物理的に隣接する点の埋め込み距離が均一である。",
        "- `qnx_t30_slope` と `sadj_t30_slope` は、部屋サイズを平均して T30=0, 0.3, 0.6 s に対して線形近似した記述的傾き。",
        "",
        "## 音源イベント別の上位・下位",
        "",
    ]
    for (model, axis), group in summary.groupby(["model", "axis"]):
        lines.append(f"### {model.upper()} / {axis}")
        q = group.sort_values("qnx_mean", ascending=False)
        s = group.sort_values("sadj_mean", ascending=True)
        lines.append("")
        lines.append(f"- QNX(2) 上位: {', '.join(f'{r.source} ({r.qnx_mean:.3f})' for r in q.head(3).itertuples())}")
        lines.append(f"- QNX(2) 下位: {', '.join(f'{r.source} ({r.qnx_mean:.3f})' for r in q.tail(3).sort_values('qnx_mean').itertuples())}")
        lines.append(f"- Sadj 上位（低い順）: {', '.join(f'{r.source} ({r.sadj_mean:.3f})' for r in s.head(3).itertuples())}")
        lines.append(f"- Sadj 下位（高い順）: {', '.join(f'{r.source} ({r.sadj_mean:.3f})' for r in s.tail(3).sort_values('sadj_mean', ascending=False).itertuples())}")
        lines.append("")

    if captions is not None and not captions.empty:
        lines.extend([
            "## 音源IDとAudioCapsイベント",
            "",
            "音源IDだけではイベント内容を解釈できないため、AudioCapsのcaptionを併記する。以下のcaptionは実験で使用した音源IDとの対応確認用であり、カテゴリ効果の統計的検定ではない。",
            "",
            "| source | caption |",
            "|---|---|",
        ])
        for row in captions.sort_values("source").itertuples():
            lines.append(f"| {row.source} | {row.caption} |")
        lines.append("")

    lines.extend([
        "## 残響に対する音源イベント別傾向",
        "",
        "傾きは、各部屋を平均した QNX(2) または Sadj を T30=0, 0.3, 0.6 s に対して近似した記述量である。QNX(2) は負の傾き、Sadj は正の傾きが残響による悪化方向を表す。",
        "",
        "| model | axis | QNXが最も低下する音源 | QNXが最も安定する音源 | Sadjが最も悪化する音源 | Sadjが最も安定する音源 |",
        "|---|---|---|---|---|---|",
    ])
    for (model, axis), group in summary.groupby(["model", "axis"]):
        q_down = group.sort_values("qnx_t30_slope").iloc[0]
        q_stable = group.sort_values("qnx_t30_slope", ascending=False).iloc[0]
        s_worse = group.sort_values("sadj_t30_slope", ascending=False).iloc[0]
        s_stable = group.sort_values("sadj_t30_slope").iloc[0]

        def event_label(row: pd.Series, column: str) -> str:
            caption = getattr(row, "caption", "")
            suffix = "" if pd.isna(caption) or not str(caption) else f"; {str(caption)[:52]}"
            return f"{row['source']} ({row[column]:+.3f}{suffix})"

        lines.append(
            f"| {model.upper()} | {axis} | {event_label(q_down, 'qnx_t30_slope')} | "
            f"{event_label(q_stable, 'qnx_t30_slope')} | {event_label(s_worse, 'sadj_t30_slope')} | "
            f"{event_label(s_stable, 'sadj_t30_slope')} |"
        )
    lines.append("")

    lines.extend([
        "## DISSE と ELSA の音源別比較",
        "",
        "QNX(2) は DISSE−ELSA が正なら DISSE が高く、Sadj は DISSE−ELSA が負なら DISSE が低く（良く）なる。次の比較は9条件平均である。",
        "",
        "| axis | source | ΔQNX (DISSE−ELSA) | ΔSadj (DISSE−ELSA) | DISSEがQNXで優位な条件割合 | DISSEがSadjで優位な条件割合 |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in comparison.itertuples():
        lines.append(f"| {row.axis} | {row.source} | {row.qnx_disse_minus_elsa_mean:.3f} | {row.sadj_disse_minus_elsa_mean:.3f} | {row.qnx_disse_better_fraction:.2f} | {row.sadj_disse_better_fraction:.2f} |")
    lines.extend([
        "",
        "## 解釈",
        "",
        "- 方位角では、DISSE の QNX(2) は10音源すべてで ELSA より高い。一方、Sadj は ELSA の方が低い音源が多く、近傍の保存と隣接距離の均一性は一致しない。",
        "- 仰角では、音源によって QNX(2) のモデル優位性が変わる。したがって、仰角については10音源平均だけで DISSE の一貫した優位性を主張せず、音源イベント別のばらつきを併記すべきである。",
        "- 音源IDごとの差は、音源内容そのものの影響と、収録音量・無音区間・音源長などの前処理差の影響を含む。IDだけから「人の声」「環境音」などの意味カテゴリを断定するには、AudioCapsのキャプション等との対応付けが必要である。",
        "- 本結果は記述統計であり、10音源だけで母集団全体への統計的有意性を主張しない。論文では音源をランダム効果とする解析、または音源ID bootstrapを追加する。",
        "",
        "## 推奨する論文での示し方",
        "",
        "1. 全体平均表に加えて、音源ID×モデル×軸のヒートマップを補足図にする。",
        "2. 本文では各軸のQNX(2)上位・下位音源を1〜2例だけ示し、全10音源表は補足表に置く。",
        "3. 音源内容による説明を行う場合は、音源IDとキャプション／カテゴリを結合し、カテゴリ別平均を別途計算する。",
    ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--captions-csv",
        type=Path,
        default=Path("AudioCaps_csv/test.csv"),
        help="Optional AudioCaps CSV with audiocap_id and caption columns.",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.summary, dtype={"source": str})
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        parser.error(f"summary is missing columns: {missing}")
    frame["QNX(2)"] = pd.to_numeric(frame["QNX(2)"], errors="raise")
    frame["Sadj"] = pd.to_numeric(frame["Sadj"], errors="raise")
    frame["target_t30_s"] = pd.to_numeric(frame["target_t30_s"], errors="raise")
    if frame["Sadj"].isna().any() or frame["QNX(2)"].isna().any():
        parser.error("QNX(2) and Sadj must be finite for source-level analysis")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    captions = None
    if args.captions_csv.exists():
        caption_frame = pd.read_csv(args.captions_csv, dtype={"audiocap_id": str})
        if {"audiocap_id", "caption"}.issubset(caption_frame.columns):
            captions = caption_frame[["audiocap_id", "caption"]].rename(columns={"audiocap_id": "source"})
            captions["source"] = captions["source"].astype(str)
            captions = captions.drop_duplicates("source")
            captions = captions[captions["source"].isin(frame["source"].astype(str).unique())].copy()
            frame = frame.merge(captions, on="source", how="left")
    frame.sort_values(["source", "model", "axis", "room", "target_t30_s"]).to_csv(
        args.output_dir / "source_condition_metrics.csv", index=False
    )
    source_summary = _make_source_summary(frame)
    if captions is not None:
        source_summary = source_summary.merge(captions, on="source", how="left")
    source_summary.to_csv(args.output_dir / "source_summary.csv", index=False)
    source_comparison = _make_model_comparison(frame)
    source_comparison.to_csv(args.output_dir / "source_model_comparison.csv", index=False)

    t30_summary = (
        frame.groupby(["source", "model", "axis", "target_t30_s"], as_index=False)[["QNX(2)", "Sadj"]]
        .mean()
        .rename(columns={"QNX(2)": "qnx_mean", "Sadj": "sadj_mean"})
    )
    t30_summary.to_csv(args.output_dir / "source_t30_summary.csv", index=False)

    room_summary = (
        frame.groupby(["source", "model", "axis", "room"], as_index=False)[["QNX(2)", "Sadj"]]
        .mean()
        .rename(columns={"QNX(2)": "qnx_mean", "Sadj": "sadj_mean"})
    )
    room_summary.to_csv(args.output_dir / "source_room_summary.csv", index=False)

    for axis in ("azimuth", "elevation"):
        axis_frame = source_summary[source_summary.axis == axis]
        for model in ("disse", "elsa"):
            model_frame = axis_frame[axis_frame.model == model]
            _heatmap(
                model_frame,
                ["qnx_mean"],
                f"{model.upper()} {axis}: per-source mean QNX(2)",
                args.output_dir / f"qnx_{model}_{axis}.png",
                cmap="viridis",
            )
            _heatmap(
                model_frame,
                ["sadj_mean"],
                f"{model.upper()} {axis}: per-source mean Sadj",
                args.output_dir / f"sadj_{model}_{axis}.png",
                cmap="magma_r",
            )
    _plot_scatter(source_summary, args.output_dir / "source_qnx_vs_sadj.png")
    if captions is not None:
        captions.to_csv(args.output_dir / "source_captions.csv", index=False)
    _write_report(args.output_dir / "report.md", source_summary, source_comparison, args.summary, captions)
    print(f"[done] source-level analysis: {len(source_summary)} source/model/axis rows")
    print(f"[saved] {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

