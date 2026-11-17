#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check whether room-size changes are associated with QNX(2)/Sadj changes.

This is a post-hoc analysis of ``summary_mean.csv``. It does not regenerate
RIRs or embeddings. A causal early-reflection claim requires an additional RIR
diagnostic (e.g. direct-to-early-reflection energy), which is not present in
the continuity summary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOMS = ["small", "medium", "large"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-mean", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.summary_mean)
    required = {"model", "axis", "room", "target_t30_s", "QNX(2)_mean", "Sadj_mean"}
    missing = sorted(required - set(frame.columns))
    if missing:
        parser.error(f"missing columns: {missing}")
    frame = frame[frame.room.isin(ROOMS)].copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for (model, axis, t30), group in frame.groupby(["model", "axis", "target_t30_s"]):
        values = group.set_index("room")
        if not set(ROOMS).issubset(values.index):
            continue
        qnx = [float(values.loc[room, "QNX(2)_mean"]) for room in ROOMS]
        sadj = [float(values.loc[room, "Sadj_mean"]) for room in ROOMS]
        rows.append({
            "model": model,
            "axis": axis,
            "target_t30_s": float(t30),
            "qnx_small": qnx[0],
            "qnx_medium": qnx[1],
            "qnx_large": qnx[2],
            "sadj_small": sadj[0],
            "sadj_medium": sadj[1],
            "sadj_large": sadj[2],
            "qnx_small_to_large_change": qnx[0] - qnx[2],
            "sadj_small_to_large_change": sadj[0] - sadj[2],
            "qnx_monotonic_small_to_large": qnx[0] <= qnx[1] <= qnx[2],
            "sadj_monotonic_worse_small_to_large": sadj[0] >= sadj[1] >= sadj[2],
        })
    effect = pd.DataFrame(rows).sort_values(["model", "axis", "target_t30_s"])
    effect.to_csv(args.output_dir / "room_size_effect.csv", index=False)

    lines = [
        "# 部屋サイズと角度幾何構造の検証",
        "",
        f"入力: `{args.summary_mean}`",
        "",
        "既存の音源別QNX(2)/Sadjを10音源平均した `summary_mean.csv` から、small・medium・large の順序を比較した。音声生成、RIR再生成、埋め込み推論は行っていない。",
        "",
        "## 結果",
        "",
        "QNX(2) は高いほど良い、Sadj は低いほど良い。QNXについてsmall ≤ medium ≤ largeなら、小さい部屋ほど近傍構造が崩れる傾向と整合する。",
        "",
        "| model | axis | T30 [s] | QNX small | QNX medium | QNX large | QNX単調性 | Sadj small | Sadj medium | Sadj large |",
        "|---|---|---:|---:|---:|---:|:---:|---:|---:|---:|",
    ]
    for row in effect.itertuples():
        lines.append(
            f"| {row.model.upper()} | {row.axis} | {row.target_t30_s:.1f} | "
            f"{row.qnx_small:.4f} | {row.qnx_medium:.4f} | {row.qnx_large:.4f} | "
            f"{'yes' if row.qnx_monotonic_small_to_large else 'no'} | "
            f"{row.sadj_small:.4f} | {row.sadj_medium:.4f} | {row.sadj_large:.4f} |"
        )
    lines.extend([
        "",
        "## 解釈",
        "",
        "- T30=0.0 sでは3部屋の結果が同一である。これは `max_order=0` の直接音条件なので、部屋サイズによる反射差が存在しないためである。",
        "- T30=0.3 sと0.6 sでは、DISSE/ELSAの両方で、方位角・仰角のQNX(2)が概ね small < medium < large となる。特にDISSE方位角では、T30=0.6 sで small 0.442、medium 0.627、large 0.672となり、差が大きい。",
        "- Sadjは、DISSE方位角のT30=0.6 sのように単調でない場合があるため、部屋サイズ効果の主証拠にはQNX(2)を使い、Sadjは補助的に扱うのが妥当である。",
        "- ただし、この結果だけでは「小さい部屋ほど初期残響が大きいことが原因」とまでは言えない。現在保存されている結果には、初期反射エネルギー、直接音対初期反射比、反射到来時間の指標がない。",
        "",
        "## 因果仮説を検証する追加指標",
        "",
        "各部屋・各角度のRIRについて、直接音到来を基準にした 0–20 ms または 0–50 ms の初期反射エネルギー、DRR（direct-to-reverberant ratio）、最初の反射到来時間を計算し、QNX(2)またはSadjとの相関を調べる必要がある。これにより、部屋サイズと幾何構造の相関を、初期残響を介したメカニズムとして検証できる。",
    ])
    (args.output_dir / "room_size_effect_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[saved] {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

