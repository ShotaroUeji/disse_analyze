#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure a simple early-reflection proxy from the calibrated room RIRs.

This diagnostic regenerates only the room impulse responses at the reference
direction (azimuth=0°, elevation=0°). It does not run DISSE/ELSA inference.
The proxy is deliberately explicit: direct energy is measured around the
expected direct arrival, and early energy is measured from 1 ms to 50 ms after
that arrival using the A-format channel energy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from continuity_experiment.acoustics import SAMPLE_RATE


def _measure(rir, distance_m: float) -> dict[str, float]:
    """Return direct/early/late energy metrics for a [4, T] RIR tensor."""
    aformat = rir.detach().cpu().numpy().astype(np.float64)
    expected = int(round(distance_m / 343.0 * SAMPLE_RATE))
    half_direct = int(round(0.0005 * SAMPLE_RATE))
    early_start = expected + int(round(0.001 * SAMPLE_RATE))
    early_end = expected + int(round(0.050 * SAMPLE_RATE))
    direct_slice = aformat[:, max(0, expected - half_direct):expected + half_direct + 1]
    early_slice = aformat[:, early_start:early_end]
    late_slice = aformat[:, early_end:]
    direct_energy = float(np.sum(direct_slice ** 2))
    early_energy = float(np.sum(early_slice ** 2))
    late_energy = float(np.sum(late_slice ** 2))
    denom = max(direct_energy, np.finfo(float).tiny)
    early_to_direct = early_energy / denom
    return {
        "expected_direct_sample": expected,
        "direct_energy": direct_energy,
        "early_energy_1_50ms": early_energy,
        "late_energy_after_50ms": late_energy,
        "early_to_direct_db": float(10.0 * np.log10(max(early_energy, np.finfo(float).tiny) / denom)),
        "direct_to_early_db": float(10.0 * np.log10(denom / max(early_energy, np.finfo(float).tiny))),
        "early_to_direct_ratio": early_to_direct,
        "early_to_direct_percent": 100.0 * early_to_direct,
        "early_fraction_direct_plus_early": float(early_energy / (direct_energy + early_energy + np.finfo(float).tiny)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--distance-m", type=float, default=1.5)
    args = parser.parse_args()

    import torch  # noqa: WPS433
    from continuous_azimuth_heatmap import generate_aformat_rir_for_azimuth_deg

    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))["conditions"]
    rows = []
    for condition, values in sorted(calibration.items()):
        target = float(values["target_t30_s"])
        if target == 0.0:
            continue
        dims = tuple(float(x) for x in values["room_dims_m"])
        rir = generate_aformat_rir_for_azimuth_deg(
            azimuth_deg=0.0,
            elevation_deg=0.0,
            distance_m=args.distance_m,
            room_dims=dims,
            alpha=float(values["alpha"]),
            max_order=int(values["max_order"]),
            fs=SAMPLE_RATE,
        )
        row = {
            "condition": condition,
            "room": values["room_id"],
            "room_dims_m": "x".join(str(x) for x in dims),
            "target_t30_s": target,
            "measured_t30_s": values.get("measured_t30_s_at_reference"),
            "alpha": values["alpha"],
            "max_order": values["max_order"],
        }
        row.update(_measure(rir, args.distance_m))
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(["target_t30_s", "room"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "early_reflection_diagnostic.csv", index=False)
    result.to_json(args.output_dir / "early_reflection_diagnostic.json", orient="records", indent=2)

    lines = [
        "# 初期残響診断",
        "",
        "直接音到来の推定位置から1–50 msのA-formatエネルギーを、直接音エネルギーに対する比として計算した。これは初期反射の簡易proxyであり、方向別の完全な残響時間ではない。",
        "",
        "| room | T30 [s] | early/direct [dB] | early/direct [%] | early/(direct+early) [%] |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in result.itertuples():
        lines.append(
            f"| {row.room} | {row.target_t30_s:.1f} | {row.early_to_direct_db:.2f} | "
            f"{row.early_to_direct_percent:.2f} | {100*row.early_fraction_direct_plus_early:.2f} |"
        )
    lines.extend([
        "",
        "early/direct が大きいほど、直接音に対する初期反射エネルギーが大きい。埋め込みQNX(2)との因果関係を調べるには、角度ごとのこの指標とQNX(2)/Sadjを結合して相関を計算する必要がある。",
    ])
    (args.output_dir / "early_reflection_diagnostic.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[saved] {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

