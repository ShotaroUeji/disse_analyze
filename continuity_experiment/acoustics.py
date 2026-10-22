"""Calibrate simulated rooms against a measured full-band T30."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SAMPLE_RATE = 48_000


def estimate_t30_s(rir: Any) -> float:
    """Estimate T30 from the FOA W RIR's Schroeder curve (-5 to -35 dB)."""
    import numpy as np

    aformat = rir.detach().cpu().numpy().astype(np.float64)
    w = aformat.sum(axis=0) / 2.0
    energy = np.cumsum(np.square(w[::-1]))[::-1]
    if energy[0] <= 0:
        raise ValueError("RIR has zero energy.")
    db = 10.0 * np.log10(np.maximum(energy / energy[0], 1e-300))
    selected = (db <= -5.0) & (db >= -35.0)
    if int(selected.sum()) < 10 or db[-1] > -35.0:
        raise ValueError("RIR does not have enough decay to estimate T30.")
    time_s = np.flatnonzero(selected) / SAMPLE_RATE
    slope = float(np.polyfit(time_s, db[selected], 1)[0])
    if slope >= 0:
        raise ValueError("RIR decay slope is nonnegative.")
    return -60.0 / slope


def calibrate_room(dims: tuple[float, float, float], target: float,
                   distance: float, tolerance: float, max_iterations: int) -> dict[str, Any]:
    import pyroomacoustics as pra
    from continuous_azimuth_heatmap import generate_aformat_rir_for_azimuth_deg

    alpha, max_order = pra.inverse_sabine(target, dims)
    lower: float | None = None  # Too much reverberation at this absorption.
    upper: float | None = None  # Too little reverberation at this absorption.
    history: list[dict[str, float]] = []
    for _ in range(max_iterations):
        rir = generate_aformat_rir_for_azimuth_deg(
            azimuth_deg=0.0, elevation_deg=0.0, distance_m=distance,
            room_dims=dims, alpha=float(alpha), max_order=int(max_order), fs=SAMPLE_RATE,
        )
        measured = estimate_t30_s(rir)
        history.append({"alpha": float(alpha), "measured_t30_s": measured})
        if abs(measured - target) <= tolerance:
            return {"target_t30_s": target, "measured_t30_s_at_reference": measured,
                    "alpha": float(alpha), "max_order": int(max_order),
                    "reference_azimuth_deg": 0.0, "reference_elevation_deg": 0.0,
                    "tolerance_s": tolerance, "calibration_history": history}
        if measured > target:
            lower = float(alpha)
        else:
            upper = float(alpha)
        if lower is not None and upper is not None:
            alpha = (lower + upper) / 2.0
        else:
            alpha = min(0.99, max(0.01, float(alpha) * measured / target))
        if history[-1]["alpha"] == alpha:
            break
    raise RuntimeError(
        f"Could not calibrate T30={target}s in room {dims} to ±{tolerance}s; "
        f"last trials: {history[-3:]}"
    )


def calibration_for(config: dict[str, Any], root: Path) -> dict[str, dict[str, Any]]:
    params = {"grid": "all_rooms_x_all_t30_v1",
              "rooms": config["rooms"], "target_t30_s": config["target_t30_s"],
              "distance_m": config["distance_m"],
              "t30_tolerance_s": config.get("t30_tolerance_s", 0.02),
              "calibration_max_iterations": config.get("calibration_max_iterations", 16)}
    fingerprint = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
    cache_path = root / "calibration.json"
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text())
        if cached.get("fingerprint") == fingerprint:
            print(f"[resume] calibration: {cache_path}", flush=True)
            return cached["conditions"]
        raise ValueError(f"Calibration settings changed; use a new output_root: {cache_path}")

    conditions: dict[str, dict[str, Any]] = {}
    for room in config["rooms"]:
        dims = tuple(float(x) for x in room["dims_m"])
        for target in config["target_t30_s"]:
            key = f"{room['id']}_t30_{str(target).replace('.', 'p')}s"
            if target == 0:
                conditions[key] = {
                    "room_id": room["id"], "room_dims_m": list(dims),
                    "target_t30_s": 0.0, "measured_t30_s_at_reference": None,
                    "alpha": 1.0, "max_order": 0,
                }
                continue
            print(f"[calibrate] room={room['id']} target={target}s", flush=True)
            conditions[key] = {
                "room_id": room["id"], "room_dims_m": list(dims),
                **calibrate_room(dims, float(target), float(config["distance_m"]),
                                 float(config.get("t30_tolerance_s", 0.02)),
                                 int(config.get("calibration_max_iterations", 16))),
            }
    root.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"fingerprint": fingerprint, "conditions": conditions},
                                     indent=2) + "\n")
    return conditions
