"""Build subprocess argument lists for the existing evaluators."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from .config import ROOT, angle_range, resolve_path


def disse_generation_command(config: dict[str, Any], source: Path, axis: str,
                             condition: dict[str, Any], out: Path) -> list[str]:
    az, el = config["azimuth"], config["elevation"]
    command = [sys.executable, str(ROOT / f"continuous_{axis}_heatmap.py"),
               "--ckpt", str(resolve_path(config["models"]["disse"]["checkpoint"])),
               "--audio_path", str(source), "--out_dir", str(out),
               "--distance_m", str(config["distance_m"]),
               "--room_dims", *(str(x) for x in condition["room_dims_m"]),
               "--alpha", str(condition["alpha"]),
               "--max_order", str(condition["max_order"])]
    if axis == "azimuth":
        command += ["--az_start_deg", str(az["start_deg"]),
                    "--az_end_deg", str(az["end_deg"]),
                    "--az_step_deg", str(az["step_deg"]),
                    "--elevation_deg", str(az["fixed_elevation_deg"])]
    else:
        n_steps = len(angle_range(float(el["min_deg"]), float(el["max_deg"]),
                                  float(el["step_deg"]), include_end=True))
        command += ["--el_min_deg", str(el["min_deg"]),
                    "--el_max_deg", str(el["max_deg"]), "--n_steps", str(n_steps),
                    "--azimuth_deg", str(el["fixed_azimuth_deg"])]
    return command


def disse_evaluation_command(pt: Path, axis: str) -> list[str]:
    return [sys.executable, str(ROOT / "evaluate_disse_continuity.py"),
            "--emb-pt", str(pt), "--axis", axis]


def elsa_command(config: dict[str, Any], source: Path, axis: str,
                 condition: dict[str, Any], out: Path) -> list[str]:
    az, el = config["azimuth"], config["elevation"]
    model = config["models"]["elsa"]
    n_steps = len(angle_range(float(el["min_deg"]), float(el["max_deg"]),
                              float(el["step_deg"]), include_end=True))
    return [sys.executable, str(ROOT / "evaluate_elsa_continuity.py"),
            "--elsa-root", str(resolve_path(model["root"])),
            "--ckpt", str(resolve_path(model["checkpoint"])),
            "--audio-path", str(source), "--out-dir", str(out),
            "--axis", axis,
            "--condition", "dry" if condition["target_t30_s"] == 0 else "reverb",
            "--distance-m", str(config["distance_m"]),
            "--room-dims", *(str(x) for x in condition["room_dims_m"]),
            "--alpha", str(condition["alpha"]),
            "--max-order", str(condition["max_order"]),
            "--batch-size", str(model.get("batch_size", 4)), "--skip-lkh",
            "--az-start-deg", str(az["start_deg"]),
            "--az-end-deg", str(az["end_deg"]),
            "--az-step-deg", str(az["step_deg"]),
            "--fixed-elevation-deg", str(az["fixed_elevation_deg"]),
            "--el-min-deg", str(el["min_deg"]),
            "--el-max-deg", str(el["max_deg"]), "--n-steps", str(n_steps),
            "--fixed-azimuth-deg", str(el["fixed_azimuth_deg"])]
