"""Load and validate the additional continuity experiment configuration."""

from __future__ import annotations

import math
from pathlib import Path
import tomllib
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/continuity_additional.toml"


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def angle_range(start: float, end: float, step: float, *, include_end: bool) -> list[float]:
    if step <= 0 or end <= start:
        raise ValueError("Angle step must be positive and end must exceed start.")
    count = (end - start) / step
    if not math.isclose(count, round(count), abs_tol=1e-9):
        raise ValueError("Angle range must be divisible by the step.")
    return [start + i * step for i in range(round(count) + int(include_end))]


def validate_positions(dims: tuple[float, float, float], distance: float,
                       az_values: list[float], el_values: list[float],
                       fixed_el: float, fixed_az: float) -> None:
    center = [d / 2 for d in dims]
    angles = [(az, fixed_el) for az in az_values] + [(fixed_az, el) for el in el_values]
    for az, el in angles:
        az_rad, el_rad = math.radians(az), math.radians(el)
        position = [center[0] + distance * math.cos(el_rad) * math.cos(az_rad),
                    center[1] + distance * math.cos(el_rad) * math.sin(az_rad),
                    center[2] + distance * math.sin(el_rad)]
        if any(not (0.1 <= x <= side - 0.1) for x, side in zip(position, dims)):
            raise ValueError(f"Source at az={az}, el={el} leaves room {dims}.")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    required = ("output_root", "audio_sources", "distance_m", "target_t30_s",
                "azimuth", "elevation", "rooms", "models")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing config entries: {missing}")
    if not config["audio_sources"] or not config["rooms"]:
        raise ValueError("At least one audio source and one room are required.")
    source_mode = config.get("audio_source_mode", "single")
    if source_mode not in ("single", "mean"):
        raise ValueError("audio_source_mode must be 'single' or 'mean'.")
    if source_mode == "single" and len(config["audio_sources"]) != 1:
        raise ValueError("audio_source_mode='single' requires exactly one audio source.")
    if source_mode == "mean" and len(config["audio_sources"]) < 2:
        raise ValueError("audio_source_mode='mean' requires at least two audio sources.")
    if float(config["distance_m"]) <= 0:
        raise ValueError("distance_m must be positive.")
    if sorted(float(x) for x in config["target_t30_s"]) != [0.0, 0.3, 0.6]:
        raise ValueError("This experiment requires target_t30_s = [0.0, 0.3, 0.6].")
    if float(config.get("t30_tolerance_s", 0.02)) <= 0:
        raise ValueError("t30_tolerance_s must be positive.")
    if int(config.get("calibration_max_iterations", 16)) < 1:
        raise ValueError("calibration_max_iterations must be positive.")

    az, el = config["azimuth"], config["elevation"]
    az_values = angle_range(float(az["start_deg"]), float(az["end_deg"]),
                            float(az["step_deg"]), include_end=False)
    el_values = angle_range(float(el["min_deg"]), float(el["max_deg"]),
                            float(el["step_deg"]), include_end=True)
    if az_values != list(range(0, 360, 5)) or el_values != list(range(-45, 46, 5)):
        raise ValueError("Expected azimuth 0..355 and elevation -45..45, both at 5 degrees.")

    ids = [room["id"] for room in config["rooms"]]
    if len(ids) != len(set(ids)) or any(not room_id.isidentifier() for room_id in ids):
        raise ValueError("Room IDs must be unique identifiers.")
    for room in config["rooms"]:
        dims = room["dims_m"]
        if len(dims) != 3 or any(float(d) <= 0 for d in dims):
            raise ValueError(f"Invalid dimensions for {room['id']}: {dims}")
        validate_positions(tuple(float(d) for d in dims), float(config["distance_m"]),
                           az_values, el_values, float(az["fixed_elevation_deg"]),
                           float(el["fixed_azimuth_deg"]))

    sources = [resolve_path(p) for p in config["audio_sources"]]
    if len({p.stem for p in sources}) != len(sources):
        raise ValueError("Audio source filenames must have unique stems.")
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(f"Audio source not found: {source}")
    for name in ("disse", "elsa"):
        model = config["models"].get(name, {})
        if model.get("enabled", False):
            checkpoint = resolve_path(model["checkpoint"])
            if not checkpoint.is_file():
                raise FileNotFoundError(f"{name} checkpoint not found: {checkpoint}")
            if name == "elsa" and not resolve_path(model["root"]).is_dir():
                raise FileNotFoundError(f"ELSA root not found: {model['root']}")
    return config
