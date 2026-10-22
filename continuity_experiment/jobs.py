"""Run, resume, and summarize a single configured sweep."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
from typing import Any

from .commands import disse_evaluation_command, disse_generation_command, elsa_command
from .config import ROOT, resolve_path


def run_command(command: list[str], *, label: str) -> None:
    print(f"[run] {label}: {shlex.join(command)}", flush=True)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/delsa-continuity-matplotlib")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def run_job(config: dict[str, Any], source: Path, model: str, axis: str,
            name: str, condition: dict[str, Any], output_root: Path) -> dict[str, Any]:
    out = output_root / source.stem / model / axis / name
    pt_name = "elsa_embeddings.pt" if model == "elsa" else f"continuous_{axis}_embeddings.pt"
    metrics_name = "metrics.json" if model == "elsa" else "continuity_metrics_audio_space.json"
    pt, metrics = out / pt_name, out / metrics_name
    spec = {"source": str(source), "model": model, "axis": axis, "condition": name,
            "condition_parameters": condition, "distance_m": config["distance_m"],
            "azimuth": config["azimuth"], "elevation": config["elevation"],
            "checkpoint": str(resolve_path(config["models"][model]["checkpoint"]))}
    if model == "elsa":
        spec["elsa_root"] = str(resolve_path(config["models"]["elsa"]["root"]))
        spec["batch_size"] = int(config["models"]["elsa"].get("batch_size", 4))
    manifest = out / "run_spec.json"
    if manifest.is_file():
        if json.loads(manifest.read_text()) != spec:
            raise ValueError(f"Existing job has different settings: {manifest}")
    else:
        if out.exists() and any(out.iterdir()):
            raise ValueError(f"Existing output has no run_spec.json: {out}")
        out.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(spec, indent=2) + "\n")

    label = f"{source.stem} {model.upper()} {axis} {name}"
    if model == "disse":
        generated = not pt.is_file()
        if generated:
            run_command(disse_generation_command(config, source, axis, condition, out),
                        label=f"{label} embeddings")
        if generated or not metrics.is_file():
            run_command(disse_evaluation_command(pt, axis), label=f"{label} scores")
    elif not pt.is_file() or not metrics.is_file():
        run_command(elsa_command(config, source, axis, condition, out), label=label)

    if not pt.is_file() or not metrics.is_file():
        raise RuntimeError(f"Job did not produce expected files: {out}")
    scores = json.loads(metrics.read_text())["scores"]
    return {"source": source.stem, "model": model, "axis": axis,
            "room": condition["room_id"],
            "target_t30_s": condition["target_t30_s"],
            "measured_t30_s_at_reference": condition["measured_t30_s_at_reference"],
            "QNX(2)": scores["QNX(2)"], "Sadj": scores["Sadj"],
            "embeddings_pt": str(pt), "metrics_json": str(metrics)}


def update_summary(row: dict[str, Any], root: Path) -> None:
    path = root / "summary.csv"
    rows: dict[str, dict[str, Any]] = {}
    if path.is_file():
        with path.open(newline="") as stream:
            for previous in csv.DictReader(stream):
                rows[previous["metrics_json"]] = previous
    rows[row["metrics_json"]] = row
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(rows[key] for key in sorted(rows))
    print(f"[summary] {path}", flush=True)


def update_mean_summary(root: Path, source_stems: list[str]) -> None:
    """Write means across all configured audio sources once a group is complete."""
    source_set = set(source_stems)
    summary_path = root / "summary.csv"
    if not summary_path.is_file():
        return

    with summary_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        if row.get("source") not in source_set:
            continue
        key = (row["model"], row["axis"], row["room"], row["target_t30_s"])
        groups.setdefault(key, []).append(row)

    mean_rows: list[dict[str, Any]] = []
    for (model, axis, room, target), group in sorted(groups.items()):
        present = {row["source"] for row in group}
        if present != source_set:
            continue
        qnx_values = [float(row["QNX(2)"]) for row in group]
        sadj_values = [float(row["Sadj"]) for row in group if row["Sadj"] not in ("", "None", "null")]
        mean_rows.append({
            "model": model,
            "axis": axis,
            "room": room,
            "target_t30_s": target,
            "n_sources": len(group),
            "QNX(2)_mean": sum(qnx_values) / len(qnx_values),
            "Sadj_mean": sum(sadj_values) / len(sadj_values) if sadj_values else None,
            "source_stems": ",".join(sorted(source_set)),
        })

    output = root / "summary_mean.csv"
    if mean_rows:
        fields = list(mean_rows[0])
        with output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(mean_rows)
        print(f"[summary] {output}", flush=True)
