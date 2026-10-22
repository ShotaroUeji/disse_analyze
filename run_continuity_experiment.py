#!/usr/bin/env python3
"""Run the configured DISSE/ELSA continuity experiment, resuming completed jobs.

Usage:
    ./.venv/bin/python run_continuity_experiment.py --dry-run
    ./.venv/bin/python run_continuity_experiment.py --limit-jobs 1
    ./.venv/bin/python run_continuity_experiment.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from continuity_experiment.acoustics import calibration_for
from continuity_experiment.config import DEFAULT_CONFIG, load_config, resolve_path
from continuity_experiment.jobs import run_job, update_mean_summary, update_summary


def condition_names(config: dict) -> list[str]:
    return [f"{room['id']}_t30_{str(target).replace('.', 'p')}s"
            for room in config["rooms"] for target in config["target_t30_s"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and list jobs without generating RIRs.")
    parser.add_argument("--models", choices=("all", "disse", "elsa"), default="all")
    parser.add_argument("--limit-jobs", type=int, help="Run only the first N jobs for a pilot.")
    args = parser.parse_args()
    if args.limit_jobs is not None and args.limit_jobs < 1:
        parser.error("--limit-jobs must be positive.")

    config = load_config(args.config.expanduser().resolve())
    root = resolve_path(config["output_root"])
    models = [name for name in ("disse", "elsa")
              if config["models"].get(name, {}).get("enabled", False)
              and args.models in ("all", name)]
    if not models:
        parser.error("No enabled model matches --models.")
    sources = [resolve_path(value) for value in config["audio_sources"]]
    jobs = [(source, model, axis, name)
            for source in sources for model in models
            for axis in ("azimuth", "elevation") for name in condition_names(config)]
    if args.limit_jobs is not None:
        jobs = jobs[:args.limit_jobs]

    print(f"[plan] {len(jobs)} jobs; output_root={root}", flush=True)
    if args.dry_run:
        for source, model, axis, name in jobs:
            print(f"  {source.stem} / {model} / {axis} / {name}")
        return

    conditions = calibration_for(config, root)
    for index, (source, model, axis, name) in enumerate(jobs, 1):
        print(f"[job {index}/{len(jobs)}] {source.stem} {model} {axis} {name}", flush=True)
        row = run_job(config, source, model, axis, name, conditions[name], root)
        update_summary(row, root)
        if config.get("audio_source_mode", "single") == "mean":
            update_mean_summary(root, [source.stem for source in sources])


if __name__ == "__main__":
    main()
