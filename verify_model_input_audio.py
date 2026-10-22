#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify one generated acoustic input shared by the DISSE and ELSA paths.

The script uses the same RIR generation and preprocessing functions as the
continuous-angle evaluators.  It saves a small, inspectable bundle containing
the dry source, A-format RIR, FOA waveform, omni channel, a waveform figure,
and JSON diagnostics.  It also checks that the DISSE and ELSA preprocessing
implementations produce identical ``i_act``, ``i_rea`` and ``omni_48k``
inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio

from continuity_experiment.acoustics import estimate_t30_s
from continuous_azimuth_heatmap import (
    FOA_SR,
    aformat_to_foa as disse_aformat_to_foa,
    generate_aformat_rir_for_azimuth_deg as disse_generate_rir,
    load_dry_audio as disse_load_dry,
    make_model_audio_input as disse_make_input,
)
from evaluate_elsa_continuity import (
    aformat_to_foa as elsa_aformat_to_foa,
    generate_aformat_rir as elsa_generate_rir,
    load_dry_audio as elsa_load_dry,
    make_elsa_audio_input,
)


def tensor_stats(value: torch.Tensor) -> dict:
    value = value.detach().float().cpu()
    finite = bool(torch.isfinite(value).all())
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": finite,
        "min": float(value.min()) if value.numel() else None,
        "max": float(value.max()) if value.numel() else None,
        "peak_abs": float(value.abs().max()) if value.numel() else None,
        "rms": float(torch.sqrt(torch.mean(value.square()))) if value.numel() else None,
        "nonzero_fraction": float(torch.count_nonzero(value).item() / value.numel()) if value.numel() else 0.0,
    }


def save_audio(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    array = waveform.detach().float().cpu().numpy()
    if array.ndim == 2:
        array = array.T  # soundfile expects [samples, channels]
    sf.write(str(path), array, sample_rate, subtype="FLOAT")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-path", type=Path, default=Path("AudioCaps_mp3/test/104937.mp3"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/audio_input_verification/104937_large_t30_0p6s"))
    parser.add_argument("--azimuth-deg", type=float, default=0.0)
    parser.add_argument("--elevation-deg", type=float, default=0.0)
    parser.add_argument("--distance-m", type=float, default=1.5)
    parser.add_argument("--room-dims", type=float, nargs=3, default=[15.0, 10.0, 3.0])
    parser.add_argument("--alpha", type=float, default=0.43240701586385316)
    parser.add_argument("--max-order", type=int, default=71)
    parser.add_argument("--target-t30-s", type=float, default=0.6)
    args = parser.parse_args()

    audio_path = args.audio_path.expanduser().resolve()
    if not audio_path.is_file():
        parser.error(f"Audio source not found: {audio_path}")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    room_dims = tuple(float(value) for value in args.room_dims)
    device = "cpu"

    dry_disse = disse_load_dry(str(audio_path), device)
    dry_elsa = elsa_load_dry(audio_path, torch.device(device))
    if dry_disse.shape != dry_elsa.shape:
        raise RuntimeError(f"DISSE/ELSA dry shapes differ: {dry_disse.shape} vs {dry_elsa.shape}")
    dry_difference = float((dry_disse - dry_elsa).abs().max())

    rir_disse = disse_generate_rir(
        azimuth_deg=args.azimuth_deg,
        elevation_deg=args.elevation_deg,
        distance_m=args.distance_m,
        room_dims=room_dims,
        alpha=args.alpha,
        max_order=args.max_order,
        fs=FOA_SR,
    )
    rir_elsa = elsa_generate_rir(
        azimuth_deg=args.azimuth_deg,
        elevation_deg=args.elevation_deg,
        distance_m=args.distance_m,
        room_dims=room_dims,
        alpha=args.alpha,
        max_order=args.max_order,
        fs=FOA_SR,
    )
    rir_difference = float((rir_disse - rir_elsa).abs().max())
    if rir_difference > 1e-5:
        raise RuntimeError(f"DISSE/ELSA RIR generation differs: max_abs_diff={rir_difference}")

    disse_input = disse_make_input(dry_disse, rir_disse, device)
    elsa_input = make_elsa_audio_input(dry_elsa, rir_elsa, torch.device(device))
    input_differences = {
        key: float((disse_input[key] - elsa_input[key]).abs().max())
        for key in ("i_act", "i_rea", "omni_48k")
    }
    if any(value > 1e-5 for value in input_differences.values()):
        raise RuntimeError(f"DISSE/ELSA model inputs differ: {input_differences}")

    wet_aformat = torchaudio.functional.fftconvolve(dry_disse, rir_disse)[..., : dry_disse.shape[-1]]
    foa_disse = disse_aformat_to_foa(wet_aformat)
    foa_elsa = elsa_aformat_to_foa(wet_aformat)
    foa_difference = float((foa_disse - foa_elsa).abs().max())

    save_audio(out_dir / "dry_mono_48k.wav", dry_disse, FOA_SR)
    save_audio(out_dir / "rir_aformat_48k.wav", rir_disse, FOA_SR)
    save_audio(out_dir / "wet_foa_48k.wav", foa_disse, FOA_SR)
    save_audio(out_dir / "model_input_omni_48k.wav", disse_input["omni_48k"], FOA_SR)

    # Plot a short, readable segment and the RIR decay.
    short_samples = min(FOA_SR, foa_disse.shape[-1])
    rir_samples = min(int(FOA_SR * 0.8), rir_disse.shape[-1])
    time_wet = np.arange(short_samples) / FOA_SR
    time_rir = np.arange(rir_samples) / FOA_SR
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    axes[0].plot(time_rir, rir_disse[0, :rir_samples].numpy(), linewidth=0.7)
    axes[0].set_title("Generated A-format RIR (microphone 0)")
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("amplitude")
    axes[1].plot(time_wet, foa_disse[0, :short_samples].numpy(), linewidth=0.5)
    axes[1].set_title("FOA W channel passed to DISSE/ELSA preprocessing (first 1 s)")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("amplitude")
    fig.savefig(out_dir / "waveform_check.png", dpi=180)
    plt.close(fig)

    measured_t30 = None if args.target_t30_s == 0.0 else float(estimate_t30_s(rir_disse))
    diagnostics = {
        "audio_path": str(audio_path),
        "model_input_is_shared": True,
        "sample_rate_hz": FOA_SR,
        "angles_deg": {"azimuth": args.azimuth_deg, "elevation": args.elevation_deg},
        "distance_m": args.distance_m,
        "room_dims_m": list(room_dims),
        "target_t30_s": args.target_t30_s,
        "measured_t30_s": measured_t30,
        "alpha": args.alpha,
        "max_order": args.max_order,
        "dry_disse_vs_elsa_max_abs_diff": dry_difference,
        "rir_disse_vs_elsa_max_abs_diff": rir_difference,
        "foa_conversion_max_abs_diff": foa_difference,
        "model_input_max_abs_diff": input_differences,
        "dry_mono_48k": tensor_stats(dry_disse),
        "rir_aformat_48k": tensor_stats(rir_disse),
        "wet_foa_48k": tensor_stats(foa_disse),
        "disse_input": {key: tensor_stats(value) for key, value in disse_input.items()},
        "elsa_input": {key: tensor_stats(value) for key, value in elsa_input.items()},
        "outputs": {
            "dry": "dry_mono_48k.wav",
            "rir": "rir_aformat_48k.wav",
            "foa": "wet_foa_48k.wav",
            "omni": "model_input_omni_48k.wav",
            "waveform": "waveform_check.png",
        },
    }
    (out_dir / "verification.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))
    print(f"[saved] {out_dir.resolve()}")


if __name__ == "__main__":
    main()
