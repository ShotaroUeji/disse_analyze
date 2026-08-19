#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
continuous_elevation_heatmap.py

Continuous elevation analysis for DISSE.

This script does NOT use precomputed csv or rir_fixed.csv.
It generates synthetic RIRs with fixed room / distance / azimuth / T30
and only changes elevation_deg.

Important unit rule:
  - All CLI arguments are in degrees when their names end with _deg.
  - All metadata and plot labels are in degrees.
  - Radians are used ONLY inside trig functions via np.deg2rad().
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from pyroomacoustics.directivities import CardioidFamily, DirectionVector
from tqdm.auto import tqdm

from model.delsa_model import DELSA
from dataset.audio_rir_dataset import foa_to_iv


FOA_SR = 48_000
IV_SR = 16_000
MAX_DURATION_SEC = 10.0


def pick_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def strip_module_prefix(sd):
    return {
        (k.replace("module.", "", 1) if k.startswith("module.") else k): v
        for k, v in sd.items()
    }


def load_model(ckpt_path: str, device: str) -> DELSA:
    model = DELSA(audio_encoder_cfg={}, text_encoder_cfg={}).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = strip_module_prefix(pick_state_dict(ckpt))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("[warn] first missing keys:", missing[:10])
    if unexpected:
        print("[warn] first unexpected keys:", unexpected[:10])
    model.eval()
    return model


def load_dry_audio(audio_path: str, device: str) -> torch.Tensor:
    wav, sr = torchaudio.load(audio_path)

    # Force mono. Shape: [1, T]
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != FOA_SR:
        wav = torchaudio.functional.resample(wav, sr, FOA_SR)

    target_len = int(FOA_SR * MAX_DURATION_SEC)
    if wav.shape[-1] < target_len:
        repeat = math.ceil(target_len / wav.shape[-1])
        wav = wav.repeat(1, repeat)[..., :target_len]
    else:
        wav = wav[..., :target_len]

    return wav.to(device)


def make_tetrahedral_directivities(radius_m: float = 0.05):
    """
    Same orientation convention as make_RIR/gen_one_rir.py.
    DirectionVector expects azimuth/colatitude in degrees when degrees=True.
    """
    r = radius_m
    v = r / math.sqrt(3.0)

    # Columns are 4 mic positions relative to center.
    tet = np.array(
        [
            [ v,  v,  v],
            [ v, -v, -v],
            [-v,  v, -v],
            [-v, -v,  v],
        ],
        dtype=float,
    ).T  # [3, 4]

    dirs = []
    for x, y, z in tet.T:
        azimuth_deg = math.degrees(math.atan2(y, x)) % 360.0
        colatitude_deg = math.degrees(math.acos(z / r))
        dirs.append(
            CardioidFamily(
                orientation=DirectionVector(
                    azimuth=azimuth_deg,
                    colatitude=colatitude_deg,
                    degrees=True,
                ),
                p=0.5,
                gain=1.0,
            )
        )

    return tet, dirs


def source_position_from_angles_deg(
    center_xyz: np.ndarray,
    distance_m: float,
    azimuth_deg: float,
    elevation_deg: float,
) -> np.ndarray:
    """
    External interface: degrees.
    Internal trig: radians.

    Coordinate:
      x = cos(el) cos(az)
      y = cos(el) sin(az)
      z = sin(el)
    """
    azimuth_rad = np.deg2rad(azimuth_deg)
    elevation_rad = np.deg2rad(elevation_deg)

    direction = np.array(
        [
            np.cos(elevation_rad) * np.cos(azimuth_rad),
            np.cos(elevation_rad) * np.sin(azimuth_rad),
            np.sin(elevation_rad),
        ],
        dtype=float,
    )

    return center_xyz + distance_m * direction


def generate_aformat_rir_for_elevation_deg(
    azimuth_deg: float,
    elevation_deg: float,
    distance_m: float,
    room_dims: Tuple[float, float, float],
    alpha: float,
    max_order: int,
    fs: int,
) -> torch.Tensor:
    """
    Generate 4-channel A-format RIR.
    Returned shape: [4, Tr]
    """
    w, h, H = room_dims
    center = np.array([w / 2.0, h / 2.0, H / 2.0], dtype=float)
    src = source_position_from_angles_deg(
        center_xyz=center,
        distance_m=distance_m,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
    )

    margin = 0.1
    if not (
        margin <= src[0] <= w - margin
        and margin <= src[1] <= h - margin
        and margin <= src[2] <= H - margin
    ):
        raise ValueError(
            f"Source outside room for elevation_deg={elevation_deg}: "
            f"azimuth_deg={azimuth_deg}, src={src.tolist()}, room_dims={room_dims}. "
            "Use smaller --distance_m, narrower elevation range, or larger --room_dims."
        )

    room = pra.ShoeBox(
        room_dims,
        fs=fs,
        materials=pra.Material(alpha),
        max_order=max_order,
    )
    room.add_source(src.tolist())

    tet, dirs = make_tetrahedral_directivities(radius_m=0.05)
    mic_positions = center.reshape(3, 1) + tet

    mic_array = pra.MicrophoneArray(
        mic_positions,
        fs=fs,
        directivity=dirs,
    )
    room.add_microphone_array(mic_array)
    room.compute_rir()

    rir_4ch = [room.rir[m][0] for m in range(4)]
    max_len = max(len(r) for r in rir_4ch)
    rir_4ch = [np.pad(r, (0, max_len - len(r))) for r in rir_4ch]
    rir_4ch = np.stack(rir_4ch, axis=0).astype(np.float32)  # [4, Tr]

    return torch.from_numpy(rir_4ch)


def aformat_to_foa(a_wav: torch.Tensor) -> torch.Tensor:
    """
    A-format 4ch -> FOA [W, Y, Z, X].
    Input shape: [4, T]
    Output shape: [4, T]
    """
    m0, m1, m2, m3 = a_wav[0], a_wav[1], a_wav[2], a_wav[3]

    W = (m0 + m1 + m2 + m3) / 2.0
    X = (m0 + m1 - m2 - m3) / 2.0
    Y = (m0 - m1 + m2 - m3) / 2.0
    Z = (m0 - m1 - m2 + m3) / 2.0

    return torch.stack([W, Y, Z, X], dim=0)


def make_model_audio_input(
    dry_mono_48k: torch.Tensor,
    rir_aformat: torch.Tensor,
    device: str,
) -> Dict[str, torch.Tensor]:
    """
    dry_mono_48k: [1, T] on device
    rir_aformat: [4, Tr] on CPU or device

    Output dict has batch dimension:
      i_act:    [1, 3, F, Tfrm]
      i_rea:    [1, 3, F, Tfrm]
      omni_48k: [1, T]
    """
    rir_aformat = rir_aformat.to(device)

    # Broadcast dry [1,T] with RIR [4,Tr] -> wet A-format [4,T+Tr-1]
    wet_a = torchaudio.functional.fftconvolve(dry_mono_48k, rir_aformat)
    wet_a = wet_a[..., : dry_mono_48k.shape[-1]]

    foa = aformat_to_foa(wet_a)  # [4, T]
    omni_48k = foa[0]            # W channel, [T]

    foa_16k = torchaudio.functional.resample(foa, orig_freq=FOA_SR, new_freq=IV_SR)
    i_act, i_rea = foa_to_iv(foa_16k.unsqueeze(0), n_fft=400, hop=100)

    return {
        "i_act": i_act.float(),
        "i_rea": i_rea.float(),
        "omni_48k": omni_48k.unsqueeze(0).float(),
    }


def cosine_distance_matrix(Z: torch.Tensor) -> np.ndarray:
    Z = F.normalize(Z.float(), dim=-1)
    D = 1.0 - (Z @ Z.T)
    return np.clip(D.detach().cpu().numpy(), 0.0, 2.0)


def elevation_distance_deg(elevations_deg: np.ndarray) -> np.ndarray:
    """
    Elevation is a bounded linear variable, not circular.
    Physical distance is absolute elevation difference.
    """
    diff = np.abs(elevations_deg[:, None] - elevations_deg[None, :])
    return diff


def upper_tri_values(M: np.ndarray) -> np.ndarray:
    idx = np.triu_indices_from(M, k=1)
    return M[idx]


def rankdata_average(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def spearmanr_np(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return float("nan")

    rx = rankdata_average(x)
    ry = rankdata_average(y)
    rx -= rx.mean()
    ry -= ry.mean()

    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom == 0:
        return float("nan")

    return float((rx * ry).sum() / denom)


def knn_overlap_score(D_emb: np.ndarray, elevations_deg: np.ndarray, k: int) -> float:
    """
    For each point, compare k nearest neighbors in embedding space
    with k nearest neighbors in physical elevation space.

    Elevation is linear:
      D_phys(i,j) = |el_i - el_j|
    """
    D_phys = elevation_distance_deg(elevations_deg)
    n = len(elevations_deg)
    scores = []

    for i in range(n):
        emb_order = np.argsort(D_emb[i])
        phys_order = np.argsort(D_phys[i])

        emb_nn = [j for j in emb_order if j != i][:k]
        phys_nn = [j for j in phys_order if j != i][:k]

        scores.append(len(set(emb_nn) & set(phys_nn)) / float(k))

    return float(np.mean(scores))


def plot_heatmap(D: np.ndarray, elevations_deg: np.ndarray, title: str, out_path: Path):
    plt.figure(figsize=(8, 7))
    im = plt.imshow(D, origin="upper", aspect="equal", interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="1 - cosine similarity")
    plt.title(title)
    plt.xlabel("Elevation [deg]")
    plt.ylabel("Elevation [deg]")

    tick_idx = list(range(0, len(elevations_deg), max(1, len(elevations_deg) // 8)))
    tick_labels = [f"{elevations_deg[i]:.1f}" for i in tick_idx]
    plt.xticks(tick_idx, tick_labels, rotation=45, ha="right")
    plt.yticks(tick_idx, tick_labels)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_scatter(D_phys: np.ndarray, D_emb: np.ndarray, title: str, out_path: Path):
    x = upper_tri_values(D_phys)
    y = upper_tri_values(D_emb)

    rho = spearmanr_np(x, y)

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, s=8, alpha=0.25)

    # Bin average for visibility
    max_x = float(np.nanmax(x)) if len(x) else 1.0
    bins = np.linspace(0, max_x, 19)
    bx, by = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (x >= lo) & (x < hi)
        if mask.sum() > 0:
            bx.append(x[mask].mean())
            by.append(y[mask].mean())
    if len(bx) > 1:
        plt.plot(bx, by, marker="o", linewidth=2)

    plt.xlabel("Absolute elevation difference [deg]")
    plt.ylabel("Embedding distance: 1 - cosine")
    plt.title(f"{title} | Spearman rho={rho:.3f}")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--audio_path", required=True)
    parser.add_argument("--caption", default="a sound")

    parser.add_argument("--out_dir", default="results/continuous_elevation")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--eval_amp", action="store_true")

    # All external angle arguments are degrees.
    # Elevation sweep:
    parser.add_argument("--azimuth_deg", type=float, default=0.0)
    parser.add_argument("--el_min_deg", type=float, default=-47.5)
    parser.add_argument("--el_max_deg", type=float, default=48.7)
    parser.add_argument("--n_steps", type=int, default=72)

    parser.add_argument("--distance_m", type=float, default=2.0)
    parser.add_argument("--room_dims", type=float, nargs=3, default=[8.0, 7.0, 3.0])
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--max_order", type=int, default=10)

    args = parser.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if device == "auto":
        device = "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[unit] All azimuth/elevation arguments and saved metadata are degrees.")
    print("[unit] Radians are used only inside np.deg2rad() for trig functions.")
    print(f"[info] device={device}")
    print(f"[info] audio_path={args.audio_path}")
    print(f"[info] fixed azimuth_deg={args.azimuth_deg}")
    print(f"[info] elevation_deg: min={args.el_min_deg}, max={args.el_max_deg}, n_steps={args.n_steps}")
    print(f"[info] distance_m={args.distance_m}")
    print(f"[info] room_dims={args.room_dims}, alpha={args.alpha}, max_order={args.max_order}")

    if not Path(args.audio_path).exists():
        raise SystemExit(f"[ERR] audio_path not found: {args.audio_path}")
    if not Path(args.ckpt).exists():
        raise SystemExit(f"[ERR] ckpt not found: {args.ckpt}")
    if args.n_steps < 2:
        raise SystemExit("[ERR] --n_steps must be >= 2.")
    if args.el_min_deg >= args.el_max_deg:
        raise SystemExit("[ERR] --el_min_deg must be smaller than --el_max_deg.")

    elevations_deg = np.linspace(
        args.el_min_deg,
        args.el_max_deg,
        args.n_steps,
        dtype=np.float32,
    )

    model = load_model(args.ckpt, device)
    dry = load_dry_audio(args.audio_path, device)

    A_SPA, A_SRC, T_SPA, T_SRC = [], [], [], []

    for el_deg in tqdm(elevations_deg, desc="Elevation sweep [deg]", dynamic_ncols=True):
        rir = generate_aformat_rir_for_elevation_deg(
            azimuth_deg=float(args.azimuth_deg),
            elevation_deg=float(el_deg),
            distance_m=float(args.distance_m),
            room_dims=tuple(args.room_dims),
            alpha=float(args.alpha),
            max_order=int(args.max_order),
            fs=FOA_SR,
        )

        audio = make_model_audio_input(dry, rir, device)
        texts = [args.caption]

        with torch.autocast(device_type=("cuda" if device == "cuda" else "cpu"), enabled=args.eval_amp):
            out = model(audio, texts)

        A_SPA.append(F.normalize(out["audio_space_emb"], dim=-1).detach().cpu())
        A_SRC.append(F.normalize(out["audio_source_emb"], dim=-1).detach().cpu())
        T_SPA.append(F.normalize(out["text_space_emb"], dim=-1).detach().cpu())
        T_SRC.append(F.normalize(out["text_source_emb"], dim=-1).detach().cpu())

    emb = {
        "audio_space": torch.cat(A_SPA, dim=0),
        "audio_source": torch.cat(A_SRC, dim=0),
        "text_space": torch.cat(T_SPA, dim=0),
        "text_source": torch.cat(T_SRC, dim=0),
    }

    D_phys = elevation_distance_deg(elevations_deg)

    metrics = {
        "unit_note": "elevations_deg are degrees; radians are used only inside trig functions",
        "sweep_name": "elevation",
        "n_elevations": int(len(elevations_deg)),
        "elevations_deg": elevations_deg.astype(float).tolist(),
        "angles_deg": elevations_deg.astype(float).tolist(),  # compatibility with old analysis scripts
        "azimuth_deg_fixed": float(args.azimuth_deg),
        "el_min_deg": float(args.el_min_deg),
        "el_max_deg": float(args.el_max_deg),
        "distance_m": float(args.distance_m),
        "room_dims": list(map(float, args.room_dims)),
        "alpha": float(args.alpha),
        "max_order": int(args.max_order),
        "audio_path": str(args.audio_path),
        "ckpt": str(args.ckpt),
        "embeddings": {},
    }

    for name, Z in emb.items():
        D_emb = cosine_distance_matrix(Z)
        rho = spearmanr_np(upper_tri_values(D_phys), upper_tri_values(D_emb))
        knn4 = knn_overlap_score(D_emb, elevations_deg, k=4)
        knn8 = knn_overlap_score(D_emb, elevations_deg, k=8)

        metrics["embeddings"][name] = {
            "spearman_vs_absolute_elevation_deg": rho,
            "knn_overlap_at_4": knn4,
            "knn_overlap_at_8": knn8,
        }

        np.save(out_dir / f"{name}_distance_matrix.npy", D_emb)

        plot_heatmap(
            D_emb,
            elevations_deg,
            title=f"{name}: continuous elevation sweep",
            out_path=out_dir / f"heatmap_{name}_elevation_deg.png",
        )

        plot_scatter(
            D_phys,
            D_emb,
            title=f"{name}",
            out_path=out_dir / f"scatter_{name}_vs_absolute_elevation_deg.png",
        )

    # Save embeddings too, in case you want to inspect later.
    torch.save(
        {
            "sweep_name": "elevation",
            "elevations_deg": torch.tensor(elevations_deg, dtype=torch.float32),
            "angles_deg": torch.tensor(elevations_deg, dtype=torch.float32),  # compatibility
            "azimuth_deg_fixed": float(args.azimuth_deg),
            "embeddings": emb,
        },
        out_dir / "continuous_elevation_embeddings.pt",
    )

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n=== Metrics ===")
    for name, vals in metrics["embeddings"].items():
        print(
            f"{name:12s} "
            f"rho={vals['spearman_vs_absolute_elevation_deg']:.4f} "
            f"KNN@4={vals['knn_overlap_at_4']:.4f} "
            f"KNN@8={vals['knn_overlap_at_8']:.4f}"
        )

    print(f"\n[✓] wrote outputs to: {out_dir.resolve()}")
    print("\nRecommended files:")
    print(f"  - {out_dir / 'heatmap_audio_space_elevation_deg.png'}")
    print(f"  - {out_dir / 'heatmap_audio_source_elevation_deg.png'}")
    print(f"  - {out_dir / 'scatter_audio_space_vs_absolute_elevation_deg.png'}")
    print(f"  - {out_dir / 'metrics.json'}")
    print(f"  - {out_dir / 'continuous_elevation_embeddings.pt'}")


if __name__ == "__main__":
    main()
