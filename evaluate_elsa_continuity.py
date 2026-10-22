#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate spatial continuity of the ELSA audio embedding.

Place this file at the root of ``disse_analyze`` (next to
``continuous_azimuth_heatmap.py``).  The ELSA implementation may remain in a
separate directory; pass that directory with ``--elsa-root``.

The script deliberately uses the same source/RIR/FOA/IV preprocessing as the
DISSE continuous-angle experiments.  It evaluates one axis and one acoustic
condition per run and writes:

* ``elsa_embeddings.pt``
* ``cosine_distance_matrix.npy``
* ``metrics.json`` containing QNX(2), Sadj, and (when LKH runs) Atour
* WordTour-compatible text, labels, TSP, and LKH parameter files
* ``wordtour_route.png`` and ``wordtour_route.csv`` when LKH is available

Examples
--------
Azimuth, dry::

    python evaluate_elsa_continuity.py \
      --elsa-root "$HOME/elsa" \
      --ckpt "$HOME/elsa/Spatial_AudioCaps/takamichi09/elsa_ckck_0711/elsa_epoch_30.pt" \
      --audio-path AudioCaps_mp3/test/107280.mp3 \
      --axis azimuth --condition dry \
      --out-dir results/elsa/azimuth_dry

Elevation, reverberant::

    python evaluate_elsa_continuity.py \
      --elsa-root "$HOME/elsa" \
      --ckpt "$HOME/elsa/Spatial_AudioCaps/takamichi09/elsa_ckck_0711/elsa_epoch_30.pt" \
      --audio-path AudioCaps_mp3/test/107280.mp3 \
      --axis elevation --condition reverb \
      --out-dir results/elsa/elevation_reverb

Notes
-----
* No file under ELSA/model is modified.
* Only ``AudioEncoder`` is instantiated; RoBERTa and the regression heads are
  unnecessary for this audio-only continuity test.
* The final ELSA checkpoint is filtered to ``audio_encoder.*`` and loaded with
  a strict key check.  This prevents accidentally evaluating partly random
  weights.
* For azimuth, WordTour is a cycle.  For elevation, a zero-cost dummy node is
  added so that removing it yields a path rather than a cycle.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pyroomacoustics as pra
import torch
import torch.nn.functional as F
import torchaudio
from pyroomacoustics.directivities import CardioidFamily, DirectionVector
from tqdm.auto import tqdm


FOA_SR = 48_000
IV_SR = 16_000
MAX_DURATION_SEC = 10.0


@lru_cache(maxsize=16)
def _cached_hann(n_fft: int, device_string: str) -> torch.Tensor:
    return torch.hann_window(n_fft, device=torch.device(device_string))


def foa_to_iv(
    foa_wave: torch.Tensor,
    n_fft: int = 400,
    hop: int = 100,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact IV preprocessing used by dataset/audio_rir_dataset.py."""
    batch_size, channels, samples = foa_wave.shape
    if channels != 4:
        raise ValueError(f"FOA waveform must have shape [B, 4, T], got {foa_wave.shape}")
    window = _cached_hann(n_fft, str(foa_wave.device))
    spectrum = torch.stft(
        foa_wave.reshape(-1, samples),
        n_fft=n_fft,
        hop_length=hop,
        window=window,
        center=True,
        return_complex=True,
    ).reshape(batch_size, 4, n_fft // 2 + 1, -1)

    w, y, z, x = spectrum[:, 0], spectrum[:, 1], spectrum[:, 2], spectrum[:, 3]
    conjugate_w = w.conj()
    i_act = torch.stack(
        [
            (conjugate_w * y).real,
            (conjugate_w * z).real,
            (conjugate_w * x).real,
        ],
        dim=1,
    )
    i_rea = torch.stack(
        [
            (conjugate_w * y).imag,
            (conjugate_w * z).imag,
            (conjugate_w * x).imag,
        ],
        dim=1,
    )
    norm = torch.linalg.norm(i_act, dim=1, keepdim=True)
    i_act = torch.where(norm > eps, i_act / norm, i_act)
    i_rea = torch.where(norm > eps, i_rea / norm, i_rea)
    return i_act.float(), i_rea.float()


# ---------------------------------------------------------------------------
# ELSA model loading
# ---------------------------------------------------------------------------

def _pick_state_dict(obj: Any, requested_key: str | None) -> Mapping[str, torch.Tensor]:
    """Resolve a state_dict from the checkpoint formats used in this project."""
    if not isinstance(obj, Mapping):
        raise TypeError("Checkpoint must be a mapping/dictionary.")

    keys: List[str] = []
    if requested_key:
        keys.append(requested_key)
    keys.extend(["model_state_dict", "state_dict", "model"])

    for key in keys:
        value = obj.get(key)
        if isinstance(value, Mapping) and value:
            return value

    if obj and all(torch.is_tensor(v) for v in obj.values()):
        return obj  # checkpoint itself is the state_dict

    raise ValueError(
        "No state_dict was found. Check --ckpt-state-key "
        "(normally 'model_state_dict')."
    )


def _strip_leading_prefixes(
    state: Mapping[str, torch.Tensor],
    prefixes: Sequence[str],
) -> Dict[str, torch.Tensor]:
    """Strip nested leading wrappers such as module.model. from checkpoint keys."""
    cleaned: Dict[str, torch.Tensor] = {}
    for original_key, value in state.items():
        key = original_key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if prefix and key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
                    break
        cleaned[key] = value
    return cleaned


def load_elsa_audio_encoder(
    *,
    elsa_root: Path,
    module_name: str,
    class_name: str,
    ckpt_path: Path,
    ckpt_state_key: str | None,
    device: torch.device,
) -> torch.nn.Module:
    """Import ELSA from a separate tree and strictly restore AudioEncoder."""
    elsa_root = elsa_root.expanduser().resolve()
    if not elsa_root.is_dir():
        raise FileNotFoundError(f"ELSA root not found: {elsa_root}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"ELSA checkpoint not found: {ckpt_path}")

    # Put ELSA before this repository, because both trees can contain a
    # top-level package named `model`.
    sys.path.insert(0, str(elsa_root))

    top_package = module_name.split(".", 1)[0]
    already_loaded = sys.modules.get(top_package)
    if already_loaded is not None:
        loaded_from = getattr(already_loaded, "__file__", None)
        if loaded_from and elsa_root not in Path(loaded_from).resolve().parents:
            raise RuntimeError(
                f"Python package '{top_package}' was already imported from "
                f"{loaded_from}, not from {elsa_root}. Start a fresh Python "
                "process and run this script directly from the disse_analyze root."
            )

    try:
        module = importlib.import_module(module_name)
        encoder_class = getattr(module, class_name)
        # These values match the ELSA config supplied for the checkpoint.
        encoder = encoder_class(hidden1=768, out_dim=512)
    except Exception as exc:
        raise RuntimeError(
            f"Could not construct {module_name}:{class_name} from {elsa_root}. "
            "The original AudioEncoder currently loads its SpatialLibriSpeech "
            "checkpoint during __init__, so that existing path must be valid. "
            "No model-source edit is otherwise required."
        ) from exc

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    full_state = _pick_state_dict(checkpoint, ckpt_state_key)
    full_state = _strip_leading_prefixes(full_state, ("module.", "model."))

    prefix = "audio_encoder."
    audio_state = {
        key[len(prefix):]: value
        for key, value in full_state.items()
        if key.startswith(prefix)
    }

    # Also support an already-filtered AudioEncoder-only checkpoint.
    if not audio_state:
        encoder_keys = set(encoder.state_dict().keys())
        audio_state = {key: value for key, value in full_state.items() if key in encoder_keys}

    if not audio_state:
        sample = list(full_state.keys())[:5]
        raise RuntimeError(
            "No audio_encoder weights were found in the checkpoint. "
            f"First checkpoint keys: {sample}"
        )

    missing, unexpected = encoder.load_state_dict(audio_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "AudioEncoder checkpoint mismatch. Evaluation was stopped to avoid "
            f"partly random weights. missing={len(missing)} {missing[:10]}, "
            f"unexpected={len(unexpected)} {unexpected[:10]}"
        )

    print(f"[load] restored AudioEncoder strictly: {len(audio_state)} tensors")
    encoder.to(device).eval()
    return encoder


# ---------------------------------------------------------------------------
# Same acoustic stimulus construction as the DISSE continuous-angle scripts
# ---------------------------------------------------------------------------

def load_dry_audio(audio_path: Path, device: torch.device) -> torch.Tensor:
    wav, sr = torchaudio.load(str(audio_path))
    if wav.numel() == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")
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
    r = radius_m
    v = r / math.sqrt(3.0)
    tetrahedron = np.array(
        [
            [v, v, v],
            [v, -v, -v],
            [-v, v, -v],
            [-v, -v, v],
        ],
        dtype=float,
    ).T

    directivities = []
    for x, y, z in tetrahedron.T:
        azimuth_deg = math.degrees(math.atan2(y, x)) % 360.0
        colatitude_deg = math.degrees(math.acos(z / r))
        directivities.append(
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
    return tetrahedron, directivities


def source_position_from_angles_deg(
    center_xyz: np.ndarray,
    distance_m: float,
    azimuth_deg: float,
    elevation_deg: float,
) -> np.ndarray:
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


def generate_aformat_rir(
    *,
    azimuth_deg: float,
    elevation_deg: float,
    distance_m: float,
    room_dims: Tuple[float, float, float],
    alpha: float,
    max_order: int,
    fs: int,
) -> torch.Tensor:
    """Generate a four-channel tetrahedral A-format RIR."""
    width, depth, height = room_dims
    center = np.array([width / 2.0, depth / 2.0, height / 2.0], dtype=float)
    source = source_position_from_angles_deg(
        center,
        distance_m,
        azimuth_deg,
        elevation_deg,
    )

    margin = 0.1
    if not (
        margin <= source[0] <= width - margin
        and margin <= source[1] <= depth - margin
        and margin <= source[2] <= height - margin
    ):
        raise ValueError(
            "Source is outside the room: "
            f"az={azimuth_deg}, el={elevation_deg}, source={source.tolist()}, "
            f"room_dims={room_dims}. Reduce distance/range or enlarge the room."
        )

    room = pra.ShoeBox(
        room_dims,
        fs=fs,
        materials=pra.Material(alpha),
        max_order=max_order,
    )
    room.add_source(source.tolist())

    tetrahedron, directivities = make_tetrahedral_directivities(radius_m=0.05)
    microphone_positions = center.reshape(3, 1) + tetrahedron
    room.add_microphone_array(
        pra.MicrophoneArray(
            microphone_positions,
            fs=fs,
            directivity=directivities,
        )
    )
    room.compute_rir()

    channels = [room.rir[microphone_index][0] for microphone_index in range(4)]
    max_length = max(len(channel) for channel in channels)
    channels = [np.pad(channel, (0, max_length - len(channel))) for channel in channels]
    return torch.from_numpy(np.stack(channels, axis=0).astype(np.float32))


def aformat_to_foa(aformat: torch.Tensor) -> torch.Tensor:
    """Convert tetrahedral A-format to FOA channel order [W, Y, Z, X]."""
    m0, m1, m2, m3 = aformat[0], aformat[1], aformat[2], aformat[3]
    w = (m0 + m1 + m2 + m3) / 2.0
    x = (m0 + m1 - m2 - m3) / 2.0
    y = (m0 - m1 + m2 - m3) / 2.0
    z = (m0 - m1 - m2 + m3) / 2.0
    return torch.stack([w, y, z, x], dim=0)


def make_elsa_audio_input(
    dry_mono_48k: torch.Tensor,
    rir_aformat: torch.Tensor,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    rir_aformat = rir_aformat.to(device)
    wet_aformat = torchaudio.functional.fftconvolve(dry_mono_48k, rir_aformat)
    wet_aformat = wet_aformat[..., : dry_mono_48k.shape[-1]]

    foa = aformat_to_foa(wet_aformat)
    omni_48k = foa[0]
    foa_16k = torchaudio.functional.resample(foa, FOA_SR, IV_SR)
    i_act, i_rea = foa_to_iv(foa_16k.unsqueeze(0), n_fft=400, hop=100)
    return {
        "i_act": i_act.float(),
        "i_rea": i_rea.float(),
        "omni_48k": omni_48k.unsqueeze(0).float(),
    }


def make_sweep_values(args: argparse.Namespace) -> np.ndarray:
    if args.axis == "azimuth":
        if args.az_step_deg <= 0:
            raise ValueError("--az-step-deg must be positive.")
        values = np.arange(
            args.az_start_deg,
            args.az_end_deg,
            args.az_step_deg,
            dtype=np.float64,
        )
    else:
        if args.n_steps < 2:
            raise ValueError("--n-steps must be at least 2.")
        if args.el_min_deg >= args.el_max_deg:
            raise ValueError("--el-min-deg must be smaller than --el-max-deg.")
        values = np.linspace(
            args.el_min_deg,
            args.el_max_deg,
            args.n_steps,
            dtype=np.float64,
        )

    if len(values) < 3:
        raise ValueError("At least three sweep points are required for QNX(2).")
    return values


def extract_embeddings(
    *,
    encoder: torch.nn.Module,
    dry: torch.Tensor,
    values_deg: np.ndarray,
    axis: str,
    fixed_azimuth_deg: float,
    fixed_elevation_deg: float,
    distance_m: float,
    room_dims: Tuple[float, float, float],
    alpha: float,
    max_order: int,
    batch_size: int,
    eval_amp: bool,
    device: torch.device,
) -> torch.Tensor:
    embeddings: List[torch.Tensor] = []
    pending: List[Dict[str, torch.Tensor]] = []

    def encode_pending() -> None:
        if not pending:
            return
        batch = {
            key: torch.cat([item[key] for item in pending], dim=0)
            for key in ("i_act", "i_rea", "omni_48k")
        }
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            enabled=eval_amp,
        ):
            # Positional arguments match AudioEncoder.forward(I_act, I_rea, Omni).
            output = encoder(batch["i_act"], batch["i_rea"], batch["omni_48k"])
        if output.ndim != 2 or output.shape[-1] != 512:
            raise RuntimeError(f"Expected AudioEncoder output [B, 512], got {tuple(output.shape)}")
        embeddings.append(F.normalize(output.float(), dim=-1).cpu())
        pending.clear()

    for value_deg in tqdm(values_deg, desc=f"ELSA {axis} sweep", dynamic_ncols=True):
        if axis == "azimuth":
            azimuth_deg = float(value_deg)
            elevation_deg = float(fixed_elevation_deg)
        else:
            azimuth_deg = float(fixed_azimuth_deg)
            elevation_deg = float(value_deg)

        rir = generate_aformat_rir(
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg,
            distance_m=distance_m,
            room_dims=room_dims,
            alpha=alpha,
            max_order=max_order,
            fs=FOA_SR,
        )
        pending.append(make_elsa_audio_input(dry, rir, device))
        if len(pending) >= batch_size:
            encode_pending()

    encode_pending()
    result = torch.cat(embeddings, dim=0)
    if result.shape[0] != len(values_deg):
        raise RuntimeError(
            f"Embedding count mismatch: {result.shape[0]} vs {len(values_deg)}"
        )
    return result


# ---------------------------------------------------------------------------
# QNX(2) and Sadj
# ---------------------------------------------------------------------------

def cosine_distance_matrix(embeddings: torch.Tensor) -> np.ndarray:
    embeddings = F.normalize(embeddings.float(), dim=-1)
    distance = 1.0 - embeddings @ embeddings.T
    result = np.clip(distance.cpu().numpy().astype(np.float64), 0.0, 2.0)
    np.fill_diagonal(result, 0.0)
    return result


def physical_distance_matrix(values_deg: np.ndarray, axis: str) -> np.ndarray:
    distance = np.abs(values_deg[:, None] - values_deg[None, :])
    if axis == "azimuth":
        distance = np.minimum(distance, 360.0 - distance)
    return distance


def qnx_at_k(
    embedding_distance: np.ndarray,
    physical_distance: np.ndarray,
    k: int,
) -> Tuple[float, List[float]]:
    """Mean k-neighborhood overlap; this is Q_NX(k)."""
    n = embedding_distance.shape[0]
    if embedding_distance.shape != (n, n) or physical_distance.shape != (n, n):
        raise ValueError("Distance matrices must both be square and have equal shape.")
    if not 1 <= k < n:
        raise ValueError(f"k must satisfy 1 <= k < N; got k={k}, N={n}")

    per_point: List[float] = []
    for index in range(n):
        emb_row = embedding_distance[index].copy()
        phys_row = physical_distance[index].copy()
        emb_row[index] = np.inf
        phys_row[index] = np.inf
        emb_neighborhood = set(np.argsort(emb_row, kind="stable")[:k].tolist())
        phys_neighborhood = set(np.argsort(phys_row, kind="stable")[:k].tolist())
        per_point.append(len(emb_neighborhood & phys_neighborhood) / float(k))
    return float(np.mean(per_point)), per_point


def adjacent_edges(values_deg: np.ndarray, axis: str) -> List[Tuple[int, int]]:
    order = np.argsort(values_deg, kind="stable").tolist()
    edges = list(zip(order[:-1], order[1:]))
    if axis == "azimuth":
        edges.append((order[-1], order[0]))
    return edges


def normalized_adjacent_dispersion(distances: np.ndarray) -> float:
    """Sadj = sum((d_i-mean(d))^2) / sum(d_i^2)."""
    distances = np.asarray(distances, dtype=np.float64)
    denominator = float(np.sum(distances ** 2))
    if denominator <= np.finfo(np.float64).eps:
        return float("nan")
    numerator = float(np.sum((distances - distances.mean()) ** 2))
    return numerator / denominator


def continuity_metrics(
    embeddings: torch.Tensor,
    values_deg: np.ndarray,
    axis: str,
) -> Tuple[Dict[str, Any], np.ndarray, List[Dict[str, float]]]:
    cosine_distance = cosine_distance_matrix(embeddings)
    physical_distance = physical_distance_matrix(values_deg, axis)
    qnx2, qnx2_per_point = qnx_at_k(cosine_distance, physical_distance, k=2)

    edges = adjacent_edges(values_deg, axis)
    adjacent_cosine = np.asarray(
        [cosine_distance[left, right] for left, right in edges],
        dtype=np.float64,
    )
    # WordTour uses Euclidean distance on L2-normalized vectors.  It is
    # sqrt(2 * cosine_distance); save its dispersion as a diagnostic while
    # retaining cosine Sadj for comparability with the DISSE table.
    adjacent_wordtour_l2 = np.sqrt(np.maximum(0.0, 2.0 * adjacent_cosine))

    upper = cosine_distance[np.triu_indices_from(cosine_distance, k=1)]
    edge_rows: List[Dict[str, float]] = []
    for (left, right), d_cos, d_l2 in zip(edges, adjacent_cosine, adjacent_wordtour_l2):
        edge_rows.append(
            {
                "from_index": int(left),
                "to_index": int(right),
                "from_deg": float(values_deg[left]),
                "to_deg": float(values_deg[right]),
                "cosine_distance": float(d_cos),
                "wordtour_l2_distance": float(d_l2),
            }
        )

    metrics: Dict[str, Any] = {
        "Atour": None,
        "QNX(2)": qnx2,
        "Sadj": normalized_adjacent_dispersion(adjacent_cosine),
        "definitions": {
            "Atour": (
                "fraction of WordTour/LKH route edges that connect physically "
                "adjacent angles (azimuth: cycle; elevation: dummy-node path)"
            ),
            "QNX(2)": (
                "mean overlap between each point's two nearest physical "
                "neighbors and two nearest embedding neighbors"
            ),
            "Sadj": (
                "sum((d_adj-mean(d_adj))^2)/sum(d_adj^2), where d_adj "
                "is 1-cosine for physically adjacent points"
            ),
        },
        "diagnostics": {
            "qnx2_per_point": qnx2_per_point,
            "adjacent_cosine_distance_mean": float(adjacent_cosine.mean()),
            "adjacent_cosine_distance_std": float(adjacent_cosine.std()),
            "adjacent_cosine_distance_cv": (
                float(adjacent_cosine.std() / adjacent_cosine.mean())
                if adjacent_cosine.mean() > 0
                else float("nan")
            ),
            "sadj_wordtour_l2_diagnostic": normalized_adjacent_dispersion(
                adjacent_wordtour_l2
            ),
            "mean_all_pairwise_cosine_distance": float(upper.mean()),
            "std_all_pairwise_cosine_distance": float(upper.std()),
        },
    }
    return metrics, cosine_distance, edge_rows


# ---------------------------------------------------------------------------
# WordTour/LKH files and Atour
# ---------------------------------------------------------------------------

def wordtour_vectors(embeddings: torch.Tensor) -> np.ndarray:
    """Match export_disse_emb_to_wordtour.py's 8-decimal normalized text."""
    normalized = F.normalize(embeddings.float(), dim=-1).cpu().numpy().astype(np.float64)
    # Formatting with %.8f is part of the existing DISSE -> WordTour pipeline.
    return np.asarray([[float(f"{value:.8f}") for value in row] for row in normalized])


def write_wordtour_text(
    vectors: np.ndarray,
    values_deg: np.ndarray,
    axis: str,
    output_path: Path,
) -> None:
    prefix = "az" if axis == "azimuth" else "el"
    with output_path.open("w", encoding="utf-8") as stream:
        for index, (value_deg, vector) in enumerate(zip(values_deg, vectors), start=1):
            if axis == "azimuth":
                label = f"az{int(round(float(value_deg))) % 360:03d}"
            else:
                label = f"el{float(value_deg):+.3f}"
            values = " ".join(f"{value:.8f}" for value in vector)
            stream.write(f"{label} {values}\n")


def wordtour_integer_weights(vectors: np.ndarray) -> np.ndarray:
    difference = vectors[:, None, :] - vectors[None, :, :]
    euclidean = np.sqrt(np.sum(difference * difference, axis=-1))
    weights = np.floor(euclidean * 1000.0).astype(np.int64)
    np.fill_diagonal(weights, 0)
    return weights


def write_labels(values_deg: np.ndarray, axis: str, output_path: Path) -> int | None:
    if axis == "azimuth":
        with output_path.open("w", encoding="utf-8") as stream:
            for node_id, value_deg in enumerate(values_deg, start=1):
                label = f"az{int(round(float(value_deg))) % 360:03d}"
                stream.write(f"{node_id}\t{label}\t{float(value_deg):.8f}\n")
        return None

    dummy_id = len(values_deg) + 1
    with output_path.open("w", encoding="utf-8") as stream:
        stream.write("node_id\tlabel\televation_deg\tis_dummy\n")
        for node_id, value_deg in enumerate(values_deg, start=1):
            stream.write(
                f"{node_id}\tel{float(value_deg):+.3f}\t"
                f"{float(value_deg):.8f}\t0\n"
            )
        stream.write(f"{dummy_id}\tdummy\tnan\t1\n")
    return dummy_id


def write_lkh_tsp(
    weights: np.ndarray,
    axis: str,
    output_path: Path,
    name: str,
) -> int | None:
    n_real = weights.shape[0]
    dummy_id = n_real + 1 if axis == "elevation" else None
    dimension = n_real + (1 if dummy_id is not None else 0)

    with output_path.open("w", encoding="utf-8") as stream:
        stream.write(f"NAME: {name}\n")
        stream.write("TYPE: TSP\n")
        stream.write(f"DIMENSION: {dimension}\n")
        stream.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
        stream.write("EDGE_WEIGHT_FORMAT: UPPER_ROW\n")
        stream.write("EDGE_WEIGHT_SECTION\n")
        for left in range(1, dimension + 1):
            row: List[str] = []
            for right in range(left + 1, dimension + 1):
                if dummy_id is not None and (left == dummy_id or right == dummy_id):
                    row.append("0")
                else:
                    row.append(str(int(weights[left - 1, right - 1])))
            if row:
                stream.write(" ".join(row) + "\n")
        stream.write("EOF\n")
    return dummy_id


def write_lkh_parameter_file(
    *,
    problem_path: Path,
    tour_path: Path,
    parameter_path: Path,
    runs: int,
    seed: int,
) -> None:
    with parameter_path.open("w", encoding="utf-8") as stream:
        stream.write(f"PROBLEM_FILE = {problem_path.resolve()}\n")
        stream.write(f"OUTPUT_TOUR_FILE = {tour_path.resolve()}\n")
        stream.write(f"RUNS = {runs}\n")
        stream.write(f"SEED = {seed}\n")


def read_lkh_tour(tour_path: Path) -> List[int]:
    node_ids: List[int] = []
    in_tour_section = False
    with tour_path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            value = line.strip()
            if value == "TOUR_SECTION":
                in_tour_section = True
                continue
            if not in_tour_section:
                continue
            if value in {"-1", "EOF"}:
                break
            if value:
                node_ids.append(int(value))
    if not node_ids:
        raise ValueError(f"No TOUR_SECTION was found in {tour_path}")
    return node_ids


def infer_physical_step(values_deg: np.ndarray, axis: str) -> float:
    unique = np.sort(np.unique(np.round(values_deg.astype(np.float64), 8)))
    if axis == "azimuth":
        differences = np.diff(np.r_[unique, unique[0] + 360.0])
    else:
        differences = np.diff(unique)
    differences = differences[differences > 1e-8]
    if len(differences) == 0:
        raise ValueError("Could not infer a positive physical step.")
    return float(np.median(differences))


def score_atour(
    *,
    tour_path: Path,
    values_deg: np.ndarray,
    axis: str,
    dummy_id: int | None,
    tolerance_deg: float | None,
) -> Dict[str, Any]:
    tour_ids = read_lkh_tour(tour_path)
    expected_real_ids = set(range(1, len(values_deg) + 1))

    if axis == "elevation":
        if dummy_id is None or dummy_id not in tour_ids:
            raise ValueError(f"Dummy node {dummy_id} is missing from the LKH tour.")
        dummy_index = tour_ids.index(dummy_id)
        route_ids = tour_ids[dummy_index + 1:] + tour_ids[:dummy_index]
    else:
        route_ids = tour_ids

    if len(route_ids) != len(values_deg) or set(route_ids) != expected_real_ids:
        raise ValueError(
            "LKH route is not a permutation of all real nodes: "
            f"N_route={len(route_ids)}, N_expected={len(values_deg)}"
        )

    route_values = np.asarray([values_deg[node_id - 1] for node_id in route_ids])
    ideal_step = infer_physical_step(values_deg, axis)
    if tolerance_deg is None:
        tolerance_deg = max(1e-3, ideal_step * 1e-4)

    if axis == "azimuth":
        next_values = np.roll(route_values, -1)
        raw_jumps = np.abs(next_values - route_values)
        jumps = np.minimum(raw_jumps, 360.0 - raw_jumps)
    else:
        jumps = np.abs(np.diff(route_values))

    matching = np.abs(jumps - ideal_step) <= tolerance_deg
    return {
        "Atour": float(np.mean(matching)),
        "ideal_step_deg": ideal_step,
        "atour_tolerance_deg": float(tolerance_deg),
        "matching_edges": int(np.sum(matching)),
        "number_of_edges": int(len(jumps)),
        "mean_physical_jump_deg": float(np.mean(jumps)),
        "median_physical_jump_deg": float(np.median(jumps)),
        "max_physical_jump_deg": float(np.max(jumps)),
        "route_node_ids": [int(node_id) for node_id in route_ids],
        "route_values_deg": [float(value) for value in route_values],
        "route_jumps_deg": [float(value) for value in jumps],
    }


def write_route_csv(
    route: Mapping[str, Any],
    axis: str,
    output_path: Path,
) -> None:
    values = route["route_values_deg"]
    jumps = route["route_jumps_deg"]
    node_ids = route["route_node_ids"]
    with output_path.open("w", encoding="utf-8") as stream:
        stream.write(f"route_step,node_id,{axis}_deg,jump_to_next_deg\n")
        for index, (node_id, value) in enumerate(zip(node_ids, values)):
            if axis == "azimuth":
                jump = jumps[index]
            else:
                jump = "" if index == len(values) - 1 else jumps[index]
            stream.write(f"{index},{node_id},{value},{jump}\n")


def plot_route(route: Mapping[str, Any], axis: str, output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] route plot skipped because matplotlib could not load: {exc}")
        return

    values = np.asarray(route["route_values_deg"], dtype=np.float64)
    if axis == "azimuth":
        theta = np.deg2rad(values)
        x = np.cos(theta)
        y = np.sin(theta)
        x_closed = np.r_[x, x[0]]
        y_closed = np.r_[y, y[0]]
        figure, axes = plt.subplots(figsize=(7, 7))
        axes.plot(x_closed, y_closed, color="#2563eb", linewidth=1.2, alpha=0.75)
        scatter = axes.scatter(x, y, c=np.arange(len(values)), cmap="viridis", s=30)
        axes.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="0.75", linewidth=1))
        axes.set_aspect("equal")
        axes.set_xlabel("cos(azimuth)")
        axes.set_ylabel("sin(azimuth)")
        axes.set_title(f"ELSA WordTour route | Atour={route['Atour']:.3f}")
        figure.colorbar(scatter, ax=axes, label="route step")
    else:
        figure, axes = plt.subplots(figsize=(9, 4.8))
        axes.plot(np.arange(len(values)), values, marker="o", markersize=3, linewidth=1.2)
        axes.set_xlabel("route step")
        axes.set_ylabel("elevation [deg]")
        axes.set_title(f"ELSA WordTour path | Atour={route['Atour']:.3f}")
        axes.grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Evaluate ELSA azimuth/elevation continuity with Atour, QNX(2), Sadj.",
    )
    parser.add_argument("--elsa-root", required=True, type=Path)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--audio-path", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--axis", required=True, choices=("azimuth", "elevation"))
    parser.add_argument("--condition", required=True, choices=("dry", "reverb"))

    parser.add_argument("--audio-encoder-module", default="model.audio_encoder")
    parser.add_argument("--audio-encoder-class", default="AudioEncoder")
    parser.add_argument("--ckpt-state-key", default="model_state_dict")

    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-amp", action="store_true")

    # Azimuth defaults reproduce the DISSE sweep: [0, 360) in 5-degree steps.
    parser.add_argument("--az-start-deg", type=float, default=0.0)
    parser.add_argument("--az-end-deg", type=float, default=360.0)
    parser.add_argument("--az-step-deg", type=float, default=5.0)
    parser.add_argument("--fixed-elevation-deg", type=float, default=0.0)

    # Elevation defaults match DISSE: -45, -40, ..., 45 degrees.
    parser.add_argument("--el-min-deg", type=float, default=-45.0)
    parser.add_argument("--el-max-deg", type=float, default=45.0)
    parser.add_argument("--n-steps", type=int, default=19)
    parser.add_argument("--fixed-azimuth-deg", type=float, default=0.0)

    # When omitted, distance is 2 m for azimuth and 1 m for elevation, matching
    # the existing controlled experiments. Acoustic-condition defaults are
    # dry=(alpha=1,max_order=0), reverb=(alpha=.2,max_order=10).
    parser.add_argument("--distance-m", type=float, default=None)
    parser.add_argument("--room-dims", type=float, nargs=3, default=(8.0, 7.0, 3.0))
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--max-order", type=int, default=None)

    parser.add_argument(
        "--lkh",
        type=Path,
        default=Path.home() / "external/wordtour/LKH-3.0.6/LKH",
        help="LKH executable. Atour stays null if it is absent or --skip-lkh is used.",
    )
    parser.add_argument("--skip-lkh", action="store_true")
    parser.add_argument("--lkh-runs", type=int, default=10)
    parser.add_argument("--lkh-seed", type=int, default=42)
    parser.add_argument("--atour-tolerance-deg", type=float, default=None)
    return parser.parse_args()


def resolve_device(device_argument: str) -> torch.device:
    if device_argument == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_argument)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {device_argument}")
    return device


def write_adjacent_csv(rows: Iterable[Mapping[str, float]], path: Path) -> None:
    columns = (
        "from_index",
        "to_index",
        "from_deg",
        "to_deg",
        "cosine_distance",
        "wordtour_l2_distance",
    )
    with path.open("w", encoding="utf-8") as stream:
        stream.write(",".join(columns) + "\n")
        for row in rows:
            stream.write(",".join(str(row[column]) for column in columns) + "\n")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1.")
    if args.lkh_runs < 1:
        raise SystemExit("--lkh-runs must be at least 1.")

    device = resolve_device(args.device)
    audio_path = args.audio_path.expanduser().resolve()
    ckpt_path = args.ckpt.expanduser().resolve()
    elsa_root = args.elsa_root.expanduser().resolve()
    output_dir = args.out_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not audio_path.is_file():
        raise SystemExit(f"Audio source not found: {audio_path}")

    values_deg = make_sweep_values(args)
    distance_m = (
        float(args.distance_m)
        if args.distance_m is not None
        else (2.0 if args.axis == "azimuth" else 1.0)
    )
    if args.condition == "dry":
        alpha = 1.0 if args.alpha is None else float(args.alpha)
        max_order = 0 if args.max_order is None else int(args.max_order)
    else:
        alpha = 0.2 if args.alpha is None else float(args.alpha)
        max_order = 10 if args.max_order is None else int(args.max_order)

    print(f"[info] device={device}")
    print(f"[info] ELSA root={elsa_root}")
    print(f"[info] axis={args.axis}, condition={args.condition}, N={len(values_deg)}")
    print(
        f"[info] room={tuple(args.room_dims)}, distance={distance_m} m, "
        f"alpha={alpha}, max_order={max_order}"
    )

    encoder = load_elsa_audio_encoder(
        elsa_root=elsa_root,
        module_name=args.audio_encoder_module,
        class_name=args.audio_encoder_class,
        ckpt_path=ckpt_path,
        ckpt_state_key=args.ckpt_state_key,
        device=device,
    )
    dry = load_dry_audio(audio_path, device)
    embeddings = extract_embeddings(
        encoder=encoder,
        dry=dry,
        values_deg=values_deg,
        axis=args.axis,
        fixed_azimuth_deg=float(args.fixed_azimuth_deg),
        fixed_elevation_deg=float(args.fixed_elevation_deg),
        distance_m=distance_m,
        room_dims=tuple(float(value) for value in args.room_dims),
        alpha=alpha,
        max_order=max_order,
        batch_size=args.batch_size,
        eval_amp=args.eval_amp,
        device=device,
    )

    scores, cosine_distance, adjacent_rows = continuity_metrics(
        embeddings,
        values_deg,
        args.axis,
    )
    np.save(output_dir / "cosine_distance_matrix.npy", cosine_distance)
    write_adjacent_csv(adjacent_rows, output_dir / "physical_adjacent_distances.csv")

    metadata: Dict[str, Any] = {
        "model": "ELSA",
        "embedding": "audio_encoder output, L2-normalized",
        "axis": args.axis,
        "condition": args.condition,
        "values_deg": [float(value) for value in values_deg],
        "fixed_azimuth_deg": float(args.fixed_azimuth_deg),
        "fixed_elevation_deg": float(args.fixed_elevation_deg),
        "distance_m": distance_m,
        "room_dims_m": [float(value) for value in args.room_dims],
        "alpha": alpha,
        "max_order": max_order,
        "audio_path": str(audio_path),
        "ckpt": str(ckpt_path),
        "elsa_root": str(elsa_root),
    }
    torch.save(
        {
            "model": "ELSA",
            "axis": args.axis,
            "angles_deg": torch.tensor(values_deg, dtype=torch.float32),
            "elevations_deg": (
                torch.tensor(values_deg, dtype=torch.float32)
                if args.axis == "elevation"
                else None
            ),
            "embeddings": {"audio_elsa": embeddings},
            "metadata": metadata,
        },
        output_dir / "elsa_embeddings.pt",
    )

    vectors = wordtour_vectors(embeddings)
    text_path = output_dir / "wordtour_embeddings.txt"
    labels_path = output_dir / "wordtour_labels.tsv"
    problem_path = output_dir / "wordtour.tsp"
    parameter_path = output_dir / "wordtour.par"
    tour_path = output_dir / "wordtour.out"

    write_wordtour_text(vectors, values_deg, args.axis, text_path)
    labels_dummy_id = write_labels(values_deg, args.axis, labels_path)
    weights = wordtour_integer_weights(vectors)
    tsp_dummy_id = write_lkh_tsp(
        weights,
        args.axis,
        problem_path,
        name=f"elsa_{args.axis}_{args.condition}",
    )
    if labels_dummy_id != tsp_dummy_id:
        raise RuntimeError("Internal dummy-node mismatch.")
    write_lkh_parameter_file(
        problem_path=problem_path,
        tour_path=tour_path,
        parameter_path=parameter_path,
        runs=args.lkh_runs,
        seed=args.lkh_seed,
    )

    route_metrics: Dict[str, Any] | None = None
    lkh_path = args.lkh.expanduser().resolve()
    if not args.skip_lkh and lkh_path.is_file() and os.access(lkh_path, os.X_OK):
        print(f"[LKH] {lkh_path} {parameter_path}")
        subprocess.run([str(lkh_path), str(parameter_path)], check=True)
        route_metrics = score_atour(
            tour_path=tour_path,
            values_deg=values_deg,
            axis=args.axis,
            dummy_id=tsp_dummy_id,
            tolerance_deg=args.atour_tolerance_deg,
        )
        scores["Atour"] = route_metrics["Atour"]
        scores["wordtour_route"] = route_metrics
        write_route_csv(route_metrics, args.axis, output_dir / "wordtour_route.csv")
        plot_route(route_metrics, args.axis, output_dir / "wordtour_route.png")
    else:
        reason = "--skip-lkh was used" if args.skip_lkh else f"LKH not executable: {lkh_path}"
        print(f"[warn] Atour not computed ({reason}).")
        print(f"[next] Run: {lkh_path} {parameter_path}")

    result = {"metadata": metadata, "scores": scores}
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=True)

    atour_text = "pending" if scores["Atour"] is None else f"{scores['Atour']:.6f}"
    print("\n=== ELSA continuity ===")
    print(f"Atour  : {atour_text}")
    print(f"QNX(2) : {scores['QNX(2)']:.6f}")
    print(f"Sadj   : {scores['Sadj']:.6f}")
    print(f"[done] {output_dir}")


if __name__ == "__main__":
    main()
