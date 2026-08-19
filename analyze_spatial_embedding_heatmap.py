#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze DISSE spatial embedding structure with RIR-wise distance heatmaps.

Purpose:
  - Extract DISSE embeddings from a checkpoint.
  - Aggregate embeddings per RIR / space_id.
  - Plot pairwise cosine-distance heatmaps.
  - Compute Spearman correlations between embedding distance and physical distances.

Recommended first report:
  - audio_space heatmap sorted by 5-factor category
  - audio_source heatmap as control
  - Spearman correlation: D_emb vs D_azimuth / D_T30 / D_distance

Example:
  python analyze_spatial_embedding_heatmap.py \
    --ckpt Spatial_AudioCaps/takamichi09/checkpoints_delsa_fromELSA/ckpt_sup_ep20.pt \
    --config config.yaml \
    --split test \
    --out_dir results/space_heatmap_ep20 \
    --batch_size 64 \
    --num_workers 4 \
    --eval_amp

If config.yaml does not contain paths, specify them explicitly:
  python analyze_spatial_embedding_heatmap.py \
    --ckpt path/to/ckpt.pt \
    --index_csv data/test_precomp/test_precomputed.csv \
    --precomp_root data/test_precomp \
    --rir_csv RIR_dataset/rir_catalog_test.csv \
    --out_dir results/space_heatmap
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import yaml
except ImportError:
    yaml = None

from model.delsa_model import DELSA
from dataset.precomputed_val_dataset import PrecomputedValDataset
from dataset.audio_rir_dataset import collate_fn


# ----------------------------
# Category mapping
# ----------------------------
def map_direction(az: float) -> str:
    if -35 <= az <= 35:
        return "front"
    if 55 <= az <= 125:
        return "right"
    if -125 <= az <= -55:
        return "left"
    if az >= 145 or az <= -145:
        return "back"
    return "ambiguous"


def map_elevation(el: float) -> str:
    if el > 40:
        return "up"
    if el < -40:
        return "down"
    return "ambiguous"


def map_distance(d: float) -> str:
    if d < 1:
        return "near"
    if d > 2:
        return "far"
    return "ambiguous"


def map_size(area: float) -> str:
    if area < 50:
        return "small"
    if area > 100:
        return "large"
    return "mid-sized"


def map_reverb(t30: float) -> str:
    if t30 < 200:
        return "dry"
    if t30 > 1000:
        return "reverberant"
    return "ambiguous"


DIR_ORDER = {"front": 0, "right": 1, "back": 2, "left": 3, "ambiguous": 99}
ELEV_ORDER = {"down": 0, "up": 1, "ambiguous": 99}
DIST_ORDER = {"near": 0, "far": 1, "ambiguous": 99}
SIZE_ORDER = {"small": 0, "mid-sized": 1, "large": 2, "ambiguous": 99}
REV_ORDER = {"dry": 0, "reverberant": 1, "ambiguous": 99}


# ----------------------------
# Utilities
# ----------------------------
def load_config(path: str | None) -> Dict[str, Any]:
    if path is None or not Path(path).exists():
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required to read config.yaml. Install with: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def pick_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        (k.replace("module.", "", 1) if k.startswith("module.") else k): v
        for k, v in sd.items()
    }


def select_device(raw: str) -> str:
    if raw == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return raw


def stack_audio(batch: Dict[str, Any], device: str) -> Dict[str, torch.Tensor]:
    return {
        "i_act": torch.stack([d["i_act"] for d in batch["audio"]]).to(device, non_blocking=True),
        "i_rea": torch.stack([d["i_rea"] for d in batch["audio"]]).to(device, non_blocking=True),
        "omni_48k": torch.stack([d["omni_48k"] for d in batch["audio"]]).to(device, non_blocking=True),
    }


def to_numpy_1d(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().reshape(-1)
    return np.asarray(x).reshape(-1)


def rankdata_average(x: np.ndarray) -> np.ndarray:
    """Small scipy-free average-rank implementation."""
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        avg = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg
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
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def cosine_distance_matrix(Z: torch.Tensor) -> np.ndarray:
    Z = F.normalize(Z.float(), dim=-1)
    D = 1.0 - (Z @ Z.T)
    D = D.detach().cpu().numpy()
    D = np.clip(D, 0.0, 2.0)
    return D


def circular_angle_distance_deg(a: np.ndarray) -> np.ndarray:
    diff = np.abs(a[:, None] - a[None, :])
    return np.minimum(diff, 360.0 - diff)


def absolute_distance_matrix(x: np.ndarray) -> np.ndarray:
    return np.abs(x[:, None] - x[None, :])


def upper_tri_values(M: np.ndarray) -> np.ndarray:
    idx = np.triu_indices_from(M, k=1)
    return M[idx]


def plot_heatmap(
    D: np.ndarray,
    title: str,
    out_path: Path,
    block_boundaries: list[int] | None = None,
    tick_positions: list[float] | None = None,
    tick_labels: list[str] | None = None,
):
    plt.figure(figsize=(8, 7))
    im = plt.imshow(D, origin="upper", aspect="equal", interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="1 - cosine similarity")
    plt.title(title)
    plt.xlabel("RIR condition")
    plt.ylabel("RIR condition")

    if tick_positions is not None and tick_labels is not None:
        plt.xticks(tick_positions, tick_labels, rotation=45, ha="right")
        plt.yticks(tick_positions, tick_labels)

    if block_boundaries is not None:
        for b in block_boundaries:
            plt.axhline(b - 0.5, color="white", linewidth=0.8)
            plt.axvline(b - 0.5, color="white", linewidth=0.8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_scatter_binned(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
    bins: int = 20,
):
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, s=8, alpha=0.25)

    if len(x) > 0:
        qs = np.linspace(0, 1, bins + 1)
        edges = np.unique(np.quantile(x, qs))
        bx, by = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (x >= lo) & (x <= hi)
            if m.sum() > 0:
                bx.append(x[m].mean())
                by.append(y[m].mean())
        if len(bx) > 1:
            plt.plot(bx, by, marker="o", linewidth=2)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


# ----------------------------
# Main extraction
# ----------------------------
@torch.no_grad()
def extract_embeddings(args) -> Dict[str, Any]:
    cfg = load_config(args.config)

    precomp_root = args.precomp_root
    index_csv = args.index_csv
    rir_csv = args.rir_csv

    if precomp_root is None:
        key = f"{args.split}_precomp_root"
        precomp_root = cfg.get(key) or cfg.get("test_precomp_root") or cfg.get("val_precomp_root")

    if index_csv is None:
        key = f"{args.split}_index_csv"
        index_csv = cfg.get(key)
        if index_csv is None and precomp_root is not None:
            index_csv = str(Path(precomp_root) / f"{args.split}_precomputed.csv")

    if rir_csv is None:
        key = f"rir_csv_{args.split}"
        rir_csv = cfg.get(key) or cfg.get("rir_csv_test") or cfg.get("rir_csv_val")

    if index_csv is None or not Path(index_csv).exists():
        raise SystemExit(
            f"[ERR] index_csv not found: {index_csv}\n"
            "Please specify --index_csv, e.g. --index_csv data/test_precomp/test_precomputed.csv"
        )

    if rir_csv is None or not Path(rir_csv).exists():
        raise SystemExit(
            f"[ERR] rir_csv not found: {rir_csv}\n"
            "Please specify --rir_csv, e.g. --rir_csv RIR_dataset/rir_catalog_test.csv"
        )

    device = select_device(args.device)

    print(f"[info] device       = {device}")
    print(f"[info] ckpt         = {args.ckpt}")
    print(f"[info] precomp_root = {precomp_root}")
    print(f"[info] index_csv    = {index_csv}")
    print(f"[info] rir_csv      = {rir_csv}")

    ds = PrecomputedValDataset(
        index_csv=index_csv,
        rir_meta_csv=rir_csv,
        root=precomp_root,
    )
    # ------------------------------------------------------------
    # Repair RIR metadata merge.
    # PrecomputedValDataset merges by raw rir_path.
    # If paths differ by prefix, e.g.
    #   RIR_dataset/RIR_data/test/auto_x.wav
    #   test_fixed_96x96/.../auto_x.wav
    # exact string matching fails. In that case, merge by wav basename.
    # ------------------------------------------------------------
    required_meta_cols = [
        "azimuth_deg",
        "elevation_deg",
        "source_distance_m",
        "area_m2",
        "fullband_T30_ms",
    ]

    def _meta_is_broken(df):
        for c in required_meta_cols:
            if c not in df.columns:
                return True
            if df[c].isna().all():
                return True
        return False

    if _meta_is_broken(ds.df):
        print("[warn] RIR metadata merge failed or incomplete. Retrying by wav basename.")

        idx_df = pd.read_csv(index_csv)
        meta_df = pd.read_csv(rir_csv)

        idx_df["_rir_name"] = idx_df["rir_path"].apply(lambda p: Path(str(p)).name)
        meta_df["_rir_name"] = meta_df["rir_path"].apply(lambda p: Path(str(p)).name)

        keep_cols = ["_rir_name"] + [
            c for c in required_meta_cols if c in meta_df.columns
        ]

        # Optional metadata, useful for inspection but not required.
        optional_cols = ["dims", "alpha", "distance", "fs", "room_id", "split"]
        keep_cols += [c for c in optional_cols if c in meta_df.columns and c not in keep_cols]

        repaired = idx_df.merge(
            meta_df[keep_cols],
            on="_rir_name",
            how="left",
        )

        missing = repaired["azimuth_deg"].isna().sum()
        print(f"[info] repaired metadata missing rows: {missing} / {len(repaired)}")
        print("[info] unique azimuth:", repaired["azimuth_deg"].nunique(dropna=True))
        print("[info] unique elevation:", repaired["elevation_deg"].nunique(dropna=True))
        print("[info] unique distance:", repaired["source_distance_m"].nunique(dropna=True))
        print("[info] unique area:", repaired["area_m2"].nunique(dropna=True))
        print("[info] unique T30:", repaired["fullband_T30_ms"].nunique(dropna=True))

        if missing == len(repaired):
            raise SystemExit(
                "[ERR] Could not merge RIR metadata even by basename. "
                "Check rir_path values in index_csv and rir_csv."
            )

        ds.df = repaired.reset_index(drop=True)
        ds._space_map = {
            p: i for i, p in enumerate(sorted(ds.df["rir_path"].unique().tolist()))
        }
    if args.source_id is not None:
        before = len(ds.df)
        ds.df = ds.df[ds.df["audiocap_id"].astype(int) == int(args.source_id)].reset_index(drop=True)
        after = len(ds.df)

        if after == 0:
            raise SystemExit(
                f"[ERR] source_id={args.source_id} not found in dataset. "
                f"Check audiocap_id values in: {index_csv}"
            )

        # source_idで絞った後、space_id mappingを作り直す
        ds._space_map = {
            p: i for i, p in enumerate(sorted(ds.df["rir_path"].unique().tolist()))
        }

        print(f"[info] source filter before forward: source_id={args.source_id}, rows={before} -> {after}")

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
        persistent_workers=False,
    )

    model = DELSA(audio_encoder_cfg={}, text_encoder_cfg={}).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    sd = strip_module_prefix(pick_state_dict(ckpt))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 0:
        print("[warn] first missing keys:", missing[:10])
    if len(unexpected) > 0:
        print("[warn] first unexpected keys:", unexpected[:10])
    model.eval()

    A_SPA, A_SRC, T_SPA, T_SRC = [], [], [], []
    SPACE_ID, SOURCE_ID = [], []
    META = {
        "azimuth_deg": [],
        "elevation_deg": [],
        "distance": [],
        "area_m2": [],
        "fullband_T30_ms": [],
    }

    for batch in tqdm(dl, desc="Extracting embeddings", dynamic_ncols=True):
        if batch is None:
            continue

        audio = stack_audio(batch, device)
        texts = batch["texts"]

        with torch.autocast(
            device_type=("cuda" if device == "cuda" else "cpu"),
            enabled=args.eval_amp,
        ):
            out = model(audio, texts)

        A_SPA.append(F.normalize(out["audio_space_emb"], dim=-1).detach().cpu())
        A_SRC.append(F.normalize(out["audio_source_emb"], dim=-1).detach().cpu())
        T_SPA.append(F.normalize(out["text_space_emb"], dim=-1).detach().cpu())
        T_SRC.append(F.normalize(out["text_source_emb"], dim=-1).detach().cpu())

        SPACE_ID.append(batch["space_id"].reshape(-1).detach().cpu())
        SOURCE_ID.append(batch["source_id"].reshape(-1).detach().cpu())

        rm = batch["rir_meta"]
        META["azimuth_deg"].append(to_numpy_1d(rm["azimuth_deg"]))
        META["elevation_deg"].append(to_numpy_1d(rm["elevation_deg"]))
        META["distance"].append(to_numpy_1d(rm["distance"]))
        META["area_m2"].append(to_numpy_1d(rm["area_m2"]))
        META["fullband_T30_ms"].append(to_numpy_1d(rm["fullband_T30_ms"]))

    data = {
        "audio_space": torch.cat(A_SPA, dim=0),
        "audio_source": torch.cat(A_SRC, dim=0),
        "text_space": torch.cat(T_SPA, dim=0),
        "text_source": torch.cat(T_SRC, dim=0),
        "space_id": torch.cat(SPACE_ID, dim=0).long(),
        "source_id": torch.cat(SOURCE_ID, dim=0).long(),
        "meta": {k: np.concatenate(v, axis=0) for k, v in META.items()},
        "paths": {
            "ckpt": args.ckpt,
            "precomp_root": precomp_root,
            "index_csv": index_csv,
            "rir_csv": rir_csv,
        },
    }

    return data


def aggregate_per_rir(data: Dict[str, Any], source_id: int | None = None) -> Tuple[Dict[str, torch.Tensor], pd.DataFrame]:
    space_id = data["space_id"]
    source_ids = data["source_id"]

    if source_id is not None:
        mask = source_ids == int(source_id)
        if mask.sum().item() == 0:
            raise SystemExit(f"[ERR] source_id={source_id} not found in extracted data.")
    else:
        mask = torch.ones_like(source_ids, dtype=torch.bool)

    used_space = space_id[mask]
    unique_spaces = torch.unique(used_space).sort().values

    agg = {k: [] for k in ["audio_space", "audio_source", "text_space", "text_source"]}
    rows = []

    meta = data["meta"]
    for sid in unique_spaces.tolist():
        idx = mask & (space_id == sid)

        for emb_key in agg.keys():
            z = data[emb_key][idx]
            z_mean = F.normalize(z.mean(dim=0, keepdim=True), dim=-1).squeeze(0)
            agg[emb_key].append(z_mean)

        np_idx = idx.detach().cpu().numpy()
        row = {
            "space_id": int(sid),
            "n_samples": int(idx.sum().item()),
            "azimuth_deg": float(np.mean(meta["azimuth_deg"][np_idx])),
            "elevation_deg": float(np.mean(meta["elevation_deg"][np_idx])),
            "distance": float(np.mean(meta["distance"][np_idx])),
            "area_m2": float(np.mean(meta["area_m2"][np_idx])),
            "fullband_T30_ms": float(np.mean(meta["fullband_T30_ms"][np_idx])),
        }
        row["direction_cat"] = map_direction(row["azimuth_deg"])
        row["elevation_cat"] = map_elevation(row["elevation_deg"])
        row["distance_cat"] = map_distance(row["distance"])
        row["size_cat"] = map_size(row["area_m2"])
        row["reverb_cat"] = map_reverb(row["fullband_T30_ms"])

        row["direction_order"] = DIR_ORDER[row["direction_cat"]]
        row["elevation_order"] = ELEV_ORDER[row["elevation_cat"]]
        row["distance_order"] = DIST_ORDER[row["distance_cat"]]
        row["size_order"] = SIZE_ORDER[row["size_cat"]]
        row["reverb_order"] = REV_ORDER[row["reverb_cat"]]
        rows.append(row)

    agg = {k: torch.stack(v, dim=0) for k, v in agg.items()}
    meta_df = pd.DataFrame(rows)

    return agg, meta_df


def build_order(meta_df: pd.DataFrame, sort_mode: str) -> np.ndarray:
    if sort_mode == "five_factor":
        cols = ["direction_order", "elevation_order", "distance_order", "size_order", "reverb_order"]
    elif sort_mode == "direction":
        cols = ["direction_order", "elevation_order", "distance_order", "size_order", "reverb_order"]
    elif sort_mode == "reverb":
        cols = ["reverb_order", "direction_order", "elevation_order", "distance_order", "size_order"]
    elif sort_mode == "distance":
        cols = ["distance_order", "direction_order", "elevation_order", "size_order", "reverb_order"]
    elif sort_mode == "azimuth_value":
        cols = ["azimuth_deg", "elevation_deg", "distance", "area_m2", "fullband_T30_ms"]
    elif sort_mode == "t30_value":
        cols = ["fullband_T30_ms", "azimuth_deg", "distance", "area_m2", "elevation_deg"]
    else:
        raise ValueError(f"Unknown sort_mode: {sort_mode}")
    return meta_df.sort_values(cols).index.to_numpy()


def boundaries_and_ticks(meta_sorted: pd.DataFrame, group_col: str) -> Tuple[list[int], list[float], list[str]]:
    vals = meta_sorted[group_col].tolist()
    boundaries = []
    tick_positions = []
    tick_labels = []

    start = 0
    for i in range(1, len(vals) + 1):
        if i == len(vals) or vals[i] != vals[start]:
            end = i
            if end < len(vals):
                boundaries.append(end)
            tick_positions.append((start + end - 1) / 2.0)
            tick_labels.append(str(vals[start]))
            start = i

    return boundaries, tick_positions, tick_labels


def analyze_and_plot(args, data: Dict[str, Any]):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agg, meta_df = aggregate_per_rir(data, source_id=None)
    meta_df.to_csv(out_dir / "rir_meta_aggregated.csv", index=False)

    print(f"[info] aggregated RIR conditions: {len(meta_df)}")
    print(f"[info] source_id filter: {args.source_id if args.source_id is not None else 'all sources averaged'}")

    # Physical distance matrices on aggregated RIRs
    az = meta_df["azimuth_deg"].to_numpy(dtype=float)
    elev = meta_df["elevation_deg"].to_numpy(dtype=float)
    dist = meta_df["distance"].to_numpy(dtype=float)
    area = meta_df["area_m2"].to_numpy(dtype=float)
    t30 = meta_df["fullband_T30_ms"].to_numpy(dtype=float)

    D_phys = {
        "azimuth_deg_circular": circular_angle_distance_deg(az),
        "elevation_deg_abs": absolute_distance_matrix(elev),
        "distance_abs": absolute_distance_matrix(dist),
        "area_m2_abs": absolute_distance_matrix(area),
        "t30_ms_abs": absolute_distance_matrix(t30),
    }

    metrics = {
        "n_rir_conditions": int(len(meta_df)),
        "source_id_filter": None if args.source_id is None else int(args.source_id),
        "paths": data["paths"],
    }

    emb_keys = ["audio_space", "audio_source", "text_space", "text_source"]
    sort_modes = ["five_factor", "direction", "reverb", "distance", "azimuth_value", "t30_value"]

    for emb_key in emb_keys:
        Z = agg[emb_key]
        D_emb = cosine_distance_matrix(Z)
        metrics[emb_key] = {}

        upper_emb = upper_tri_values(D_emb)
        for phys_name, Dp in D_phys.items():
            rho = spearmanr_np(upper_tri_values(Dp), upper_emb)
            metrics[emb_key][f"spearman_vs_{phys_name}"] = rho

        # save raw distance matrix
        np.save(out_dir / f"{emb_key}_distance_matrix.npy", D_emb)

        for sort_mode in sort_modes:
            order = build_order(meta_df, sort_mode)
            D_sorted = D_emb[np.ix_(order, order)]
            meta_sorted = meta_df.iloc[order].reset_index(drop=True)

            if sort_mode in ("five_factor", "direction", "azimuth_value"):
                group_col = "direction_cat"
            elif sort_mode in ("reverb", "t30_value"):
                group_col = "reverb_cat"
            elif sort_mode == "distance":
                group_col = "distance_cat"
            else:
                group_col = "direction_cat"

            boundaries, tick_pos, tick_labels = boundaries_and_ticks(meta_sorted, group_col)

            title = f"{emb_key}: pairwise distance sorted by {sort_mode}"
            out_path = out_dir / f"heatmap_{emb_key}_sort-{sort_mode}.png"
            plot_heatmap(
                D_sorted,
                title=title,
                out_path=out_path,
                block_boundaries=boundaries,
                tick_positions=tick_pos,
                tick_labels=tick_labels,
            )

        # scatter / binned plots for report
        for phys_name, Dp in D_phys.items():
            x = upper_tri_values(Dp)
            y = upper_emb
            rho = metrics[emb_key][f"spearman_vs_{phys_name}"]
            plot_scatter_binned(
                x=x,
                y=y,
                xlabel=phys_name,
                ylabel="Embedding distance: 1 - cosine",
                title=f"{emb_key}: rho={rho:.3f}",
                out_path=out_dir / f"scatter_{emb_key}_vs_{phys_name}.png",
                bins=20,
            )

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # readable summary
    rows = []
    for emb_key in emb_keys:
        row = {"embedding": emb_key}
        row.update(metrics[emb_key])
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "metrics_summary.csv", index=False)

    print("\n=== Spearman summary ===")
    print(summary_df.to_string(index=False))
    print(f"\n[✓] wrote outputs to: {out_dir.resolve()}")
    print("\nRecommended files for MTG:")
    print(f"  - {out_dir / 'heatmap_audio_space_sort-five_factor.png'}")
    print(f"  - {out_dir / 'heatmap_audio_source_sort-five_factor.png'}")
    print(f"  - {out_dir / 'scatter_audio_space_vs_azimuth_deg_circular.png'}")
    print(f"  - {out_dir / 'metrics_summary.csv'}")


def build_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--ckpt", required=True, help="Path to DISSE checkpoint .pt")
    p.add_argument("--config", default="config.yaml", help="config.yaml path. Used only to infer paths.")

    p.add_argument("--split", default="test", choices=["val", "test"], help="Used to infer config keys and filenames.")
    p.add_argument("--precomp_root", default=None, help="Root directory of precomputed features.")
    p.add_argument("--index_csv", default=None, help="precomputed csv, e.g. test_precomputed.csv")
    p.add_argument("--rir_csv", default=None, help="RIR catalog csv, e.g. RIR_dataset/rir_catalog_test.csv")

    p.add_argument("--out_dir", default="results/spatial_embedding_heatmap")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--eval_amp", action="store_true")

    p.add_argument(
        "--source_id",
        type=int,
        default=None,
        help=(
            "If specified, use only one source. "
            "If omitted, average embeddings over all sources for each RIR."
        ),
    )

    return p


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    args = build_parser().parse_args()

    if not Path(args.ckpt).exists():
        raise SystemExit(f"[ERR] checkpoint not found: {args.ckpt}")

    data = extract_embeddings(args)
    analyze_and_plot(args, data)


if __name__ == "__main__":
    main()