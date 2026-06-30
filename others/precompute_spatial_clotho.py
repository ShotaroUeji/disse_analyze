#!/usr/bin/env python3
import argparse
import math
import random
import sys
from pathlib import Path

# /home/takamichi-lab-pc09/DELSA を import path に追加
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm.auto import tqdm

from dataset.audio_rir_dataset import foa_to_iv

FOA_SR = 48_000
IV_SR = 16_000
MAX_DURATION_SEC = 10.0


def _clean(s: str) -> str:
    return str(s).strip()


def _safe_stem(file_name: str) -> str:
    stem = Path(file_name).stem
    for ch in ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]:
        stem = stem.replace(ch, "_")
    return stem.replace(" ", "_")


def load_dry(path: Path, target_sr: int = FOA_SR, max_duration_sec: float = MAX_DURATION_SEC):
    wav, sr = torchaudio.load(str(path))

    # Clotho は基本 mono だが、安全のため stereo は mono 化
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    target_len = int(target_sr * max_duration_sec)

    if wav.shape[-1] < target_len:
        rpt = math.ceil(target_len / wav.shape[-1])
        wav = wav.repeat(1, rpt)[:, :target_len]
    else:
        wav = wav[:, :target_len]

    return wav


def apply_rir_to_foa(dry: torch.Tensor, rir_path: str):
    """
    dry: [1, T]
    rir: [4, Tr] A-format RIR
    return: [4, T] FOA [W, Y, Z, X]
    """
    rir, sr = torchaudio.load(str(rir_path))

    if sr != FOA_SR:
        rir = torchaudio.functional.resample(rir, sr, FOA_SR)

    if rir.shape[0] != 4:
        raise ValueError(f"RIR must have 4 channels, got {rir.shape[0]}: {rir_path}")

    wet = torchaudio.functional.fftconvolve(dry, rir)
    wet = wet[..., : dry.shape[-1]]

    m0, m1, m2, m3 = wet[0], wet[1], wet[2], wet[3]

    W = (m0 + m1 + m2 + m3) / 2
    X = (m0 + m1 - m2 - m3) / 2
    Y = (m0 - m1 + m2 - m3) / 2
    Z = (m0 - m1 - m2 + m3) / 2

    return torch.stack([W, Y, Z, X], dim=0)


# ----------------------------
# Random caption template
# ----------------------------

def choice(xs):
    return random.choice(xs)


def map_distance_phrase(d):
    d = float(d)
    if d <= 2.0:
        return choice(["near", "nearby", "close"])
    return choice(["far", "distant"])


def map_elevation_phrase(el):
    el = float(el)
    if el > 40:
        return choice(["up", "above"])
    if el < -40:
        return choice(["down", "below"])
    return ""


def map_direction_phrase(az):
    az = float(az)

    if -35 <= az <= 35:
        return choice(["front", "in front"])
    if 55 <= az <= 125:
        return choice(["right", "to the right"])
    if -125 <= az <= -55:
        return choice(["left", "to the left"])
    if az >= 145 or az <= -145:
        return choice(["back", "behind"])

    return ""


def map_size_phrase(area):
    area = float(area)

    if area < 50:
        return choice(["small", "tiny"])
    if area > 100:
        return choice(["large", "spacious"])

    return choice(["medium", "mid-sized"])


def map_reverb_phrase(t30):
    t30 = float(t30)

    if t30 < 200:
        return choice(["acoustically dampened", "dry-sounding"])
    if t30 > 1000:
        return choice(["highly reverberant", "echoey"])

    return ""


def build_location(meta):
    parts = [
        map_distance_phrase(meta["source_distance_m"]),
        map_elevation_phrase(meta["elevation_deg"]),
        map_direction_phrase(meta["azimuth_deg"]),
    ]
    return " ".join(p for p in parts if p).strip()


def build_room(meta):
    size = map_size_phrase(meta["area_m2"])
    reverb = map_reverb_phrase(meta["fullband_T30_ms"])

    if reverb:
        return f"{size} {reverb} room"
    return f"{size} room"


def rewrite_caption_random_template(orig: str, meta: dict):
    orig = _clean(orig)
    loc = build_location(meta)
    room = build_room(meta)

    templates = [
        "The sound: {orig} is coming from the {loc} of a {room}.",
        "The sound: {orig} originates in the {loc} of a {room}.",
        "You can hear {orig} from the {loc}, inside a {room}.",
    ]

    return choice(templates).format(orig=orig, loc=loc, room=room).replace("  ", " ")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--clotho_csv",
        type=str,
        required=True,
        help="e.g. /home/takamichi09/datasets/clotho/clotho_captions_development.csv",
    )
    parser.add_argument(
        "--clotho_audio_dir",
        type=str,
        required=True,
        help="e.g. /home/takamichi09/datasets/clotho/development",
    )
    parser.add_argument(
        "--rir_csv",
        type=str,
        required=True,
        help="RIR catalog CSV containing rir_path and meta columns",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output root, e.g. /home/takamichi09/datasets/spatial_clotho/development",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        required=True,
        help="Output CSV path",
    )
    parser.add_argument(
        "--n_views",
        type=int,
        default=1,
        help="Number of RIRs per Clotho audio. <=0 means all RIRs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--duration_sec",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--caption_mode",
        choices=["five_rows", "random_one"],
        default="five_rows",
        help="five_rows: caption_1..5 all become rows. random_one: choose one caption per audio/RIR.",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    clotho_csv = Path(args.clotho_csv)
    clotho_audio_dir = Path(args.clotho_audio_dir)
    out_root = Path(args.out_dir)
    out_csv = Path(args.out_csv)

    foa_dir = out_root / "foa"
    feat_dir = out_root / "feat"
    foa_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    clotho_df = pd.read_csv(clotho_csv)
    rir_df = pd.read_csv(args.rir_csv)

    required_clotho_cols = ["file_name", "caption_1", "caption_2", "caption_3", "caption_4", "caption_5"]
    for col in required_clotho_cols:
        if col not in clotho_df.columns:
            raise ValueError(f"Missing Clotho column: {col}")

    required_rir_cols = [
        "rir_path",
        "area_m2",
        "source_distance_m",
        "fullband_T30_ms",
        "azimuth_deg",
        "elevation_deg",
    ]
    for col in required_rir_cols:
        if col not in rir_df.columns:
            raise ValueError(f"Missing RIR column: {col}")

    rir_df = rir_df.sort_values("rir_path").reset_index(drop=True)

    if args.n_views <= 0:
        selected_rir_df = rir_df
    else:
        selected_rir_df = rir_df.iloc[: args.n_views]

    records = []
    bad_records = []

    total = len(clotho_df) * len(selected_rir_df)
    pbar = tqdm(total=total, desc="Precomputing SpatialClotho", unit="view")

    torch.set_grad_enabled(False)

    with torch.no_grad():
        for audio_idx, row in clotho_df.iterrows():
            file_name = row["file_name"]
            audio_path = clotho_audio_dir / file_name

            if not audio_path.exists():
                # CSV 側が &amp; で、実ファイルが & のケースを救う
                alt_name = file_name.replace("&amp;", "&")
                alt_path = clotho_audio_dir / alt_name
                if alt_path.exists():
                    audio_path = alt_path
                    file_name = alt_name
                else:
                    bad_records.append((file_name, "missing_audio"))
                    pbar.update(len(selected_rir_df))
                    continue

            try:
                dry = load_dry(
                    audio_path,
                    target_sr=FOA_SR,
                    max_duration_sec=args.duration_sec,
                )
            except Exception as e:
                bad_records.append((file_name, f"load_error: {type(e).__name__}: {e}"))
                pbar.update(len(selected_rir_df))
                continue

            audio_stem = _safe_stem(file_name)

            caption_cols = ["caption_1", "caption_2", "caption_3", "caption_4", "caption_5"]
            captions = [_clean(row[c]) for c in caption_cols if isinstance(row[c], str) and _clean(row[c])]

            if len(captions) == 0:
                bad_records.append((file_name, "no_caption"))
                pbar.update(len(selected_rir_df))
                continue

            for rir_i, rir_row in selected_rir_df.iterrows():
                meta = rir_row.to_dict()
                rir_path = meta["rir_path"]

                try:
                    foa = apply_rir_to_foa(dry, rir_path)
                except Exception as e:
                    bad_records.append((file_name, f"rir_error: {type(e).__name__}: {e}"))
                    pbar.update(1)
                    continue

                rir_stem = Path(rir_path).stem
                key = f"{audio_idx:06d}_{audio_stem}_{rir_stem}"

                foa_rel = f"foa/{key}.wav"
                feat_rel = f"feat/{key}.pt"

                torchaudio.save(
                    str(out_root / foa_rel),
                    foa.cpu(),
                    FOA_SR,
                    format="wav",
                    encoding="PCM_S",
                    bits_per_sample=16,
                )

                foa16 = torchaudio.functional.resample(foa, FOA_SR, IV_SR)
                i_act, i_rea = foa_to_iv(foa16.unsqueeze(0))
                i_act = i_act.squeeze(0).cpu()
                i_rea = i_rea.squeeze(0).cpu()

                torch.save(
                    {"i_act": i_act, "i_rea": i_rea},
                    out_root / feat_rel,
                )

                if args.caption_mode == "random_one":
                    selected_captions = [choice(captions)]
                    selected_caption_ids = [captions.index(selected_captions[0]) + 1]
                else:
                    selected_captions = captions
                    selected_caption_ids = list(range(1, len(captions) + 1))

                for caption_id, cap in zip(selected_caption_ids, selected_captions):
                    spatial_caption = rewrite_caption_random_template(cap, meta)

                    records.append(
                        {
                            "clotho_id": audio_idx,
                            "file_name": file_name,
                            "caption_id": caption_id,
                            "rir_path": rir_path,
                            "foa_path": foa_rel,
                            "feat_path": feat_rel,
                            "caption": spatial_caption,
                            "original_caption": cap,
                            "area_m2": meta["area_m2"],
                            "source_distance_m": meta["source_distance_m"],
                            "fullband_T30_ms": meta["fullband_T30_ms"],
                            "azimuth_deg": meta["azimuth_deg"],
                            "elevation_deg": meta["elevation_deg"],
                        }
                    )

                pbar.set_postfix({"audio": file_name[:20], "rir": rir_stem})
                pbar.update(1)

    pbar.close()

    out_df = pd.DataFrame(records)
    out_df.to_csv(out_csv, index=False)

    if bad_records:
        bad_path = out_csv.with_suffix(".bad.txt")
        with bad_path.open("w", encoding="utf-8") as f:
            for name, reason in bad_records:
                f.write(f"{name}\t{reason}\n")
        print(f"[WARN] bad records written to {bad_path}")

    print(f"[OK] Wrote {len(out_df)} rows")
    print(f"[OK] FOA/feat root: {out_root}")
    print(f"[OK] CSV: {out_csv}")


if __name__ == "__main__":
    main()