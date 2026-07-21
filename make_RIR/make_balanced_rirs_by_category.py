#!/usr/bin/env python3
# make_balanced_rirs_by_category.py
"""
RIRをカテゴリごとに生成するスクリプト。

デフォルト:
  空欄カテゴリ '' を含めない。
  つまり以下の 96 通りを各 PER_COMBO 個ずつ作る。

    direction: front / right / left / back
    elevation: up / down
    distance : near / far
    size     : small / mid-sized / large
    reverb   : acoustically dampened / highly reverberant

  4 * 2 * 2 * 3 * 2 = 96

以前のように空欄カテゴリも含めたい場合:
  --include-blank を付ける。

例:
  python make_RIR/make_balanced_rirs_by_category.py --split test

  python make_RIR/make_balanced_rirs_by_category.py --split test --include-blank

安全に完全作り直し:
  python make_RIR/make_balanced_rirs_by_category.py --split test --clean
"""

from pathlib import Path
import argparse
import csv
import math
import random
import shutil
from collections import Counter, defaultdict
from itertools import product

import numpy as np
import tqdm
import yaml

from gen_one_rir import gen_one_rir, _rand_room, _rand_position, POOL


# ─────────────────────────────
# 1. caption用カテゴリマッピング
# ─────────────────────────────
def map_direction(az: float) -> str:
    if -35 <= az <= 35:
        return "front"
    if 55 <= az <= 125:
        return "right"
    if -125 <= az <= -55:
        return "left"
    if az >= 145 or az <= -145:
        return "back"
    return ""


def map_elevation(el: float) -> str:
    if el > 40:
        return "up"
    if el < -40:
        return "down"
    return ""


def map_distance(d: float) -> str:
    if d < 1:
        return "near"
    if d > 2:
        return "far"
    return ""


def map_size(a: float) -> str:
    if a < 50:
        return "small"
    if a > 100:
        return "large"
    return "mid-sized"


def map_reverb(t30: float) -> str:
    if t30 < 200:
        return "acoustically dampened"
    if t30 > 1000:
        return "highly reverberant"
    return ""


def make_key(meta_or_values) -> tuple[str, str, str, str, str]:
    """
    dict または値tupleからカテゴリキーを作る。
    key = (direction, elevation, distance, size, reverb)
    """
    if isinstance(meta_or_values, dict):
        az = float(meta_or_values["azimuth_deg"])
        el = float(meta_or_values["elevation_deg"])
        dist = float(meta_or_values["source_distance_m"])
        area = float(meta_or_values["area_m2"])
        t30 = float(meta_or_values["fullband_T30_ms"])
    else:
        az, el, dist, area, t30 = meta_or_values

    return (
        map_direction(az),
        map_elevation(el),
        map_distance(dist),
        map_size(area),
        map_reverb(t30),
    )


# ─────────────────────────────
# 2. 設定読み込み
# ─────────────────────────────
def load_spatial_ranges() -> dict:
    """
    実行場所がリポジトリrootでも make_RIR/ 内でも動くように探す。
    """
    candidates = [
        Path("make_RIR/spatial_ranges.yml"),
        Path("spatial_ranges.yml"),
        Path(__file__).resolve().with_name("spatial_ranges.yml"),
    ]

    for path in candidates:
        if path.exists():
            return yaml.safe_load(path.read_text())

    raise FileNotFoundError(
        "spatial_ranges.yml not found. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def range_intersects(lo: float, hi: float, a: float, b: float) -> bool:
    """
    [lo, hi] と [a, b] が交差するか。
    """
    return max(lo, a) <= min(hi, b)


# ─────────────────────────────
# 3. ターゲットカテゴリ構築
# ─────────────────────────────
def build_nonblank_sets() -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    """
    空欄なし: valと同じ96カテゴリ。
    """
    direction = ["front", "right", "left", "back"]
    elevation = ["up", "down"]
    distance = ["near", "far"]
    size = ["small", "mid-sized", "large"]
    reverb = ["acoustically dampened", "highly reverberant"]
    return direction, elevation, distance, size, reverb


def build_blank_allowed_sets(split: str) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    """
    空欄あり: 以前の版に近い挙動。
    ただし、実際に出現し得るラベルだけを入れる。
    """
    rooms = POOL[split]
    cfg = load_spatial_ranges()
    rng = cfg["TEST"] if split == "test" else cfg["TRAINVAL"]

    el_min = float(rng["EL_MIN"])
    el_max = float(rng["EL_MAX"])
    dist_min = float(rng["DIST_MIN"])
    dist_max = float(rng["DIST_MAX"])

    direction = ["front", "right", "left", "back", ""]

    elevation = []
    if el_max > 40:
        elevation.append("up")
    if el_min < -40:
        elevation.append("down")
    if range_intersects(el_min, el_max, -40, 40):
        elevation.append("")

    distance = []
    if dist_min < 1:
        distance.append("near")
    if dist_max > 2:
        distance.append("far")
    if range_intersects(dist_min, dist_max, 1, 2):
        distance.append("")

    size = sorted({map_size(float(rc["area_m2"])) for rc in rooms})
    size = [x for x in size if x != ""]

    reverb = sorted({map_reverb(float(rc["T30_ms"])) for rc in rooms})

    return direction, elevation, distance, size, reverb


def build_targets(per_combo: int, split: str, include_blank: bool) -> dict[tuple, int]:
    if include_blank:
        DIR, ELEV, DIST, SIZE, REV = build_blank_allowed_sets(split)
    else:
        DIR, ELEV, DIST, SIZE, REV = build_nonblank_sets()

    targets = {}
    for key in product(DIR, ELEV, DIST, SIZE, REV):
        targets[key] = per_combo

    return targets


# ─────────────────────────────
# 4. 生成可能性チェック
# ─────────────────────────────
def check_feasibility(split: str, include_blank: bool):
    """
    空欄なし生成時に、spatial_ranges.yml や room_pool のせいで
    そもそも生成不可能なカテゴリがないか確認する。
    """
    if include_blank:
        return

    rooms = POOL[split]
    cfg = load_spatial_ranges()
    rng = cfg["TEST"] if split == "test" else cfg["TRAINVAL"]

    el_min = float(rng["EL_MIN"])
    el_max = float(rng["EL_MAX"])
    dist_min = float(rng["DIST_MIN"])
    dist_max = float(rng["DIST_MAX"])

    possible_elev = set()
    if el_max > 40:
        possible_elev.add("up")
    if el_min < -40:
        possible_elev.add("down")

    possible_dist = set()
    if dist_min < 1:
        possible_dist.add("near")
    if dist_max > 2:
        possible_dist.add("far")

    possible_size = {map_size(float(rc["area_m2"])) for rc in rooms}
    possible_reverb = {map_reverb(float(rc["T30_ms"])) for rc in rooms}

    required_elev = {"up", "down"}
    required_dist = {"near", "far"}
    required_size = {"small", "mid-sized", "large"}
    required_reverb = {"acoustically dampened", "highly reverberant"}

    errors = []

    if not required_elev <= possible_elev:
        errors.append(
            f"elevation impossible: required={sorted(required_elev)}, "
            f"possible={sorted(possible_elev)}. "
            f"Check {split} EL_MIN/EL_MAX in spatial_ranges.yml."
        )

    if not required_dist <= possible_dist:
        errors.append(
            f"distance impossible: required={sorted(required_dist)}, "
            f"possible={sorted(possible_dist)}. "
            f"Check {split} DIST_MIN/DIST_MAX in spatial_ranges.yml."
        )

    if not required_size <= possible_size:
        errors.append(
            f"room size impossible: required={sorted(required_size)}, "
            f"possible={sorted(possible_size)}. "
            f"Check room_pool for split={split}."
        )

    if not required_reverb <= possible_reverb:
        errors.append(
            f"reverb impossible: required={sorted(required_reverb)}, "
            f"possible={sorted(possible_reverb)}. "
            f"Check room_pool T30_ms for split={split}."
        )

    if errors:
        msg = "\n".join(f"  - {e}" for e in errors)
        raise SystemExit(
            "[error] Non-blank 96-category generation is not feasible with current settings.\n"
            + msg
            + "\n\n"
            + "For test, you probably need something like:\n"
            + "  TEST:\n"
            + "    EL_MIN: -47.5\n"
            + "    EL_MAX:  48.7\n"
            + "    DIST_MIN: 0.5\n"
            + "    DIST_MAX: 4.0\n"
        )


# ─────────────────────────────
# 5. 既存メタ読み込み
# ─────────────────────────────
def read_existing_by_key(meta_dir: Path, targets: dict[tuple, int]):
    """
    既存YAMLをカテゴリ別に読む。
    targetsに含まれないkeyは無視する。
    """
    by_key = defaultdict(list)
    counts_all = Counter()

    for yml_path in sorted(meta_dir.glob("*.yml")):
        meta = yaml.safe_load(yml_path.read_text())
        key = make_key(meta)
        counts_all[key] += 1

        if key in targets and len(by_key[key]) < targets[key]:
            by_key[key].append(yml_path)

    return by_key, counts_all


def existing_counts_for_targets(by_key: dict, targets: dict[tuple, int]) -> Counter:
    cnt = Counter()
    for key in targets:
        cnt[key] = len(by_key.get(key, []))
    return cnt


# ─────────────────────────────
# 6. CSV作成
# ─────────────────────────────
def center_dist(dims, src) -> float:
    cx, cy, cz = [d / 2 for d in dims]
    return round(math.dist((cx, cy, cz), src), 3)


def build_selected_catalog(
    base_dir: Path,
    split: str,
    selected_meta_paths: list[Path],
    out_csv: Path | None = None,
):
    """
    selected_meta_paths だけをCSV化する。
    既存ディレクトリに余計な空欄RIRが残っていても、CSVには入れない。
    """
    data_dir = base_dir / "RIR_data" / split
    rows = []

    for yml_path in sorted(selected_meta_paths, key=lambda p: p.stem):
        rid = yml_path.stem
        wav_path = data_dir / f"{rid}.wav"
        if not wav_path.exists():
            print(f"[warn] wav not found, skipped: {wav_path}")
            continue

        meta = yaml.safe_load(yml_path.read_text())

        rows.append(
            {
                "rir_path": str(wav_path),
                "dims": meta["dims"],
                "area_m2": meta["area_m2"],
                "alpha": meta["alpha"],
                "fullband_T30_ms": meta["fullband_T30_ms"],
                "source_distance_m": meta["source_distance_m"],
                "azimuth_deg": meta["azimuth_deg"],
                "elevation_deg": meta["elevation_deg"],
                "source_pos_xyz": meta["source_pos_xyz"],
                "distance": center_dist(meta["dims"], meta["source_pos_xyz"]),
                "fs": meta["fs"],
                "room_id": meta["room_id"],
                "split": split,
            }
        )

    if out_csv is None:
        out_csv = base_dir / f"rir_catalog_{split}.csv"

    if not rows:
        print(f"[empty] no selected rows for split={split}")
        return

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_csv} ({len(rows)} rows)")


# ─────────────────────────────
# 7. ID生成
# ─────────────────────────────
def make_next_id_func(data_dir: Path, meta_dir: Path):
    """
    既存ファイルと被らない auto_XXXXXX を返す。
    """
    serial = 0

    def next_id():
        nonlocal serial
        while True:
            rid = f"auto_{serial:06d}"
            serial += 1
            wav_path = data_dir / f"{rid}.wav"
            yml_path = meta_dir / f"{rid}.yml"
            if not wav_path.exists() and not yml_path.exists():
                return rid

    return next_id


# ─────────────────────────────
# 8. メイン
# ─────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("RIR_dataset"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--per-combo", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=10_000_000)

    parser.add_argument(
        "--include-blank",
        action="store_true",
        help="空欄カテゴリ '' も有効カテゴリとして含める。指定しない場合は空欄なし96カテゴリ。",
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="既存の RIR_data/<split> と RIR_meta/<split> を削除してから作り直す。",
    )

    parser.add_argument(
        "--catalog-all",
        action="store_true",
        help="選ばれたtargetだけでなく、meta_dir内の全RIRをCSV化する。通常は指定しない。",
    )

    args = parser.parse_args()

    BASE = args.base
    SPLIT = args.split
    SEED = args.seed
    PER_COMBO = args.per_combo
    MAX_ATTEMPTS = args.max_attempts

    random.seed(SEED)

    data_dir = BASE / "RIR_data" / SPLIT
    meta_dir = BASE / "RIR_meta" / SPLIT

    if args.clean:
        if data_dir.exists():
            shutil.rmtree(data_dir)
        if meta_dir.exists():
            shutil.rmtree(meta_dir)

    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    check_feasibility(SPLIT, include_blank=args.include_blank)

    targets = build_targets(
        per_combo=PER_COMBO,
        split=SPLIT,
        include_blank=args.include_blank,
    )

    selected_by_key, counts_all = read_existing_by_key(meta_dir, targets)
    have = existing_counts_for_targets(selected_by_key, targets)

    remaining = {k: max(0, targets[k] - have.get(k, 0)) for k in targets}
    total_rem = sum(remaining.values())

    mode = "include_blank" if args.include_blank else "nonblank_96"
    print(f"[info] mode={mode}")
    print(f"[info] split={SPLIT}")
    print(f"[info] combos={len(targets)}")
    print(f"[info] per_combo={PER_COMBO}")
    print(f"[info] already_selected={sum(have.values())}")
    print(f"[info] remaining={total_rem}")

    if not args.include_blank:
        blank_existing = sum(
            c for k, c in counts_all.items()
            if "" in k
        )
        if blank_existing > 0:
            print(
                f"[warn] Existing blank-category RIRs found in meta_dir: {blank_existing}. "
                "They will not be included in the selected catalog. "
                "Use --clean if you want to remove them physically."
            )

    next_id = make_next_id_func(data_dir, meta_dir)

    pbar = tqdm.tqdm(total=total_rem, desc="Generating RIRs")
    attempts = 0

    while total_rem > 0 and attempts < MAX_ATTEMPTS:
        attempts += 1

        # 軽量プレビュー: 部屋と位置だけ先にサンプルする。
        rng_state = random.getstate()

        room_cfg = _rand_room(SPLIT)
        w, h, H = room_cfg["dims"]
        ctr = np.array([w / 2, h / 2, H / 2])

        _, dist, az_deg, el_deg = _rand_position(
            SPLIT,
            ctr,
            dims=(w, h, H),
        )

        key = (
            map_direction(az_deg),
            map_elevation(el_deg),
            map_distance(dist),
            map_size(room_cfg["area_m2"]),
            map_reverb(room_cfg["T30_ms"]),
        )

        # デフォルトでは空欄カテゴリは完全にreject
        if not args.include_blank and "" in key:
            continue

        if key not in remaining or remaining[key] <= 0:
            continue

        # 受理: 乱数状態を戻して、gen_one_rir内で同じ部屋・位置を再現する。
        random.setstate(rng_state)

        rid = next_id()
        meta_path = gen_one_rir(
            base_dir=BASE,
            id=rid,
            split=SPLIT,
        )

        meta_path = Path(meta_path)
        selected_by_key[key].append(meta_path)

        remaining[key] -= 1
        total_rem -= 1
        pbar.update(1)

    pbar.close()

    if total_rem > 0:
        lacking = {k: v for k, v in remaining.items() if v > 0}
        print("[!] Could not fill all quotas. Lacking combos:")
        for k, v in list(lacking.items())[:30]:
            print(f"    {k}: {v}")
        if len(lacking) > 30:
            print(f"    ... and {len(lacking) - 30} more")

        raise SystemExit(
            f"[error] generation stopped with remaining={total_rem}. "
            "Increase --max-attempts or check spatial_ranges.yml / room_pool."
        )

    selected_meta_paths = []
    for key in sorted(targets.keys()):
        selected_meta_paths.extend(selected_by_key[key][:targets[key]])

    print("Rebuilding CSV...")

    if args.catalog_all:
        # 旧create_csv_for_rir.pyと同じ挙動に近い。
        # 注意: meta_dir内の空欄RIRも全部入る。
        from create_csv_for_rir import build_catalog
        build_catalog(BASE, SPLIT)
    else:
        # デフォルト: targetに選ばれたRIRだけCSV化する。
        build_selected_catalog(BASE, SPLIT, selected_meta_paths)

    print("[✓] Done.")


if __name__ == "__main__":
    main()