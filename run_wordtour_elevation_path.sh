#!/usr/bin/env bash
set -euo pipefail

DELSA_DIR="$HOME/DELSA"
WORDTOUR_DIR="$HOME/external/wordtour"
LKH_DIR="$WORDTOUR_DIR/LKH-3.0.6"

DRY_PT="$DELSA_DIR/results/no_reverb_continuous_elevation_ep20_src107280_dry/continuous_elevation_embeddings.pt"
REVERB_PT="$DELSA_DIR/results/reverb_continuous_elevation_ep20_src107280/continuous_elevation_embeddings.pt"

if [ ! -x "$LKH_DIR/LKH" ]; then
  echo "[error] LKH executable not found: $LKH_DIR/LKH"
  echo "Build LKH first:"
  echo "  cd $LKH_DIR/SRC"
  echo "  make clean"
  echo "  make CFLAGS=\"-O3 -Wall -IINCLUDE -DTWO_LEVEL_TREE -g -fcommon\""
  exit 1
fi

run_one () {
  local pt_path="$1"
  local key="$2"
  local tag="$3"

  echo
  echo "============================================================"
  echo "[run] elevation path: tag=$tag key=$key"
  echo "============================================================"

  local base="${tag}_${key}_elevation_path"
  local tsp="$LKH_DIR/${base}.tsp"
  local labels="$WORDTOUR_DIR/${base}.labels.tsv"
  local par="$LKH_DIR/${base}.par"
  local out="$LKH_DIR/${base}.out"
  local result_dir="$DELSA_DIR/results/wordtour_${tag}_${key}_elevation_path"

  cd "$DELSA_DIR"

  echo "[1/4] Create dummy-node LKH TSP"
  python make_lkh_path_tsp_from_elevation_embeddings.py \
    --emb_pt "$pt_path" \
    --key "$key" \
    --out_tsp "$tsp" \
    --out_labels "$labels" \
    --name "$base"

  echo "[2/4] Write LKH par file"
  cd "$LKH_DIR"
  cat > "$par" <<PAR
PROBLEM_FILE = ${base}.tsp
OUTPUT_TOUR_FILE = ${base}.out
RUNS = 10
PAR

  echo "[3/4] Run LKH"
  ./LKH "$par"

  echo "[4/4] Evaluate dummy path"
  cd "$DELSA_DIR"
  python evaluate_lkh_dummy_path_output.py \
    --labels "$labels" \
    --tour_out "$out" \
    --out_dir "$result_dir" \
    --name "$base"

  echo "[done] $result_dir"
}

run_one "$DRY_PT"    "audio_space"  "dry"
run_one "$DRY_PT"    "audio_source" "dry"
run_one "$REVERB_PT" "audio_space"  "reverb"
run_one "$REVERB_PT" "audio_source" "reverb"

echo
echo "All elevation path analyses done."
