cd ~/DELSA

cat > run_wordtour_azimuth.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

# ===== Paths =====
DELSA_DIR="$HOME/DELSA"
WORDTOUR_DIR="$HOME/external/wordtour"
LKH_DIR="$WORDTOUR_DIR/LKH-3.0.6"

# ===== Inputs =====
DRY_PT="$DELSA_DIR/results/no_reverb_continuous_azimuth_ep20_src107280_dry/continuous_azimuth_embeddings.pt"
REVERB_PT="$DELSA_DIR/results/reverb_continuous_azimuth_ep20_src107280/continuous_azimuth_embeddings.pt"

# ===== Check LKH =====
if [ ! -x "$LKH_DIR/LKH" ]; then
  echo "[error] LKH executable not found: $LKH_DIR/LKH"
  echo "Build it first, for example:"
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
  echo "[run] tag=$tag key=$key"
  echo "============================================================"

  local txt="$WORDTOUR_DIR/${tag}_${key}.txt"
  local labels="$WORDTOUR_DIR/${tag}_${key}.labels.txt"
  local tsp="$LKH_DIR/${tag}_${key}.tsp"
  local par="$LKH_DIR/${tag}_${key}.par"
  local out="$LKH_DIR/${tag}_${key}.out"
  local result_dir="$DELSA_DIR/results/wordtour_${tag}_${key}"

  cd "$DELSA_DIR"

  echo "[1/4] Export embeddings to WordTour format"
  python export_disse_emb_to_wordtour.py \
    --emb_pt "$pt_path" \
    --key "$key" \
    --out_txt "$txt" \
    --normalize

  echo "[2/4] Create LKH TSP file"
  cd "$WORDTOUR_DIR"
  ./make_LKH_file "$txt" 72 > "$tsp"

  echo "[3/4] Run LKH"
  cd "$LKH_DIR"
  cat > "$par" <<PAR
PROBLEM_FILE = ${tag}_${key}.tsp
OUTPUT_TOUR_FILE = ${tag}_${key}.out
RUNS = 10
PAR

  ./LKH "$par"

  echo "[4/4] Evaluate tour"
  cd "$DELSA_DIR"
  python evaluate_wordtour_lkh_output.py \
    --labels "$labels" \
    --tour_out "$out" \
    --out_dir "$result_dir" \
    --name "${tag}_${key}"

  echo "[done] $result_dir"
}

# ===== Run all =====
run_one "$DRY_PT"    "audio_space"  "dry"
run_one "$DRY_PT"    "audio_source" "dry"
run_one "$REVERB_PT" "audio_space"  "reverb"
run_one "$REVERB_PT" "audio_source" "reverb"

echo
echo "All done."
echo "Results:"
echo "  $DELSA_DIR/results/wordtour_dry_audio_space"
echo "  $DELSA_DIR/results/wordtour_dry_audio_source"
echo "  $DELSA_DIR/results/wordtour_reverb_audio_space"
echo "  $DELSA_DIR/results/wordtour_reverb_audio_source"
EOF

chmod +x run_wordtour_azimuth.sh