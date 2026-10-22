#!/usr/bin/env bash
set -euo pipefail

# Run the four ELSA baseline conditions used in the DISSE comparison.
# Place this file and evaluate_elsa_continuity.py in the disse_analyze root.
#
# Override any path without editing the script, for example:
#   ELSA_ROOT="$HOME/ELSA" CKPT="/path/to/elsa_epoch_30.pt" bash run_elsa_continuity_all.sh

DISSE_ROOT="${DISSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ELSA_ROOT="${ELSA_ROOT:-$HOME/elsa}"
CKPT="${CKPT:-$ELSA_ROOT/Spatial_AudioCaps/takamichi09/elsa_ckck_0711/elsa_epoch_30.pt}"
AUDIO_PATH="${AUDIO_PATH:-$DISSE_ROOT/AudioCaps_mp3/test/107280.mp3}"
LKH="${LKH:-$HOME/external/wordtour/LKH-3.0.6/LKH}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BATCH_SIZE="${BATCH_SIZE:-4}"

if [[ ! -d "$ELSA_ROOT" ]]; then
  echo "[error] ELSA_ROOT not found: $ELSA_ROOT" >&2
  echo "Set it explicitly, e.g. ELSA_ROOT=\"$HOME/ELSA\"" >&2
  exit 1
fi
if [[ ! -f "$CKPT" ]]; then
  echo "[error] CKPT not found: $CKPT" >&2
  exit 1
fi
if [[ ! -f "$AUDIO_PATH" ]]; then
  echo "[error] AUDIO_PATH not found: $AUDIO_PATH" >&2
  exit 1
fi

run_one() {
  local axis="$1"
  local condition="$2"
  local out_dir="$DISSE_ROOT/results/elsa/${axis}_${condition}"

  echo "[run] axis=$axis condition=$condition"
  "$PYTHON_BIN" "$DISSE_ROOT/evaluate_elsa_continuity.py" \
    --elsa-root "$ELSA_ROOT" \
    --ckpt "$CKPT" \
    --audio-path "$AUDIO_PATH" \
    --axis "$axis" \
    --condition "$condition" \
    --out-dir "$out_dir" \
    --lkh "$LKH" \
    --batch-size "$BATCH_SIZE"
}

run_one azimuth dry
run_one azimuth reverb
run_one elevation dry
run_one elevation reverb

echo "[done] Results: $DISSE_ROOT/results/elsa"
