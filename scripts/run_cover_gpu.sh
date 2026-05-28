#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

INPUT_AUDIO="${1:-ringo.mp3}"
OUTPUT_ROOT="${2:-assets/output/cover_gpu_test}"

SOUNDFONT="${SOUNDFONT:-$REPO_ROOT/assets/soundfonts/extracted/usr/share/sounds/sf2/FluidR3_GM.sf2}"
FLUIDSYNTH_BIN="${FLUIDSYNTH_BIN:-$REPO_ROOT/assets/soundfonts/fluidsynth_pkg/usr/bin/fluidsynth}"
FLUIDSYNTH_LIB_DIR="${FLUIDSYNTH_LIB_DIR:-$REPO_ROOT/assets/soundfonts/runtime_libs/usr/lib/x86_64-linux-gnu}"

if [[ -d "$SOUNDFONT" ]]; then
  mapfile -t soundfont_candidates < <(find "$SOUNDFONT" -type f -name '*.sf2' | sort)
  if [[ "${#soundfont_candidates[@]}" -eq 1 ]]; then
    SOUNDFONT="${soundfont_candidates[0]}"
  elif [[ "${#soundfont_candidates[@]}" -eq 0 ]]; then
    echo "SoundFont directory contains no .sf2 files: $SOUNDFONT" >&2
    exit 1
  else
    echo "SoundFont directory contains multiple .sf2 files; set SOUNDFONT to one file:" >&2
    printf '  %s\n' "${soundfont_candidates[@]}" >&2
    exit 1
  fi
fi

if [[ ! -f "$INPUT_AUDIO" ]]; then
  echo "Input audio not found: $INPUT_AUDIO" >&2
  exit 1
fi

if [[ ! -f "$SOUNDFONT" ]]; then
  echo "SoundFont not found: $SOUNDFONT" >&2
  exit 1
fi

if [[ ! -x "$FLUIDSYNTH_BIN" ]]; then
  echo "FluidSynth binary not found or not executable: $FLUIDSYNTH_BIN" >&2
  exit 1
fi

if [[ ! -d "$FLUIDSYNTH_LIB_DIR" ]]; then
  echo "FluidSynth library directory not found: $FLUIDSYNTH_LIB_DIR" >&2
  exit 1
fi

if ! python - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
then
  echo "CUDA is not available in this Python environment. Activate pianoformer-cover inside a GPU session." >&2
  exit 1
fi

python scripts/run_ai_piano_cover.py \
  --input "$INPUT_AUDIO" \
  --output-root "$OUTPUT_ROOT" \
  --device cuda \
  --pop2piano-device cuda \
  --soundfont "$SOUNDFONT" \
  --fluidsynth-bin "$FLUIDSYNTH_BIN" \
  --fluidsynth-lib-dir "$FLUIDSYNTH_LIB_DIR"

stem="$(basename "$INPUT_AUDIO")"
stem="${stem%.*}"
echo
echo "Done: $OUTPUT_ROOT/final/${stem}_ai_cover_piano.wav"
