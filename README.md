# AI Cover + Pop2Piano

Main setup guide: `README.zh.md`.

This repository's current primary workflow generates an AI vocal cover plus Pop2Piano piano accompaniment from one input song.

Main entry point:

```bash
python scripts/run_ai_piano_cover.py ...
```

FastAPI backend entry point:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Backend role/API config lives in `configs/roles.yaml`.
The default backend preload mode is `torch_cuda`: startup loads Demucs, all configured DDSP role runtimes, RMVPE, and Pop2Piano into the backend process on CUDA, and backend jobs reuse those in-process models instead of loading them in model subprocesses.
Startup also runs `warmup.mp3` through the full cover pipeline into `assets/output/jobs/_startup_warmup` with non-default pitch shift and mix volumes, then prints `startup complete`.

## Quick Setup

Use the cover environment:

```bash
conda env create -f environment.pianoformer-cover.yml
conda activate pianoformer-cover
```

Large runtime assets are expected to be copied from an existing machine rather than committed to Git:

- `third_party/ai_cover/demucs/`
- `third_party/ai_cover/DDSP-SVC/`
- DDSP checkpoint and pretrained assets under `third_party/ai_cover/DDSP-SVC/exp/` and `third_party/ai_cover/DDSP-SVC/pretrain/`
- FluidSynth runtime and SoundFont under `assets/soundfonts/`
- Pop2Piano model, recommended at `models/pop2piano/sweetcocoa-pop2piano/`

## Smoke Test

```bash
python scripts/run_ai_piano_cover.py \
  --input sample.mp3 \
  --device cuda \
  --pop2piano-device cuda \
  --pop2piano-model models/pop2piano/sweetcocoa-pop2piano \
  --dry-run
```

## Run

```bash
export SOUNDFONT=assets/soundfonts/extracted/usr/share/sounds/sf2/FluidR3_GM.sf2
export FLUIDSYNTH_BIN=assets/soundfonts/fluidsynth_pkg/usr/bin/fluidsynth
export FLUIDSYNTH_LIB_DIR=$(pwd)/assets/soundfonts/runtime_libs/usr/lib/x86_64-linux-gnu

python scripts/run_ai_piano_cover.py \
  --input sample.mp3 \
  --output-root assets/output/cover \
  --device cuda \
  --pop2piano-device cuda \
  --pop2piano-model models/pop2piano/sweetcocoa-pop2piano \
  --soundfont "$SOUNDFONT" \
  --fluidsynth-bin "$FLUIDSYNTH_BIN" \
  --fluidsynth-lib-dir "$FLUIDSYNTH_LIB_DIR"
```

Expected final output:

```text
assets/output/cover/final/sample_ai_cover_piano.wav
```

The previous audio-to-MIDI / PianistTransformer workflow is documented in `docs/audio_to_midi_legacy.zh.md`.
