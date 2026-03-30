# Fingerstyle Guitar Audio-to-MIDI MVP

Chinese version: `README.zh.md`

This repository contains a lightweight Linux-first MVP for converting single-track fingerstyle guitar audio into:

1. a transcribed MIDI file
2. a piano-rendered WAV file

The current pipeline is intentionally simple and CPU-friendly:

1. preprocess input audio into a canonical WAV
2. transcribe the WAV to raw MIDI with `basic-pitch`
3. remove very short MIDI notes with `pretty_midi`
4. render the cleaned MIDI to WAV with `midi2audio` + `FluidSynth`

This repo does not include training code, MT3 integration, batch processing, or any frontend/UI.

## Current Status

The MVP is runnable end-to-end on Linux and has been validated in this repo with:

- Python `3.10`
- `basic-pitch`
- `pretty_midi`
- `midi2audio`
- `FluidSynth`
- a GM `.sf2` soundfont

The current implementation is CPU-first. A GPU is available on the target server, but this stage does not require CUDA or GPU-specific libraries.

## Pipeline Overview

Input audio:

- accepted as one file path, for example `sample.mp3`

Generated outputs:

- preprocessed WAV: `assets/output/convert/<name>.wav`
- raw MIDI: `assets/output/raw/<name>_basic_pitch.mid`
- cleaned MIDI: `assets/output/clean/<name>_clean.mid`
- rendered WAV: `assets/output/rendered/<name>.wav`

The main entry point is:

```bash
python scripts/run_pipeline.py ...
```

It prints logs, shows a 4-stage progress bar, and prints the final output paths at the end.

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── default.yaml
├── assets/
│   ├── input/
│   ├── output/
│   │   ├── clean/
│   │   ├── convert/
│   │   ├── raw/
│   │   └── rendered/
│   └── soundfonts/
├── scripts/
│   ├── run_cleanup_midi.py
│   ├── run_pipeline.py
│   ├── run_preprocess_audio.py
│   ├── run_render.py
│   └── run_transcription.py
├── src/
│   ├── __init__.py
│   ├── audio_preprocess.py
│   ├── midi_cleanup.py
│   ├── render.py
│   ├── transcription.py
│   └── utils.py
└── tests/
    └── test_smoke.py
```

Directory roles:

- `assets/input/`: optional place to store input audio files
- `assets/output/convert/`: canonical mono WAV files at `22050 Hz`
- `assets/output/raw/`: raw MIDI predicted by `basic-pitch`
- `assets/output/clean/`: cleaned MIDI after short-note removal
- `assets/output/rendered/`: final rendered WAV files
- `assets/soundfonts/`: soundfont assets and optional local FluidSynth runtime files
- `scripts/`: user-facing CLI entry points, including standalone stage runners and the end-to-end pipeline
- `src/`: implementation modules for each stage

## Environment Setup

### 1. Create the Python environment

Use Python `3.10`.

Recommended:

```bash
conda env create -f environment.yml
conda activate dl
```

Equivalent manual setup:

```bash
conda create -n dl python=3.10 pip -y
conda activate dl
python -m pip install -r requirements.txt
```

Notes:

- `requirements.txt` pins `numpy<2` because `basic-pitch` + `tflite-runtime` can fail with NumPy `2.x`.
- `environment.yml` is the preferred reproducible environment entrypoint for teammates.
- If `conda` is not already initialized in your shell, source your local `conda.sh` first.

Example:

```bash
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate dl
```

### 2. Confirm Python dependencies

```bash
python - <<'PY'
import librosa
import soundfile
import basic_pitch
import pretty_midi
import midi2audio
import numpy
print("python_deps_ok")
PY
```

## FluidSynth and SoundFont Setup

Rendering requires two things:

1. a `fluidsynth` executable
2. a `.sf2` soundfont file

You have two ways to provide them.

### Option A: Use the local runtime files already present in this repo

If you already prepared the following local files, you can use them directly:

- soundfont:
  `assets/soundfonts/extracted/usr/share/sounds/sf2/FluidR3_GM.sf2`
- FluidSynth binary:
  `assets/soundfonts/fluidsynth_pkg/usr/bin/fluidsynth`
- runtime libs:
  `assets/soundfonts/runtime_libs/usr/lib/x86_64-linux-gnu`

Export them for convenience:

```bash
export SOUNDFONT=assets/soundfonts/extracted/usr/share/sounds/sf2/FluidR3_GM.sf2
export FLUIDSYNTH_BIN=assets/soundfonts/fluidsynth_pkg/usr/bin/fluidsynth
export FLUIDSYNTH_LIB_DIR=$(pwd)/assets/soundfonts/runtime_libs/usr/lib/x86_64-linux-gnu
```

### Option B: Rebuild the local runtime without sudo

If those files are missing, you can recreate them on Ubuntu/Debian-like systems without root:

```bash
mkdir -p assets/soundfonts
cd assets/soundfonts

apt download fluid-soundfont-gm
apt download fluidsynth
apt download libfluidsynth3
apt download libsdl2-2.0-0
apt download libinstpatch-1.0-2
apt download libdecor-0-0

mkdir -p extracted fluidsynth_pkg runtime_libs

dpkg-deb -x fluid-soundfont-gm_*.deb extracted
dpkg-deb -x fluidsynth_*.deb fluidsynth_pkg

for pkg in libfluidsynth3_*.deb libsdl2-2.0-0_*.deb libinstpatch-1.0-2_*.deb libdecor-0-0_*.deb; do
  dpkg-deb -x "$pkg" runtime_libs
done

cd ../..
```

Then export:

```bash
export SOUNDFONT=assets/soundfonts/extracted/usr/share/sounds/sf2/FluidR3_GM.sf2
export FLUIDSYNTH_BIN=assets/soundfonts/fluidsynth_pkg/usr/bin/fluidsynth
export FLUIDSYNTH_LIB_DIR=$(pwd)/assets/soundfonts/runtime_libs/usr/lib/x86_64-linux-gnu
```

### 3. Confirm FluidSynth runtime

```bash
LD_LIBRARY_PATH="$FLUIDSYNTH_LIB_DIR" "$FLUIDSYNTH_BIN" --version
```

Expected output should include something like:

```text
FluidSynth runtime version 2.x
```

## Quick Start

Assuming:

- you already activated `dl`
- you already set `SOUNDFONT`, `FLUIDSYNTH_BIN`, and `FLUIDSYNTH_LIB_DIR`
- you want to use the example file `sample.mp3`

Run the full MVP:

```bash
python scripts/run_pipeline.py \
  --input sample.mp3 \
  --output-root assets/output \
  --soundfont "$SOUNDFONT" \
  --fluidsynth-bin "$FLUIDSYNTH_BIN" \
  --fluidsynth-lib-dir "$FLUIDSYNTH_LIB_DIR"
```

Expected generated files:

- `assets/output/convert/sample.wav`
- `assets/output/raw/sample_basic_pitch.mid`
- `assets/output/clean/sample_clean.mid`
- `assets/output/rendered/sample.wav`

## Step-by-Step Usage

### 1. Audio preprocessing

Convert one input audio file into a canonical WAV:

```bash
python scripts/run_preprocess_audio.py \
  --input sample.mp3 \
  --output assets/output/convert/sample.wav \
  --sr 22050
```

Behavior:

- converts to mono
- resamples to `22050 Hz`
- writes a `.wav`

### 2. Transcription

Transcribe one preprocessed WAV file to raw MIDI:

```bash
python scripts/run_transcription.py \
  --input assets/output/convert/sample.wav \
  --output-dir assets/output/raw
```

Expected output:

- `assets/output/raw/sample_basic_pitch.mid`

### 3. MIDI cleanup

Remove notes shorter than the configured threshold:

```bash
python scripts/run_cleanup_midi.py \
  --input-midi assets/output/raw/sample_basic_pitch.mid \
  --output-midi assets/output/clean/sample_clean.mid \
  --min-note-duration 0.05
```

Note:

- today, cleanup only removes very short notes
- the behavior is deterministic and intentionally minimal
- standalone cleanup is also included inside `scripts/run_pipeline.py`

### 4. Rendering

Render one cleaned MIDI file to WAV:

```bash
python scripts/run_render.py \
  --input-midi assets/output/clean/sample_clean.mid \
  --output-wav assets/output/rendered/sample_clean.wav \
  --soundfont "$SOUNDFONT" \
  --fluidsynth-bin "$FLUIDSYNTH_BIN" \
  --fluidsynth-lib-dir "$FLUIDSYNTH_LIB_DIR"
```

## End-to-End Command

This is the one command a teammate should be able to run after environment setup:

```bash
python scripts/run_pipeline.py \
  --input sample.mp3 \
  --output-root assets/output \
  --soundfont "$SOUNDFONT" \
  --fluidsynth-bin "$FLUIDSYNTH_BIN" \
  --fluidsynth-lib-dir "$FLUIDSYNTH_LIB_DIR"
```

What it does:

1. preprocesses `sample.mp3` to `assets/output/convert/sample.wav`
2. transcribes it to `assets/output/raw/sample_basic_pitch.mid`
3. cleans it to `assets/output/clean/sample_clean.mid`
4. renders it to `assets/output/rendered/sample.wav`

What it prints:

- stage logs
- a `tqdm` progress bar
- final output paths

## Example Files in This Repo

Suggested first run:

```bash
python scripts/run_pipeline.py \
  --input /path/to/your/input.mp3 \
  --output-root assets/output \
  --soundfont "$SOUNDFONT" \
  --fluidsynth-bin "$FLUIDSYNTH_BIN" \
  --fluidsynth-lib-dir "$FLUIDSYNTH_LIB_DIR"
```

## Configuration Notes

There is a placeholder config file at:

```text
configs/default.yaml
```

Current CLI scripts are still primarily argument-driven. The YAML file is useful as a reference for default path conventions, but the main pipeline is not yet fully config-driven.

Current keys:

- `input_audio`
- `preprocess_output_dir`
- `preprocessed_audio`
- `transcription_output_dir`
- `cleaned_midi_output_dir`
- `render_output_dir`
- `output_dir`
- `soundfont_path`
- `min_note_duration`
- `preprocess_sample_rate`
- `preprocess_mono`

## Linux Notes

- This MVP is designed for Linux.
- The current implementation does not require CUDA.
- The current implementation does not use the GPU, even if one is available.
- Rendering depends on `FluidSynth` plus a valid `.sf2` file.
- If you use a local extracted FluidSynth binary instead of a system install, also pass `--fluidsynth-lib-dir`.

## Known Limitations

- single-file processing only
- no batch mode
- no MT3 integration
- no frontend
- no training code
- cleanup only removes notes shorter than a threshold
- rendering uses a general MIDI soundfont, not a guitar-specific instrument model

## Troubleshooting

### `basic-pitch` fails with a NumPy / TFLite error

Make sure your environment uses `numpy<2`:

```bash
python -m pip install "numpy<2"
python -m pip install -r requirements.txt
```

### `basic-pitch` command not found

Make sure the `dl` environment is activated:

```bash
conda activate dl
which basic-pitch
```

### `fluidsynth` not found

Either:

- install `fluidsynth` system-wide, or
- use the local runtime method in this README and pass `--fluidsynth-bin`

### SoundFont file not found

Check that:

```bash
ls "$SOUNDFONT"
```

returns the `.sf2` file you expect.

### Rendering fails because of shared library errors

If you are using the local extracted FluidSynth runtime, make sure you passed:

```bash
--fluidsynth-lib-dir "$FLUIDSYNTH_LIB_DIR"
```

and that `FLUIDSYNTH_LIB_DIR` points to:

```text
assets/soundfonts/runtime_libs/usr/lib/x86_64-linux-gnu
```

### The pipeline is slow

That is expected for the current MVP:

- transcription runs on CPU
- rendering a long MIDI file can also take time

The current priority is simplicity and reproducibility, not speed.

## Minimal Verification Checklist

After setup, a teammate should be able to run:

```bash
conda activate dl
python scripts/run_pipeline.py \
  --input sample.mp3 \
  --output-root assets/output \
  --soundfont "$SOUNDFONT" \
  --fluidsynth-bin "$FLUIDSYNTH_BIN" \
  --fluidsynth-lib-dir "$FLUIDSYNTH_LIB_DIR"
```

and verify that these files exist:

```bash
ls assets/output/convert/sample.wav
ls assets/output/raw/sample_basic_pitch.mid
ls assets/output/clean/sample_clean.mid
ls assets/output/rendered/sample.wav
```

If all four files exist, the current MVP is working end-to-end.
