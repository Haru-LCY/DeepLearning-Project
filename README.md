# Audio-to-MIDI MVP with Optional PianistTransformer Rendering

Chinese version: `README.zh.md`

This repository contains a Linux-first symbolic music pipeline that converts one input audio file into:

1. a transcribed MIDI file
2. a cleaned MIDI file
3. an optional expressive piano MIDI rendered by PianistTransformer
4. a final piano WAV file

The baseline pipeline still works on its own. PianistTransformer is now an optional stage inside the same Python environment by default, while still allowing an explicit `--pt-python` override when you want to call a different interpreter.

## Current Status

The repository now supports a single-environment workflow centered on Python `3.11`:

- `basic-pitch`
- `pretty_midi`
- `midi2audio`
- `FluidSynth`
- `PianistTransformer`
- PyTorch `2.7.1` with CUDA `11.8` wheels

The intended server workflow is:

- run the full baseline pipeline in the `pianist-transformer` environment
- enable PianistTransformer with a CLI flag when expressive piano rendering is desired
- use the current Python interpreter by default for the PT stage

On systems with newer NVIDIA drivers, the official PyTorch `cu118` wheels work well even if the system CUDA runtime is newer.

## Pipeline Overview

Baseline path:

1. preprocess audio into canonical mono WAV
2. transcribe WAV to raw MIDI with `basic-pitch`
3. clean MIDI with `pretty_midi`
4. render cleaned MIDI to WAV with `midi2audio` + FluidSynth

Optional expressive path:

1. preprocess audio
2. transcribe to raw MIDI
3. clean MIDI
4. render expressive raw MIDI with PianistTransformer
5. map expressive timing back to score-aligned MIDI
6. render mapped expressive MIDI to WAV

Default output locations:

- preprocessed WAV: `assets/output/convert/<stem>.wav`
- raw MIDI: `assets/output/raw/<stem>_basic_pitch.mid`
- cleaned MIDI: `assets/output/clean/<stem>_clean.mid`
- expressive raw MIDI: `assets/output/expressive/raw/<stem>_pt_raw.mid`
- expressive mapped MIDI: `assets/output/expressive/mapped/<stem>_pt_mapped.mid`
- rendered WAV: `assets/output/rendered/<stem>.wav`

Main entry points:

- `python scripts/run_pipeline.py ...`
- `python scripts/run_expressive_render.py ...`

## Project Structure

```text
.
├── README.md
├── README.zh.md
├── environment.yml
├── requirements.txt
├── PianistTransformer/
├── configs/
├── assets/
├── scripts/
│   ├── run_cleanup_midi.py
│   ├── run_expressive_render.py
│   ├── run_pipeline.py
│   ├── run_preprocess_audio.py
│   ├── run_render.py
│   └── run_transcription.py
├── src/
│   ├── audio_preprocess.py
│   ├── expressive_render.py
│   ├── midi_cleanup.py
│   ├── render.py
│   ├── transcription.py
│   └── utils.py
└── tests/
```

Key directories:

- `assets/output/convert/`: canonical mono WAV files at `22050 Hz`
- `assets/output/raw/`: raw MIDI predicted by `basic-pitch`
- `assets/output/clean/`: cleaned MIDI after short-note removal
- `assets/output/expressive/raw/`: raw expressive MIDI from PianistTransformer
- `assets/output/expressive/mapped/`: mapped/editable expressive MIDI used for default PT rendering
- `assets/output/rendered/`: final rendered WAV files

## Environment Setup

### Recommended single environment

Use the provided environment file:

```bash
conda env create -f environment.yml
conda activate pianist-transformer
```

Equivalent manual setup:

```bash
conda create -n pianist-transformer python=3.11 pip -y
conda activate pianist-transformer
python -m pip install -r requirements.txt
```

Notes:

- `requirements.txt` includes both the baseline stack and the PianistTransformer runtime stack
- `numpy<2` remains pinned to preserve `basic-pitch` compatibility
- the file also adds the official PyTorch `cu118` index for GPU wheels
- on Python `3.11`, `basic-pitch` may install the TensorFlow backend instead of the older `tflite-runtime` path

Confirm imports:

```bash
python - <<'PY'
import basic_pitch
import pretty_midi
import midi2audio
import librosa
import soundfile
import torch
import transformers
import accelerate
import miditoolkit
import partitura
import numpy
print("python_deps_ok")
print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)
print(numpy.__version__)
PY
```

## FluidSynth and SoundFont Setup

Rendering requires:

1. a `fluidsynth` executable
2. a `.sf2` soundfont

The repository already supports a local no-sudo runtime layout:

- soundfont:
  `assets/soundfonts/extracted/usr/share/sounds/sf2/FluidR3_GM.sf2`
- FluidSynth binary:
  `assets/soundfonts/fluidsynth_pkg/usr/bin/fluidsynth`
- runtime libs:
  `assets/soundfonts/runtime_libs/usr/lib/x86_64-linux-gnu`

Recommended shell variables:

```bash
export SOUNDFONT=assets/soundfonts/extracted/usr/share/sounds/sf2/FluidR3_GM.sf2
export FLUIDSYNTH_BIN=assets/soundfonts/fluidsynth_pkg/usr/bin/fluidsynth
export FLUIDSYNTH_LIB_DIR=$(pwd)/assets/soundfonts/runtime_libs/usr/lib/x86_64-linux-gnu
```

Quick runtime check:

```bash
LD_LIBRARY_PATH="$FLUIDSYNTH_LIB_DIR" "$FLUIDSYNTH_BIN" --version
```

## Baseline End-to-End Command

This runs the original 4-stage pipeline in the single `pianist-transformer` environment:

```bash
python scripts/run_pipeline.py \
  --input sample.mp3 \
  --output-root assets/output \
  --soundfont "$SOUNDFONT" \
  --fluidsynth-bin "$FLUIDSYNTH_BIN" \
  --fluidsynth-lib-dir "$FLUIDSYNTH_LIB_DIR"
```

Expected artifacts:

- `assets/output/convert/sample.wav`
- `assets/output/raw/sample_basic_pitch.mid`
- `assets/output/clean/sample_clean.mid`
- `assets/output/rendered/sample.wav`

## Optional PianistTransformer in the Main Pipeline

Enable the expressive stage with:

```bash
python scripts/run_pipeline.py \
  --input sample.mp3 \
  --output-root assets/output \
  --soundfont "$SOUNDFONT" \
  --fluidsynth-bin "$FLUIDSYNTH_BIN" \
  --fluidsynth-lib-dir "$FLUIDSYNTH_LIB_DIR" \
  --enable-pianist-transformer
```

By default, the PT stage uses the current Python interpreter. This is the recommended single-environment workflow.

Optional PT-specific flags:

- `--pt-python /path/to/python`
- `--pt-model-dir PianistTransformer/models/sft`
- `--pt-device auto|cuda|cpu`
- `--pt-temperature 1.0`
- `--pt-top-p 0.95`
- `--pt-max-tempo 300`

When PT is enabled, the pipeline additionally creates:

- `assets/output/expressive/raw/<stem>_pt_raw.mid`
- `assets/output/expressive/mapped/<stem>_pt_mapped.mid`

The mapped expressive MIDI is the default render input for the final WAV.

## Standalone Expressive Rendering

If you already have a cleaned MIDI file and want to run only the PT stage:

```bash
python scripts/run_expressive_render.py \
  --input-midi assets/output/clean/hoshi_clean.mid \
  --output-root assets/output \
  --render \
  --soundfont "$SOUNDFONT" \
  --fluidsynth-bin "$FLUIDSYNTH_BIN" \
  --fluidsynth-lib-dir "$FLUIDSYNTH_LIB_DIR"
```

This writes:

- `assets/output/expressive/raw/hoshi_pt_raw.mid`
- `assets/output/expressive/mapped/hoshi_pt_mapped.mid`
- `assets/output/rendered/hoshi_pt.wav`

## Linux and Slurm Notes

- The repository is designed for Linux.
- The baseline stages do not require GPU.
- PianistTransformer benefits from GPU when available.
- On some clusters, `torch.cuda.is_available()` may be `False` in a plain shell but `True` inside a Slurm allocation.

If you already have an interactive GPU job, you can run PT inside it, for example:

```bash
srun --jobid=<your_jobid> --overlap bash -lc 'python scripts/run_expressive_render.py ...'
```

## Troubleshooting

### `basic-pitch` fails because of NumPy compatibility

Keep `numpy<2`:

```bash
python -m pip install "numpy<2"
python -m pip install -r requirements.txt
```

### TensorFlow prints CUDA/XLA warnings during import

This can happen because `basic-pitch` installs TensorFlow in the unified Python `3.11` environment. These warnings are noisy but do not necessarily indicate a failure in the PT stage or the baseline stage.

### `basic-pitch` command not found

Make sure the environment is active:

```bash
conda activate pianist-transformer
which basic-pitch
```

### `fluidsynth` or shared libraries are missing

Pass both:

```bash
--fluidsynth-bin "$FLUIDSYNTH_BIN" \
--fluidsynth-lib-dir "$FLUIDSYNTH_LIB_DIR"
```

### PT model files are missing

The local model directory must contain:

- `config.json`
- `generation_config.json`
- `model.safetensors`

under:

```text
PianistTransformer/models/sft/
```

## Minimal Verification Checklist

Baseline:

```bash
conda activate pianist-transformer
python scripts/run_pipeline.py \
  --input sample.mp3 \
  --output-root assets/output \
  --soundfont "$SOUNDFONT" \
  --fluidsynth-bin "$FLUIDSYNTH_BIN" \
  --fluidsynth-lib-dir "$FLUIDSYNTH_LIB_DIR"
```

Expressive:

```bash
python scripts/run_pipeline.py \
  --input sample.mp3 \
  --output-root assets/output \
  --soundfont "$SOUNDFONT" \
  --fluidsynth-bin "$FLUIDSYNTH_BIN" \
  --fluidsynth-lib-dir "$FLUIDSYNTH_LIB_DIR" \
  --enable-pianist-transformer
```

If both commands finish and the expected files exist, the single-environment workflow is ready.
