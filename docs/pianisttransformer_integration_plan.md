# PianistTransformer Integration Plan

## Scope of this note

This note is a read-only design pass. It does not implement the integration yet.

Goal:

- keep the current baseline pipeline working exactly as it does today when expressive rendering is disabled
- add PianistTransformer later as an optional stage with explicit intermediate files
- make the smallest safe change set for a Linux server workflow

## Current repository summary

### Current pipeline stages

The current repository implements a simple 4-stage file pipeline:

1. input audio -> canonical mono WAV
2. canonical WAV -> raw MIDI via `basic-pitch`
3. raw MIDI -> cleaned MIDI via `pretty_midi`
4. cleaned MIDI -> piano WAV via `midi2audio` + FluidSynth

This is stated in [README.md](../README.md) and wired directly in [`scripts/run_pipeline.py`](../scripts/run_pipeline.py).

### Main entry points

User-facing scripts:

- [`scripts/run_pipeline.py`](../scripts/run_pipeline.py): end-to-end pipeline
- [`scripts/run_preprocess_audio.py`](../scripts/run_preprocess_audio.py): standalone audio preprocessing
- [`scripts/run_transcription.py`](../scripts/run_transcription.py): standalone transcription
- [`scripts/run_cleanup_midi.py`](../scripts/run_cleanup_midi.py): standalone MIDI cleanup
- [`scripts/run_render.py`](../scripts/run_render.py): standalone MIDI-to-WAV rendering

### Where MIDI is generated

Raw MIDI is generated in [`src/transcription.py`](../src/transcription.py).

- `transcribe_audio(...)` shells out to `basic-pitch`
- output directory is provided by the caller
- expected output filename is `<stem>_basic_pitch.mid`

The end-to-end caller is [`scripts/run_pipeline.py`](../scripts/run_pipeline.py), which stores raw MIDI under `assets/output/raw/`.

### Where MIDI is cleaned

Cleaned MIDI is produced in [`src/midi_cleanup.py`](../src/midi_cleanup.py).

- `clean_midi(...)` loads the MIDI with `pretty_midi`
- it only removes notes shorter than `min_note_duration`
- it writes a new MIDI file

The end-to-end caller stores this under `assets/output/clean/<stem>_clean.mid`.

### Where MIDI is rendered to WAV

Final rendering happens in [`src/render.py`](../src/render.py).

- `render_midi_to_wav(...)` uses `midi2audio.FluidSynth`
- it requires a `.sf2` soundfont
- it optionally accepts a non-system FluidSynth binary and library directory

The end-to-end caller currently renders the cleaned MIDI directly to `assets/output/rendered/<stem>.wav`.

### Current output artifacts

The current repo already uses explicit intermediate files:

- preprocessed WAV: `assets/output/convert/<stem>.wav`
- raw MIDI: `assets/output/raw/<stem>_basic_pitch.mid`
- cleaned MIDI: `assets/output/clean/<stem>_clean.mid`
- rendered WAV: `assets/output/rendered/<stem>.wav`

This is a good fit for adding one more optional explicit MIDI artifact.

### Likely integration points

The least invasive places to extend later are:

- [`scripts/run_pipeline.py`](../scripts/run_pipeline.py)
  - add one optional stage and one extra output path
  - keep default behavior identical when disabled
- `build_output_paths(...)` in [`scripts/run_pipeline.py`](../scripts/run_pipeline.py)
  - define explicit output path(s) for expressive MIDI
- a new module under `src/`
  - likely `src/expressive_render.py` or `src/pianist_transformer.py`
  - isolate all PianistTransformer-specific logic from baseline modules
- a new standalone CLI script
  - likely `scripts/run_expressive_render.py`
  - useful for debugging and explicit stage-by-stage operation
- config handling
  - [`configs/default.yaml`](../configs/default.yaml) already exists
  - [`src/utils.py`](../src/utils.py) has a stub `load_config(...)`, but config loading is not currently used by the pipeline
  - the current real control surface is CLI flags, not YAML parsing

## PianistTransformer summary

### What task it solves

PianistTransformer is not an audio transcription model. It is a symbolic music model for expressive piano performance rendering.

Based on its README and code, the intended task is:

- input: score-like piano MIDI
- output: expressive piano performance MIDI

This is described in:

- [`PianistTransformer/README.md`](../PianistTransformer/README.md)
- `batch_performance_render(...)` in [`PianistTransformer/src/model/generate.py`](../PianistTransformer/src/model/generate.py)

### What inputs it expects

The shipped inference path loads MIDI files from:

- `PianistTransformer/data/midis/testset/score/*.mid`

The naming and README language consistently call these files "score" MIDI.

Important code-level observations:

- `batch_performance_render(model, score_midi_objs, ...)` takes a list of `miditoolkit.MidiFile` objects
- those MIDI objects are converted by `midi_to_ids(...)`
- `midi_to_ids(...)` calls `normalize_midi(...)` by default
- `normalize_midi(...)` merges all non-drum instruments into a single piano instrument, converts timing into a normalized 500 TPB / 120 BPM space, and preserves only sustain pedal CC64

This means the model effectively expects a piano-score-like symbolic input, not arbitrary multi-instrument MIDI with detailed controller data.

### What outputs it produces

There are two practically relevant output forms in the codebase:

1. generated expressive performance MIDI
2. mapped/editable expressive MIDI aligned back onto the original score grid

Details:

- `batch_performance_render(...)` returns a generated `miditoolkit.MidiFile`
- `ids_to_midi(...)` reconstructs notes plus generated pedal CC64 in normalized time
- the demo inference script then calls `map_midi(score_midi_obj, performance_midi_obj)`
- `map_midi(...)` transfers expressive timing, velocity, and pedal back onto the original score timeline as a DAW-friendlier MIDI with tempo changes

For this repository, the second form is likely more useful as the file to hand off to FluidSynth:

- it preserves the original note identity/order
- it writes expressive timing through tempo changes and note durations
- it is the output form the upstream README recommends as "editable"

### How inference is launched today

The provided shell entry point is very thin:

- [`PianistTransformer/script/inference.sh`](../PianistTransformer/script/inference.sh)
  - sets `PYTHONPATH=.`
  - runs `python src/inference/inference.py`

The actual script at [`PianistTransformer/src/inference/inference.py`](../PianistTransformer/src/inference/inference.py):

- loads model weights from `models/sft/`
- loads a hard-coded example score from `data/midis/testset/score/0.mid`
- runs `batch_performance_render(...)`
- immediately runs `map_midi(...)`
- writes output to `data/midis/testset/inference/0.mid`

This is a demo script, not a reusable CLI for arbitrary paths.

### Environment and dependency assumptions

PianistTransformer has materially different environment assumptions from the current repo:

- current repo baseline:
  - Python 3.10 in [`environment.yml`](../environment.yml)
  - `basic-pitch`, `pretty_midi`, `midi2audio`
- PianistTransformer:
  - Python 3.11 in [`PianistTransformer/README.md`](../PianistTransformer/README.md)
  - `torch==2.7.1`
  - `transformers==4.54.0`
  - `datasets`, `accelerate`, `miditoolkit`, `partitura`
  - optional GUI deps `PyQt5`, `pygame`

Model checkpoint assumptions:

- expected local model directory: `PianistTransformer/models/sft/`
- expected files include `generation_config.json`, `config.json`, `model.safetensors`
- the repo currently does not contain downloaded model files
- download helper exists at [`PianistTransformer/src/utils/download_model.py`](../PianistTransformer/src/utils/download_model.py)

Inference hardware assumptions:

- README says CPU is sufficient for inference
- code defaults to `device="cpu"` in the demo path
- GPU is optional, not required for the initial integration design

### Score-like MIDI vs expressive MIDI

My conclusion from the code is:

- input should be score-like piano MIDI
- output is expressive piano performance MIDI

Why:

- README repeatedly says "render the score MIDI"
- `batch_performance_render(...)` argument name is `score_midi_objs`
- `map_midi(score, performance)` assumes the generated performance has the same note sequence/cardinality as the source score after normalization
- upstream GUI labels the input as `Load MIDI` for score and distinguishes "Rendered MIDI" from "Editable Rendered MIDI"

### Python API vs subprocess

Technically, PianistTransformer can be called from Python directly:

- `PianoT5Gemma.from_pretrained(...)`
- `batch_performance_render(...)`
- `map_midi(...)`

However, for this repository, it is better treated initially as an isolated subprocess stage, not an in-process import.

Reason:

- the environment requirements differ substantially from the baseline pipeline
- the repo currently centers around shelling out for external tools already (`basic-pitch`)
- subprocess isolation reduces the risk of breaking the current baseline environment
- the shipped upstream inference entry point is script-oriented, even though it needs a wrapper for arbitrary inputs

## Recommended integration point

### Recommended stage placement

Recommended placement:

`audio -> preprocess -> raw MIDI -> MIDI cleanup -> optional PianistTransformer -> expressive piano MIDI -> WAV render`

In other words:

- insert the new stage after cleanup
- render WAV from expressive MIDI when enabled
- render WAV from cleaned MIDI when disabled

### Why this is the right place

This is the least invasive and most semantically correct insertion point.

Reasons:

1. the current cleanup stage is the first place where the transcription output becomes a more stable score-like symbolic artifact
2. PianistTransformer is a score-to-expressive-performance renderer, not an audio model and not a post-WAV effect
3. the render stage already only needs a MIDI file path, so it can remain unchanged if the pipeline simply swaps which MIDI path is passed into it
4. baseline behavior can stay identical by default:
   - if disabled: render `cleaned_midi`
   - if enabled: render `expressive_midi`

### Alternatives considered

#### Before MIDI cleanup

Not recommended.

Why:

- raw `basic-pitch` MIDI is likely noisier and less score-like
- cleanup currently removes obvious garbage notes
- feeding noisier note streams into a score-conditioned performance model is higher risk

#### After WAV rendering

Not meaningful.

- PianistTransformer operates on MIDI, not audio

#### Inside the render stage

Not recommended.

- it mixes symbolic performance generation with WAV synthesis concerns
- it would make debugging harder and hide an important intermediate artifact

## Expected adaptation between cleaned MIDI and PianistTransformer input

### Is the cleaned MIDI already score-like enough?

Maybe partially, but not reliably enough to assume zero adaptation.

What helps:

- the cleaned MIDI is already symbolic
- short-note removal should reduce some transcription artifacts

What still makes this uncertain:

- `basic-pitch` output comes from audio transcription, not symbolic score entry
- transcription may produce dense overlaps, fragmented notes, spurious notes, or velocity patterns unlike a clean score
- the current cleanup only removes short notes; it does not normalize instruments, channels, pedal, overlap, tempo, or note density

So my current judgment is:

- the cleaned MIDI is the right candidate input
- but a lightweight adapter/normalizer will likely still be needed before invoking PianistTransformer

### Adaptation likely needed

At minimum, the integration should verify or normalize these areas before calling PianistTransformer:

1. piano-only structure
   - PianistTransformer collapses non-drum instruments into one piano instrument internally
   - current transcribed MIDI may already be simple, but the integration should not assume instrument naming/programs are meaningful

2. non-drum single-stream note layout
   - upstream normalization merges notes across instruments
   - it also trims overlapping same-pitch notes
   - current cleanup does not do this

3. tempo and tick normalization
   - upstream `normalize_midi(...)` already converts to 500 TPB / 120 BPM internally
   - this does not necessarily require a separate external preprocessing file, but it is part of the true input contract

4. pedal data
   - PianistTransformer models CC64 sustain pedal
   - your cleaned MIDI from transcription likely has no meaningful pedal
   - that is probably acceptable, because the model generates pedal in the output path

5. note order stability
   - `map_midi(...)` asserts that normalized score note count equals normalized generated performance note count
   - this likely holds when using their own generation path, but bad upstream MIDI structure could still create edge cases

6. channels and track structure
   - safest later approach is to hand PianistTransformer a normalized piano-only score MIDI, even if the raw cleaned MIDI contains more structure than needed

### Recommended adaptation approach

Least invasive recommendation:

- keep the existing cleaned MIDI file unchanged as the baseline artifact
- add a separate explicit adapter output for the optional stage, for example:
  - `assets/output/pt_input/<stem>_pt_input.mid`

This adapter file should be produced only when expressive rendering is enabled, and should be a normalized piano-score MIDI specifically intended for PianistTransformer.

That keeps:

- baseline artifact history intact
- debugging easy
- the handoff contract explicit

## Recommended calling strategy

### Recommendation

Use a subprocess boundary around a small local wrapper script that lives in this repository, not the upstream demo script directly.

Recommended shape:

1. current pipeline writes cleaned MIDI
2. current repo writes a normalized PT input MIDI
3. current repo calls a dedicated wrapper script in `PianistTransformer/` or a local helper script with explicit arguments
4. wrapper loads the PT model and writes expressive MIDI to an explicit output path
5. baseline render stage renders that expressive MIDI

### Why not call the upstream demo script directly?

Because `PianistTransformer/src/inference/inference.py` is hard-coded to:

- fixed model path
- fixed example input path
- fixed output path
- fixed generation parameters

It is not suitable as the integration surface without modification or replacement.

### Why not import PT in-process right away?

Possible, but not recommended for the first integration step.

A subprocess wrapper is safer because:

- it avoids forcing the baseline Python 3.10 environment to satisfy the PT stack
- it allows later use of a dedicated PT environment if needed
- it mirrors how `basic-pitch` is already integrated
- failures remain localized and easier to diagnose

## Recommended new files and modules for a later implementation

This is the smallest structure I would add.

### In the main repo

- `src/expressive_render.py`
  - orchestration function for the optional PT stage
  - likely responsible for:
    - validating paths
    - optionally preparing PT input MIDI
    - invoking the PT subprocess wrapper
    - returning expressive MIDI output path

- `scripts/run_expressive_render.py`
  - standalone CLI for one MIDI in -> one expressive MIDI out
  - useful for debugging without running the full audio pipeline

- pipeline updates in [`scripts/run_pipeline.py`](../scripts/run_pipeline.py)
  - new flag such as `--enable-expressive-rendering`
  - optional PT-related path/config args
  - progress bar expanded from 4 stages to 5 when enabled, or kept logically conditional

- output directory addition
  - `assets/output/expressive/`
  - optional: `assets/output/pt_input/`

### Near or inside PianistTransformer

One new wrapper script is likely needed, for example:

- `PianistTransformer/src/inference/render_file.py`

Suggested responsibility:

- accept input MIDI path, output MIDI path, model path, device, temperature, top-p, and maybe `max_tempo`
- load `miditoolkit.MidiFile`
- call `batch_performance_render(...)`
- optionally call `map_midi(...)`
- write the final expressive MIDI to the requested output path

This is much safer than trying to bend the existing demo script into a pipeline API.

## Risks and blockers

### 1. Environment mismatch

This is the biggest concrete blocker.

Current repo:

- Python 3.10
- `basic-pitch` stack

PianistTransformer:

- Python 3.11
- PyTorch 2.7.1
- Transformers 4.54.0

It is not safe to assume one environment will satisfy both stacks without validation.

Implication:

- the optional stage should be designed so it can run in a separate environment if necessary

### 2. Missing model weights

The PT repository currently has no `models/sft/` files checked in.

Implication:

- integration must handle "feature enabled but model missing" as a clear runtime error
- this should not affect baseline behavior when the feature is disabled

### 3. Input domain mismatch

PianistTransformer is designed for piano score rendering.

Your current pipeline starts from arbitrary single-source audio and transcribes it with a guitar-oriented MVP description in the top-level README.

Implication:

- even if the code integrates correctly, musical quality may vary a lot when the cleaned MIDI is not truly piano-score-like
- this is a product-quality risk, not only an engineering risk

### 4. Note-count / mapping assumptions

`map_midi(...)` assumes stable note correspondence between score and generated performance after normalization.

Implication:

- malformed or extremely noisy transcribed MIDI may cause poor mapping quality or edge-case failures

### 5. Long-sequence and latency considerations

`batch_performance_render(...)` uses a max context length of 4096 tokens with sliding windows.

Implication:

- long dense MIDI files may increase latency
- this is probably acceptable for an optional offline render stage

## Open questions

These do not block the design note, but they should be resolved before implementation.

1. Which expressive output should become the default artifact when PT is enabled?
   - raw PT performance MIDI
   - mapped/editable MIDI

My recommendation: render WAV from the mapped/editable MIDI, and optionally keep the raw PT output too if debugging is needed.

2. Should the optional stage require a dedicated external Python executable or conda env name?
   - this is likely the safest operational model on Linux

3. Do you want PT-specific generation controls exposed immediately?
   - `temperature`
   - `top_p`
   - `device`
   - `max_tempo`

My recommendation: start with fixed defaults or a very small set of flags.

4. How strict should the adapter be about rejecting non-piano-like MIDI?
   - permissive pass-through
   - or validation warnings/errors

My recommendation: start permissive, log warnings, and keep artifacts explicit.

5. Do you want a YAML-driven config path soon, or should the first implementation stay CLI-first?

My recommendation: CLI-first, because config parsing is not active in the current pipeline yet.

## Small-step implementation plan

Recommended later implementation order:

1. Add a design-minimal standalone PT wrapper for `input MIDI -> expressive MIDI`.
2. Add a new repo-local module to call that wrapper as a subprocess.
3. Add explicit output paths for:
   - PT input MIDI
   - expressive output MIDI
4. Add one standalone script in the main repo for the optional stage.
5. Extend `scripts/run_pipeline.py` with an opt-in flag only.
6. When disabled, keep the current path exactly:
   - cleaned MIDI -> render WAV
7. When enabled, use:
   - cleaned MIDI -> PT input MIDI -> expressive MIDI -> render WAV
8. Add smoke tests for:
   - baseline path unchanged when feature disabled
   - path selection logic when feature enabled
   - clear failure if PT model or executable/env is missing

## Final recommendation

The least invasive integration path is:

- keep the baseline pipeline unchanged by default
- insert PianistTransformer after MIDI cleanup and before WAV rendering
- treat it as an optional subprocess-based symbolic rendering stage
- preserve explicit intermediate files
- isolate PT-specific logic in new files rather than modifying existing baseline modules heavily

This fits the current repository structure well and minimizes the risk of breaking the MVP.
