"""End-to-end AI vocal cover plus Pop2Piano accompaniment orchestration."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
import gc
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Iterator


REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = REPO_ROOT / "third_party" / "ai_cover"
DEMUCS_ROOT = VENDOR_ROOT / "demucs"
DDSP_ROOT = VENDOR_ROOT / "DDSP-SVC"
DEFAULT_DDSP_MODEL = DDSP_ROOT / "exp" / "reflow-test" / "model_30000.pt"
DEFAULT_LOCAL_POP2PIANO_MODEL = Path(
    "/home/fit/alex/WORK/pretrained_models/models--sweetcocoa--pop2piano/"
    "snapshots/142e8ed35614bcf77a3515b979e48ed528342349"
)
DEFAULT_POP2PIANO_MODEL = (
    str(DEFAULT_LOCAL_POP2PIANO_MODEL)
    if DEFAULT_LOCAL_POP2PIANO_MODEL.exists()
    else "sweetcocoa/pop2piano"
)
DEFAULT_POP2PIANO_BEAT_CHECKPOINT = REPO_ROOT / "pop2piano" / "models" / "final0.ckpt"
DEFAULT_SOUNDFONT = REPO_ROOT / "assets" / "soundfonts" / "extracted" / "usr" / "share" / "sounds" / "sf2" / "FluidR3_GM.sf2"
DEFAULT_FLUIDSYNTH_BIN = REPO_ROOT / "assets" / "soundfonts" / "fluidsynth_pkg" / "usr" / "bin" / "fluidsynth"
DEFAULT_FLUIDSYNTH_LIB_DIR = REPO_ROOT / "assets" / "soundfonts" / "runtime_libs" / "usr" / "lib" / "x86_64-linux-gnu"
ProgressCallback = Callable[[str, int, str], None]
USE_PARALLEL_COVER_STAGES = True
_CUDA_STREAM_LOCK = threading.Lock()
_CUDA_STREAMS: dict[tuple[str, str], Any] = {}


@dataclass(frozen=True)
class CommandSpec:
    cwd: Path
    argv: list[str]
    env: dict[str, str] | None = None

    def pretty(self) -> str:
        return shlex.join(self.argv)


@dataclass(frozen=True)
class CoverArtifacts:
    preprocessed_audio: Path
    vocals: Path
    no_vocals: Path
    ddsp_vocals: Path
    piano_midi: Path
    piano_wav: Path
    final_mix: Path


@dataclass(frozen=True)
class CoverRunResult:
    artifacts: CoverArtifacts
    stage_timings: dict[str, float]


def build_cover_artifacts(input_audio: Path, output_root: Path) -> CoverArtifacts:
    stem = input_audio.stem
    return CoverArtifacts(
        preprocessed_audio=output_root / "preprocessed" / f"{stem}.wav",
        vocals=output_root / "separated" / f"{stem}_vocals.wav",
        no_vocals=output_root / "separated" / f"{stem}_no_vocals.wav",
        ddsp_vocals=output_root / "vocals" / f"{stem}_ddsp_vocals.wav",
        piano_midi=output_root / "piano" / f"{stem}_pop2piano.mid",
        piano_wav=output_root / "piano" / f"{stem}_pop2piano.wav",
        final_mix=output_root / "final" / f"{stem}_ai_cover_piano.wav",
    )


def _merged_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")
    if extra:
        env.update(extra)
    return env


def _pythonpath_with(*paths: Path) -> str:
    parts = [str(path) for path in paths]
    if os.environ.get("PYTHONPATH"):
        parts.append(os.environ["PYTHONPATH"])
    return os.pathsep.join(parts)


def _run(spec: CommandSpec, dry_run: bool = False) -> None:
    print(f"Working directory: {spec.cwd}")
    print(f"Command: {spec.pretty()}")
    if dry_run:
        return
    subprocess.run(spec.argv, cwd=spec.cwd, env=_merged_env(spec.env), check=True)


def _emit_progress(
    progress_callback: ProgressCallback | None,
    stage_name: str,
    progress: int,
    message: str,
) -> None:
    if progress_callback is not None:
        progress_callback(stage_name, progress, message)


def _is_cuda_device(device: str | None) -> bool:
    if device is None:
        return False
    try:
        import torch

        torch_device = torch.device(device)
        return torch_device.type == "cuda" and torch.cuda.is_available()
    except Exception:  # noqa: BLE001 - stream support is an optimization only
        return False


@contextmanager
def _cuda_stream_context(device: str | None, branch_name: str) -> Iterator[None]:
    if not _is_cuda_device(device):
        yield
        return

    import torch

    torch_device = torch.device(device)
    with torch.cuda.device(torch_device):
        stream = _get_cuda_stream(torch_device, branch_name)
        stream.wait_stream(torch.cuda.current_stream(torch_device))
        print(f"Using CUDA stream for {branch_name} branch on {torch_device}")
        try:
            with torch.cuda.stream(stream):
                yield
        finally:
            stream.synchronize()


def _get_cuda_stream(torch_device: Any, branch_name: str) -> Any:
    key = (str(torch_device), branch_name)
    with _CUDA_STREAM_LOCK:
        stream = _CUDA_STREAMS.get(key)
        if stream is None:
            import torch

            stream = torch.cuda.Stream(device=torch_device)
            _CUDA_STREAMS[key] = stream
        return stream


def _component_device(component: Any | None, fallback: str | None) -> str | None:
    if component is None:
        return fallback
    if isinstance(component, dict) and "device" in component:
        return str(component["device"])
    device = getattr(component, "device", None)
    if device is not None:
        return str(device)
    private_device = getattr(component, "_device", None)
    if private_device is not None:
        return str(private_device)
    return fallback


def _cuda_branch_device(*devices: str | None) -> str | None:
    for device in devices:
        if _is_cuda_device(device):
            return device
    return None


def _cleanup_cuda_memory(reason: str) -> None:
    gc.collect()
    try:
        import torch

        if not torch.cuda.is_available():
            return
        for device_index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device_index)
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
        print(f"Cleaned CUDA cache after {reason}")
    except Exception as exc:  # noqa: BLE001 - cleanup must not hide pipeline errors
        print(f"CUDA cleanup skipped after {reason}: {exc}")


def _move_torch_object(obj: Any, device: str) -> Any:
    if obj is None:
        return None

    mover = getattr(obj, "to", None)
    if callable(mover):
        moved = mover(device)
        if moved is not None:
            obj = moved
    return obj


def _move_demucs_separator(separator: Any | None, device: str, reason: str, strict: bool = True) -> None:
    if separator is None:
        return
    try:
        _move_torch_object(getattr(separator, "model", None), device)
        updater = getattr(separator, "update_parameter", None)
        if callable(updater):
            updater(device=device, progress=False)
        elif hasattr(separator, "_device"):
            separator._device = device
        print(f"Moved Demucs separator to {device} for {reason}")
    except Exception as exc:  # noqa: BLE001 - device setup should not hide the primary stage error
        if strict:
            raise
        print(f"Demucs separator move skipped for {reason}: {exc}")


def _ffmpeg_runtime_env() -> dict[str, str] | None:
    prefix_lib = Path(sys.prefix) / "lib"
    required_openh264 = prefix_lib / "libopenh264.so.5"
    fallback_openh264 = prefix_lib / "libopenh264.so.6"
    if required_openh264.exists() or not fallback_openh264.exists():
        return None

    compat_dir = Path("/tmp") / "pianoformer_cover_ffmpeg_libs"
    compat_dir.mkdir(parents=True, exist_ok=True)
    compat_openh264 = compat_dir / "libopenh264.so.5"
    if not compat_openh264.exists():
        shutil.copy2(fallback_openh264, compat_openh264)

    parts = [str(compat_dir)]
    if os.environ.get("LD_LIBRARY_PATH"):
        parts.append(os.environ["LD_LIBRARY_PATH"])
    return {"LD_LIBRARY_PATH": os.pathsep.join(parts)}


def build_preprocess_command(input_audio: Path, output_wav: Path, pre_pitch_shift: float = 0.0) -> CommandSpec:
    if not math.isfinite(pre_pitch_shift):
        raise ValueError("Preprocess pitch shift must be a finite number.")

    argv = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_audio),
    ]
    if pre_pitch_shift != 0.0:
        pitch_ratio = 2 ** (pre_pitch_shift / 12)
        argv.extend(["-filter:a", f"rubberband=pitch={pitch_ratio:.12g}"])
    argv.extend([
        "-ar",
        "44100",
        "-ac",
        "2",
        "-c:a",
        "pcm_s16le",
        str(output_wav),
    ])

    return CommandSpec(
        cwd=REPO_ROOT,
        argv=argv,
        env=_ffmpeg_runtime_env(),
    )


def build_demucs_command(input_audio: Path, raw_output_root: Path, device: str) -> CommandSpec:
    return CommandSpec(
        cwd=DEMUCS_ROOT,
        argv=[
            sys.executable,
            "-m",
            "demucs",
            "-n",
            "htdemucs",
            "-o",
            str(raw_output_root),
            "--two-stems",
            "vocals",
            "-d",
            device,
            str(input_audio),
        ],
        env={"PYTHONPATH": _pythonpath_with(VENDOR_ROOT, DEMUCS_ROOT)},
    )


def copy_demucs_outputs(preprocessed_audio: Path, raw_output_root: Path, artifacts: CoverArtifacts) -> None:
    track_dir = raw_output_root / "htdemucs" / preprocessed_audio.stem
    vocals_src = track_dir / "vocals.wav"
    no_vocals_src = track_dir / "no_vocals.wav"
    if not vocals_src.exists():
        raise FileNotFoundError(f"Expected Demucs vocals output was not found: {vocals_src}")
    if not no_vocals_src.exists():
        raise FileNotFoundError(f"Expected Demucs no_vocals output was not found: {no_vocals_src}")
    artifacts.vocals.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(vocals_src, artifacts.vocals)
    shutil.copy2(no_vocals_src, artifacts.no_vocals)


def run_demucs_inprocess(input_audio: Path, artifacts: CoverArtifacts, separator: Any, device: str) -> None:
    """Separate vocals with an already-loaded Demucs separator."""
    from demucs.api import save_audio

    separator_device = device
    _move_demucs_separator(separator, separator_device, "Demucs stage")
    _, sources = separator.separate_audio_file(input_audio)
    if "vocals" not in sources:
        raise RuntimeError("Demucs did not return a vocals stem.")

    vocals = sources["vocals"]
    accompaniment_sources = [source for name, source in sources.items() if name != "vocals"]
    if not accompaniment_sources:
        raise RuntimeError("Demucs did not return accompaniment stems.")

    # no_vocals = accompaniment_sources[0].clone()
    # for source in accompaniment_sources[1:]:
    #     no_vocals += source

    artifacts.vocals.parent.mkdir(parents=True, exist_ok=True)
    save_audio(vocals, str(artifacts.vocals), samplerate=separator.samplerate)
    # save_audio(no_vocals, str(artifacts.no_vocals), samplerate=separator.samplerate)
    del vocals, accompaniment_sources, sources


def resolve_ddsp_assets(model_ckpt: Path, repair: bool = True) -> None:
    import yaml

    if not model_ckpt.exists():
        raise FileNotFoundError(f"DDSP model checkpoint was not found: {model_ckpt}")
    config_path = model_ckpt.with_name("config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"DDSP config.yaml was not found next to model: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    encoder_ckpt = (DDSP_ROOT / config["data"]["encoder_ckpt"]).resolve()
    vocoder_ckpt = (DDSP_ROOT / config["vocoder"]["ckpt"]).resolve()
    vocoder_config = vocoder_ckpt.parent / "config.json"
    if not encoder_ckpt.exists():
        raise FileNotFoundError(f"DDSP encoder checkpoint was not found: {encoder_ckpt}")

    if not vocoder_ckpt.exists() or not vocoder_config.exists():
        fallback_ckpt = DDSP_ROOT / "pretrain" / "nsf_hifigan" / "model" / "model.ckpt"
        fallback_config = DDSP_ROOT / "pretrain" / "nsf_hifigan" / "config.json"
        if not fallback_ckpt.exists() or not fallback_config.exists():
            raise FileNotFoundError("DDSP vocoder assets are missing and fallback assets were not found.")
        if not repair:
            raise FileNotFoundError(
                "DDSP vocoder assets are missing. Re-run without --dry-run to repair them from fallback assets."
            )
        vocoder_ckpt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fallback_ckpt, vocoder_ckpt)
        shutil.copy2(fallback_config, vocoder_config)


def build_ddsp_command(
    input_vocals: Path,
    output_vocals: Path,
    model_ckpt: Path,
    device: str,
    spk_id: int,
    key: int,
    pitch_extractor: str,
    cache_dir: Path,
) -> CommandSpec:
    return CommandSpec(
        cwd=DDSP_ROOT,
        argv=[
            sys.executable,
            "main_reflow.py",
            "-m",
            os.path.relpath(model_ckpt, DDSP_ROOT),
            "-i",
            os.path.relpath(input_vocals, DDSP_ROOT),
            "-o",
            os.path.relpath(output_vocals, DDSP_ROOT),
            "-id",
            str(spk_id),
            "-k",
            str(key),
            "-pe",
            pitch_extractor,
            "-d",
            device,
        ],
        env={
            "NUMBA_CACHE_DIR": str(cache_dir / "numba"),
            "MPLCONFIGDIR": str(cache_dir / "matplotlib"),
        },
    )


def build_pop2piano_command(
    input_audio: Path,
    output_midi: Path,
    model: str,
    composer: str,
    device: str,
    max_length: int,
    cache_dir: Path,
) -> CommandSpec:
    env = {"NUMBA_CACHE_DIR": str(cache_dir / "numba")}
    if Path(model).exists():
        env.update({
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HTTP_PROXY": "",
            "HTTPS_PROXY": "",
            "http_proxy": "",
            "https_proxy": "",
        })

    return CommandSpec(
        cwd=REPO_ROOT,
        argv=[
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_pop2piano.py"),
            "--input",
            str(input_audio),
            "--output",
            str(output_midi),
            "--model",
            model,
            "--composer",
            composer,
            "--device",
            device,
            "--max-length",
            str(max_length),
        ],
        env=env,
    )


def _resolve_pop2piano_device(device: str) -> str:
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported Pop2Piano device: {device}")

    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def run_pop2piano_inprocess(
    input_audio: Path,
    output_midi: Path,
    model: str,
    composer: str,
    device: str,
    max_length: int,
    beat_checkpoint: Path | None = None,
    preloaded_components: dict[str, Any] | None = None,
    sampling_rate: int = 44100,
) -> dict[str, Any]:
    """Run Pop2Piano without spawning a second Python interpreter."""
    import librosa
    import numpy as np
    import torch
    import types

    timings: dict[str, Any] = {}
    total_start = time.perf_counter()
    resolved_device = (
        str(preloaded_components["device"])
        if preloaded_components is not None and "device" in preloaded_components
        else _resolve_pop2piano_device(device)
    )

    start = time.perf_counter()
    if preloaded_components is None:
        from transformers import Pop2PianoForConditionalGeneration, Pop2PianoProcessor

        processor = Pop2PianoProcessor.from_pretrained(model)
        pop2piano_model = Pop2PianoForConditionalGeneration.from_pretrained(model)
    else:
        processor = preloaded_components["processor"]
        pop2piano_model = preloaded_components["model"]

    pop2piano_model = _move_torch_object(pop2piano_model, resolved_device)
    pop2piano_model.eval()
    timings["load_model"] = round(time.perf_counter() - start, 3)
    print(f'[T] load_model: {timings["load_model"]}s')

    start = time.perf_counter()
    audio, sr = librosa.load(input_audio, sr=sampling_rate, mono=True)
    timings["load_audio"] = round(time.perf_counter() - start, 3)
    print(f'[T] load_audio: {timings["load_audio"]}s')

    start = time.perf_counter()
    if preloaded_components is not None and "beat_tracker" in preloaded_components:
        beat_tracker = preloaded_components["beat_tracker"]
    else:
        if beat_checkpoint is None:
            raise ValueError("Pop2Piano Audio2Beats checkpoint is required for in-process inference.")
        from beat_this.inference import Audio2Beats

        beat_tracker = Audio2Beats(checkpoint_path=str(beat_checkpoint), device=resolved_device)
    beats, _downbeats = beat_tracker(audio, sr)
    beat_times = np.asarray(beats, dtype=np.float32)
    beat_intervals = np.diff(beat_times)
    bpm = float(60.0 / np.median(beat_intervals)) if beat_intervals.size else 120.0
    timings["rhythm_extract"] = round(time.perf_counter() - start, 3)
    print(f'[T] rhythm_extract (beat_this, device={resolved_device}): {timings["rhythm_extract"]}s')

    def patched_extract_rhythm(self, audio):
        return bpm, beat_times, 1.0, [], beat_intervals

    processor.feature_extractor.extract_rhythm = types.MethodType(
        patched_extract_rhythm,
        processor.feature_extractor,
    )

    start = time.perf_counter()
    inputs = processor(
        audio=audio,
        sampling_rate=sr,
        return_tensors="pt",
    )
    inputs = {key: value.to(resolved_device) if hasattr(value, "to") else value for key, value in inputs.items()}
    timings["preprocess"] = round(time.perf_counter() - start, 3)
    print(f'[T] preprocess: {timings["preprocess"]}s')

    start = time.perf_counter()
    with torch.inference_mode():
        generated = pop2piano_model.generate(
            input_features=inputs["input_features"],
            attention_mask=inputs.get("attention_mask"),
            composer=composer,
            max_length=max_length,
        )
    timings["model_generate"] = round(time.perf_counter() - start, 3)
    print(f'[T] model_generate (max_length={max_length}): {timings["model_generate"]}s')

    start = time.perf_counter()
    generated = generated.cpu()
    decoder_inputs = {
        key: value.cpu() if hasattr(value, "cpu") else value
        for key, value in inputs.items()
    }
    del inputs
    decoded = processor.batch_decode(generated, feature_extractor_output=decoder_inputs)
    midi = decoded["pretty_midi_objects"][0]
    timings["decode"] = round(time.perf_counter() - start, 3)
    print(f'[T] decode: {timings["decode"]}s')

    start = time.perf_counter()
    output_midi.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(output_midi))
    timings["save_midi"] = round(time.perf_counter() - start, 3)
    print(f'[T] save_midi: {timings["save_midi"]}s')
    print(output_midi)

    timings["device"] = resolved_device
    timings["total"] = round(time.perf_counter() - total_start, 3)

    timing_path = output_midi.parent / "pop2piano_timings.json"
    timing_path.write_text(json.dumps(timings, indent=2), encoding="utf-8")
    print(f"[T] Timing report saved to {timing_path}")
    return timings


def build_mix_command(
    vocals_audio: Path,
    piano_audio: Path,
    output_audio: Path,
    vocals_volume: float,
    piano_volume: float,
) -> CommandSpec:
    for name, value in (("vocals_volume", vocals_volume), ("piano_volume", piano_volume)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")
        if value < 0:
            raise ValueError(f"{name} must be greater than or equal to 0.")

    filter_complex = (
        f"[0:a]aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo,volume={vocals_volume}[vocals];"
        f"[1:a]aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo,volume={piano_volume}[piano];"
        "[vocals][piano]amix=inputs=2:duration=longest:dropout_transition=0[mix]"
    )
    return CommandSpec(
        cwd=REPO_ROOT,
        argv=[
            "ffmpeg",
            "-y",
            "-i",
            str(vocals_audio),
            "-i",
            str(piano_audio),
            "-filter_complex",
            filter_complex,
            "-map",
            "[mix]",
            "-ar",
            "44100",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(output_audio),
        ],
        env=_ffmpeg_runtime_env(),
    )


def mix_cover_audio(
    vocals_audio: Path,
    piano_audio: Path,
    output_audio: Path,
    vocals_volume: float = 1.0,
    piano_volume: float = 1.0,
    dry_run: bool = False,
) -> Path:
    if not dry_run:
        output_audio.parent.mkdir(parents=True, exist_ok=True)
    _run(
        build_mix_command(
            vocals_audio=vocals_audio,
            piano_audio=piano_audio,
            output_audio=output_audio,
            vocals_volume=vocals_volume,
            piano_volume=piano_volume,
        ),
        dry_run=dry_run,
    )
    return output_audio


def run_ai_piano_cover(
    input_audio: Path,
    output_root: Path,
    device: str = "cuda",
    spk_id: int = 1,
    key: int = 0,
    pre_pitch_shift: float = 0.0,
    pitch_extractor: str = "rmvpe",
    vocals_volume: float = 1.0,
    piano_volume: float = 1.0,
    ddsp_model_ckpt: Path = DEFAULT_DDSP_MODEL,
    pop2piano_model: str = DEFAULT_POP2PIANO_MODEL,
    pop2piano_composer: str = "composer1",
    pop2piano_device: str = "cpu",
    pop2piano_max_length: int = 256,
    pop2piano_beat_checkpoint: Path | None = DEFAULT_POP2PIANO_BEAT_CHECKPOINT,
    demucs_separator: Any | None = None,
    ddsp_runtime: Any | None = None,
    pop2piano_components: dict[str, Any] | None = None,
    soundfont: Path = DEFAULT_SOUNDFONT,
    fluidsynth_bin: Path = DEFAULT_FLUIDSYNTH_BIN,
    fluidsynth_lib_dir: Path = DEFAULT_FLUIDSYNTH_LIB_DIR,
    use_parallel_stages: bool = USE_PARALLEL_COVER_STAGES,
    dry_run: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> CoverRunResult:
    input_audio = input_audio.resolve()
    output_root = output_root.resolve()
    ddsp_model_ckpt = ddsp_model_ckpt.resolve()
    artifacts = build_cover_artifacts(input_audio, output_root)
    raw_demucs_root = output_root / "demucs_raw"
    runtime_cache_dir = output_root / "cache"
    stage_timings: dict[str, float] = {}
    total_start = time.perf_counter()

    if not input_audio.exists():
        raise FileNotFoundError(f"Input audio was not found: {input_audio}")
    if not DEMUCS_ROOT.exists():
        raise FileNotFoundError(f"Demucs vendor directory was not found: {DEMUCS_ROOT}")
    if not DDSP_ROOT.exists():
        raise FileNotFoundError(f"DDSP-SVC vendor directory was not found: {DDSP_ROOT}")

    resolve_ddsp_assets(ddsp_model_ckpt, repair=not dry_run)

    if not dry_run:
        for path in [
            artifacts.preprocessed_audio,
            artifacts.vocals,
            artifacts.ddsp_vocals,
            artifacts.piano_midi,
            artifacts.piano_wav,
            artifacts.final_mix,
            raw_demucs_root / ".keep",
            runtime_cache_dir / "numba" / ".keep",
            runtime_cache_dir / "matplotlib" / ".keep",
        ]:
            path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input audio: {input_audio}")
    print(f"Output root: {output_root}")
    print(f"Final mix: {artifacts.final_mix}")
    print("")
    _emit_progress(progress_callback, "preprocess", 0, "Starting audio preprocessing")

    print("Stage 0: preprocess audio")
    if pre_pitch_shift != 0.0:
        print(f"Preprocess pitch shift: {pre_pitch_shift:+g} semitones")
    stage_start = time.perf_counter()
    _run(
        build_preprocess_command(
            input_audio,
            artifacts.preprocessed_audio,
            pre_pitch_shift=pre_pitch_shift,
        ),
        dry_run=dry_run,
    )
    stage_timings["preprocess"] = round(time.perf_counter() - stage_start, 3)
    _emit_progress(progress_callback, "separate_vocals", 10, "Preprocessed audio")

    pop2piano_input_audio = artifacts.preprocessed_audio if pre_pitch_shift != 0.0 else input_audio

    def run_vocal_branch() -> dict[str, float]:
        branch_timings: dict[str, float] = {}
        stream_device = (
            _cuda_branch_device(
                _component_device(ddsp_runtime, None),
                _component_device(demucs_separator, None),
                device,
            )
            if use_parallel_stages and (demucs_separator is not None or ddsp_runtime is not None)
            else None
        )
        with _cuda_stream_context(stream_device, "vocal"):
            print("")
            print("Stage 1: separate vocals with Demucs")
            _emit_progress(progress_callback, "separate_vocals", 15, "Separating vocals")
            stage_start = time.perf_counter()
            if demucs_separator is not None:
                if dry_run:
                    print(f"Would run Demucs in-process with preloaded model -> {artifacts.vocals}")
                else:
                    run_demucs_inprocess(artifacts.preprocessed_audio, artifacts, demucs_separator, device)
                    print(f"Vocals: {artifacts.vocals}")
                    # print(f"No vocals: {artifacts.no_vocals}")
            else:
                _run(build_demucs_command(artifacts.preprocessed_audio, raw_demucs_root, device), dry_run=dry_run)
                if not dry_run:
                    copy_demucs_outputs(artifacts.preprocessed_audio, raw_demucs_root, artifacts)
                    print(f"Vocals: {artifacts.vocals}")
                    print(f"No vocals: {artifacts.no_vocals}")
            branch_timings["separate_vocals"] = round(time.perf_counter() - stage_start, 3)
            _emit_progress(progress_callback, "separate_vocals", 25, "Separated vocals and accompaniment")
            if not use_parallel_stages:
                _cleanup_cuda_memory("Demucs stage")

            print("")
            print("Stage 2: convert vocals with DDSP-SVC")
            _emit_progress(progress_callback, "voice_conversion", 30, "Starting voice conversion")
            stage_start = time.perf_counter()
            if ddsp_runtime is not None:
                if dry_run:
                    print(f"Would run DDSP-SVC in-process with preloaded model -> {artifacts.ddsp_vocals}")
                else:
                    from src.ddsp_inprocess import run_ddsp_inprocess

                    run_ddsp_inprocess(
                        input_vocals=artifacts.vocals,
                        output_vocals=artifacts.ddsp_vocals,
                        runtime=ddsp_runtime,
                        spk_id=spk_id,
                        key=key,
                        pitch_extractor=pitch_extractor,
                        cache_dir=runtime_cache_dir,
                    )
            else:
                _run(
                    build_ddsp_command(
                        input_vocals=artifacts.vocals,
                        output_vocals=artifacts.ddsp_vocals,
                        model_ckpt=ddsp_model_ckpt,
                        device=device,
                        spk_id=spk_id,
                        key=key,
                        pitch_extractor=pitch_extractor,
                        cache_dir=runtime_cache_dir,
                    ),
                    dry_run=dry_run,
                )
            branch_timings["voice_conversion"] = round(time.perf_counter() - stage_start, 3)
            _emit_progress(progress_callback, "voice_conversion", 50, "Converted vocals")
            if not use_parallel_stages:
                _cleanup_cuda_memory("DDSP stage")
        return branch_timings

    def run_piano_branch() -> dict[str, float]:
        branch_timings: dict[str, float] = {}
        stream_device = (
            _cuda_branch_device(_component_device(pop2piano_components, None), pop2piano_device)
            if use_parallel_stages and pop2piano_components is not None
            else None
        )
        with _cuda_stream_context(stream_device, "piano"):
            print("")
            print("Stage 3: generate Pop2Piano accompaniment")
            _emit_progress(progress_callback, "piano_cover", 55, "Starting piano cover generation")
            stage_start = time.perf_counter()
            if pop2piano_components is not None:
                if dry_run:
                    print(f"Would run Pop2Piano in-process with preloaded model -> {artifacts.piano_midi}")
                else:
                    run_pop2piano_inprocess(
                        input_audio=pop2piano_input_audio,
                        output_midi=artifacts.piano_midi,
                        model=pop2piano_model,
                        composer=pop2piano_composer,
                        device=pop2piano_device,
                        max_length=pop2piano_max_length,
                        beat_checkpoint=pop2piano_beat_checkpoint,
                        preloaded_components=pop2piano_components,
                    )
            else:
                _run(
                    build_pop2piano_command(
                        input_audio=pop2piano_input_audio,
                        output_midi=artifacts.piano_midi,
                        model=pop2piano_model,
                        composer=pop2piano_composer,
                        device=pop2piano_device,
                        max_length=pop2piano_max_length,
                        cache_dir=runtime_cache_dir,
                    ),
                    dry_run=dry_run,
                )
            branch_timings["piano_cover"] = round(time.perf_counter() - stage_start, 3)
            _emit_progress(progress_callback, "piano_cover", 72, "Generated piano MIDI")
            if not use_parallel_stages:
                _cleanup_cuda_memory("Pop2Piano stage")

            print("")
            print("Stage 4: render piano MIDI to WAV")
            _emit_progress(progress_callback, "piano_cover", 75, "Rendering piano audio")
            stage_start = time.perf_counter()
            if not dry_run:
                from src.render import render_midi_to_wav

                render_midi_to_wav(
                    input_midi=artifacts.piano_midi,
                    output_wav=artifacts.piano_wav,
                    soundfont_path=soundfont,
                    fluidsynth_bin=fluidsynth_bin,
                    fluidsynth_lib_dir=fluidsynth_lib_dir,
                )
                print(f"Piano WAV: {artifacts.piano_wav}")
            else:
                print(f"Would render {artifacts.piano_midi} -> {artifacts.piano_wav}")
            branch_timings["render_piano"] = round(time.perf_counter() - stage_start, 3)
            _emit_progress(progress_callback, "piano_cover", 85, "Rendered piano audio")
        return branch_timings

    if use_parallel_stages:
        print("")
        print("Running Stage 1+2 and Stage 3+4 in parallel")
        _emit_progress(progress_callback, "separate_vocals", 12, "Starting vocal and piano branches in parallel")
        parallel_start = time.perf_counter()
        branch_errors: list[tuple[str, Exception]] = []
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="cover-branch") as executor:
            futures = {
                executor.submit(run_vocal_branch): "vocal",
                executor.submit(run_piano_branch): "piano",
            }
            for future in as_completed(futures):
                branch_name = futures[future]
                try:
                    stage_timings.update(future.result())
                except Exception as exc:  # noqa: BLE001 - re-raised with branch context
                    branch_errors.append((branch_name, exc))

        stage_timings["parallel_branches"] = round(time.perf_counter() - parallel_start, 3)
        if branch_errors:
            branch_name, exc = branch_errors[0]
            _cleanup_cuda_memory(f"{branch_name} branch failure")
            raise RuntimeError(f"{branch_name} branch failed: {exc}") from exc
    else:
        print("")
        print("Running Stage 1+2 and Stage 3+4 sequentially")
        _emit_progress(progress_callback, "separate_vocals", 12, "Starting vocal and piano branches sequentially")
        branch_start = time.perf_counter()
        try:
            stage_timings.update(run_vocal_branch())
            _cleanup_cuda_memory("vocal branch")
            stage_timings.update(run_piano_branch())
            _cleanup_cuda_memory("piano branch")
        except Exception as exc:  # noqa: BLE001 - re-raised with branch context
            _cleanup_cuda_memory("sequential branch failure")
            raise RuntimeError(f"sequential branch failed: {exc}") from exc
        stage_timings["sequential_branches"] = round(time.perf_counter() - branch_start, 3)

    try:
        print("")
        print("Stage 5: mix DDSP vocals with piano accompaniment")
        _emit_progress(progress_callback, "merge", 90, "Starting final mix")
        stage_start = time.perf_counter()
        mix_cover_audio(
            vocals_audio=artifacts.ddsp_vocals,
            piano_audio=artifacts.piano_wav,
            output_audio=artifacts.final_mix,
            vocals_volume=vocals_volume,
            piano_volume=piano_volume,
            dry_run=dry_run,
        )
        stage_timings["merge"] = round(time.perf_counter() - stage_start, 3)
        stage_timings["total"] = round(time.perf_counter() - total_start, 3)
        if not dry_run:
            timings_path = output_root / "timings.json"
            timings_path.write_text(json.dumps(stage_timings, indent=2), encoding="utf-8")
        _emit_progress(progress_callback, "completed", 100, "Cover pipeline completed")

        return CoverRunResult(artifacts=artifacts, stage_timings=stage_timings)
    finally:
        _cleanup_cuda_memory("cover pipeline")
