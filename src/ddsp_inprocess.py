"""In-process DDSP-SVC runtime used by the FastAPI backend."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from ast import literal_eval
from typing import Any, Iterator

import numpy as np

from src.ai_cover import DDSP_ROOT, VENDOR_ROOT


_DDSP_VOCODER_CACHE: dict[tuple[str, str, str], Any] = {}
_DDSP_UNITS_ENCODER_CACHE: dict[tuple[str, str, int, int, int, str], Any] = {}


@dataclass
class DDSPRuntime:
    model_ckpt: Path
    device: str
    model: Any
    vocoder: Any
    args: Any
    units_encoder: Any
    pitch_extractor: str


def _configure_imports() -> None:
    for path in (str(VENDOR_ROOT), str(DDSP_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


@contextmanager
def _ddsp_working_directory() -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(DDSP_ROOT)
    try:
        yield
    finally:
        os.chdir(previous)


def preload_ddsp_pitch_extractor(pitch_extractor: str, device: str) -> Any | None:
    """Prime vendor-global pitch extractor state on the requested device."""
    if pitch_extractor != "rmvpe":
        return None

    _configure_imports()
    with _ddsp_working_directory():
        from ddsp.vocoder import F0_Extractor

        extractor = F0_Extractor("rmvpe", sample_rate=44100, hop_size=512, f0_min=50, f0_max=1100)
        extractor.rmvpe.model = extractor.rmvpe.model.to(device)
        extractor.rmvpe.mel_extractor = extractor.rmvpe.mel_extractor.to(device)
        return extractor


def load_ddsp_runtime(model_ckpt: Path, device: str, pitch_extractor: str) -> DDSPRuntime:
    """Load one DDSP role, its encoder, and its vocoder into process memory."""
    _configure_imports()
    model_ckpt = model_ckpt.resolve()
    with _ddsp_working_directory():
        from ddsp.vocoder import Units_Encoder
        from nsf_hifigan.models import load_model as load_hifigan_model
        from reflow.vocoder import load_model_vocoder

        model, vocoder, args = load_model_vocoder(str(model_ckpt), device=device)
        model.eval()

        # DDSP-SVC lazily loads HifiGAN on first vocoder inference. Force it at
        # backend startup so the job path does not hit disk/model load again.
        hifigan = vocoder.vocoder
        vocoder_key = (args.vocoder.type, str(Path(hifigan.model_path).resolve()), device)
        if vocoder_key in _DDSP_VOCODER_CACHE:
            vocoder = _DDSP_VOCODER_CACHE[vocoder_key]
        else:
            if getattr(hifigan, "model", None) is None:
                hifigan.model, hifigan.h = load_hifigan_model(hifigan.model_path, device=hifigan.device)
            hifigan.model.eval()
            _DDSP_VOCODER_CACHE[vocoder_key] = vocoder

        if args.data.encoder == "cnhubertsoftfish":
            cnhubertsoft_gate = args.data.cnhubertsoft_gate
        else:
            cnhubertsoft_gate = 10
        units_encoder_key = (
            args.data.encoder,
            str(Path(args.data.encoder_ckpt).resolve()),
            int(args.data.encoder_sample_rate),
            int(args.data.encoder_hop_size),
            int(cnhubertsoft_gate),
            device,
        )
        if units_encoder_key in _DDSP_UNITS_ENCODER_CACHE:
            units_encoder = _DDSP_UNITS_ENCODER_CACHE[units_encoder_key]
        else:
            units_encoder = Units_Encoder(
                args.data.encoder,
                args.data.encoder_ckpt,
                args.data.encoder_sample_rate,
                args.data.encoder_hop_size,
                cnhubertsoft_gate=cnhubertsoft_gate,
                device=device,
            )
            _DDSP_UNITS_ENCODER_CACHE[units_encoder_key] = units_encoder
        preload_ddsp_pitch_extractor(pitch_extractor, device)

    return DDSPRuntime(
        model_ckpt=model_ckpt,
        device=device,
        model=model,
        vocoder=vocoder,
        args=args,
        units_encoder=units_encoder,
        pitch_extractor=pitch_extractor,
    )


def run_ddsp_inprocess(
    input_vocals: Path,
    output_vocals: Path,
    runtime: DDSPRuntime,
    spk_id: int,
    key: int,
    pitch_extractor: str,
    cache_dir: Path,
    formant_shift_key: float = 0.0,
    vocal_register_shift_key: float = 0.0,
    f0_min: float = 50.0,
    f0_max: float = 1100.0,
    threhold: float = -60.0,
    infer_step: str = "auto",
    method: str = "auto",
    t_start: str = "auto",
    spk_mix_dict: str = "None",
) -> dict[str, Any]:
    """Run DDSP-SVC inference with a preloaded runtime."""
    _configure_imports()

    input_vocals = input_vocals.resolve()
    output_vocals = output_vocals.resolve()
    cache_dir = cache_dir.resolve()
    output_vocals.parent.mkdir(parents=True, exist_ok=True)

    with _ddsp_working_directory():
        import librosa
        import soundfile as sf
        import torch
        from ddsp.core import upsample
        from ddsp.vocoder import F0_Extractor, Volume_Extractor
        from main_reflow import cross_fade, split
        from tqdm import tqdm

        timings: dict[str, Any] = {}
        total_start = time.perf_counter()
        device = runtime.device
        args = runtime.args

        start = time.perf_counter()
        audio, sample_rate = librosa.load(input_vocals, sr=None)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        timings["load_audio"] = round(time.perf_counter() - start, 3)
        print(f'[T] load_audio: {timings["load_audio"]}s')

        hop_size = args.data.block_size * sample_rate / args.data.sampling_rate
        win_size = args.data.volume_smooth_size * sample_rate / args.data.sampling_rate

        with input_vocals.open("rb") as handle:
            md5_hash = hashlib.md5(handle.read()).hexdigest()
        print("MD5: " + md5_hash)

        f0_cache_dir = cache_dir / "ddsp_f0"
        f0_cache_dir.mkdir(parents=True, exist_ok=True)
        f0_cache_path = f0_cache_dir / f"{pitch_extractor}_{hop_size}_{f0_min}_{f0_max}_{md5_hash}.npy"

        start = time.perf_counter()
        is_cache_available = f0_cache_path.exists()
        if is_cache_available:
            print("Loading pitch curves for input audio from cache directory...")
            f0 = np.load(f0_cache_path, allow_pickle=False)
        else:
            print("Pitch extractor type: " + pitch_extractor)
            f0_runtime = F0_Extractor(
                pitch_extractor,
                sample_rate,
                hop_size,
                float(f0_min),
                float(f0_max),
            )
            print("Extracting the pitch curve of the input audio...")
            f0 = f0_runtime.extract(audio, uv_interp=True, device=device)
            np.save(f0_cache_path, f0, allow_pickle=False)
        timings["f0_extract"] = round(time.perf_counter() - start, 3)
        timings["f0_cached"] = is_cache_available
        print(f'[T] f0_extract (cached={is_cache_available}): {timings["f0_extract"]}s')

        f0 = torch.from_numpy(f0).float().to(device).unsqueeze(-1).unsqueeze(0)
        f0 = f0 * 2 ** (float(key) / 12)
        formant_shift = torch.from_numpy(np.array([[float(formant_shift_key)]])).float().to(device)

        if runtime.vocoder.vocoder.h.pc_aug:
            vocal_register_factor = 2 ** (float(vocal_register_shift_key) / 12)
        else:
            print("Vocal register shift is not supported for current vocoder!")
            vocal_register_factor = 1

        print("Extracting the volume envelope of the input audio...")
        start = time.perf_counter()
        volume_extractor = Volume_Extractor(hop_size, win_size)
        volume = volume_extractor.extract(audio)
        timings["volume_extract"] = round(time.perf_counter() - start, 3)
        print(f'[T] volume_extract: {timings["volume_extract"]}s')

        mask = (volume > 10 ** (float(threhold) / 20)).astype("float")
        mask = torch.from_numpy(mask).float().to(device).unsqueeze(-1).unsqueeze(0)
        mask = upsample(mask, args.data.block_size).squeeze(-1)
        volume = torch.from_numpy(volume).float().to(device).unsqueeze(-1).unsqueeze(0)

        parsed_spk_mix_dict = literal_eval(spk_mix_dict)
        speaker_id = torch.LongTensor(np.array([[int(spk_id)]])).to(device)
        if parsed_spk_mix_dict is not None:
            print("Mix-speaker mode")
        else:
            print("Speaker ID: " + str(int(spk_id)))

        resolved_method = args.infer.method if method == "auto" else method
        resolved_infer_step = args.infer.infer_step if infer_step == "auto" else int(infer_step)

        if t_start == "auto":
            resolved_t_start = float(args.model.t_start) if args.model.t_start is not None else 0.0
        else:
            resolved_t_start = float(t_start)
            if args.model.t_start is not None and resolved_t_start < args.model.t_start:
                resolved_t_start = args.model.t_start

        if resolved_infer_step > 0:
            print("Sampling method: " + resolved_method)
            print("infer step: " + str(resolved_infer_step))
            print("t_start: " + str(resolved_t_start))
        elif resolved_infer_step < 0:
            raise ValueError("infer_step cannot be negative")

        result = np.zeros(0)
        current_length = 0
        start = time.perf_counter()
        segments = split(audio, sample_rate, hop_size)
        timings["split_audio"] = round(time.perf_counter() - start, 3)
        print(f'[T] split_audio: {timings["split_audio"]}s')
        print("Cut the input audio into " + str(len(segments)) + " slices")
        timings["n_segments"] = len(segments)

        cum_units_encode = 0.0
        cum_model_forward = 0.0
        cum_vocoder_infer = 0.0
        cum_seg_loop = 0.0

        with torch.no_grad():
            for segment in tqdm(segments):
                segment_start = time.perf_counter()
                start_frame = segment[0]
                seg_input = torch.from_numpy(segment[1]).float().unsqueeze(0).to(device)

                units_start = time.perf_counter()
                seg_units = runtime.units_encoder.encode(seg_input, sample_rate, hop_size)
                cum_units_encode += time.perf_counter() - units_start

                seg_f0 = f0[:, start_frame : start_frame + seg_units.size(1), :]
                seg_volume = volume[:, start_frame : start_frame + seg_units.size(1), :]

                model_start = time.perf_counter()
                seg_mel = runtime.model(
                    seg_units,
                    seg_f0 / vocal_register_factor,
                    seg_volume,
                    spk_id=speaker_id,
                    spk_mix_dict=parsed_spk_mix_dict,
                    aug_shift=formant_shift,
                    vocoder=runtime.vocoder,
                    infer_step=resolved_infer_step,
                    method=resolved_method,
                    t_start=resolved_t_start,
                )
                cum_model_forward += time.perf_counter() - model_start

                vocoder_start = time.perf_counter()
                seg_output = runtime.vocoder.infer(seg_mel, seg_f0)
                cum_vocoder_infer += time.perf_counter() - vocoder_start

                seg_output *= mask[
                    :,
                    start_frame * args.data.block_size : (start_frame + seg_units.size(1)) * args.data.block_size,
                ]
                seg_output = seg_output.squeeze().cpu().numpy()

                silent_length = round(start_frame * args.data.block_size) - current_length
                if silent_length >= 0:
                    result = np.append(result, np.zeros(silent_length))
                    result = np.append(result, seg_output)
                else:
                    result = cross_fade(result, seg_output, current_length + silent_length)
                current_length = current_length + silent_length + len(seg_output)
                cum_seg_loop += time.perf_counter() - segment_start

        timings["seg_loop_total"] = round(cum_seg_loop, 3)
        timings["units_encode_total"] = round(cum_units_encode, 3)
        timings["model_forward_total"] = round(cum_model_forward, 3)
        timings["vocoder_infer_total"] = round(cum_vocoder_infer, 3)
        timings["overhead"] = round(cum_seg_loop - cum_units_encode - cum_model_forward - cum_vocoder_infer, 3)
        print(f'[T] seg_loop_total ({len(segments)} segs): {timings["seg_loop_total"]}s')
        print(f'[T]   units_encode: {timings["units_encode_total"]}s')
        print(f'[T]   model_forward (DDSP+Reflow): {timings["model_forward_total"]}s')
        print(f'[T]   vocoder_infer: {timings["vocoder_infer_total"]}s')
        print(f'[T]   overhead (mask/copy/fade): {timings["overhead"]}s')

        start = time.perf_counter()
        sf.write(str(output_vocals), result, args.data.sampling_rate)
        timings["save_output"] = round(time.perf_counter() - start, 3)
        print(f'[T] save_output: {timings["save_output"]}s')

        timings["total"] = round(time.perf_counter() - total_start, 3)
        timing_path = output_vocals.parent / "ddsp_timings.json"
        timing_path.write_text(json.dumps(timings, indent=2), encoding="utf-8")
        print(f"[T] Timing report saved to {timing_path}")
        return timings
