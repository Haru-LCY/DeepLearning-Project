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
    f0_extractor: Any | None


@dataclass
class _EncodedSegment:
    index: int
    start_frame: int
    frames: int
    units: Any
    f0: Any
    volume: Any


@dataclass
class _SegmentMeta:
    index: int
    start_frame: int
    frames: int


@dataclass
class _RenderedBatch:
    segments: list[_SegmentMeta]
    output: Any


def _segment_batch_size(configured_value: int | None = None, default: int = 4) -> int:
    if configured_value is not None:
        if configured_value < 1:
            print(f"Configured DDSP segment batch size must be >= 1; using {default}")
            configured_value = default
        default = configured_value

    raw_value = os.environ.get("DDSP_SEGMENT_BATCH_SIZE")
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError:
        print(f"Invalid DDSP_SEGMENT_BATCH_SIZE={raw_value!r}; using {default}")
        return default

    if value < 1:
        print(f"DDSP_SEGMENT_BATCH_SIZE must be >= 1; using {default}")
        return default
    return value


def _length_sorted_batches(segments: list[_EncodedSegment], batch_size: int) -> Iterator[list[_EncodedSegment]]:
    if batch_size <= 1:
        for segment in segments:
            yield [segment]
        return

    sorted_segments = sorted(segments, key=lambda segment: segment.frames)
    for start in range(0, len(sorted_segments), batch_size):
        yield sorted_segments[start : start + batch_size]


def _move_torch_object(obj: Any, device: str) -> Any:
    if obj is None:
        return None

    mover = getattr(obj, "to", None)
    if callable(mover):
        moved = mover(device)
        if moved is not None:
            obj = moved
    return obj


def _move_attr(obj: Any, attr: str, device: str) -> None:
    if obj is None or not hasattr(obj, attr):
        return
    setattr(obj, attr, _move_torch_object(getattr(obj, attr), device))


def _move_mapping_values(mapping: Any, device: str) -> None:
    if not isinstance(mapping, dict):
        return
    for key, value in list(mapping.items()):
        mapping[key] = _move_torch_object(value, device)


def _move_stft_cache(stft: Any, device: str) -> None:
    if stft is None:
        return
    _move_mapping_values(getattr(stft, "mel_basis", None), device)
    _move_mapping_values(getattr(stft, "hann_window", None), device)


def _move_ddsp_vocoder(vocoder: Any, device: str) -> None:
    if vocoder is None:
        return

    vocoder.device = device
    _move_mapping_values(getattr(vocoder, "resample_kernel", None), device)

    inner_vocoder = getattr(vocoder, "vocoder", None)
    if inner_vocoder is not None:
        inner_vocoder.device = device
        _move_torch_object(inner_vocoder, device)
        _move_attr(inner_vocoder, "model", device)
        _move_stft_cache(getattr(inner_vocoder, "stft", None), device)


def _move_units_encoder(units_encoder: Any, device: str) -> None:
    if units_encoder is None:
        return

    units_encoder.device = device
    _move_mapping_values(getattr(units_encoder, "resample_kernel", None), device)

    model = getattr(units_encoder, "model", None)
    if model is None:
        return

    _move_torch_object(model, device)
    if hasattr(model, "device"):
        model.device = device
    for attr in ("hubert", "model", "proj", "fcpe"):
        _move_attr(model, attr, device)


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
            _move_ddsp_vocoder(vocoder, device)
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
            _move_units_encoder(units_encoder, device)
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
        f0_extractor = preload_ddsp_pitch_extractor(pitch_extractor, device)

    return DDSPRuntime(
        model_ckpt=model_ckpt,
        device=device,
        model=model,
        vocoder=vocoder,
        args=args,
        units_encoder=units_encoder,
        pitch_extractor=pitch_extractor,
        f0_extractor=f0_extractor,
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
    segment_batch_size: int | None = None,
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
            f0_runtime = runtime.f0_extractor if pitch_extractor == runtime.pitch_extractor else None
            if f0_runtime is None:
                f0_runtime = F0_Extractor(
                    pitch_extractor,
                    sample_rate,
                    hop_size,
                    float(f0_min),
                    float(f0_max),
                )
            else:
                f0_runtime.sample_rate = sample_rate
                f0_runtime.hop_size = hop_size
                f0_runtime.f0_min = float(f0_min)
                f0_runtime.f0_max = float(f0_max)
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

        start = time.perf_counter()
        segments = split(audio, sample_rate, hop_size)
        timings["split_audio"] = round(time.perf_counter() - start, 3)
        print(f'[T] split_audio: {timings["split_audio"]}s')
        n_segments = len(segments)
        print("Cut the input audio into " + str(n_segments) + " slices")
        timings["n_segments"] = n_segments

        cum_units_encode = 0.0
        cum_model_forward = 0.0
        cum_vocoder_infer = 0.0
        cum_cpu_transfer = 0.0
        cum_assemble = 0.0
        requested_segment_batch_size = _segment_batch_size(segment_batch_size)
        if bool(getattr(args.model, "use_attention", False)) and requested_segment_batch_size > 1:
            print("DDSP segment batching disabled because model.use_attention has no padding mask")
            requested_segment_batch_size = 1
        segment_batch_size = min(requested_segment_batch_size, max(n_segments, 1))
        timings["segment_batch_size"] = segment_batch_size
        block_size = int(args.data.block_size)

        def pad_time_batch(tensors: list[Any], target_frames: int) -> Any:
            padded_tensors = []
            for tensor in tensors:
                pad_frames = target_frames - tensor.size(1)
                if pad_frames > 0:
                    tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_frames))
                padded_tensors.append(tensor)
            return torch.cat(padded_tensors, dim=0)

        def batch_mask_for_output(
            batch_segments: list[_EncodedSegment],
            output_samples: int,
            output_dims: int,
        ) -> Any:
            mask_slices = []
            for encoded_segment in batch_segments:
                mask_slice = mask[
                    :,
                    encoded_segment.start_frame
                    * block_size : (encoded_segment.start_frame + encoded_segment.frames)
                    * block_size,
                ]
                if mask_slice.size(-1) < output_samples:
                    mask_slice = torch.nn.functional.pad(mask_slice, (0, output_samples - mask_slice.size(-1)))
                elif mask_slice.size(-1) > output_samples:
                    mask_slice = mask_slice[:, :output_samples]
                mask_slices.append(mask_slice)
            batch_mask = torch.cat(mask_slices, dim=0)
            while batch_mask.dim() < output_dims:
                batch_mask = batch_mask.unsqueeze(1)
            return batch_mask

        def render_segment_batch(batch_segments: list[_EncodedSegment]) -> None:
            nonlocal cum_model_forward, cum_vocoder_infer
            max_frames = max(encoded_segment.frames for encoded_segment in batch_segments)
            batch_units = pad_time_batch([encoded_segment.units for encoded_segment in batch_segments], max_frames)
            batch_f0 = pad_time_batch([encoded_segment.f0 for encoded_segment in batch_segments], max_frames)
            batch_volume = pad_time_batch(
                [encoded_segment.volume for encoded_segment in batch_segments],
                max_frames,
            )
            batch_speaker_id = speaker_id.expand(len(batch_segments), speaker_id.size(1))
            # Vendor Unit2Control adds aug_shift to [B, T, C], so keep a singleton time axis.
            batch_formant_shift = formant_shift.expand(len(batch_segments), formant_shift.size(1)).unsqueeze(1)

            model_start = time.perf_counter()
            batch_mel = runtime.model(
                batch_units,
                batch_f0 / vocal_register_factor,
                batch_volume,
                spk_id=batch_speaker_id,
                spk_mix_dict=parsed_spk_mix_dict,
                aug_shift=batch_formant_shift,
                vocoder=runtime.vocoder,
                infer_step=resolved_infer_step,
                method=resolved_method,
                t_start=resolved_t_start,
                use_tqdm=False,
            )
            cum_model_forward += time.perf_counter() - model_start

            vocoder_start = time.perf_counter()
            batch_output_tensor = runtime.vocoder.infer(batch_mel, batch_f0)
            cum_vocoder_infer += time.perf_counter() - vocoder_start

            batch_output_tensor *= batch_mask_for_output(
                batch_segments,
                batch_output_tensor.size(-1),
                batch_output_tensor.dim(),
            )
            rendered_batches.append(
                _RenderedBatch(
                    segments=[
                        _SegmentMeta(
                            index=encoded_segment.index,
                            start_frame=encoded_segment.start_frame,
                            frames=encoded_segment.frames,
                        )
                        for encoded_segment in batch_segments
                    ],
                    output=batch_output_tensor.detach(),
                )
            )
            del batch_units, batch_f0, batch_volume, batch_mel, batch_output_tensor

        seg_loop_start = time.perf_counter()
        encoded_segments: list[_EncodedSegment] = []
        rendered_batches: list[_RenderedBatch] = []
        with torch.inference_mode():
            for index, segment in enumerate(tqdm(segments, desc="encode segments")):
                start_frame = segment[0]
                seg_input = torch.from_numpy(segment[1]).float().unsqueeze(0).to(device)

                units_start = time.perf_counter()
                seg_units = runtime.units_encoder.encode(seg_input, sample_rate, hop_size)
                cum_units_encode += time.perf_counter() - units_start

                seg_frames = seg_units.size(1)
                seg_f0 = f0[:, start_frame : start_frame + seg_frames, :]
                seg_volume = volume[:, start_frame : start_frame + seg_frames, :]
                encoded_segments.append(
                    _EncodedSegment(
                        index=index,
                        start_frame=start_frame,
                        frames=seg_frames,
                        units=seg_units,
                        f0=seg_f0,
                        volume=seg_volume,
                    )
                )
                del seg_input

            segment_batches = list(_length_sorted_batches(encoded_segments, segment_batch_size))
            timings["n_segment_batches"] = len(segment_batches)
            print(
                "Rendering "
                + str(n_segments)
                + " segments in "
                + str(len(segment_batches))
                + " DDSP batches"
            )

            batch_segments: list[_EncodedSegment] = []
            for batch_segments in tqdm(segment_batches, desc="render segment batches"):
                try:
                    render_segment_batch(batch_segments)
                except RuntimeError as exc:
                    if len(batch_segments) == 1:
                        raise
                    print(
                        "DDSP segment batch render failed for "
                        + str(len(batch_segments))
                        + " segments; retrying individually: "
                        + str(exc)
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    for single_segment in batch_segments:
                        render_segment_batch([single_segment])

        del encoded_segments, segment_batches, batch_segments

        segment_outputs: list[np.ndarray | None] = [None] * n_segments
        transfer_start = time.perf_counter()
        for rendered_batch in rendered_batches:
            batch_output = rendered_batch.output.cpu().numpy()
            for batch_index, segment_meta in enumerate(rendered_batch.segments):
                true_samples = segment_meta.frames * block_size
                segment_output = np.asarray(batch_output[batch_index]).squeeze().reshape(-1)
                segment_outputs[segment_meta.index] = segment_output[:true_samples].astype(np.float32, copy=True)
        cum_cpu_transfer = time.perf_counter() - transfer_start

        assemble_start = time.perf_counter()
        result_array: np.ndarray | None = None
        result_chunks: list[np.ndarray] = []
        current_length = 0

        def materialize_result() -> np.ndarray:
            nonlocal result_array, result_chunks
            if result_array is None:
                if result_chunks:
                    result_array = np.concatenate(result_chunks).astype(np.float32, copy=False)
                else:
                    result_array = np.zeros(0, dtype=np.float32)
            elif result_chunks:
                result_array = np.concatenate([result_array, *result_chunks]).astype(np.float32, copy=False)
            result_chunks = []
            return result_array

        for segment_output, segment in zip(segment_outputs, segments):
            if segment_output is None:
                continue
            start_frame = segment[0]
            silent_length = round(start_frame * block_size) - current_length
            if silent_length >= 0:
                if silent_length > 0:
                    result_chunks.append(np.zeros(silent_length, dtype=np.float32))
                result_chunks.append(segment_output)
            else:
                result_array = cross_fade(
                    materialize_result(),
                    segment_output,
                    current_length + silent_length,
                ).astype(np.float32, copy=False)
            current_length = current_length + silent_length + len(segment_output)
        result = materialize_result()
        cum_assemble = time.perf_counter() - assemble_start
        cum_seg_loop = time.perf_counter() - seg_loop_start

        del f0, volume, mask, speaker_id, formant_shift, segments, audio, rendered_batches, segment_outputs

        timings["seg_loop_total"] = round(cum_seg_loop, 3)
        timings["units_encode_total"] = round(cum_units_encode, 3)
        timings["model_forward_total"] = round(cum_model_forward, 3)
        timings["vocoder_infer_total"] = round(cum_vocoder_infer, 3)
        timings["cpu_transfer_total"] = round(cum_cpu_transfer, 3)
        timings["assemble_total"] = round(cum_assemble, 3)
        timings["overhead"] = round(cum_seg_loop - cum_units_encode - cum_model_forward - cum_vocoder_infer, 3)
        print(f'[T] seg_loop_total ({n_segments} segs): {timings["seg_loop_total"]}s')
        print(f'[T]   DDSP segment batch size: {segment_batch_size}')
        print(f'[T]   units_encode: {timings["units_encode_total"]}s')
        print(f'[T]   model_forward (DDSP+Reflow): {timings["model_forward_total"]}s')
        print(f'[T]   vocoder_infer: {timings["vocoder_infer_total"]}s')
        print(f'[T]   cpu_transfer: {timings["cpu_transfer_total"]}s')
        print(f'[T]   assemble: {timings["assemble_total"]}s')
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
