"""Run Hugging Face Pop2Piano inference on one audio file."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
import tempfile

import librosa
import torch
from transformers import Pop2PianoForConditionalGeneration, Pop2PianoProcessor


DEFAULT_LOCAL_MODEL = Path(
    "/home/fit/alex/WORK/pretrained_models/models--sweetcocoa--pop2piano/"
    "snapshots/142e8ed35614bcf77a3515b979e48ed528342349"
)
DEFAULT_MODEL = str(DEFAULT_LOCAL_MODEL) if DEFAULT_LOCAL_MODEL.exists() else "sweetcocoa/pop2piano"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate piano-cover MIDI with Pop2Piano.")
    parser.add_argument("--input", required=True, type=Path, help="Input audio file, e.g. ringo.mp3.")
    parser.add_argument("--output", required=True, type=Path, help="Output MIDI path.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model id or local model directory.",
    )
    parser.add_argument(
        "--composer",
        default="composer1",
        help="Composer token, usually composer1..composer21 for sweetcocoa/pop2piano.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Inference device.",
    )
    parser.add_argument("--sampling-rate", default=44100, type=int, help="Audio loading sampling rate.")
    parser.add_argument("--max-length", default=256, type=int, help="Maximum generated token length.")
    return parser.parse_args()


def main() -> None:
    pop2piano_timings = {}
    t_total_start = time.perf_counter()
    
    args = parse_args()
    os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "pianoformer_cover_numba"))
    if Path(args.model).exists():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ.pop(name, None)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    t0 = time.perf_counter()
    processor = Pop2PianoProcessor.from_pretrained(args.model)
    model = Pop2PianoForConditionalGeneration.from_pretrained(args.model).to(device)
    model.eval()
    pop2piano_timings['load_model'] = round(time.perf_counter() - t0, 3)
    print(f'[T] load_model: {pop2piano_timings["load_model"]}s')

    t0 = time.perf_counter()
    audio, sr = librosa.load(args.input, sr=args.sampling_rate, mono=True)
    pop2piano_timings['load_audio'] = round(time.perf_counter() - t0, 3)
    print(f'[T] load_audio: {pop2piano_timings["load_audio"]}s')
    
    t0 = time.perf_counter()
    inputs = processor(
        audio=audio,
        sampling_rate=sr,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    pop2piano_timings['preprocess'] = round(time.perf_counter() - t0, 3)
    print(f'[T] preprocess: {pop2piano_timings["preprocess"]}s')

    t0 = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(
            input_features=inputs["input_features"],
            attention_mask=inputs.get("attention_mask"),
            composer=args.composer,
            max_length=args.max_length,
        )
    pop2piano_timings['model_generate'] = round(time.perf_counter() - t0, 3)
    print(f'[T] model_generate (max_length={args.max_length}): {pop2piano_timings["model_generate"]}s')

    t0 = time.perf_counter()
    generated = generated.cpu()
    decoder_inputs = {
        key: value.cpu() if hasattr(value, "cpu") else value
        for key, value in inputs.items()
    }
    decoded = processor.batch_decode(generated, feature_extractor_output=decoder_inputs)
    midi = decoded["pretty_midi_objects"][0]
    pop2piano_timings['decode'] = round(time.perf_counter() - t0, 3)
    print(f'[T] decode: {pop2piano_timings["decode"]}s')
    
    t0 = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(args.output))
    pop2piano_timings['save_midi'] = round(time.perf_counter() - t0, 3)
    print(f'[T] save_midi: {pop2piano_timings["save_midi"]}s')
    print(args.output)

    pop2piano_timings['device'] = device
    pop2piano_timings['total'] = round(time.perf_counter() - t_total_start, 3)
    
    timing_path = args.output.parent / 'pop2piano_timings.json'
    timing_path.write_text(json.dumps(pop2piano_timings, indent=2), encoding='utf-8')
    print(f'[T] Timing report saved to {timing_path}')


if __name__ == "__main__":
    main()
