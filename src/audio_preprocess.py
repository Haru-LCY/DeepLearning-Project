"""Utilities for canonicalizing input audio before transcription."""

from __future__ import annotations

from pathlib import Path

import librosa
import soundfile as sf


def preprocess_audio(input_path: Path, output_path: Path, sr: int = 22050) -> Path:
    """Convert one audio file into a mono WAV file at the target sample rate."""
    if sr <= 0:
        raise ValueError("Sample rate must be a positive integer.")

    if not input_path.exists():
        raise FileNotFoundError(f"Input audio file not found: {input_path}")

    if not input_path.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")

    if output_path.suffix.lower() != ".wav":
        raise ValueError("Output path must end with '.wav'.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        audio, sample_rate = librosa.load(input_path, sr=sr, mono=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to load audio from {input_path}: {exc}") from exc

    if audio.size == 0:
        raise ValueError(f"Loaded audio is empty: {input_path}")

    try:
        sf.write(output_path, audio, sample_rate, format="WAV")
    except Exception as exc:
        raise RuntimeError(f"Failed to write WAV to {output_path}: {exc}") from exc

    return output_path
