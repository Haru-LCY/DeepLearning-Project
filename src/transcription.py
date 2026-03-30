"""Utilities for audio-to-MIDI transcription via basic-pitch."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess


def transcribe_audio(input_audio: Path, output_dir: Path) -> Path:
    """Transcribe one audio file to MIDI using the basic-pitch CLI."""
    if not input_audio.exists():
        raise FileNotFoundError(f"Input audio file not found: {input_audio}")

    if not input_audio.is_file():
        raise ValueError(f"Input path is not a file: {input_audio}")

    basic_pitch_executable = shutil.which("basic-pitch")
    if basic_pitch_executable is None:
        raise RuntimeError(
            "Could not find 'basic-pitch' in PATH. Activate the 'dl' environment first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        basic_pitch_executable,
        "--model-serialization",
        "tflite",
        "--save-midi",
        str(output_dir),
        str(input_audio),
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "No stderr captured."
        raise RuntimeError(f"basic-pitch transcription failed: {stderr}") from exc

    expected_output = output_dir / f"{input_audio.stem}_basic_pitch.mid"
    if expected_output.exists():
        return expected_output

    midi_candidates = sorted(output_dir.glob(f"{input_audio.stem}*.mid"))
    if len(midi_candidates) == 1:
        return midi_candidates[0]

    raise FileNotFoundError(
        f"Transcription finished but no MIDI file was found in {output_dir}"
    )
