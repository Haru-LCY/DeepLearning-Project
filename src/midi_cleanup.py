"""Utilities for minimal MIDI post-processing."""

from __future__ import annotations

from pathlib import Path

import pretty_midi


def clean_midi(input_midi: Path, output_midi: Path, min_note_duration: float) -> Path:
    """Remove notes shorter than the configured minimum duration and save a new MIDI."""
    if min_note_duration < 0:
        raise ValueError("Minimum note duration must be non-negative.")

    if not input_midi.exists():
        raise FileNotFoundError(f"Input MIDI file not found: {input_midi}")

    if not input_midi.is_file():
        raise ValueError(f"Input path is not a file: {input_midi}")

    output_midi.parent.mkdir(parents=True, exist_ok=True)

    try:
        midi = pretty_midi.PrettyMIDI(str(input_midi))
    except Exception as exc:
        raise RuntimeError(f"Failed to load MIDI from {input_midi}: {exc}") from exc

    for instrument in midi.instruments:
        filtered_notes = []
        for note in instrument.notes:
            note_duration = note.end - note.start
            if note_duration >= min_note_duration:
                filtered_notes.append(note)
        instrument.notes = filtered_notes

    try:
        midi.write(str(output_midi))
    except Exception as exc:
        raise RuntimeError(f"Failed to write cleaned MIDI to {output_midi}: {exc}") from exc

    return output_midi
