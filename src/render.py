"""Utilities for rendering MIDI to WAV with midi2audio and FluidSynth."""

from __future__ import annotations

import os
from pathlib import Path
import shutil

from midi2audio import FluidSynth


def render_midi_to_wav(
    input_midi: Path,
    output_wav: Path,
    soundfont_path: Path,
    fluidsynth_bin: Path | None = None,
    fluidsynth_lib_dir: Path | None = None,
) -> Path:
    """Render one MIDI file to one WAV file using a required SoundFont."""
    if not input_midi.exists():
        raise FileNotFoundError(f"Input MIDI file not found: {input_midi}")

    if not input_midi.is_file():
        raise ValueError(f"Input path is not a file: {input_midi}")

    if not soundfont_path.exists():
        raise FileNotFoundError(f"SoundFont file not found: {soundfont_path}")

    if not soundfont_path.is_file():
        raise ValueError(f"SoundFont path is not a file: {soundfont_path}")

    if output_wav.suffix.lower() != ".wav":
        raise ValueError("Output path must end with '.wav'.")

    resolved_fluidsynth = fluidsynth_bin
    if resolved_fluidsynth is None:
        executable = shutil.which("fluidsynth")
        if executable is None:
            raise RuntimeError(
                "Could not find 'fluidsynth' in PATH. Install it on Linux or pass --fluidsynth-bin."
            )
        resolved_fluidsynth = Path(executable)

    if not resolved_fluidsynth.exists():
        raise FileNotFoundError(
            f"FluidSynth executable not found: {resolved_fluidsynth}"
        )

    if not resolved_fluidsynth.is_file():
        raise ValueError(
            f"FluidSynth executable path is not a file: {resolved_fluidsynth}"
        )

    resolved_lib_dir = None
    if fluidsynth_lib_dir is not None:
        resolved_lib_dir = fluidsynth_lib_dir.resolve()
        if not resolved_lib_dir.exists():
            raise FileNotFoundError(
                f"FluidSynth library directory not found: {resolved_lib_dir}"
            )
        if not resolved_lib_dir.is_dir():
            raise ValueError(
                f"FluidSynth library path is not a directory: {resolved_lib_dir}"
            )

    output_wav.parent.mkdir(parents=True, exist_ok=True)

    original_path = os.environ.get("PATH", "")
    original_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")

    try:
        resolved_bin_dir = str(resolved_fluidsynth.resolve().parent)
        os.environ["PATH"] = (
            resolved_bin_dir if not original_path else f"{resolved_bin_dir}:{original_path}"
        )
        if resolved_lib_dir is not None:
            os.environ["LD_LIBRARY_PATH"] = (
                str(resolved_lib_dir)
                if not original_ld_library_path
                else f"{resolved_lib_dir}:{original_ld_library_path}"
            )
        synthesizer = FluidSynth(sound_font=str(soundfont_path))
        synthesizer.midi_to_audio(str(input_midi), str(output_wav))
    except Exception as exc:
        raise RuntimeError(f"Failed to render WAV from {input_midi}: {exc}") from exc
    finally:
        os.environ["PATH"] = original_path
        if original_ld_library_path:
            os.environ["LD_LIBRARY_PATH"] = original_ld_library_path
        else:
            os.environ.pop("LD_LIBRARY_PATH", None)

    if not output_wav.exists():
        raise FileNotFoundError(f"Expected rendered WAV was not created: {output_wav}")

    return output_wav
