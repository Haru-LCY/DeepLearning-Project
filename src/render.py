"""Utilities for rendering MIDI to WAV with midi2audio and FluidSynth."""

from __future__ import annotations

import os
from pathlib import Path
import shutil

from midi2audio import FluidSynth


def resolve_soundfont_path(soundfont_path: Path) -> Path:
    """Resolve a SoundFont file, accepting either a .sf2 file or a directory containing one."""
    if not soundfont_path.exists():
        raise FileNotFoundError(f"SoundFont file not found: {soundfont_path}")

    if soundfont_path.is_file():
        return soundfont_path

    if soundfont_path.is_dir():
        candidates = sorted(soundfont_path.rglob("*.sf2"))
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise ValueError(f"SoundFont directory does not contain any .sf2 files: {soundfont_path}")
        choices = ", ".join(str(path) for path in candidates)
        raise ValueError(f"SoundFont directory contains multiple .sf2 files; pass one explicitly: {choices}")

    raise ValueError(f"SoundFont path is neither a file nor a directory: {soundfont_path}")


def resolve_fluidsynth_bin(fluidsynth_bin: Path | None) -> Path:
    """Resolve FluidSynth, accepting either an executable path or a directory containing it."""
    if fluidsynth_bin is None:
        executable = shutil.which("fluidsynth")
        if executable is None:
            raise RuntimeError(
                "Could not find 'fluidsynth' in PATH. Install it on Linux or pass --fluidsynth-bin."
            )
        return Path(executable)

    if not fluidsynth_bin.exists():
        raise FileNotFoundError(f"FluidSynth executable not found: {fluidsynth_bin}")

    if fluidsynth_bin.is_file():
        return fluidsynth_bin

    if fluidsynth_bin.is_dir():
        direct_candidate = fluidsynth_bin / "bin" / "fluidsynth"
        if direct_candidate.is_file():
            return direct_candidate
        candidates = sorted(path for path in fluidsynth_bin.rglob("fluidsynth") if path.is_file())
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise ValueError(f"FluidSynth directory does not contain a fluidsynth executable: {fluidsynth_bin}")
        choices = ", ".join(str(path) for path in candidates)
        raise ValueError(f"FluidSynth directory contains multiple executables; pass one explicitly: {choices}")

    raise ValueError(f"FluidSynth path is neither a file nor a directory: {fluidsynth_bin}")


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

    soundfont_path = resolve_soundfont_path(soundfont_path)

    if output_wav.suffix.lower() != ".wav":
        raise ValueError("Output path must end with '.wav'.")

    resolved_fluidsynth = resolve_fluidsynth_bin(fluidsynth_bin)

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
