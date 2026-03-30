"""CLI entry point for the MIDI-to-WAV rendering step."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.render import render_midi_to_wav


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for MIDI rendering."""
    parser = argparse.ArgumentParser(
        description="Render one cleaned MIDI file to one WAV file."
    )
    parser.add_argument("--input-midi", required=True, type=Path, help="Input MIDI path.")
    parser.add_argument("--output-wav", required=True, type=Path, help="Output WAV path.")
    parser.add_argument(
        "--soundfont",
        required=True,
        type=Path,
        help="Explicit path to a .sf2 SoundFont file.",
    )
    parser.add_argument(
        "--fluidsynth-bin",
        type=Path,
        default=None,
        help="Optional path to the FluidSynth executable if it is not on PATH.",
    )
    parser.add_argument(
        "--fluidsynth-lib-dir",
        type=Path,
        default=None,
        help="Optional library directory needed by a non-system FluidSynth binary.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the rendering stage."""
    args = parse_args()
    output_wav = render_midi_to_wav(
        input_midi=args.input_midi,
        output_wav=args.output_wav,
        soundfont_path=args.soundfont,
        fluidsynth_bin=args.fluidsynth_bin,
        fluidsynth_lib_dir=args.fluidsynth_lib_dir,
    )
    print(output_wav)


if __name__ == "__main__":
    main()
