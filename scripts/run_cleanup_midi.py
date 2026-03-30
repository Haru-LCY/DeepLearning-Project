"""CLI entry point for the standalone MIDI cleanup stage."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.midi_cleanup import clean_midi


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for MIDI cleanup."""
    parser = argparse.ArgumentParser(
        description="Remove short notes from one MIDI file."
    )
    parser.add_argument("--input-midi", required=True, type=Path, help="Input MIDI path.")
    parser.add_argument(
        "--output-midi",
        required=True,
        type=Path,
        help="Output cleaned MIDI path.",
    )
    parser.add_argument(
        "--min-note-duration",
        default=0.05,
        type=float,
        help="Minimum note duration in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the standalone MIDI cleanup stage."""
    args = parse_args()
    output_midi = clean_midi(
        input_midi=args.input_midi,
        output_midi=args.output_midi,
        min_note_duration=args.min_note_duration,
    )
    print(output_midi)


if __name__ == "__main__":
    main()
