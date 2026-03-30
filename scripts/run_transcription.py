"""CLI entry point for the audio-to-MIDI transcription step."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transcription import transcribe_audio


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for transcription."""
    parser = argparse.ArgumentParser(
        description="Transcribe one preprocessed audio file to MIDI with basic-pitch."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to one input audio file, typically under assets/output/convert.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("assets/output/raw"),
        type=Path,
        help="Directory for raw MIDI outputs.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the transcription stage."""
    args = parse_args()
    output_midi = transcribe_audio(input_audio=args.input, output_dir=args.output_dir)
    print(output_midi)


if __name__ == "__main__":
    main()
