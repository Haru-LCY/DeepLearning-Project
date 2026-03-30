"""CLI entry point for audio preprocessing."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.audio_preprocess import preprocess_audio


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for audio preprocessing."""
    parser = argparse.ArgumentParser(
        description="Convert one audio file into a canonical mono WAV."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input audio file path.")
    parser.add_argument("--output", required=True, type=Path, help="Output WAV file path.")
    parser.add_argument(
        "--sr",
        default=22050,
        type=int,
        help="Target sample rate for the output WAV.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the preprocessing stage from the command line."""
    args = parse_args()
    preprocess_audio(input_path=args.input, output_path=args.output, sr=args.sr)


if __name__ == "__main__":
    main()
