"""CLI entry point for the end-to-end MVP pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.audio_preprocess import preprocess_audio
from src.midi_cleanup import clean_midi
from src.render import render_midi_to_wav
from src.transcription import transcribe_audio


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the full MVP pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the full MVP pipeline from input audio to rendered WAV."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input audio file path.")
    parser.add_argument(
        "--output-root",
        default=Path("assets/output"),
        type=Path,
        help="Root directory for all generated outputs.",
    )
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
        help="Optional library directory for a non-system FluidSynth binary.",
    )
    parser.add_argument(
        "--sr",
        default=22050,
        type=int,
        help="Target sample rate for preprocessing.",
    )
    parser.add_argument(
        "--min-note-duration",
        default=0.05,
        type=float,
        help="Minimum note duration in seconds for MIDI cleanup.",
    )
    return parser.parse_args()


def build_output_paths(output_root: Path, input_audio: Path) -> dict[str, Path]:
    """Build explicit output paths for each stage of the pipeline."""
    stem = input_audio.stem
    convert_dir = output_root / "convert"
    raw_dir = output_root / "raw"
    clean_dir = output_root / "clean"
    rendered_dir = output_root / "rendered"

    return {
        "preprocessed_audio": convert_dir / f"{stem}.wav",
        "raw_midi_dir": raw_dir,
        "cleaned_midi": clean_dir / f"{stem}_clean.mid",
        "rendered_wav": rendered_dir / f"{stem}.wav",
    }


def configure_logging() -> None:
    """Configure simple console logging for pipeline progress."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def main() -> None:
    """Run preprocessing, transcription, cleanup, and rendering in sequence."""
    configure_logging()
    args = parse_args()
    outputs = build_output_paths(output_root=args.output_root, input_audio=args.input)

    LOGGER.info("Starting MVP pipeline")
    LOGGER.info("Input audio: %s", args.input)
    LOGGER.info("Output root: %s", args.output_root)

    with tqdm(total=4, desc="Pipeline", unit="stage") as progress:
        progress.set_description("Preprocessing audio")
        LOGGER.info(
            "Preprocessing audio to %s at %d Hz",
            outputs["preprocessed_audio"],
            args.sr,
        )
        preprocessed_audio = preprocess_audio(
            input_path=args.input,
            output_path=outputs["preprocessed_audio"],
            sr=args.sr,
        )
        progress.update(1)

        progress.set_description("Transcribing to MIDI")
        LOGGER.info(
            "Running transcription on %s into %s",
            preprocessed_audio,
            outputs["raw_midi_dir"],
        )
        raw_midi = transcribe_audio(
            input_audio=preprocessed_audio,
            output_dir=outputs["raw_midi_dir"],
        )
        progress.update(1)

        progress.set_description("Cleaning MIDI")
        LOGGER.info(
            "Cleaning MIDI %s with min note duration %.3f s",
            raw_midi,
            args.min_note_duration,
        )
        cleaned_midi = clean_midi(
            input_midi=raw_midi,
            output_midi=outputs["cleaned_midi"],
            min_note_duration=args.min_note_duration,
        )
        progress.update(1)

        progress.set_description("Rendering WAV")
        LOGGER.info(
            "Rendering %s to %s using soundfont %s",
            cleaned_midi,
            outputs["rendered_wav"],
            args.soundfont,
        )
        rendered_wav = render_midi_to_wav(
            input_midi=cleaned_midi,
            output_wav=outputs["rendered_wav"],
            soundfont_path=args.soundfont,
            fluidsynth_bin=args.fluidsynth_bin,
            fluidsynth_lib_dir=args.fluidsynth_lib_dir,
        )
        progress.update(1)

    LOGGER.info("Pipeline finished successfully")

    print(f"preprocessed_audio: {preprocessed_audio}")
    print(f"raw_midi: {raw_midi}")
    print(f"cleaned_midi: {cleaned_midi}")
    print(f"rendered_wav: {rendered_wav}")


if __name__ == "__main__":
    main()
