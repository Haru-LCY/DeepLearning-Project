"""CLI entry point for standalone PianistTransformer rendering."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.expressive_render import build_expressive_output_paths, run_pianist_transformer
from src.render import render_midi_to_wav


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the expressive rendering stage."""
    parser = argparse.ArgumentParser(
        description="Render one cleaned MIDI file with PianistTransformer."
    )
    parser.add_argument("--input-midi", required=True, type=Path, help="Input cleaned MIDI path.")
    parser.add_argument(
        "--output-root",
        default=Path("assets/output"),
        type=Path,
        help="Root directory for generated expressive artifacts.",
    )
    parser.add_argument(
        "--pt-python",
        default=None,
        type=Path,
        help="Optional path to the Python executable used for PianistTransformer. Defaults to the current interpreter.",
    )
    parser.add_argument(
        "--pt-model-dir",
        default=Path("PianistTransformer/models/sft"),
        type=Path,
        help="Path to the local PianistTransformer model directory.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Execution device for PianistTransformer.",
    )
    parser.add_argument(
        "--temperature",
        default=1.0,
        type=float,
        help="Sampling temperature for generation.",
    )
    parser.add_argument(
        "--top-p",
        default=0.95,
        type=float,
        help="Top-p sampling parameter for generation.",
    )
    parser.add_argument(
        "--max-tempo",
        default=300,
        type=int,
        help="Maximum tempo used when mapping the expressive MIDI back to score time.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Also render the mapped expressive MIDI to WAV.",
    )
    parser.add_argument(
        "--soundfont",
        default=None,
        type=Path,
        help="Required when --render is set. Path to a .sf2 SoundFont file.",
    )
    parser.add_argument(
        "--fluidsynth-bin",
        default=None,
        type=Path,
        help="Optional path to the FluidSynth executable if it is not on PATH.",
    )
    parser.add_argument(
        "--fluidsynth-lib-dir",
        default=None,
        type=Path,
        help="Optional library directory for a non-system FluidSynth binary.",
    )
    return parser.parse_args()


def main() -> None:
    """Run standalone expressive rendering and optionally render to WAV."""
    args = parse_args()
    outputs = build_expressive_output_paths(args.output_root, args.input_midi)
    raw_midi, mapped_midi = run_pianist_transformer(
        input_midi=args.input_midi,
        output_raw_midi=outputs["raw_expressive_midi"],
        output_mapped_midi=outputs["mapped_expressive_midi"],
        pt_python=args.pt_python,
        pt_model_dir=args.pt_model_dir,
        device=args.device,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tempo=args.max_tempo,
    )

    print(f"raw_expressive_midi: {raw_midi}")
    print(f"mapped_expressive_midi: {mapped_midi}")

    if args.render:
        if args.soundfont is None:
            raise ValueError("--soundfont is required when --render is set.")

        render_stem = mapped_midi.stem.removesuffix("_pt_mapped")
        rendered_wav = args.output_root / "rendered" / f"{render_stem}_pt.wav"
        rendered_wav = render_midi_to_wav(
            input_midi=mapped_midi,
            output_wav=rendered_wav,
            soundfont_path=args.soundfont,
            fluidsynth_bin=args.fluidsynth_bin,
            fluidsynth_lib_dir=args.fluidsynth_lib_dir,
        )
        print(f"rendered_wav: {rendered_wav}")


if __name__ == "__main__":
    main()
