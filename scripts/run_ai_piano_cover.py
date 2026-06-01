"""Run AI vocal cover plus Pop2Piano piano accompaniment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import site
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PYTHONNOUSERSITE", "1")
user_site = site.getusersitepackages()
if user_site in sys.path:
    sys.path.remove(user_site)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai_cover import (
    DEFAULT_DDSP_MODEL,
    DEFAULT_FLUIDSYNTH_BIN,
    DEFAULT_FLUIDSYNTH_LIB_DIR,
    DEFAULT_POP2PIANO_MODEL,
    DEFAULT_SOUNDFONT,
    run_ai_piano_cover,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an AI cover vocal track and Pop2Piano accompaniment from one audio file."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input song audio path.")
    parser.add_argument(
        "--output-root",
        default=Path("assets/output/cover"),
        type=Path,
        help="Root directory for generated cover artifacts.",
    )
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"), help="Device for Demucs and DDSP-SVC.")
    parser.add_argument("--spk-id", default=1, type=int, help="DDSP-SVC speaker id.")
    parser.add_argument("--key", default=0, type=int, help="DDSP-SVC pitch shift in semitones.")
    parser.add_argument(
        "--pre-pitch-shift",
        default=0.0,
        type=float,
        help="Pitch shift applied during preprocessing before separation/conversion, in semitones.",
    )
    parser.add_argument("--pitch-extractor", default="rmvpe", help="DDSP-SVC pitch extractor.")
    parser.add_argument("--vocals-volume", default=1.0, type=float, help="Final mix vocal gain.")
    parser.add_argument("--piano-volume", default=1.0, type=float, help="Final mix piano gain.")
    parser.add_argument("--ddsp-model-ckpt", default=DEFAULT_DDSP_MODEL, type=Path, help="DDSP checkpoint path.")
    parser.add_argument("--pop2piano-model", default=DEFAULT_POP2PIANO_MODEL, help="Pop2Piano model id or path.")
    parser.add_argument("--pop2piano-composer", default="composer1", help="Pop2Piano composer token.")
    parser.add_argument(
        "--pop2piano-device",
        default="cpu",
        choices=("cpu", "cuda", "auto"),
        help="Pop2Piano inference device. CPU is the stable default.",
    )
    parser.add_argument("--pop2piano-max-length", default=256, type=int, help="Pop2Piano max generated token length.")
    parser.add_argument("--soundfont", default=DEFAULT_SOUNDFONT, type=Path, help="SoundFont used for piano rendering.")
    parser.add_argument(
        "--fluidsynth-bin",
        default=DEFAULT_FLUIDSYNTH_BIN,
        type=Path,
        help="FluidSynth executable path.",
    )
    parser.add_argument(
        "--fluidsynth-lib-dir",
        default=DEFAULT_FLUIDSYNTH_LIB_DIR,
        type=Path,
        help="FluidSynth runtime library directory.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_ai_piano_cover(
        input_audio=args.input,
        output_root=args.output_root,
        device=args.device,
        spk_id=args.spk_id,
        key=args.key,
        pre_pitch_shift=args.pre_pitch_shift,
        pitch_extractor=args.pitch_extractor,
        vocals_volume=args.vocals_volume,
        piano_volume=args.piano_volume,
        ddsp_model_ckpt=args.ddsp_model_ckpt,
        pop2piano_model=args.pop2piano_model,
        pop2piano_composer=args.pop2piano_composer,
        pop2piano_device=args.pop2piano_device,
        pop2piano_max_length=args.pop2piano_max_length,
        soundfont=args.soundfont,
        fluidsynth_bin=args.fluidsynth_bin,
        fluidsynth_lib_dir=args.fluidsynth_lib_dir,
        dry_run=args.dry_run,
    )
    artifacts = result.artifacts
    print("")
    print(f"preprocessed_audio: {artifacts.preprocessed_audio}")
    print(f"ddsp_vocals: {artifacts.ddsp_vocals}")
    print(f"piano_midi: {artifacts.piano_midi}")
    print(f"piano_wav: {artifacts.piano_wav}")
    print(f"final_mix: {artifacts.final_mix}")
    print(f"stage_timings: {result.stage_timings}")


if __name__ == "__main__":
    main()
