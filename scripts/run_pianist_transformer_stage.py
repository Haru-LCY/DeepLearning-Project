"""Internal CLI wrapper for one-file PianistTransformer rendering."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PT_ROOT = REPO_ROOT / "PianistTransformer"
DEFAULT_MODEL_DIR = PT_ROOT / "models" / "sft"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for one-file expressive rendering."""
    parser = argparse.ArgumentParser(
        description="Render one input MIDI into raw and mapped expressive MIDI outputs."
    )
    parser.add_argument("--input-midi", required=True, type=Path, help="Input score-like MIDI.")
    parser.add_argument(
        "--output-raw-midi",
        required=True,
        type=Path,
        help="Output path for the raw expressive performance MIDI.",
    )
    parser.add_argument(
        "--output-mapped-midi",
        required=True,
        type=Path,
        help="Output path for the mapped/editable expressive MIDI.",
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        type=Path,
        help="Path to the local PianistTransformer model directory.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Execution device. Defaults to auto-detect.",
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
    return parser.parse_args()


def resolve_device(device_arg: str, torch_module: object) -> str:
    """Resolve the concrete execution device from the CLI argument."""
    if device_arg == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment.")
    return device_arg


def validate_model_dir(model_dir: Path) -> None:
    """Ensure the local model directory contains the expected files."""
    required_files = ("config.json", "generation_config.json", "model.safetensors")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not model_dir.is_dir():
        raise ValueError(f"Model directory path is not a directory: {model_dir}")
    missing_files = [name for name in required_files if not (model_dir / name).exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Model directory is missing required files: {', '.join(missing_files)}"
        )


def main() -> None:
    """Run one-file expressive rendering and print generated artifact paths."""
    args = parse_args()

    try:
        import torch
        from miditoolkit import MidiFile

        sys.path.insert(0, str(PT_ROOT))
        from src.model.generate import batch_performance_render, map_midi
        from src.model.pianoformer import PianoT5Gemma
    except Exception as exc:
        raise RuntimeError(
            "Failed to import PianistTransformer runtime dependencies. "
            "Run this script inside the configured PianistTransformer environment."
        ) from exc

    if not args.input_midi.exists():
        raise FileNotFoundError(f"Input MIDI file not found: {args.input_midi}")
    if not args.input_midi.is_file():
        raise ValueError(f"Input MIDI path is not a file: {args.input_midi}")

    validate_model_dir(args.model_dir)

    args.output_raw_midi.parent.mkdir(parents=True, exist_ok=True)
    args.output_mapped_midi.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device, torch)
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = PianoT5Gemma.from_pretrained(str(args.model_dir), torch_dtype=torch_dtype)
    if device == "cuda":
        model = model.to(device)

    score_midi = MidiFile(str(args.input_midi))
    raw_output = batch_performance_render(
        model,
        [score_midi],
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
    )[0]
    raw_output.dump(str(args.output_raw_midi))

    mapped_output = map_midi(score_midi, raw_output, max_tempo=args.max_tempo)
    mapped_output.dump(str(args.output_mapped_midi))

    print(f"device: {device}")
    print(f"raw_midi: {args.output_raw_midi}")
    print(f"mapped_midi: {args.output_mapped_midi}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"PianistTransformer rendering failed: {exc}", file=sys.stderr)
        sys.exit(1)
