"""Subprocess orchestration for optional PianistTransformer rendering."""

from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PT_ROOT = REPO_ROOT / "PianistTransformer"
PT_WRAPPER = REPO_ROOT / "scripts" / "run_pianist_transformer_stage.py"
DEFAULT_PT_MODEL_DIR = PT_ROOT / "models" / "sft"


def derive_output_stem(input_midi: Path) -> str:
    """Derive a stable artifact stem from an input MIDI filename."""
    stem = input_midi.stem
    if stem.endswith("_clean"):
        return stem[: -len("_clean")]
    return stem


def build_expressive_output_paths(output_root: Path, input_midi: Path) -> dict[str, Path]:
    """Build explicit output paths for expressive rendering artifacts."""
    stem = derive_output_stem(input_midi)
    raw_dir = output_root / "expressive" / "raw"
    mapped_dir = output_root / "expressive" / "mapped"
    return {
        "raw_expressive_midi": raw_dir / f"{stem}_pt_raw.mid",
        "mapped_expressive_midi": mapped_dir / f"{stem}_pt_mapped.mid",
    }


def validate_model_dir(model_dir: Path) -> None:
    """Ensure the PT model directory exists and contains the expected files."""
    required_files = ("config.json", "generation_config.json", "model.safetensors")
    if not model_dir.exists():
        raise FileNotFoundError(f"PianistTransformer model directory not found: {model_dir}")
    if not model_dir.is_dir():
        raise ValueError(f"PianistTransformer model path is not a directory: {model_dir}")
    missing_files = [name for name in required_files if not (model_dir / name).exists()]
    if missing_files:
        raise FileNotFoundError(
            "PianistTransformer model directory is missing required files: "
            + ", ".join(missing_files)
        )


def run_pianist_transformer(
    input_midi: Path,
    output_raw_midi: Path,
    output_mapped_midi: Path,
    pt_python: Path | None = None,
    pt_model_dir: Path | None = None,
    device: str = "auto",
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_tempo: int = 300,
) -> tuple[Path, Path]:
    """Run PianistTransformer in a subprocess and return output artifact paths."""
    if not input_midi.exists():
        raise FileNotFoundError(f"Input MIDI file not found: {input_midi}")
    if not input_midi.is_file():
        raise ValueError(f"Input MIDI path is not a file: {input_midi}")

    resolved_python = Path(sys.executable) if pt_python is None else pt_python
    if not resolved_python.exists():
        raise FileNotFoundError(f"PianistTransformer Python executable not found: {resolved_python}")
    if not resolved_python.is_file():
        raise ValueError(f"PianistTransformer Python path is not a file: {resolved_python}")

    if not PT_WRAPPER.exists():
        raise FileNotFoundError(f"PianistTransformer wrapper script not found: {PT_WRAPPER}")

    resolved_model_dir = DEFAULT_PT_MODEL_DIR if pt_model_dir is None else pt_model_dir
    validate_model_dir(resolved_model_dir)

    output_raw_midi.parent.mkdir(parents=True, exist_ok=True)
    output_mapped_midi.parent.mkdir(parents=True, exist_ok=True)

    command = [
        str(resolved_python),
        str(PT_WRAPPER),
        "--input-midi",
        str(input_midi.resolve()),
        "--output-raw-midi",
        str(output_raw_midi.resolve()),
        "--output-mapped-midi",
        str(output_mapped_midi.resolve()),
        "--model-dir",
        str(resolved_model_dir.resolve()),
        "--device",
        device,
        "--temperature",
        str(temperature),
        "--top-p",
        str(top_p),
        "--max-tempo",
        str(max_tempo),
    ]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(PT_ROOT)
        if not existing_pythonpath
        else f"{PT_ROOT}:{existing_pythonpath}"
    )

    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            cwd=PT_ROOT,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "No stderr captured."
        raise RuntimeError(f"PianistTransformer subprocess failed: {stderr}") from exc

    if not output_raw_midi.exists():
        raise FileNotFoundError(
            f"PianistTransformer finished but raw MIDI was not created: {output_raw_midi}"
        )
    if not output_mapped_midi.exists():
        raise FileNotFoundError(
            "PianistTransformer finished but mapped MIDI was not created: "
            f"{output_mapped_midi}"
        )

    return output_raw_midi, output_mapped_midi
