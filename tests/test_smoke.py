"""Tiny smoke checks for the starter repository layout."""

from __future__ import annotations

from pathlib import Path


def test_expected_paths_exist() -> None:
    """Ensure the initial project skeleton is present."""
    repo_root = Path(__file__).resolve().parents[1]

    assert (repo_root / "README.md").exists()
    assert (repo_root / "configs" / "default.yaml").exists()
    assert (repo_root / "assets" / "output" / "convert").exists()
    assert (repo_root / "assets" / "output" / "clean").exists()
    assert (repo_root / "assets" / "output" / "raw").exists()
    assert (repo_root / "scripts" / "run_cleanup_midi.py").exists()
    assert (repo_root / "scripts" / "run_render.py").exists()
    assert (repo_root / "scripts" / "run_preprocess_audio.py").exists()
    assert (repo_root / "scripts" / "run_transcription.py").exists()
    assert (repo_root / "src" / "audio_preprocess.py").exists()
    assert (repo_root / "src" / "render.py").exists()
    assert (repo_root / "src" / "transcription.py").exists()
