"""Shared helper placeholders for configuration and paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    # TODO: Parse YAML config once implementation begins.
    raise NotImplementedError("Config loading is not implemented yet.")
