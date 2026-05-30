"""Backend configuration loading."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import yaml

from src.ai_cover import (
    DEFAULT_FLUIDSYNTH_BIN,
    DEFAULT_FLUIDSYNTH_LIB_DIR,
    DEFAULT_POP2PIANO_MODEL,
    DEFAULT_SOUNDFONT,
    REPO_ROOT,
)


@dataclass(frozen=True)
class RoleConfig:
    id: str
    name: str
    ddsp_model_ckpt: Path
    spk_id: int = 1
    default_pre_pitch_shift: int = 0
    avatar: str | None = None


@dataclass(frozen=True)
class RuntimeConfig:
    upload_root: Path
    output_root: Path
    device: str
    pop2piano_model: str
    pop2piano_composer: str
    pop2piano_device: str
    pop2piano_max_length: int
    soundfont: Path
    fluidsynth_bin: Path
    fluidsynth_lib_dir: Path
    pitch_extractor: str
    pre_pitch_shift: float
    preload_mode: str
    require_ready_models: bool
    pre_pitch_shift_min: int
    pre_pitch_shift_max: int
    volume_min: float
    volume_max: float
    cors_origins: list[str]


@dataclass(frozen=True)
class BackendConfig:
    roles: list[RoleConfig]
    runtime: RuntimeConfig


def _resolve_path(value: str | Path, base: Path = REPO_ROOT) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return base / path


def _resolve_model_id_or_path(value: str | Path) -> str:
    model = str(value)
    path = Path(model).expanduser()
    if path.is_absolute() and path.exists():
        return str(path)
    repo_path = REPO_ROOT / path
    if repo_path.exists():
        return str(repo_path)
    return model


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: list[str]) -> list[str]:
    value = os.environ.get(name)
    if value is None:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def _runtime_from_dict(data: dict[str, Any]) -> RuntimeConfig:
    constraints = data.get("constraints", {})
    return RuntimeConfig(
        upload_root=_resolve_path(os.environ.get("COVER_UPLOAD_ROOT", data.get("upload_root", "assets/uploads"))),
        output_root=_resolve_path(os.environ.get("COVER_OUTPUT_ROOT", data.get("output_root", "assets/output/jobs"))),
        device=os.environ.get("COVER_DEVICE", data.get("device", "cuda")),
        pop2piano_model=_resolve_model_id_or_path(
            os.environ.get("COVER_POP2PIANO_MODEL", data.get("pop2piano_model", DEFAULT_POP2PIANO_MODEL))
        ),
        pop2piano_composer=os.environ.get("COVER_POP2PIANO_COMPOSER", data.get("pop2piano_composer", "composer1")),
        pop2piano_device=os.environ.get("COVER_POP2PIANO_DEVICE", data.get("pop2piano_device", "cpu")),
        pop2piano_max_length=int(os.environ.get("COVER_POP2PIANO_MAX_LENGTH", data.get("pop2piano_max_length", 256))),
        soundfont=_resolve_path(os.environ.get("COVER_SOUNDFONT", data.get("soundfont", DEFAULT_SOUNDFONT))),
        fluidsynth_bin=_resolve_path(os.environ.get("COVER_FLUIDSYNTH_BIN", data.get("fluidsynth_bin", DEFAULT_FLUIDSYNTH_BIN))),
        fluidsynth_lib_dir=_resolve_path(
            os.environ.get("COVER_FLUIDSYNTH_LIB_DIR", data.get("fluidsynth_lib_dir", DEFAULT_FLUIDSYNTH_LIB_DIR))
        ),
        pitch_extractor=os.environ.get("COVER_PITCH_EXTRACTOR", data.get("pitch_extractor", "rmvpe")),
        pre_pitch_shift=float(os.environ.get("COVER_PRE_PITCH_SHIFT", data.get("pre_pitch_shift", 0.0))),
        preload_mode=os.environ.get("COVER_PRELOAD_MODE", data.get("preload_mode", "validate")),
        require_ready_models=_env_bool("COVER_REQUIRE_READY_MODELS", bool(data.get("require_ready_models", False))),
        pre_pitch_shift_min=int(os.environ.get("COVER_PRE_PITCH_SHIFT_MIN", constraints.get("pre_pitch_shift", {}).get("min", -12))),
        pre_pitch_shift_max=int(os.environ.get("COVER_PRE_PITCH_SHIFT_MAX", constraints.get("pre_pitch_shift", {}).get("max", 12))),
        volume_min=float(os.environ.get("COVER_VOLUME_MIN", constraints.get("volume", {}).get("min", 0.0))),
        volume_max=float(os.environ.get("COVER_VOLUME_MAX", constraints.get("volume", {}).get("max", 2.0))),
        cors_origins=_env_list("COVER_CORS_ORIGINS", data.get("cors_origins", ["*"])),
    )


def load_backend_config(config_path: Path | None = None) -> BackendConfig:
    path = config_path or _resolve_path(os.environ.get("COVER_ROLES_CONFIG", "configs/roles.yaml"))
    if not path.exists():
        raise FileNotFoundError(f"Backend roles config was not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    role_items = data.get("roles") or []
    if not role_items:
        raise ValueError(f"Backend roles config does not define any roles: {path}")

    roles = [
        RoleConfig(
            id=str(item["id"]),
            name=str(item.get("name", item["id"])),
            ddsp_model_ckpt=_resolve_path(item["ddsp_model_ckpt"]),
            spk_id=int(item.get("spk_id", 1)),
            default_pre_pitch_shift=int(item.get("default_pre_pitch_shift", 0)),
            avatar=item.get("avatar"),
        )
        for item in role_items
    ]
    duplicate_ids = {role.id for role in roles if sum(1 for item in roles if item.id == role.id) > 1}
    if duplicate_ids:
        raise ValueError(f"Duplicate role ids in backend config: {', '.join(sorted(duplicate_ids))}")

    runtime = _runtime_from_dict(data.get("runtime", {}))
    return BackendConfig(roles=roles, runtime=runtime)
