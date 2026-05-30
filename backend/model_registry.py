"""Startup model registry and readiness checks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.ai_cover import DDSP_ROOT, DEMUCS_ROOT, resolve_ddsp_assets

from backend.settings import BackendConfig, RoleConfig, load_backend_config


@dataclass(frozen=True)
class RoleStatus:
    id: str
    name: str
    checkpoint: Path
    spk_id: int
    default_pre_pitch_shift: int
    avatar: str | None
    ready: bool
    loaded: bool
    error: str | None


class ModelRegistry:
    """Loads role config at startup and tracks model readiness."""

    def __init__(self) -> None:
        self.config: BackendConfig | None = None
        self.roles: dict[str, RoleConfig] = {}
        self.role_status: dict[str, RoleStatus] = {}
        self.components: dict[str, dict[str, Any]] = {}
        self.loaded_checkpoints: dict[str, Any] = {}
        self.loaded_at: str | None = None

    def load(self) -> None:
        config = load_backend_config()
        self.config = config
        self.roles = {role.id: role for role in config.roles}
        self.components = self._check_components(config)
        self.role_status = {}
        self.loaded_checkpoints = {}

        for role in config.roles:
            self.role_status[role.id] = self._load_role(role, config.runtime.preload_mode)

        self.loaded_at = datetime.now(timezone.utc).isoformat()

        if config.runtime.require_ready_models:
            failures = [status for status in self.role_status.values() if not status.ready]
            component_failures = [name for name, status in self.components.items() if not status["ready"]]
            if failures or component_failures:
                role_errors = "; ".join(f"{item.id}: {item.error}" for item in failures)
                component_errors = "; ".join(f"{name}: {self.components[name]['error']}" for name in component_failures)
                details = "; ".join(item for item in (role_errors, component_errors) if item)
                raise RuntimeError(f"Backend model readiness check failed: {details}")

    @property
    def runtime(self):
        if self.config is None:
            raise RuntimeError("Model registry has not been loaded.")
        return self.config.runtime

    def get_role(self, role_id: str) -> RoleConfig:
        if role_id not in self.roles:
            raise KeyError(f"Unknown role_id: {role_id}")
        return self.roles[role_id]

    def get_ready_role(self, role_id: str) -> RoleConfig:
        role = self.get_role(role_id)
        status = self.role_status[role_id]
        if not status.ready:
            raise RuntimeError(f"Role '{role_id}' is not ready: {status.error}")
        return role

    def public_config(self) -> dict[str, Any]:
        runtime = self.runtime
        return {
            "roles": [self._public_role(status) for status in self.role_status.values()],
            "constraints": {
                "pre_pitch_shift": {"min": runtime.pre_pitch_shift_min, "max": runtime.pre_pitch_shift_max, "step": 1},
                "vocals_volume": {"min": runtime.volume_min, "max": runtime.volume_max, "step": 0.05, "default": 1.0},
                "piano_volume": {"min": runtime.volume_min, "max": runtime.volume_max, "step": 0.05, "default": 1.0},
            },
            "stages": [
                {"id": 1, "name": "separate_vocals", "label": "Separate vocals"},
                {"id": 2, "name": "voice_conversion", "label": "Voice conversion"},
                {"id": 3, "name": "piano_cover", "label": "Piano cover"},
                {"id": 4, "name": "merge", "label": "Final mix"},
            ],
        }

    def public_status(self) -> dict[str, Any]:
        return {
            "loaded_at": self.loaded_at,
            "preload_mode": self.runtime.preload_mode,
            "components": self.components,
            "roles": [self._public_role(status, include_internal=True) for status in self.role_status.values()],
        }

    def _check_components(self, config: BackendConfig) -> dict[str, dict[str, Any]]:
        runtime = config.runtime
        components: dict[str, dict[str, Any]] = {}
        components["demucs"] = self._path_status(DEMUCS_ROOT, expect_dir=True)
        components["ddsp"] = self._path_status(DDSP_ROOT, expect_dir=True)
        components["soundfont"] = self._path_status(runtime.soundfont, expect_dir=False)
        components["fluidsynth_bin"] = self._path_status(runtime.fluidsynth_bin, expect_dir=False)
        components["fluidsynth_lib_dir"] = self._path_status(runtime.fluidsynth_lib_dir, expect_dir=True)

        pop2piano_path = Path(runtime.pop2piano_model)
        if pop2piano_path.exists():
            components["pop2piano"] = self._path_status(pop2piano_path, expect_dir=True)
        else:
            components["pop2piano"] = {
                "ready": True,
                "path": runtime.pop2piano_model,
                "error": None,
                "note": "Model id will be resolved by transformers at runtime.",
            }
        return components

    def _load_role(self, role: RoleConfig, preload_mode: str) -> RoleStatus:
        loaded = False
        error = None
        ready = True

        if not role.ddsp_model_ckpt.exists():
            ready = False
            error = f"Checkpoint file does not exist: {role.ddsp_model_ckpt}"
        elif not role.ddsp_model_ckpt.is_file():
            ready = False
            error = f"Checkpoint path is not a file: {role.ddsp_model_ckpt}"
        else:
            try:
                resolve_ddsp_assets(role.ddsp_model_ckpt, repair=False)
            except Exception as exc:  # noqa: BLE001 - surfaced through readiness status
                ready = False
                error = str(exc)

        if ready and preload_mode == "torch_cpu":
            try:
                import torch

                self.loaded_checkpoints[role.id] = torch.load(role.ddsp_model_ckpt, map_location="cpu")
                loaded = True
            except Exception as exc:  # noqa: BLE001 - surfaced through readiness status
                ready = False
                error = f"Failed to torch-load checkpoint: {exc}"
        elif ready and preload_mode in {"validate", "none"}:
            loaded = False
        elif ready:
            ready = False
            error = f"Unsupported preload mode: {preload_mode}"

        return RoleStatus(
            id=role.id,
            name=role.name,
            checkpoint=role.ddsp_model_ckpt,
            spk_id=role.spk_id,
            default_pre_pitch_shift=role.default_pre_pitch_shift,
            avatar=role.avatar,
            ready=ready,
            loaded=loaded,
            error=error,
        )

    def _path_status(self, path: Path, expect_dir: bool) -> dict[str, Any]:
        ready = path.exists() and (path.is_dir() if expect_dir else path.is_file())
        expected = "directory" if expect_dir else "file"
        return {
            "ready": ready,
            "path": str(path),
            "error": None if ready else f"Expected {expected} not found: {path}",
        }

    def _public_role(self, status: RoleStatus, include_internal: bool = False) -> dict[str, Any]:
        data = {
            "id": status.id,
            "name": status.name,
            "avatar": status.avatar,
            "default_pre_pitch_shift": status.default_pre_pitch_shift,
            "ready": status.ready,
            "loaded": status.loaded,
            "error": status.error,
        }
        if include_internal:
            data.update({"checkpoint": str(status.checkpoint), "spk_id": status.spk_id})
        return data
