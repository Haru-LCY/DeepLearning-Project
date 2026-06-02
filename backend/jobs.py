"""Job queue and execution layer for the cover backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import queue
import threading
import traceback
from typing import Any
from uuid import uuid4

from src.ai_cover import CoverArtifacts, mix_cover_audio, run_ai_piano_cover

from backend.model_registry import MODEL_PRELOAD_MODES, ModelRegistry
from backend.progress import ProgressBroker
from backend.settings import RuntimeConfig


STAGE_NUMBERS = {
    "preprocess": 1,
    "separate_vocals": 1,
    "voice_conversion": 2,
    "piano_cover": 3,
    "render_piano": 3,
    "merge": 4,
    "completed": 4,
}

FILE_KINDS = {"input", "vocals", "piano", "final"}


@dataclass(frozen=True)
class JobParams:
    role_id: str
    pre_pitch_shift: int
    vocals_volume: float
    piano_volume: float
    original_filename: str | None = None


@dataclass
class ArtifactPaths:
    preprocessed_audio: Path | None = None
    vocals: Path | None = None
    no_vocals: Path | None = None
    ddsp_vocals: Path | None = None
    piano_midi: Path | None = None
    piano_wav: Path | None = None
    final_mix: Path | None = None

    @classmethod
    def from_cover_artifacts(cls, artifacts: CoverArtifacts) -> "ArtifactPaths":
        return cls(
            preprocessed_audio=artifacts.preprocessed_audio,
            vocals=artifacts.vocals,
            no_vocals=artifacts.no_vocals,
            ddsp_vocals=artifacts.ddsp_vocals,
            piano_midi=artifacts.piano_midi,
            piano_wav=artifacts.piano_wav,
            final_mix=artifacts.final_mix,
        )


@dataclass
class JobRecord:
    id: str
    params: JobParams
    input_path: Path
    output_root: Path
    status: str = "queued"
    stage: int | None = None
    stage_name: str | None = None
    progress: int = 0
    message: str = "Queued"
    artifacts: ArtifactPaths = field(default_factory=ArtifactPaths)
    stage_timings: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    cancel_requested: bool = False


class JobManager:
    def __init__(self, registry: ModelRegistry, broker: ProgressBroker) -> None:
        self.registry = registry
        self.broker = broker
        self.runtime: RuntimeConfig = registry.runtime
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None

    def start(self) -> None:
        self.runtime.upload_root.mkdir(parents=True, exist_ok=True)
        self.runtime.output_root.mkdir(parents=True, exist_ok=True)
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._run_loop, name="cover-job-worker", daemon=True)
            self._worker.start()

    def shutdown(self) -> None:
        self._stop.set()
        self._queue.put(None)
        if self._worker is not None:
            self._worker.join(timeout=5)

    def create_job_id(self) -> str:
        return str(uuid4())

    def enqueue(self, job_id: str, input_path: Path, params: JobParams) -> JobRecord:
        output_root = self.runtime.output_root / job_id
        output_root.mkdir(parents=True, exist_ok=True)
        record = JobRecord(id=job_id, params=params, input_path=input_path, output_root=output_root)
        with self._lock:
            self._jobs[job_id] = record
        self._publish(record, "queued")
        self._queue.put(job_id)
        return record

    def get_job(self, job_id: str) -> JobRecord:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(f"Unknown job_id: {job_id}")
            return record

    def public_job(self, job_id: str) -> dict[str, Any]:
        return self._serialize_job(self.get_job(job_id))

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        record = self.get_job(job_id)
        if record.status == "queued":
            self._update(job_id, status="cancelled", message="Cancelled before processing", progress=0)
            self._publish(self.get_job(job_id), "cancelled")
        elif record.status == "running":
            self._update(job_id, cancel_requested=True, message="Cancellation requested; running model stages cannot be interrupted safely")
            self._publish(self.get_job(job_id), "progress")
        return self.public_job(job_id)

    def remix_job(self, job_id: str, vocals_volume: float, piano_volume: float) -> dict[str, Any]:
        record = self.get_job(job_id)
        if record.status not in {"completed", "failed"}:
            raise RuntimeError("Only completed jobs can be remixed.")
        if record.artifacts.ddsp_vocals is None or record.artifacts.piano_wav is None or record.artifacts.final_mix is None:
            raise RuntimeError("Job does not have the required vocals and piano artifacts yet.")
        if not record.artifacts.ddsp_vocals.exists() or not record.artifacts.piano_wav.exists():
            raise FileNotFoundError("Required source artifacts are missing on disk.")

        self._update(
            job_id,
            status="running",
            stage=4,
            stage_name="merge",
            progress=90,
            message="Remixing existing vocals and piano audio",
            error=None,
        )
        try:
            started_at = datetime.now(timezone.utc)
            mix_cover_audio(
                vocals_audio=record.artifacts.ddsp_vocals,
                piano_audio=record.artifacts.piano_wav,
                output_audio=record.artifacts.final_mix,
                vocals_volume=vocals_volume,
                piano_volume=piano_volume,
            )
        except Exception as exc:  # noqa: BLE001 - propagated through API after state update
            self._update(job_id, status="failed", message=str(exc), error=str(exc))
            self._publish(self.get_job(job_id), "failed")
            raise
        updated_params = JobParams(
            role_id=record.params.role_id,
            pre_pitch_shift=record.params.pre_pitch_shift,
            vocals_volume=vocals_volume,
            piano_volume=piano_volume,
            original_filename=record.params.original_filename,
        )
        self._update(
            job_id,
            params=updated_params,
            stage_timings={**record.stage_timings, "merge": round((datetime.now(timezone.utc) - started_at).total_seconds(), 3)},
            status="completed",
            stage=4,
            stage_name="completed",
            progress=100,
            message="Remix completed",
        )
        record = self.get_job(job_id)
        self._publish(record, "completed")
        return self.public_job(job_id)

    def file_path(self, job_id: str, kind: str) -> Path:
        if kind not in FILE_KINDS:
            raise KeyError(f"Unknown file kind: {kind}")
        record = self.get_job(job_id)
        path = {
            "input": record.input_path,
            "vocals": record.artifacts.ddsp_vocals,
            "piano": record.artifacts.piano_wav,
            "final": record.artifacts.final_mix,
        }[kind]
        if path is None:
            raise FileNotFoundError(f"Artifact is not available yet: {kind}")
        if not path.exists():
            raise FileNotFoundError(f"Artifact file does not exist: {path}")
        return path

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            job_id = self._queue.get()
            if job_id is None:
                self._queue.task_done()
                return
            try:
                record = self.get_job(job_id)
                if record.status == "cancelled":
                    continue
                self._process_job(job_id)
            finally:
                self._queue.task_done()

    def _process_job(self, job_id: str) -> None:
        try:
            record = self.get_job(job_id)
            role = self.registry.get_ready_role(record.params.role_id)
            demucs_separator = self._loaded_component("demucs")
            ddsp_runtime = self._loaded_role_runtime(role.id)
            pop2piano_components = self._loaded_component("pop2piano")
            self._update(job_id, status="running", stage=1, stage_name="separate_vocals", progress=0, message="Starting job")

            result = run_ai_piano_cover(
                input_audio=record.input_path,
                output_root=record.output_root,
                device=self.runtime.device,
                spk_id=role.spk_id,
                key=0,
                pre_pitch_shift=float(record.params.pre_pitch_shift),
                pitch_extractor=self.runtime.pitch_extractor,
                vocals_volume=record.params.vocals_volume,
                piano_volume=record.params.piano_volume,
                ddsp_model_ckpt=role.ddsp_model_ckpt,
                pop2piano_model=self.runtime.pop2piano_model,
                pop2piano_composer=self.runtime.pop2piano_composer,
                pop2piano_device=self.runtime.pop2piano_device,
                pop2piano_max_length=self.runtime.pop2piano_max_length,
                pop2piano_beat_checkpoint=self.runtime.pop2piano_beat_checkpoint,
                demucs_separator=demucs_separator,
                ddsp_runtime=ddsp_runtime,
                pop2piano_components=pop2piano_components,
                soundfont=self.runtime.soundfont,
                fluidsynth_bin=self.runtime.fluidsynth_bin,
                fluidsynth_lib_dir=self.runtime.fluidsynth_lib_dir,
                progress_callback=lambda stage_name, progress, message: self._handle_progress(
                    job_id, stage_name, progress, message
                ),
            )

            self._set_artifacts(job_id, ArtifactPaths.from_cover_artifacts(result.artifacts))
            self._update(
                job_id,
                stage_timings=result.stage_timings,
                status="completed",
                stage=4,
                stage_name="completed",
                progress=100,
                message="Completed",
                error=None,
            )
            self._publish(self.get_job(job_id), "completed")
        except Exception as exc:  # noqa: BLE001 - failures are returned to the API client
            self._update(
                job_id,
                status="failed",
                message=str(exc),
                error=str(exc),
            )
            error_path = self.get_job(job_id).output_root / "error.log"
            error_path.parent.mkdir(parents=True, exist_ok=True)
            error_path.write_text(traceback.format_exc(), encoding="utf-8")
            self._publish(self.get_job(job_id), "failed")

    def _loaded_component(self, name: str) -> Any | None:
        if self.runtime.preload_mode not in MODEL_PRELOAD_MODES:
            return None
        component = self.registry.loaded_components.get(name)
        if component is None:
            raise RuntimeError(f"Preloaded component is not available: {name}")
        return component

    def _loaded_role_runtime(self, role_id: str) -> Any | None:
        if self.runtime.preload_mode not in MODEL_PRELOAD_MODES:
            return None
        runtime = self.registry.loaded_checkpoints.get(role_id)
        if runtime is None:
            raise RuntimeError(f"Preloaded DDSP runtime is not available for role: {role_id}")
        return runtime

    def _handle_progress(self, job_id: str, stage_name: str, progress: int, message: str) -> None:
        self._update(
            job_id,
            status="running" if stage_name != "completed" else "running",
            stage=STAGE_NUMBERS.get(stage_name),
            stage_name=stage_name,
            progress=progress,
            message=message,
        )
        self._publish(self.get_job(job_id), "progress")

    def _set_artifacts(self, job_id: str, artifacts: ArtifactPaths) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.artifacts = artifacts
            record.updated_at = datetime.now(timezone.utc).isoformat()

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            record = self._jobs[job_id]
            for key, value in changes.items():
                setattr(record, key, value)
            record.updated_at = datetime.now(timezone.utc).isoformat()

    def _publish(self, record: JobRecord, event_type: str) -> None:
        self.broker.publish(record.id, {"type": event_type, **self._serialize_job(record)})

    def _serialize_job(self, record: JobRecord) -> dict[str, Any]:
        return {
            "job_id": record.id,
            "status": record.status,
            "stage": record.stage,
            "stage_name": record.stage_name,
            "progress": record.progress,
            "message": record.message,
            "params": {
                "role_id": record.params.role_id,
                "pre_pitch_shift": record.params.pre_pitch_shift,
                "vocals_volume": record.params.vocals_volume,
                "piano_volume": record.params.piano_volume,
                "original_filename": record.params.original_filename,
            },
            "artifacts": self._artifact_urls(record),
            "stage_timings": record.stage_timings,
            "error": record.error,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "cancel_requested": record.cancel_requested,
        }

    def _artifact_urls(self, record: JobRecord) -> dict[str, str | None]:
        base = f"/api/jobs/{record.id}/files"
        return {
            "input_audio": f"{base}/input" if record.input_path.exists() else None,
            "vocals": f"{base}/vocals" if record.artifacts.ddsp_vocals and record.artifacts.ddsp_vocals.exists() else None,
            "piano": f"{base}/piano" if record.artifacts.piano_wav and record.artifacts.piano_wav.exists() else None,
            "final": f"{base}/final" if record.artifacts.final_mix and record.artifacts.final_mix.exists() else None,
        }
