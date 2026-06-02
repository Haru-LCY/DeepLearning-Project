"""FastAPI entry point for the AI cover backend."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import math
from pathlib import Path
import shutil
from typing import AsyncIterator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from backend.jobs import JobManager, JobParams
from backend.model_registry import MODEL_PRELOAD_MODES, ModelRegistry
from backend.progress import ProgressBroker, queue_get_with_timeout
from src.ai_cover import run_ai_piano_cover


ALLOWED_AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"}


class RemixRequest(BaseModel):
    vocals_volume: float
    piano_volume: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry = ModelRegistry()
    registry.load()
    await asyncio.to_thread(_run_startup_warmup, registry)
    broker = ProgressBroker()
    manager = JobManager(registry=registry, broker=broker)
    manager.start()

    app.state.registry = registry
    app.state.broker = broker
    app.state.jobs = manager
    print("startup complete")

    yield

    manager.shutdown()


app = FastAPI(title="AI Cover Backend", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _run_startup_warmup(registry: ModelRegistry) -> None:
    runtime = registry.runtime
    if not runtime.warmup_enabled:
        return
    if not runtime.warmup_audio.exists():
        raise FileNotFoundError(f"Warmup audio was not found: {runtime.warmup_audio}")

    role_id = runtime.warmup_role_id or _first_ready_role_id(registry)
    role = registry.get_ready_role(role_id)
    demucs_separator = _loaded_warmup_component(registry, "demucs")
    ddsp_runtime = _loaded_warmup_role_runtime(registry, role.id)
    pop2piano_components = _loaded_warmup_component(registry, "pop2piano")

    print(
        "Running startup warmup with "
        f"{runtime.warmup_audio} as role {role.id} "
        f"(pre_pitch_shift={runtime.warmup_pre_pitch_shift:+g}, "
        f"vocals_volume={runtime.warmup_vocals_volume:g}, "
        f"piano_volume={runtime.warmup_piano_volume:g})"
    )
    run_ai_piano_cover(
        input_audio=runtime.warmup_audio,
        output_root=runtime.warmup_output_root,
        device=runtime.device,
        spk_id=role.spk_id,
        key=0,
        pre_pitch_shift=runtime.warmup_pre_pitch_shift,
        pitch_extractor=runtime.pitch_extractor,
        vocals_volume=runtime.warmup_vocals_volume,
        piano_volume=runtime.warmup_piano_volume,
        ddsp_model_ckpt=role.ddsp_model_ckpt,
        ddsp_segment_batch_size=role.ddsp_segment_batch_size,
        pop2piano_model=runtime.pop2piano_model,
        pop2piano_composer=runtime.pop2piano_composer,
        pop2piano_device=runtime.pop2piano_device,
        pop2piano_max_length=runtime.pop2piano_max_length,
        pop2piano_beat_checkpoint=runtime.pop2piano_beat_checkpoint,
        demucs_separator=demucs_separator,
        ddsp_runtime=ddsp_runtime,
        pop2piano_components=pop2piano_components,
        soundfont=runtime.soundfont,
        fluidsynth_bin=runtime.fluidsynth_bin,
        fluidsynth_lib_dir=runtime.fluidsynth_lib_dir,
        use_parallel_stages=runtime.use_parallel_stages,
    )


def _first_ready_role_id(registry: ModelRegistry) -> str:
    for status in registry.role_status.values():
        if status.ready:
            return status.id
    raise RuntimeError("No ready role is available for startup warmup.")


def _loaded_warmup_component(registry: ModelRegistry, name: str):
    if registry.runtime.preload_mode not in MODEL_PRELOAD_MODES:
        return None
    component = registry.loaded_components.get(name)
    if component is None:
        raise RuntimeError(f"Preloaded warmup component is not available: {name}")
    return component


def _loaded_warmup_role_runtime(registry: ModelRegistry, role_id: str):
    if registry.runtime.preload_mode not in MODEL_PRELOAD_MODES:
        return None
    runtime = registry.loaded_checkpoints.get(role_id)
    if runtime is None:
        raise RuntimeError(f"Preloaded warmup DDSP runtime is not available for role: {role_id}")
    return runtime


@app.middleware("http")
async def add_no_buffer_header(request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Accel-Buffering", "no")
    return response


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config")
async def config() -> dict:
    return app.state.registry.public_config()


@app.get("/api/models/status")
async def models_status() -> dict:
    return app.state.registry.public_status()


@app.post("/api/jobs")
async def create_job(
    audio: UploadFile = File(...),
    role_id: str = Form(...),
    pre_pitch_shift: int = Form(0),
    vocals_volume: float = Form(1.0),
    piano_volume: float = Form(1.0),
) -> dict:
    registry: ModelRegistry = app.state.registry
    manager: JobManager = app.state.jobs
    _validate_job_params(registry, role_id, pre_pitch_shift, vocals_volume, piano_volume)

    suffix = _validated_audio_suffix(audio.filename)
    job_id = manager.create_job_id()
    upload_dir = registry.runtime.upload_root / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    input_path = upload_dir / f"input{suffix}"
    try:
        with input_path.open("wb") as handle:
            shutil.copyfileobj(audio.file, handle)
    finally:
        await audio.close()

    params = JobParams(
        role_id=role_id,
        pre_pitch_shift=pre_pitch_shift,
        vocals_volume=vocals_volume,
        piano_volume=piano_volume,
        original_filename=audio.filename,
    )
    record = manager.enqueue(job_id=job_id, input_path=input_path, params=params)
    return manager.public_job(record.id)


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    try:
        return app.state.jobs.public_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    manager: JobManager = app.state.jobs
    broker: ProgressBroker = app.state.broker
    try:
        initial = manager.public_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    async def stream() -> AsyncIterator[str]:
        yield _sse_event("snapshot", initial)
        subscriber = broker.subscribe(job_id)
        try:
            while True:
                event = await asyncio.to_thread(queue_get_with_timeout, subscriber, 15.0)
                if event is None:
                    yield ": keepalive\n\n"
                    continue
                yield _sse_event(event.get("type", "message"), event)
                if event.get("type") in {"completed", "failed", "cancelled"}:
                    break
        finally:
            broker.unsubscribe(job_id, subscriber)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict:
    try:
        return app.state.jobs.cancel_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/jobs/{job_id}/remix")
async def remix_job(job_id: str, request: RemixRequest) -> dict:
    registry: ModelRegistry = app.state.registry
    _validate_volume(registry, request.vocals_volume, "vocals_volume")
    _validate_volume(registry, request.piano_volume, "piano_volume")
    try:
        return await asyncio.to_thread(
            app.state.jobs.remix_job,
            job_id,
            request.vocals_volume,
            request.piano_volume,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}/files/{kind}")
async def get_job_file(job_id: str, kind: str) -> FileResponse:
    try:
        path = app.state.jobs.file_path(job_id, kind)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    media_type = "audio/wav" if path.suffix.lower() == ".wav" else "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=path.name)


def _validate_job_params(
    registry: ModelRegistry,
    role_id: str,
    pre_pitch_shift: int,
    vocals_volume: float,
    piano_volume: float,
) -> None:
    try:
        registry.get_ready_role(role_id)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    runtime = registry.runtime
    if pre_pitch_shift < runtime.pre_pitch_shift_min or pre_pitch_shift > runtime.pre_pitch_shift_max:
        raise HTTPException(
            status_code=400,
            detail=f"pre_pitch_shift must be between {runtime.pre_pitch_shift_min} and {runtime.pre_pitch_shift_max}",
        )
    _validate_volume(registry, vocals_volume, "vocals_volume")
    _validate_volume(registry, piano_volume, "piano_volume")


def _validate_volume(registry: ModelRegistry, value: float, name: str) -> None:
    runtime = registry.runtime
    if not math.isfinite(value):
        raise HTTPException(status_code=400, detail=f"{name} must be a finite number")
    if value < runtime.volume_min or value > runtime.volume_max:
        raise HTTPException(
            status_code=400,
            detail=f"{name} must be between {runtime.volume_min} and {runtime.volume_max}",
        )


def _validated_audio_suffix(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_AUDIO_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format. Allowed suffixes: {', '.join(sorted(ALLOWED_AUDIO_SUFFIXES))}",
        )
    return suffix


def _sse_event(event_name: str, data: dict) -> str:
    return f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
