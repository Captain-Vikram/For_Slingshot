from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

try:
    from shared.utils import normalize_label_mode
    from shared.config import load_config, ConfigManager
except ModuleNotFoundError:
    from yolodex.shared.utils import normalize_label_mode
    from yolodex.shared.config import load_config, ConfigManager


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT_CONFIG_PATH = REPO_ROOT / "config.json"
RUNS_DIR = REPO_ROOT / "runs"
PROJECT_CONFIG_DIR = RUNS_DIR / "_configs" # Keep for legacy or global project configs?
# JOBS_DIR = REPO_ROOT / "runs" / "_jobs" # Deprecated
PROJECT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
# JOBS_DIR.mkdir(parents=True, exist_ok=True)

# New helper to find job path
def _find_job_path(job_id: str, project: str | None = None) -> Path | None:
    if project:
        # Check new location
        candidate = RUNS_DIR / project / "_jobs" / f"{job_id}.json"
        if candidate.exists():
            return candidate
        # Check legacy location
        candidate_old = RUNS_DIR / project / job_id / "job.json"
        if candidate_old.exists():
            return candidate_old
    
    # Fallback: scan all projects
    if RUNS_DIR.exists():
        for proj_dir in RUNS_DIR.iterdir():
            if proj_dir.is_dir() and not proj_dir.name.startswith("_"):
                candidate = proj_dir / "_jobs" / f"{job_id}.json"
                if candidate.exists():
                    return candidate
                
                candidate_old = proj_dir / job_id / "job.json"
                if candidate_old.exists():
                    return candidate_old
    
    # Check legacy global _jobs
    legacy = RUNS_DIR / "_jobs" / f"{job_id}.json"
    if legacy.exists():
        return legacy
            
    return None

class ProjectConfigUpsert(BaseModel):
    project: str = Field(min_length=1)
    video_url: str | None = None
    classes: list[str] | None = None
    label_mode: str | None = None
    model: str | None = None
    gemini_model: str | None = None
    openai_model: str | None = None
    local_model_name: str | None = None
    local_vlm_url: str | None = None
    local_vlm_api_key: str | None = None
    target_accuracy: float | None = None
    max_iterations: int | None = None
    num_agents: int | None = None
    fps: int | None = None
    yolo_model: str | None = None
    epochs: int | None = None
    train_split: float | None = None


RunMode = Literal["full", "collect", "label", "augment", "train", "eval"]


class RunRequest(BaseModel):
    mode: RunMode = "full"
    gemini_api_key: str | None = None
    google_api_key: str | None = None
    openai_api_key: str | None = None
    local_vlm_api_key: str | None = None
    source_job_id: str | None = None


@dataclass
class JobInfo:
    id: str
    project: str
    requested_mode: RunMode
    phases: list[str]
    source_job_id: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    status: Literal["queued", "running", "completed", "failed"] = "queued"
    current_phase: str | None = None
    phase_index: int = 0
    total_phases: int = 0
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    error_code: str | None = None
    error_hint: str | None = None
    retryable: bool | None = None
    exit_code: int | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    event_seq: int = 0

    def push_event(self, level: str, message: str) -> None:
        self.event_seq += 1
        self.events.append(
            {
                "seq": self.event_seq,
                "time": utc_now_iso(),
                "level": level,
                "message": message.rstrip("\n"),
            }
        )
        if len(self.events) > 5000:
            self.events = self.events[-2000:]


import logging

# Filter Uvicorn access logs for frequent polling endpoints
class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.args and len(record.args) >= 3:
            path = str(tuple(record.args)[2])
            if any(x in path for x in ["/status", "/artifacts", "/jobs/"]):
                return False
        return True

logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

app = FastAPI(title="Yolodex API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_state_lock = threading.RLock()
_config_lock = threading.Lock()
_jobs: dict[str, JobInfo] = {}
_last_save_time: dict[str, float] = {}
_project_latest_job: dict[str, str] = {}
_project_running_job: dict[str, str] = {}
_active_job_id: str | None = None
_job_env_overrides: dict[str, dict[str, str]] = {}
_active_processes: dict[str, subprocess.Popen] = {} # Map job_id to Popen object


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _project_config_path(project: str) -> Path:
    # Always prefer the new structure for writes or if it exists
    project_dir = RUNS_DIR / project
    project_dir.mkdir(parents=True, exist_ok=True)
    return project_dir / "config.json"


def _new_job_path(project: str, job_id: str) -> Path:
    # Store job metadata in a hidden _jobs folder to keep the project clean
    (RUNS_DIR / project / "_jobs").mkdir(parents=True, exist_ok=True)
    return RUNS_DIR / project / "_jobs" / f"{job_id}.json"


def _param_job_path(job_id: str) -> Path:
    # Best-effort path resolution for existing jobs
    with _state_lock:
        job = _jobs.get(job_id)
        if job:
            return _new_job_path(job.project, job_id)
    
    found = _find_job_path(job_id)
    if found:
        return found
    
    # Check all projects if not found directly
    if RUNS_DIR.exists():
        for proj_dir in RUNS_DIR.iterdir():
            if proj_dir.is_dir() and not proj_dir.name.startswith("_"):
                candidate = proj_dir / "_jobs" / f"{job_id}.json"
                if candidate.exists():
                    return candidate
                # Check legacy: project/job_id/job.json
                candidate_old = proj_dir / job_id / "job.json"
                if candidate_old.exists():
                    return candidate_old

    return RUNS_DIR / "_jobs" / f"{job_id}.json"


def _parse_job(payload: dict[str, Any]) -> JobInfo:
    normalized = dict(payload)
    normalized.setdefault("created_at", utc_now_iso())
    normalized.setdefault("events", [])
    normalized.setdefault("event_seq", 0)
    normalized.setdefault("phase_index", 0)
    normalized.setdefault("total_phases", len(normalized.get("phases", [])))
    return JobInfo(**normalized)


def _job_sort_key(job: JobInfo) -> datetime:
    stamp = job.started_at or job.created_at or utc_now_iso()
    try:
        return datetime.fromisoformat(stamp)
    except ValueError:
        return datetime.fromtimestamp(0, tz=timezone.utc)


def _save_job(job: JobInfo, force: bool = False) -> None:
    now = time.time()
    with _state_lock:
        _jobs[job.id] = job
        last = _last_save_time.get(job.id, 0.0)
        should_save = force or job.status in {"completed", "failed"} or (now - last > 1.0)
        if should_save:
            _last_save_time[job.id] = now

        # Simple cache eviction: keep only recent/active jobs in memory
        if len(_jobs) > 50:
            # Sort by last save time or created_at? Last save/access is better.
            # But here we only have _save_job.
            # Keep active jobs
            candidates = [
                jid for jid, j in _jobs.items() 
                if j.status not in {"queued", "running"} and jid != job.id
            ]
            if len(candidates) > 10:
                # Remove oldest 5
                # Using created_at for simplicity as proxy for age
                candidates.sort(key=lambda jid: _jobs[jid].created_at)
                for jid in candidates[:5]:
                    _jobs.pop(jid, None)
                    _last_save_time.pop(jid, None)

    if should_save:
        try:
            _write_json(_new_job_path(job.project, job.id), asdict(job))
        except Exception:  # noqa: BLE001
            # If disk write fails and we're not shutting down, just continue.
            pass


def _load_job(job_id: str) -> JobInfo | None:
    with _state_lock:
        cached = _jobs.get(job_id)
        if cached:
            return cached

    path = _find_job_path(job_id)
    if not path or not path.exists():
        return None  # Early return if no file

    try:
        job = _parse_job(_read_json(path))
        return job
    except Exception:  # noqa: BLE001
        return None


def _collect_jobs(project: str | None = None) -> list[JobInfo]:
    by_id: dict[str, JobInfo] = {}

    # Scan new structure: runs/<project>/_jobs/<job_id>.json
    try:
        if not RUNS_DIR.exists():
            return []

        if project:
            proj_dir = RUNS_DIR / project
            projects = [proj_dir] if proj_dir.exists() else []
        else:
            projects = [p for p in RUNS_DIR.iterdir() if p.is_dir() and not p.name.startswith("_")]
        
        for proj_dir in projects:
            # Check metadata dir
            meta_dir = proj_dir / "_jobs"
            if meta_dir.exists():
                for j_path in meta_dir.glob("*.json"):
                    try:
                        job_data = _read_json(j_path)
                        # Ensure project field is set correctly from dir structure if missing
                        if "project" not in job_data: job_data["project"] = proj_dir.name
                        
                        parsed = _parse_job(job_data)
                        if project and parsed.project != project: continue
                        by_id[parsed.id] = parsed
                    except Exception: pass
            
            # Check legacy: runs/project/job_id/job.json
            for job_dir in proj_dir.iterdir():
                if job_dir.name == "_jobs" or not job_dir.is_dir(): continue
                j_path = job_dir / "job.json"
                if j_path.exists():
                     try:
                         job_data = _read_json(j_path)
                         if "project" not in job_data: job_data["project"] = proj_dir.name
                         parsed = _parse_job(job_data)
                         # Prefer newer if duplicate ID (unlikely but safe)
                         if parsed.id not in by_id:
                             if project and parsed.project != project: continue
                             by_id[parsed.id] = parsed
                     except Exception: pass

    except Exception:
        # log error?
        pass

    # Load from disk legacy (baseline)
    legacy_dir = RUNS_DIR / "_jobs"
    if legacy_dir.exists():
        for path in legacy_dir.glob("*.json"):
            try:
                parsed = _parse_job(_read_json(path))
            except Exception:  # noqa: BLE001
                continue
            if project and parsed.project != project:
                continue
            by_id[parsed.id] = parsed

    # Overlay in-memory state (fresher)
    with _state_lock:
        for item in _jobs.values():
            if project and item.project != project:
                continue
            by_id[item.id] = item

    return list(by_id.values())


def _latest_job_for_project(project: str) -> JobInfo | None:
    jobs = _collect_jobs(project)
    if not jobs:
        return None
    return max(jobs, key=_job_sort_key)


def _running_job_for_project(project: str) -> JobInfo | None:
    jobs = [j for j in _collect_jobs(project) if j.status in {"queued", "running"}]
    if not jobs:
        return None
    return max(jobs, key=_job_sort_key)


def _active_job() -> JobInfo | None:
    jobs = [j for j in _collect_jobs() if j.status in {"queued", "running"}]
    if not jobs:
        return None
    return max(jobs, key=_job_sort_key)

def _cleanup_stale_jobs():
    # Cleanup stale jobs by iterating over all collected jobs
    # This runs on server start/reload.
    for job in _collect_jobs():
        if job.status in {"queued", "running"}:
            job.status = "failed"
            job.error = "Job interrupted by server restart."
            job.finished_at = utc_now_iso()
            job.push_event("error", "Server restarted during job execution.")
            
            # Save to disk (in-place update if possible)
            path = _find_job_path(job.id, job.project)
            if not path:
                 # If not found (weird), skip or default to new path
                 path = _new_job_path(job.project, job.id)
            
            try:
                _write_json(path, asdict(job))
            except Exception:
                pass

# Run cleanup immediately on module load (server start)
_cleanup_stale_jobs()


def _load_root_config() -> dict[str, Any]:
    return _read_json(ROOT_CONFIG_PATH)


def _load_project_config(project: str) -> dict[str, Any] | None:
    path = _project_config_path(project)
    if not path.exists():
        return None
    return _read_json(path)


def _effective_output_dir(config: dict[str, Any]) -> Path:
    if config.get("project"):
        return REPO_ROOT / "runs" / str(config["project"])
    return REPO_ROOT / str(config.get("output_dir", "output"))


def _sync_project_to_root(project: str) -> dict[str, Any]:
    with _config_lock:
        project_cfg = _load_project_config(project)
        if not project_cfg:
            raise HTTPException(status_code=404, detail=f"Project '{project}' not found")
        _write_json(ROOT_CONFIG_PATH, project_cfg)
        return project_cfg


def _resolve_label_command(config: dict[str, Any]) -> list[str]:
    mode = normalize_label_mode(config.get("label_mode", "gemini"))
    num_agents = int(config.get("num_agents", 1) or 1)
    is_windows = os.name == "nt"

    if mode == "cua+sam":
        return _script_command(".agents/skills/label/scripts/label_cua_sam.py")

    if is_windows and num_agents > 1:
        if mode in {"gemini", "gpt"}:
            return _script_command(".agents/skills/label/scripts/run_batch.py")
        return _script_command(".agents/skills/label/scripts/run.py")

    if num_agents > 1 and mode in {"gemini", "gpt", "codex"}:
        bash = shutil.which("bash")
        if bash:
            return [bash, ".agents/skills/label/scripts/dispatch.sh", str(num_agents)]

    return _script_command(".agents/skills/label/scripts/run.py")


def _script_command(script_path: str) -> list[str]:
    # Use python directly for simplicity, assuming deps installed or uv run if preferred
    uv_bin = shutil.which("uv")
    if uv_bin:
        return [uv_bin, "run", script_path]
    return [sys.executable, script_path]


def _latest_job_with_frames(project: str, exclude_job_id: str | None = None) -> JobInfo | None:
    jobs = _collect_jobs(project)
    jobs.sort(key=_job_sort_key, reverse=True)
    for j in jobs:
        if j.id == exclude_job_id: continue
        path = _new_job_path(j.project, j.id).parent / "artifacts" / "frames"
        if path.exists() and any(path.iterdir()):
            return j
    return None

def _phase_command(phase: str, job: JobInfo, config: dict[str, Any]) -> list[str]:
    # Unified pipeline execution
    # Use a single shared artifacts folder per project (simpler session model)
    output_dir = RUNS_DIR / job.project / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # If not starting from scratch (i.e. not 'collect'), and frames exist, utilize them directly.
    # The pipeline will pick up existing frames in output_dir/frames.
    
    pipeline_script = str(REPO_ROOT / "pipeline" / "main.py")
    
    video_url = config.get("video_url")
    if not video_url:
        # Fallback or error?
        # If collect phase needs it, it will fail inside pipeline.
        video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    label_mode = normalize_label_mode(config.get("label_mode", "gemini"))
    if label_mode == "gpt":
        provider = "openai"
    elif label_mode == "local":
        provider = "local"
    else:
        provider = "gemini"
    
    if provider == "gemini":
        model = config.get("gemini_model") or "gemini-2.5-flash-lite"
    elif provider == "openai":
        model = config.get("openai_model") or "gpt-4o"
    elif provider == "local":
        model = config.get("local_model_name") or "local-model"
    else:
        model = "gemini-2.5-flash-lite"

    cmd = [
        sys.executable, pipeline_script,
        video_url,
        "--output-dir", str(output_dir),
        "--phase", phase,
        "--provider", provider,
        "--model", str(model)
    ]
    return cmd


def _phases_for_mode(mode: RunMode) -> list[str]:
    if mode == "full":
        return ["collect", "label", "augment", "train", "eval"]
    return [mode]


def _run_subprocess(job: JobInfo, cmd: list[str], env_overrides: dict[str, str] | None = None) -> int:
    job.push_event("info", f"$ {' '.join(cmd)}")
    _save_job(job)
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except Exception as e:
        job.push_event("error", f"Failed to start subprocess: {e}")
        _save_job(job)
        return 1

    with _state_lock:
        _active_processes[job.id] = proc

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            job.push_event("log", line.rstrip("\n"))
            _save_job(job)
        
        proc.wait()
        return int(proc.returncode)
    finally:
        with _state_lock:
            _active_processes.pop(job.id, None)


def _classify_failure(job: JobInfo, phase: str, fallback_message: str) -> tuple[str, str, str, bool]:
    recent = "\n".join(event.get("message", "") for event in job.events[-300:])
    text = recent.lower()

    checks: list[tuple[str, str, str, bool]] = [
        (
            "missing_api_key_primary",
            "Primary Provider API key is missing.",
            "Set the primary API key in the GUI or backend environment and retry.",
            True,
        ),
        (
            "missing_api_key_secondary",
            "Secondary Provider API key is missing.",
            "Set the secondary API key in the GUI or backend environment and retry.",
            True,
        ),
        (
            "quota_exceeded",
            "Provider quota exceeded.",
            "Reduce requests (lower FPS / fewer frames), wait for quota reset, or use a billed API project.",
            True,
        ),
        (
            "model_unavailable",
            "Selected model is unavailable for current provider/API version.",
            "Pick a supported model (for this setup prefer `gemini-2.5-flash`) and retry.",
            True,
        ),
        (
            "malformed_model_json",
            "Model returned malformed JSON for too many frames.",
            "Retry with a stronger model or simplify class set/prompt; empty-frame fallback is already enabled.",
            True,
        ),
        (
            "network_or_provider_unavailable",
            "Provider/network temporary failure.",
            "Check internet connectivity/provider status and retry.",
            True,
        ),
        (
            "missing_dependency",
            "A required dependency is missing.",
            "Install missing packages (`uv sync`) and ensure tools like ffmpeg/yt-dlp are on PATH.",
            False,
        ),
        (
            "missing_executable",
            "A required executable was not found.",
            "Install missing executables (`ffmpeg`, `yt-dlp`, `uv`) and restart the server.",
            False,
        ),
        (
            "no_frames",
            "No frames were available for labeling.",
            "Run collect successfully first and verify frame extraction output.",
            False,
        ),
    ]

    if "gemini_api_key or google_api_key is not set" in text:
        return checks[0]
    if "openai_api_key is not set" in text:
        return checks[1]
    if "quota exceeded" in text or "exceeded your current quota" in text or re.search(r"\berror:\s*429\b", text):
        return checks[2]
    if "models/" in text and "not found" in text:
        return checks[3]
    if "malformed model json" in text:
        return checks[4]
    if "permissiondenied" in text or "resourceexhausted" in text or "timed out" in text or "ssl" in text:
        return checks[5]
    if "no module named" in text or "is not installed" in text:
        return checks[6]
    if "required executable not found" in text or "[winerror 2]" in text:
        return checks[7]
    if "no frames found" in text:
        return checks[8]

    return (
        "phase_failed",
        f"Phase '{phase}' failed.",
        fallback_message,
        True,
    )


def _cleanup_partial_artifacts(job: JobInfo):
    """Clean up artifacts created during a failed or cancelled job phase.
    
    This is a destructive operation requested to ensure clean state after cancellation.
    - collect: deletes the frames/ directory (all frames).
    - train: deletes the weights/ directory and detect/ runs.
    - label: does NOT bulk delete to avoid destroying paid API results.
    - augment: deletes the augmented/ directory.
    """
    try:
        if not job.project: # Should not happen
             return
        
        # New artifact location (shared per project)
        output_dir = RUNS_DIR / job.project / "artifacts"
        phase = job.current_phase
        
        if not phase:
            return

        if phase == "collect":
            frames_dir = output_dir / "frames"
            if frames_dir.exists():
                shutil.rmtree(frames_dir, ignore_errors=True)
                job.push_event("info", f"Cleanup: Deleted partial frames in {frames_dir}")

        elif phase == "augment":
            aug_dir = output_dir / "augmented"
            if aug_dir.exists():
                shutil.rmtree(aug_dir, ignore_errors=True)
                job.push_event("info", f"Cleanup: Deleted partial augmented data in {aug_dir}")

        elif phase == "train":
            weights_dir = output_dir / "weights"
            detect_dir = output_dir / "detect"
            if weights_dir.exists():
                shutil.rmtree(weights_dir, ignore_errors=True)
            if detect_dir.exists():
                shutil.rmtree(detect_dir, ignore_errors=True)
            job.push_event("info", "Cleanup: Deleted partial training weights and logs.")

        elif phase == "eval":
            # Just results json?
            res = output_dir / "eval_results.json"
            if res.exists():
                res.unlink()
                job.push_event("info", "Cleanup: Deleted partial evaluation results.")

    except Exception as e:
        job.push_event("error", f"Cleanup failed: {e}")


def _job_runner(job_id: str) -> None:
    global _active_job_id
    with _state_lock:
        job = _load_job(job_id)
        if job is None:
            return
        job.status = "running"
        job.started_at = utc_now_iso()
        _active_job_id = job_id
        _save_job(job, force=True)

    try:
        config = _sync_project_to_root(job.project)
        with _state_lock:
            env_overrides = dict(_job_env_overrides.get(job_id, {}))

        for index, phase in enumerate(job.phases, start=1):
            with _state_lock:
                job.phase_index = index
                job.current_phase = phase
                _save_job(job, force=True)
            job.push_event("info", f"Starting phase {index}/{job.total_phases}: {phase}")
            _save_job(job)

            # Updated to pass 'job'
            cmd = _phase_command(phase, job, config)
            code = _run_subprocess(job, cmd, env_overrides=env_overrides)
            if code != 0:
                with _state_lock:
                    fallback = "Inspect job logs for details and retry after resolving the underlying issue."
                    error_code, friendly_error, error_hint, retryable = _classify_failure(job, phase, fallback)
                    job.status = "failed"
                    job.exit_code = code
                    job.error_code = error_code
                    job.error = f"{friendly_error} (exit code {code})"
                    job.error_hint = error_hint
                    job.retryable = retryable
                    job.finished_at = utc_now_iso()
                    _project_running_job.pop(job.project, None)
                    _active_job_id = None
                    _job_env_overrides.pop(job_id, None)
                job.push_event("error", job.error)
                job.push_event("error", f"Hint: {job.error_hint}")
                _save_job(job)
                # Remove large artifacts in the shared project artifacts folder to prevent bloat for failed jobs
                try:
                    artifacts_dir = RUNS_DIR / job.project / "artifacts"
                    if artifacts_dir.exists():
                        shutil.rmtree(artifacts_dir, ignore_errors=True)
                        job.push_event("info", f"Cleanup: deleted artifacts for failed job {job.id}")
                except Exception:
                    pass
                return

        with _state_lock:
            job.status = "completed"
            job.exit_code = 0
            job.finished_at = utc_now_iso()
            _project_running_job.pop(job.project, None)
            _active_job_id = None
            _job_env_overrides.pop(job_id, None)
        job.push_event("info", "Job completed successfully")
        _save_job(job)

    except Exception as exc:  # noqa: BLE001
        with _state_lock:
            fallback = f"Unhandled exception: {exc}"
            error_code, friendly_error, error_hint, retryable = _classify_failure(job, str(job.current_phase or "unknown"), fallback)
            job.status = "failed"
            job.error_code = error_code
            job.error = f"{friendly_error} ({exc})"
            job.error_hint = error_hint
            job.retryable = retryable
            job.finished_at = utc_now_iso()
            _project_running_job.pop(job.project, None)
            _active_job_id = None
            _job_env_overrides.pop(job_id, None)
        job.push_event("error", f"Unhandled error: {exc}")
        job.push_event("error", f"Hint: {job.error_hint}")
        _save_job(job)
        # Cleanup artifacts for failed job to avoid disk bloat (shared project artifacts)
        try:
            artifacts_dir = RUNS_DIR / job.project / "artifacts"
            if artifacts_dir.exists():
                shutil.rmtree(artifacts_dir, ignore_errors=True)
                job.push_event("info", f"Cleanup: deleted artifacts for failed job {job.id}")
        except Exception:
            pass


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/system/cancel_active")
def cancel_active_job() -> dict[str, Any]:
    global _active_job_id
    with _state_lock:
        if not _active_job_id:
            # If no active job ID is known, but process might be stuck, we can't do much without ID.
            # But let's check widely if any job thinks it is running.
            running_jobs = [j for j in _jobs.values() if j.status == "running"]
            if running_jobs:
                 # Pick the one that started most recently
                 job = max(running_jobs, key=_job_sort_key)
                 running_job_id = job.id
                 _active_job_id = running_job_id # Re-anchor state
            else:
                 raise HTTPException(status_code=404, detail="No active global job found")
        else:
            running_job_id = _active_job_id
            job = _jobs.get(running_job_id)
        
        if not job: # Active ID set but job missing from memory?
            _active_job_id = None
            raise HTTPException(status_code=404, detail="Active job state inconsistent")

    # Cancel whatever job is running (could belong to any project)
    job.status = "failed"
    job.error = "Job cancelled by global system request."
    job.push_event("error", "Global cancellation requested.")
    _save_job(job, force=True)

    proc = _active_processes.get(running_job_id)
    if proc:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:
            pass 
            
    # Cleanup artifacts for the interrupted phase
    _cleanup_partial_artifacts(job)
    
    # Cleanup global state
    with _state_lock:
        if job.project in _project_running_job:
            _project_running_job.pop(job.project, None)
        
        if _active_job_id == running_job_id:
            _active_job_id = None
        
        _job_env_overrides.pop(running_job_id, None)

    return {"status": "cancelled", "job_id": running_job_id}


@app.post("/api/projects")
def upsert_project(payload: ProjectConfigUpsert) -> dict[str, Any]:
    with _config_lock:
        base = _load_project_config(payload.project) or _load_root_config()
        updates = payload.model_dump(exclude_none=True)
        for key, value in updates.items():
            base[key] = value

        base["project"] = payload.project
        if "output_dir" in base:
            base["output_dir"] = "output"

        project_cfg_path = _project_config_path(payload.project)
        _write_json(project_cfg_path, base)
        # Decouple: Do not overwrite global config indiscriminately.
        # _write_json(ROOT_CONFIG_PATH, base) 

    output_dir = _effective_output_dir(base)
    return {
        "project": payload.project,
        "config_path": str(project_cfg_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "output_dir": str(output_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
        "config": load_config(ROOT_CONFIG_PATH),
    }


@app.get("/api/projects/{project}")
def get_project(project: str) -> dict[str, Any]:
    cfg = _load_project_config(project)
    if not cfg:
        raise HTTPException(status_code=404, detail=f"Project '{project}' not found")

    running = _running_job_for_project(project)
    latest = _latest_job_for_project(project)

    output_dir = _effective_output_dir(cfg)
    return {
        "project": project,
        "config": cfg,
        "output_dir": str(output_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
        "running_job_id": running.id if running else None,
        "latest_job_id": latest.id if latest else None,
    }


@app.post("/api/projects/{project}/run")
def run_project(project: str, payload: RunRequest) -> JSONResponse:
    global _active_job_id
    cfg = _load_project_config(project)
    if not cfg:
        raise HTTPException(status_code=404, detail=f"Project '{project}' not found")

    with _state_lock:
        active = _active_job()
        if active is not None:
            raise HTTPException(status_code=409, detail=f"Another job is already running: {active.id}")

        if project in _project_running_job:
            active = _project_running_job[project]
            raise HTTPException(status_code=409, detail=f"Project already running job: {active}")

        phases = _phases_for_mode(payload.mode)
        job_id = uuid.uuid4().hex
        job = JobInfo(
            id=job_id,
            project=project,
            requested_mode=payload.mode,
            phases=phases,
            source_job_id=payload.source_job_id,
            total_phases=len(phases),
        )
        job.push_event("info", f"Job queued with mode={payload.mode}")
        _jobs[job_id] = job
        _project_latest_job[project] = job_id
        _project_running_job[project] = job_id
        _active_job_id = job_id
        _save_job(job)
        env_overrides: dict[str, str] = {}
        if payload.gemini_api_key:
            env_overrides["GEMINI_API_KEY"] = payload.gemini_api_key
        if payload.google_api_key:
            env_overrides["GOOGLE_API_KEY"] = payload.google_api_key
        if payload.openai_api_key:
            env_overrides["OPENAI_API_KEY"] = payload.openai_api_key
        if payload.local_vlm_api_key:
            env_overrides["LOCAL_VLM_API_KEY"] = payload.local_vlm_api_key
        _job_env_overrides[job_id] = env_overrides

    try:
        thread = threading.Thread(target=_job_runner, args=(job_id,), daemon=True)
        thread.start()
    except Exception as exc:  # noqa: BLE001
        with _state_lock:
            failed = _jobs.get(job_id)
            if failed:
                failed.status = "failed"
                failed.error = f"Failed to start worker thread: {exc}"
                failed.error_code = "worker_start_failed"
                failed.error_hint = "Check server resources and thread/runtime configuration, then retry."
                failed.retryable = True
                failed.finished_at = utc_now_iso()
                _save_job(failed)
            _project_running_job.pop(project, None)
            if _active_job_id == job_id:
                _active_job_id = None
            _job_env_overrides.pop(job_id, None)
        raise HTTPException(status_code=500, detail=f"Failed to start job worker: {exc}") from exc

    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "project": project,
            "status": "queued",
            "phases": phases,
        },
    )


@app.post("/api/projects/{project}/cancel")
def cancel_job(project: str) -> dict[str, Any]:
    global _active_job_id
    with _state_lock:
        running_job_id = _project_running_job.get(project)
        # Also check if it's the globally running job
        if not running_job_id and _active_job_id:
            active_job = _jobs.get(_active_job_id)
            if active_job and active_job.project == project:
                 running_job_id = _active_job_id

        if not running_job_id:
             raise HTTPException(status_code=404, detail="No running job found to cancel")

        job = _jobs.get(running_job_id)
        if job:
            job.status = "failed" # Mark as failed/cancelled
            job.error = "Job cancelled by user request."
            job.push_event("error", "Job cancellation requested.")
            _save_job(job, force=True)

        # Terminate the process if it exists
        proc = _active_processes.get(running_job_id)
        if proc:
            try:
                proc.terminate()
                # Give it a moment to terminate gracefully
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception:
                pass 
        
        # Cleanup
        if job:
            _cleanup_partial_artifacts(job)
        
        # Cleanup state
        _project_running_job.pop(project, None)
        if _active_job_id == running_job_id:
             _active_job_id = None
        _job_env_overrides.pop(running_job_id, None)

    return {"status": "cancelled", "job_id": running_job_id}


@app.get("/api/projects/{project}/jobs/{job_id}")
def get_job(project: str, job_id: str) -> dict[str, Any]:
    with _state_lock:
        job = _load_job(job_id)
    if not job or job.project != project:
        raise HTTPException(status_code=404, detail="Job not found")
    return asdict(job)


def _find_metrics(project: str, job_id: str | None = None) -> dict[str, Any] | None:
    if job_id:
        job = _load_job(job_id)
        candidate_jobs = [job] if (job and job.project == project) else []
    else:
        candidate_jobs = _collect_jobs(project)
        # Sort so we check newest first (most recently created/started)
        candidate_jobs.sort(key=_job_sort_key, reverse=True)
    
    for job in candidate_jobs:
        # Check new path: runs/<project>/<job_id>/artifacts/eval_results.json
        job_path = _find_job_path(job.id, project) or _new_job_path(project, job.id)
        job_dir = job_path.parent
        path = job_dir / "artifacts" / "eval_results.json"
        
        if path.exists():
            try:
                data = _read_json(path)
                if data: return data
            except Exception: pass
            
    # Legacy fallback: check project root
    project_dir = RUNS_DIR / project
    legacy_path = project_dir / "eval_results.json"
    if legacy_path.exists():
        try:
             return _read_json(legacy_path)
        except Exception: pass

    return None

@app.get("/api/projects/{project}/status")
def get_status(project: str) -> dict[str, Any]:
    cfg = _load_project_config(project)
    if not cfg:
        raise HTTPException(status_code=404, detail=f"Project '{project}' not found")

    metrics = _find_metrics(project)

    with _state_lock:
        running_job = _running_job_for_project(project)
        latest_job = _latest_job_for_project(project)
        active_job = _active_job()
        running_state = asdict(running_job) if running_job else None

    return {
        "project": project,
        "active_job_id": active_job.id if active_job else None,
        "running_job_id": running_job.id if running_job else None,
        "latest_job_id": latest_job.id if latest_job else None,
        "running_job": running_state,
        "metrics": metrics,
    }


@app.get("/api/projects/{project}/metrics")
def get_metrics(project: str, job_id: str | None = None) -> dict[str, Any]:
    metrics = _find_metrics(project, job_id)
    if metrics is None:
        raise HTTPException(status_code=404, detail="No evaluation results found")
    return metrics


@app.get("/api/projects/{project}/artifacts")
def get_artifacts(project: str, job_id: str | None = None) -> dict[str, Any]:
    # Single shared artifact folder per project
    output_dir = RUNS_DIR / project / "artifacts"
    
    frames_dir = output_dir / "frames"
    preview_dir = frames_dir / "preview"
    weights_dir = output_dir / "weights"

    def count_files(p: Path, pattern: str) -> int:
        if not p.exists(): return 0
        try:
             return len([f for f in p.glob(pattern) if f.is_file()])
        except Exception: return 0

    return {
        "job_id": job_id or "shared",
        "output_dir": str(output_dir.relative_to(REPO_ROOT)).replace("\\", "/") if output_dir.exists() else "",
        "classes_file": (output_dir / "classes.txt").exists(),
        "dataset_yaml": (output_dir / "dataset.yaml").exists(),
        "eval_results": (output_dir / "eval_results.json").exists(),
        "frame_count": count_files(frames_dir, "*.jpg"),
        "label_count": count_files(frames_dir, "*.txt"),
        "preview_count": count_files(preview_dir, "*.jpg"),
        "best_weight": (weights_dir / "best.pt").exists(),
    }


@app.get("/api/projects/{project}/events")
async def stream_events(
    project: str,
    job_id: str | None = Query(default=None),
    since: int = Query(default=0, ge=0),
) -> StreamingResponse:
    if job_id is None:
        latest = _latest_job_for_project(project)
        job_id = latest.id if latest else None
    if not job_id:
        raise HTTPException(status_code=404, detail="No jobs found for project")

    with _state_lock:
        job = _load_job(job_id)
    if not job or job.project != project:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_gen() -> AsyncGenerator[str, None]:
        cursor = since
        while True:
            with _state_lock:
                local = _load_job(job_id)
                if not local:
                    break
                new_events = [e for e in local.events if e["seq"] > cursor]
                done = local.status in {"completed", "failed"}

            for event in new_events:
                cursor = int(event["seq"])
                payload = json.dumps(event)
                yield f"id: {cursor}\nevent: {event['level']}\ndata: {payload}\n\n"

            if done and not new_events:
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(event_gen(), media_type="text/event-stream")
