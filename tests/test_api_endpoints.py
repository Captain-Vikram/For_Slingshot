from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
import sys

# Ensure repo root is on sys.path so sibling packages like `shared` can be imported
repo_root = Path(__file__).resolve().parents[1]
# Add repo root and the `phase1_2` package dir so `shared` (phase1_2/shared) is importable as top-level
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "phase1_2"))

import phase1_2.api.main as api_main


class DummyThread:
    def __init__(self, target=None, args=None, daemon=None):
        self.target = target
        self.args = args or ()

    def start(self):
        return None


class APITestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.repo_root = Path(self._tmp_dir.name)
        (self.repo_root / "runs" / "_configs").mkdir(parents=True, exist_ok=True)
        (self.repo_root / "runs" / "_jobs").mkdir(parents=True, exist_ok=True)

        base_config = {
            "project": "",
            "video_url": "",
            "classes": [],
            "target_accuracy": 0.75,
            "max_iterations": 10,
            "num_agents": 1,
            "fps": 1,
            "output_dir": "output",
            "label_mode": "gemini",
            "model": "gemini-2.5-flash",
            "openai_model": "gpt-5-nano",
            "gemini_model": "gemini-2.5-flash",
            "yolo_model": "yolov8n.pt",
            "epochs": 5,
            "train_split": 0.8,
        }
        (self.repo_root / "config.json").write_text(json.dumps(base_config), encoding="utf-8")

        self.repo_patch = patch.object(api_main, "REPO_ROOT", self.repo_root)
        self.root_cfg_patch = patch.object(api_main, "ROOT_CONFIG_PATH", self.repo_root / "config.json")
        self.runs_dir_patch = patch.object(api_main, "RUNS_DIR", self.repo_root / "runs")

        self.repo_patch.start()
        self.root_cfg_patch.start()
        self.runs_dir_patch.start()

        self._reset_runtime_state()
        self.client = TestClient(api_main.app)

    def tearDown(self) -> None:
        self.repo_patch.stop()
        self.root_cfg_patch.stop()
        self.runs_dir_patch.stop()
        self._tmp_dir.cleanup()

    def _reset_runtime_state(self) -> None:
        api_main._jobs.clear()
        api_main._project_latest_job.clear()
        api_main._project_running_job.clear()
        api_main._active_job_id = None
        api_main._job_env_overrides.clear()
        # Clean up jobs in runs dir
        runs_dir = self.repo_root / "runs"
        if runs_dir.exists():
            for proj in runs_dir.iterdir():
                if proj.is_dir():
                    for job_dir in proj.iterdir():
                         if job_dir.is_dir() and (job_dir / "job.json").exists():
                             import shutil
                             shutil.rmtree(job_dir, ignore_errors=True)

    def _create_project(self, name: str = "demo-project") -> dict:
        payload = {
            "project": name,
            "video_url": "https://www.youtube.com/watch?v=abc123",
            "classes": ["player", "weapon"],
            "label_mode": "gemini",
            "model": "gemini-2.5-flash",
            "num_agents": 2,
        }
        response = self.client.post("/api/projects", json=payload)
        self.assertEqual(response.status_code, 200)
        return response.json()

    def test_health_route(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_project_upsert_and_get(self) -> None:
        created = self._create_project("fruit-api")
        self.assertEqual(created["project"], "fruit-api")
        self.assertEqual(created["output_dir"], "runs/fruit-api")

        get_response = self.client.get("/api/projects/fruit-api")
        self.assertEqual(get_response.status_code, 200)
        body = get_response.json()
        self.assertEqual(body["project"], "fruit-api")
        self.assertEqual(body["output_dir"], "runs/fruit-api")

    def test_run_route_queues_job(self) -> None:
        self._create_project("run-check")

        with patch("api.main.threading.Thread", DummyThread):
            response = self.client.post("/api/projects/run-check/run", json={"mode": "full"})

        self.assertEqual(response.status_code, 202)
        data = response.json()
        self.assertEqual(data["project"], "run-check")
        self.assertEqual(data["status"], "queued")
        self.assertEqual(data["phases"], ["collect", "label", "augment", "train", "eval"])

    def test_run_route_blocks_when_project_already_running(self) -> None:
        self._create_project("busy-project")
        api_main._project_running_job["busy-project"] = "busy-job"

        response = self.client.post("/api/projects/busy-project/run", json={"mode": "eval"})
        self.assertEqual(response.status_code, 409)

    def test_metrics_route_not_found_then_success(self) -> None:
        self._create_project("metrics-project")

        not_found = self.client.get("/api/projects/metrics-project/metrics")
        self.assertEqual(not_found.status_code, 404)

        # Create dummy job+artifacts
        job_id = "job-metrics-1"
        job_dir = self.repo_root / "runs" / "metrics-project" / job_id
        artifacts = job_dir / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        
        job_data = {
            "id": job_id,
            "project": "metrics-project",
            "requested_mode": "eval",
            "phases": ["eval"],
            "status": "completed",
            "created_at": "2024-01-01T00:00:00+00:00"
        }
        (job_dir / "job.json").write_text(json.dumps(job_data), encoding="utf-8")

        metric_payload = {"map50": 0.88, "meets_target": True}
        (artifacts / "eval_results.json").write_text(json.dumps(metric_payload), encoding="utf-8")

        ok = self.client.get("/api/projects/metrics-project/metrics")
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(ok.json()["map50"], 0.88)

    def test_artifacts_route_counts_outputs(self) -> None:
        self._create_project("artifact-project")
        
        job_id = "job-art-1"
        job_dir = self.repo_root / "runs" / "artifact-project" / job_id
        output_dir = job_dir / "artifacts"
        frames = output_dir / "frames"
        preview = frames / "preview"
        weights = output_dir / "weights"

        frames.mkdir(parents=True, exist_ok=True)
        preview.mkdir(parents=True, exist_ok=True)
        weights.mkdir(parents=True, exist_ok=True)

        job_data = {
            "id": job_id,
            "project": "artifact-project",
            "requested_mode": "full",
            "phases": ["collect", "label"],
            "status": "completed",
            "created_at": "2024-01-01T00:00:00+00:00"
        }
        (job_dir / "job.json").write_text(json.dumps(job_data), encoding="utf-8")

        (output_dir / "classes.txt").write_text("player\nweapon\n", encoding="utf-8")
        (output_dir / "dataset.yaml").write_text("names: [player, weapon]", encoding="utf-8")
        (output_dir / "eval_results.json").write_text('{"map50":0.5}', encoding="utf-8")
        (weights / "best.pt").write_bytes(b"weights")

        for idx in range(2):
            (frames / f"frame_{idx:06d}.jpg").write_bytes(b"jpg")
            (frames / f"frame_{idx:06d}.txt").write_text("0 0.5 0.5 0.2 0.2", encoding="utf-8")
        (preview / "frame_000000_preview.jpg").write_bytes(b"preview")

        # Explicitly query for this job ID to ensure test isolation
        response = self.client.get(f"/api/projects/artifact-project/artifacts?job_id={job_id}")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["frame_count"], 2)
        self.assertEqual(body["label_count"], 2)
        self.assertEqual(body["preview_count"], 1)
        self.assertTrue(body["best_weight"])

    def test_status_includes_running_job_and_metrics(self) -> None:
        self._create_project("status-project")
        
        # Create a completed job with metrics to verify they are picked up
        completed_job_id = "job-status-0"
        completed_job_dir = self.repo_root / "runs" / "status-project" / completed_job_id
        artifacts = completed_job_dir / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        (artifacts / "eval_results.json").write_text('{"map50":0.91}', encoding="utf-8")
        
        completed_job_info = {
             "id": completed_job_id, 
             "project": "status-project",
             "requested_mode": "eval",
             "phases": ["eval"], 
             "status": "completed",
             "created_at": "2024-01-01T00:00:00+00:00"
        }
        (completed_job_dir / "job.json").write_text(json.dumps(completed_job_info), encoding="utf-8")

        job = api_main.JobInfo(
            id="job-status-1",
            project="status-project",
            requested_mode="eval",
            phases=["eval"],
            status="running",
            total_phases=1,
            phase_index=1,
            current_phase="eval",
        )
        api_main._save_job(job)

        response = self.client.get("/api/projects/status-project/status")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["active_job_id"], "job-status-1")
        self.assertEqual(body["running_job_id"], "job-status-1")
        # Ensure metrics are picked up from the completed job history
        self.assertEqual(body["metrics"]["map50"], 0.91)

    def test_run_route_blocks_when_another_project_job_is_active(self) -> None:
        self._create_project("project-a")
        self._create_project("project-b")

        job = api_main.JobInfo(
            id="active-global-job",
            project="project-a",
            requested_mode="collect",
            phases=["collect"],
            status="running",
            total_phases=1,
            phase_index=1,
            current_phase="collect",
        )
        api_main._save_job(job)

        response = self.client.post("/api/projects/project-b/run", json={"mode": "eval"})
        self.assertEqual(response.status_code, 409)
        self.assertIn("Another job is already running", response.json()["detail"])

    def test_events_route_streams_job_events(self) -> None:
        self._create_project("event-project")

        job = api_main.JobInfo(
            id="job-event-1",
            project="event-project",
            requested_mode="eval",
            phases=["eval"],
            status="completed",
            total_phases=1,
            phase_index=1,
            current_phase="eval",
        )
        job.push_event("info", "Job queued")
        job.push_event("info", "Job completed successfully")

        api_main._save_job(job)

        with self.client.stream("GET", "/api/projects/event-project/events") as response:
            self.assertEqual(response.status_code, 200)
            payload = "".join(response.iter_text())

        self.assertIn("event: info", payload)
        self.assertIn("Job completed successfully", payload)


if __name__ == "__main__":
    unittest.main()
