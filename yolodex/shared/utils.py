"""Shared utilities for the Yolodex pipeline."""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class BoundingBox:
    class_name: str
    x: float
    y: float
    width: float
    height: float


class PipelineError(RuntimeError):
    """Raised when a pipeline step fails."""


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OPENAI_MODEL = "gpt-5-nano"

GEMINI_MODEL_ALIASES = {
    "gemini-1.5-flash": "gemini-1.5-flash-latest",
    "gemini-1.5-flash-lite": "gemini-1.5-flash-8b",
}


def run_command(cmd: list[str]) -> None:
    """Run a subprocess command and raise with readable context on failure."""
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise PipelineError(f"Required executable not found: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise PipelineError(f"Command failed ({exc.returncode}): {' '.join(cmd)}") from exc


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load config.json from the repo root. Resolves output_dir to runs/<project>/ when project is set."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("project"):
        config["output_dir"] = f"runs/{config['project']}"
    return config


def normalize_label_mode(raw_mode: Any) -> str:
    mode = str(raw_mode or "").strip().lower()
    aliases = {
        "": "gemini",
        "google": "gemini",
        "google-gemini": "gemini",
        "openai": "gpt",
        "cua_sam": "cua+sam",
        "sam": "cua+sam",
        "lm_studio": "local",
        "local_vlm": "local",
    }
    return aliases.get(mode, mode)


def resolve_label_backend(config: dict[str, Any]) -> tuple[str, str, str | None]:
    mode = normalize_label_mode(config.get("label_mode", "gemini"))
    generic_model = str(config.get("model", "")).strip()
    gemini_model = str(config.get("gemini_model", "")).strip()
    openai_model = str(config.get("openai_model", "")).strip()
    local_model = str(config.get("local_model_name", "")).strip()

    if mode == "gemini":
        requested = gemini_model or generic_model or DEFAULT_GEMINI_MODEL
        mapped = GEMINI_MODEL_ALIASES.get(requested, requested)

        if mapped.startswith("gemma-") or mapped.startswith("gemini-3"):
            note = (
                f"Model '{requested}' is not supported by the current Gemini SDK path; "
                f"falling back to '{DEFAULT_GEMINI_MODEL}'."
            )
            return ("gemini", DEFAULT_GEMINI_MODEL, note)

        if mapped != requested:
            note = f"Model '{requested}' mapped to '{mapped}' for compatibility."
            return ("gemini", mapped, note)

        return ("gemini", mapped, None)

    if mode == "gpt":
        return ("openai", openai_model or generic_model or DEFAULT_OPENAI_MODEL, None)

    if mode == "local":
        return ("local", local_model or generic_model or "local-model", None)

    return (mode, generic_model, None)


def encode_image_base64(image_path: Path) -> str:
    image_bytes = image_path.read_bytes()
    return base64.b64encode(image_bytes).decode("utf-8")


def extract_json_from_text(text: str) -> dict[str, Any]:
    """Parse JSON directly; fallback to extracting from markdown code fences or custom markers."""
    # 1. Strip out <think>...</think> blocks if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

    # 2. Extract content between <|begin_of_box|> and <|end_of_box|> if present
    box_match = re.search(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", text, flags=re.DOTALL)
    if box_match:
        text = box_match.group(1).strip()

    # 3. Prefer TOON-decoded responses when available (LLM interactions may use TOON).
    try:
        from shared.toon_integration import decode as toon_decode, toon_available  # type: ignore

        if toon_available():
            try:
                decoded = toon_decode(text)
                if isinstance(decoded, dict):
                    return decoded
            except Exception:
                # Fall back to JSON parsing below
                pass
    except Exception:
        pass

    # 4. Try standard JSON parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 5. Fallback: extract JSON from markdown code fences or loose braces
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            # If plain text fails and no JSON/TOON found, raise error
            raise PipelineError(f"Model did not return valid JSON or TOON. Output was: {text[:200]}...")
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise PipelineError("Model returned malformed JSON.") from exc


def read_image_dimensions(frame_path: Path) -> tuple[int, int]:
    """Read width/height via ffprobe to avoid extra Python imaging deps."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(frame_path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        return int(stream["width"]), int(stream["height"])
    except Exception as exc:  # noqa: BLE001
        raise PipelineError(f"Failed to read dimensions for {frame_path}") from exc
