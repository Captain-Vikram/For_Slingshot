#!/usr/bin/env python3
"""Label skill (single-agent mode): label frames sequentially using Gemini or OpenAI."""

from __future__ import annotations

import os
import re
import sys
import subprocess
import time
import ast
from pathlib import Path
from typing import Any

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from openai import OpenAI

from shared.utils import (
    BoundingBox,
    DEFAULT_GEMINI_MODEL,
    PipelineError,
    clamp,
    encode_image_base64,
    extract_json_from_text,
    load_config,
    normalize_label_mode,
    read_image_dimensions,
    resolve_label_backend,
)

MULTI_CLASS_PROMPT_TEMPLATE = """
Detect every visible object in this image and return bounding boxes.
{class_hint}
Rules:
- x,y,width,height must be pixel values in the original image.
- x,y is top-left corner.
- Include all salient objects.
""".strip()

SINGLE_CLASS_PROMPT_TEMPLATE = """
Detect only objects of class "{class_name}" in this image and return bounding boxes.
Rules:
- Return only "{class_name}" objects. Ignore every other class.
- If no "{class_name}" is visible, return an empty list.
- x,y,width,height must be pixel values in the original image.
- x,y is top-left corner.
""".strip()

# Structured output schema — the API enforces this, no more JSON parsing failures
RESPONSE_SCHEMA = {
    "type": "json_schema",
    "name": "bounding_boxes",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "objects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "class_name": {"type": "string"},
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "width": {"type": "number"},
                        "height": {"type": "number"},
                    },
                    "required": ["class_name", "x", "y", "width", "height"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["objects"],
        "additionalProperties": False,
    },
}


def build_prompt(classes: list[str]) -> str:
    if classes:
        hint = f"Focus on these classes: {', '.join(classes)}."
    else:
        hint = "Include all salient objects (people, vehicles, UI elements, weapons, items, enemies, etc.)."
    return MULTI_CLASS_PROMPT_TEMPLATE.format(class_hint=hint)


def build_single_class_prompt(class_name: str) -> str:
    return SINGLE_CLASS_PROMPT_TEMPLATE.format(class_name=class_name)


def detect_objects_openai(client: OpenAI, model: str, frame_path: Path, prompt: str) -> list[BoundingBox]:
    image_b64 = encode_image_base64(frame_path)

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{image_b64}",
                    },
                ],
            }
        ],
        text={"format": RESPONSE_SCHEMA},
    )

    payload = extract_json_from_text(response.output_text)
    objects = payload.get("objects", [])

    boxes: list[BoundingBox] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        try:
            boxes.append(
                BoundingBox(
                    class_name=str(obj["class_name"]).strip().lower().replace(" ", "_"),
                    x=float(obj["x"]),
                    y=float(obj["y"]),
                    width=float(obj["width"]),
                    height=float(obj["height"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return boxes


def detect_objects_gemini(model: Any, frame_path: Path, classes: list[str]) -> list[BoundingBox]:
    from PIL import Image

    img = Image.open(frame_path)
    img_w, img_h = img.size

    class_hint = f"Focus on these classes: {', '.join(classes)}." if classes else "Detect all visible objects."
    prompt = (
        "Detect objects in this image and return bounding boxes. "
        f"{class_hint}\n\n"
        'Return strict JSON: {"objects": [{"label": "class_name", "box_2d": [y_min, x_min, y_max, x_max]}]}\n'
        "box_2d coordinates must be in 0-1000 normalized scale."
    )

    response = model.generate_content([prompt, img])
    payload = extract_json_from_text(response.text)

    objects = payload.get("objects", [])
    boxes: list[BoundingBox] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        try:
            label = str(obj.get("label", "")).strip().lower().replace(" ", "_")
            box_2d = obj["box_2d"]
            y_min = float(box_2d[0]) / 1000.0 * img_h
            x_min = float(box_2d[1]) / 1000.0 * img_w
            y_max = float(box_2d[2]) / 1000.0 * img_h
            x_max = float(box_2d[3]) / 1000.0 * img_w
            if not label:
                continue
            boxes.append(
                BoundingBox(
                    class_name=label,
                    x=x_min,
                    y=y_min,
                    width=x_max - x_min,
                    height=y_max - y_min,
                )
            )
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    return boxes


def detect_objects_local(client: OpenAI, model: str, frame_path: Path, prompt: str) -> list[BoundingBox]:
    image_b64 = encode_image_base64(frame_path)
    
    # Enrich prompt with explicit schema instruction for local models that don't support strict schemas
    width, height = read_image_dimensions(frame_path)
    # Create a simplified prompt because some models get confused by too much instruction
    full_prompt = (
        f"{prompt}\n\n"
        f"Return the result ONLY as a JSON object. No markdown, no explanations.\n"
        f"Schema: {{'objects': [{{'class_name': 'string', 'x': int, 'y': int, 'width': int, 'height': int}}]}}\n"
        f"Image size is {width}x{height} pixels. Coordinates must be absolute."
    )

    # Debug: Confirm to logs that we are sending request
    print(f"[label] Sending request to Local VLM ({model}) for {frame_path.name}...")

    try:
        # Retry logic for local models often helps
        response = None
        for attempt in range(2):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": full_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                                },
                            ],
                        }
                    ],
                    temperature=0.1, 
                    max_tokens=2048,
                )
                break
            except Exception as e:
                print(f"[label] Local attempt {attempt+1} failed: {e}")
                time.sleep(1)
        
        if not response:
            return []

        content = response.choices[0].message.content or "{}"
        
        # Debug: Print the raw content to see what is happening
        print(f"[label] RAW RESPONSE ({frame_path.name}): {content[:200]}...") # Limit length to avoid spamming logs

        if not content.strip():
            print(f"[label] Warning: Local VLM returned empty response for {frame_path.name}")
            return []

        try:
            payload = extract_json_from_text(content)
        except Exception:
             # Fallback: sometimes local models output valid python dict string instead of json
             try:
                 import ast
                 # clean markdown
                 clean = content.replace("```json", "").replace("```", "").strip()
                 payload = ast.literal_eval(clean)
             except Exception:
                 print(f"[label] Warning: Could not parse JSON from local model response on {frame_path.name}. Raw: {content[:50]}...")
                 return []
                 
        objects = payload.get("objects", [])

        boxes: list[BoundingBox] = []
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            try:
                # Handle potential variations in key names from local models
                # Common local model hallucinations: 'label', 'box_2d', 'bbox', 'class'
                cls = obj.get("class_name") or obj.get("label") or obj.get("object") or obj.get("class") or "unknown"
                
                # Check for different coordinate formats
                x = obj.get("x")
                y = obj.get("y")
                w = obj.get("width")
                h = obj.get("height")

                # Some models return [x, y, w, h] or similar lists
                if x is None and "bbox" in obj:
                     continue # TODO: handle bbox list if commonly returned
                
                # Ensure we have numbers
                if x is not None and y is not None and w is not None and h is not None:
                     boxes.append(
                        BoundingBox(
                            class_name=str(cls).strip().lower().replace(" ", "_"),
                            x=float(x),
                            y=float(y),
                            width=float(w),
                            height=float(h),
                        )
                    )
            except (TypeError, ValueError) as ve:
                continue
        
        if not boxes:
             print(f"[label] Info: No objects detected in {frame_path.name} (or parsing failed).")
             
        return boxes
        
    except Exception as e:
        # Check if it's a connection error or critical API failure
        error_msg = str(e).lower()
        if "connection" in error_msg or "refused" in error_msg or "404" in error_msg:
             raise PipelineError(f"Local VLM Connection Failed: {e}. Check if the server is running at {client.base_url} and model '{model}' is loaded.") from e
        
        # For model output errors (parsing, etc.), log warning and return empty to continue job
        print(f"[label] Warning: Local VLM error on {frame_path.name} (skipping frame): {e}")
        return []


def detect_objects_for_class(
    client: OpenAI,
    model: str,
    frame_path: Path,
    class_name: str,
) -> list[BoundingBox]:
    prompt = build_single_class_prompt(class_name)
    boxes = detect_objects_openai(client, model, frame_path, prompt)
    normalized_name = class_name.strip().lower().replace(" ", "_")
    return [
        BoundingBox(
            class_name=normalized_name,
            x=box.x,
            y=box.y,
            width=box.width,
            height=box.height,
        )
        for box in boxes
    ]


def to_yolo_line(box: BoundingBox, class_id: int, img_w: int, img_h: int) -> str:
    x = clamp(box.x, 0.0, float(img_w))
    y = clamp(box.y, 0.0, float(img_h))
    w = clamp(box.width, 0.0, float(img_w))
    h = clamp(box.height, 0.0, float(img_h))

    center_x = clamp((x + (w / 2.0)) / float(img_w), 0.0, 1.0)
    center_y = clamp((y + (h / 2.0)) / float(img_h), 0.0, 1.0)
    norm_w = clamp(w / float(img_w), 0.0, 1.0)
    norm_h = clamp(h / float(img_h), 0.0, 1.0)

    return f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}"


def write_yolo_labels(
    frame_path: Path,
    boxes: list[BoundingBox],
    class_to_id: dict[str, int],
) -> None:
    img_w, img_h = read_image_dimensions(frame_path)

    lines: list[str] = []
    for box in boxes:
        if box.class_name not in class_to_id:
            class_to_id[box.class_name] = len(class_to_id)
        class_id = class_to_id[box.class_name]
        lines.append(to_yolo_line(box, class_id, img_w, img_h))

    label_path = frame_path.with_suffix(".txt")
    label_path.write_text("\n".join(lines), encoding="utf-8")


def write_class_map(class_to_id: dict[str, int], output_path: Path) -> None:
    names = [name for name, _ in sorted(class_to_id.items(), key=lambda item: item[1])]
    output_path.write_text("\n".join(names), encoding="utf-8")


def _is_gemini_model_not_found(exc: Exception) -> bool:
    text = str(exc).lower()
    return "models/" in text and "not found" in text


def _is_malformed_json_error(exc: Exception) -> bool:
    return "malformed json" in str(exc).lower()


def _is_quota_exceeded_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "resourceexhausted" in text
        or "quota exceeded" in text
        or "exceeded your current quota" in text
        or "error: 429" in text
    )


def _extract_retry_delay_seconds(exc: Exception) -> int | None:
    text = str(exc)
    match = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", text, flags=re.IGNORECASE)
    if match:
        return max(1, int(float(match.group(1)) + 1))
    match = re.search(r"seconds:\s*(\d+)", text)
    if match:
        return max(1, int(match.group(1)))
    return None


def main() -> int:
    config = load_config()
    label_mode = normalize_label_mode(config.get("label_mode", "gemini"))
    provider, model, model_note = resolve_label_backend(config)

    if model_note:
        print(f"[label] {model_note}")

    if label_mode not in {"gemini", "gpt", "local"}:
        print(f"Error: run.py supports label_mode=gemini, gpt, or local (got: {label_mode}).", file=sys.stderr)
        return 1

    openai_client: OpenAI | None = None
    gemini_model: Any = None
    gemini_mod: Any = None
    gemini_fallback_used = False

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("Error: GEMINI_API_KEY or GOOGLE_API_KEY is not set.", file=sys.stderr)
            return 1
        try:
            import google.generativeai as genai
        except ImportError:
            print("Error: google-generativeai is not installed. Run: uv sync", file=sys.stderr)
            return 1
        gemini_mod = genai
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel(model)
    elif provider == "local":
        local_url = config.get("local_vlm_url")
        if not local_url:
            print("Error: local_vlm_url is not set in config (required for local mode).", file=sys.stderr)
            return 1
        local_api_key = os.getenv("LOCAL_VLM_API_KEY") or config.get("local_vlm_api_key") or "lm-studio"
        openai_client = OpenAI(base_url=local_url, api_key=local_api_key)
        
        # Auto-detect model if set to "auto" or empty
        # Note: Frontend handles this primarily, but good to have fallback
        if not model or model == "auto":
             try:
                models_list = openai_client.models.list()
                if models_list.data:
                    model = models_list.data[0].id
                    print(f"[label] Auto-detected local model: {model}")
                else:
                    print("[label] Warning: Local server returned no models. Using 'local-model'.")
                    model = "local-model"
             except Exception as e:
                 print(f"[label] Warning: Failed to list models from {local_url}: {e}")
                 model = "local-model"

        print(f"[label] Local VLM initialized at {local_url} with model {model}")
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY is not set.", file=sys.stderr)
            return 1
        openai_client = OpenAI(api_key=api_key)

    classes = config.get("classes", [])
    output_dir = Path(config.get("output_dir", "output"))
    frames_dir = output_dir / "frames"

    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        print("Error: No frames found. Run the collect skill first.", file=sys.stderr)
        return 1

    # Skip already-labeled frames
    unlabeled = [f for f in frames if not f.with_suffix(".txt").exists()]
    if not unlabeled:
        print("[label] All frames already labeled.")
        return 0

    fallback_prompt = build_prompt(classes) if not classes else ""
    class_to_id: dict[str, int] = {}

    # Load existing class map if present
    class_map_path = output_dir / "classes.txt"
    if class_map_path.exists():
        for idx, name in enumerate(class_map_path.read_text().strip().split("\n")):
            if name:
                class_to_id[name] = idx
    elif classes:
        for class_name in classes:
            normalized = str(class_name).strip().lower().replace(" ", "_")
            if normalized and normalized not in class_to_id:
                class_to_id[normalized] = len(class_to_id)

    print(f"[label] Labeling {len(unlabeled)} frames with {provider}:{model}...")
    try:
        for idx, frame_path in enumerate(unlabeled, start=1):
            print(f"  - Frame {idx}/{len(unlabeled)}: {frame_path.name}")
            if provider == "gemini":
                try:
                    boxes = detect_objects_gemini(gemini_model, frame_path, classes)
                except Exception as exc:  # noqa: BLE001
                    if _is_malformed_json_error(exc):
                        print(
                            f"[label] Warning: malformed model JSON for {frame_path.name}; "
                            "writing empty label and continuing."
                        )
                        boxes = []
                        write_yolo_labels(frame_path, boxes, class_to_id)
                        continue

                    if _is_quota_exceeded_error(exc):
                        retry_delay = _extract_retry_delay_seconds(exc)
                        if retry_delay is not None and retry_delay <= 60:
                            print(f"[label] Quota exhausted; waiting {retry_delay}s and retrying frame {frame_path.name}.")
                            time.sleep(retry_delay)
                            try:
                                boxes = detect_objects_gemini(gemini_model, frame_path, classes)
                                write_yolo_labels(frame_path, boxes, class_to_id)
                                continue
                            except Exception as retry_exc:  # noqa: BLE001
                                if _is_quota_exceeded_error(retry_exc):
                                    raise PipelineError(
                                        f"Provider quota exceeded. Retry after {retry_delay}s or use a billed key/project."
                                    ) from retry_exc
                                raise PipelineError(str(retry_exc)) from retry_exc

                        raise PipelineError(
                            "Provider quota exceeded. Reduce requests (lower FPS/fewer frames), "
                            "wait for reset, or use a billed key/project."
                        ) from exc

                    if (
                        not gemini_fallback_used
                        and model != DEFAULT_GEMINI_MODEL
                        and _is_gemini_model_not_found(exc)
                        and gemini_mod is not None
                    ):
                        gemini_fallback_used = True
                        previous_model = model
                        model = DEFAULT_GEMINI_MODEL
                        print(
                            f"[label] Gemini model '{previous_model}' unavailable; "
                            f"retrying with '{model}'."
                        )
                        gemini_model = gemini_mod.GenerativeModel(model)
                        try:
                            boxes = detect_objects_gemini(gemini_model, frame_path, classes)
                        except Exception as fallback_exc:  # noqa: BLE001
                            if _is_quota_exceeded_error(fallback_exc):
                                retry_delay = _extract_retry_delay_seconds(fallback_exc)
                                if retry_delay is not None and retry_delay <= 60:
                                    print(f"[label] Quota exhausted after fallback; waiting {retry_delay}s and retrying frame {frame_path.name}.")
                                    time.sleep(retry_delay)
                                    try:
                                        boxes = detect_objects_gemini(gemini_model, frame_path, classes)
                                    except Exception as second_retry_exc:  # noqa: BLE001
                                        raise PipelineError(
                                            "Provider quota exceeded. Retry later or use a billed key/project."
                                        ) from second_retry_exc
                                else:
                                    raise PipelineError(
                                        "Provider quota exceeded. Retry later or use a billed key/project."
                                    ) from fallback_exc
                            else:
                                raise PipelineError(str(fallback_exc)) from fallback_exc
                    else:
                        raise PipelineError(str(exc)) from exc
            elif classes:
                boxes: list[BoundingBox] = []
                for class_name in classes:
                    prompt = build_single_class_prompt(str(class_name))
                    if provider == "local":
                        # Local VLM likely doesn't support strict schema enforcing "responses", use chat completions
                        c_boxes = detect_objects_local(openai_client, model, frame_path, prompt)
                    else:
                        # Cloud OpenAI uses structured outputs
                        c_boxes = detect_objects_openai(openai_client, model, frame_path, prompt)
                    
                    # Normalize naming
                    norm_cls = str(class_name).strip().lower().replace(" ", "_")
                    for b in c_boxes:
                        b.class_name = norm_cls
                        boxes.append(b)
            else:
                if provider == "local":
                    boxes = detect_objects_local(openai_client, model, frame_path, fallback_prompt)
                else:
                    boxes = detect_objects_openai(openai_client, model, frame_path, fallback_prompt)
            write_yolo_labels(frame_path, boxes, class_to_id)
    except PipelineError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    write_class_map(class_to_id, class_map_path)
    print(f"[label] Done. {len(unlabeled)} frames labeled. Classes: {class_map_path}")
    _maybe_generate_previews(output_dir)
    return 0


def _maybe_generate_previews(output_dir: Path) -> None:
    frames_dir = output_dir / "frames"
    classes_path = output_dir / "classes.txt"
    if not frames_dir.exists() or not classes_path.exists():
        return

    preview_dir = frames_dir / "preview"
    video_out = preview_dir / "preview.mp4"

    cmd = [
        "uv",
        "run",
        ".agents/skills/eval/scripts/preview_labels.py",
        str(frames_dir),
        "--classes",
        str(classes_path),
        "--out-dir",
        str(preview_dir),
        "--limit",
        "0",
        "--video-out",
        str(video_out),
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"[label] Warning: preview generation failed ({exc})", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
