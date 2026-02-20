#!/usr/bin/env python3
"""Label skill (subagent batch mode): label frames in worktrees using Gemini or OpenAI."""

from __future__ import annotations

import os
import re
import sys
import time
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

RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
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
        text=RESPONSE_SCHEMA,
        temperature=0,
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
        print(f"[batch] {model_note}")

    if label_mode not in {"gemini", "gpt"}:
        print(f"Error: run_batch.py supports label_mode=gemini or gpt (got: {label_mode}).", file=sys.stderr)
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
        print("[batch] No frames in this worktree. Nothing to label.")
        return 0

    # Skip already-labeled frames
    unlabeled = [f for f in frames if not f.with_suffix(".txt").exists()]
    if not unlabeled:
        print("[batch] All frames in this worktree already labeled.")
        return 0

    fallback_prompt = build_prompt(classes) if not classes else ""
    class_to_id: dict[str, int] = {}
    for class_name in classes:
        normalized = str(class_name).strip().lower().replace(" ", "_")
        if normalized and normalized not in class_to_id:
            class_to_id[normalized] = len(class_to_id)

    print(f"[batch] Labeling {len(unlabeled)} frames with {provider}:{model}...")
    try:
        for idx, frame_path in enumerate(unlabeled, start=1):
            print(f"  - Frame {idx}/{len(unlabeled)}: {frame_path.name}")
            if provider == "gemini":
                try:
                    boxes = detect_objects_gemini(gemini_model, frame_path, classes)
                except Exception as exc:  # noqa: BLE001
                    if _is_malformed_json_error(exc):
                        print(
                            f"[batch] Warning: malformed model JSON for {frame_path.name}; "
                            "writing empty label and continuing."
                        )
                        boxes = []
                        write_yolo_labels(frame_path, boxes, class_to_id)
                        continue

                    if _is_quota_exceeded_error(exc):
                        retry_delay = _extract_retry_delay_seconds(exc)
                        if retry_delay is not None and retry_delay <= 60:
                            print(f"[batch] Quota exhausted; waiting {retry_delay}s and retrying frame {frame_path.name}.")
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
                            f"[batch] Gemini model '{previous_model}' unavailable; "
                            f"retrying with '{model}'."
                        )
                        gemini_model = gemini_mod.GenerativeModel(model)
                        try:
                            boxes = detect_objects_gemini(gemini_model, frame_path, classes)
                        except Exception as fallback_exc:  # noqa: BLE001
                            if _is_quota_exceeded_error(fallback_exc):
                                retry_delay = _extract_retry_delay_seconds(fallback_exc)
                                if retry_delay is not None and retry_delay <= 60:
                                    print(f"[batch] Quota exhausted after fallback; waiting {retry_delay}s and retrying frame {frame_path.name}.")
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
                    boxes.extend(detect_objects_for_class(openai_client, model, frame_path, str(class_name)))
            else:
                boxes = detect_objects_openai(openai_client, model, frame_path, fallback_prompt)
            write_yolo_labels(frame_path, boxes, class_to_id)
    except PipelineError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    write_class_map(class_to_id, output_dir / "classes.txt")
    print(f"[batch] Done. {len(unlabeled)} frames labeled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
