#!/usr/bin/env python3
"""End-to-end YouTube -> frames -> Gemini/OpenAI labels -> YOLO annotations pipeline."""

from __future__ import annotations

import argparse
import base64
import importlib
import json
import os
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from openai import OpenAI
try:
    from shared.toon_integration import wrap_prompt, toon_available  # type: ignore
except Exception:
    def wrap_prompt(instruction: str, schema: Any | None = None) -> str:  # type: ignore
        return instruction
    def toon_available() -> bool:  # type: ignore
        return False


@dataclass
class BoundingBox:
    class_name: str
    x: float
    y: float
    width: float
    height: float


PROMPT = """
Detect every visible object in this image and return bounding boxes.

Return strict JSON using this exact schema:
{
  "objects": [
    {
      "class_name": "string",
      "x": 0,
      "y": 0,
      "width": 0,
      "height": 0
    }
  ]
}

Rules:
- x,y,width,height must be pixel values in the original image.
- x,y is top-left corner.
- Include all salient objects (people, vehicles, UI elements, weapons, items, enemies, etc.).
- Do not include explanation text.
""".strip()


class PipelineError(RuntimeError):
    """Raised when a pipeline step fails."""


def run_command(cmd: list[str]) -> None:
    """Run a subprocess command and raise with readable context on failure."""
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise PipelineError(f"Required executable not found: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise PipelineError(f"Command failed ({exc.returncode}): {' '.join(cmd)}") from exc


def download_video(url: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_command([
        "yt-dlp",
        "-f",
        "bestvideo+bestaudio/best",
        "--merge-output-format",
        "mp4",
        "-o",
        str(output_path),
        url,
    ])

    if output_path.exists():
        return output_path

    candidates = sorted(output_path.parent.glob(f"{output_path.name}.*"))
    if candidates:
        return candidates[0]

    stem_candidates = sorted(output_path.parent.glob(f"{output_path.stem}.*"))
    if stem_candidates:
        return stem_candidates[0]

    raise PipelineError(f"Downloaded video not found near expected path: {output_path}")


def extract_frames(video_path: Path, frames_dir: Path, fps: int = 1) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_pattern = frames_dir / "frame_%06d.jpg"
    run_command([
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        str(frame_pattern),
    ])
    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        raise PipelineError("No frames extracted. Check video input and ffmpeg installation.")
    return frames


def flip_horizontal(img: Image.Image, label_lines: list[str]) -> tuple[Image.Image, list[str]]:
    """Flip image horizontally and mirror bounding box x-coordinates."""
    flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
    new_lines: list[str] = []
    for line in label_lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id, cx, cy, w, h = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        new_cx = 1.0 - cx
        new_lines.append(f"{cls_id} {new_cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return flipped, new_lines


def adjust_brightness(img: Image.Image, factor: float) -> Image.Image:
    """Adjust brightness by a factor (0.5-1.5 typical)."""
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def adjust_contrast(img: Image.Image, factor: float) -> Image.Image:
    """Adjust contrast by a factor."""
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)


def add_noise(img: Image.Image, intensity: float = 15.0) -> Image.Image:
    """Add Gaussian noise to image."""
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, intensity, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


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


def encode_image_base64(image_path: Path) -> str:
    image_bytes = image_path.read_bytes()
    return base64.b64encode(image_bytes).decode("utf-8")


def extract_json_from_text(text: str) -> dict[str, Any]:
    """Parse JSON directly; fallback to extracting from markdown code fences or custom markers."""
    if not text:
        return {}
        
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
                pass
    except Exception:
        pass

    # 4. Try standard JSON parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 5. Fallback: extract JSON from markdown code fences or loose braces
        # Look for ```json ... ``` or just ``` ... ```
        match = re.search(r"```(?:json)?(.*?)```", text, flags=re.DOTALL)
        if match:
             candidate = match.group(1).strip()
             try:
                 return json.loads(candidate)
             except json.JSONDecodeError:
                 pass
        
        # Look for { ... } at outer level
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            # If plain text fails and no JSON/TOON found, raise error
            raise PipelineError(f"Model did not return valid JSON or TOON. Output was: {text[:200]}...")
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
             # Last ditch effort: Try to fix trailing commas or single quotes?
             candidate = match.group(0).replace("'", '"')
             # Remove trailing commas
             candidate = re.sub(r",\s*([\]}])", r"\1", candidate)
             try:
                 return json.loads(candidate)
             except json.JSONDecodeError:
                 raise PipelineError("Model returned malformed JSON.") from exc


def detect_objects_gemini(model: Any, frame_path: Path, allowed_classes: list[str] | None = None) -> list[BoundingBox]:
    from PIL import Image
    import time
    
    try:
        image = Image.open(frame_path)
        image_w, image_h = image.size
    except Exception as e:
        print(f"Error opening image {frame_path}: {e}")
        return []

    base_prompt = (
        "Detect every visible object in this image. "
        "Return a JSON object with a single key 'objects' containing a list of objects. "
        "Each object must have a 'label' (string) and 'box_2d' (array of 4 integers: [ymin, xmin, ymax, xmax]). "
        "The box coordinates must be on a 0-1000 scale relative to the image dimensions. "
        "Order: y_min, x_min, y_max, x_max. "
    )

    if allowed_classes:
        base_prompt += f" ONLY detect objects belonging to these classes: {', '.join(allowed_classes)}. Do not return any other objects."
    
    example_prompt = "Example: {'objects': [{'label': 'person', 'box_2d': [100, 200, 500, 600]}]}"
    
    # Retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Vary prompt slightly on retries if needed, or simply retry
            current_prompt = base_prompt + example_prompt
            if attempt > 0:
                 current_prompt += f" Ensure valid JSON. Attempt {attempt+1}."

            try:
                response = model.generate_content(
                    [current_prompt, image],
                    generation_config={"response_mime_type": "application/json"}
                )
            except TypeError:
                 # fallback for older SDK versions
                 response = model.generate_content([current_prompt, image])
            
            if not response.text:
                 raise ValueError("Empty response from Gemini")

            payload = extract_json_from_text(response.text)
            objects = payload.get("objects", [])
            
            # If payload is empty list but we expected objects, maybe the model didn't find any.
            # But if it's malformed structure, objects might be None.
            if objects is None: objects = []

            boxes: list[BoundingBox] = []
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                try:
                    label = str(obj.get("label", "")).strip().lower().replace(" ", "_")
                    box_2d = obj.get("box_2d")
                    
                    if not box_2d or not isinstance(box_2d, list) or len(box_2d) != 4: 
                        continue
                    
                    # Gemini returns [ymin, xmin, ymax, xmax] in 0-1000 scale
                    y_min_norm = float(box_2d[0])
                    x_min_norm = float(box_2d[1])
                    y_max_norm = float(box_2d[2])
                    x_max_norm = float(box_2d[3])
                    
                    y_min = y_min_norm / 1000.0 * image_h
                    x_min = x_min_norm / 1000.0 * image_w
                    y_max = y_max_norm / 1000.0 * image_h
                    x_max = x_max_norm / 1000.0 * image_w
                    
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
            
            if not boxes and attempt < max_retries - 1:
                 # If no boxes found, it MIGHT be correct, but let's double check if we suspect failure?
                 # Actually, for object detection, 0 boxes is a valid result.
                 # Only retry if we failed to parse or got an error.
                 pass

            return boxes
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to detect objects in {frame_path.name} after {max_retries} attempts: {e}")
                return []
            time.sleep(1 + attempt) # exponential backoff
    return []


def detect_objects(client: OpenAI, model: str, frame_path: Path, allowed_classes: list[str] | None = None) -> list[BoundingBox]:
    import time
    image_b64 = encode_image_base64(frame_path)
    
    classes_instruction = ""
    if allowed_classes:
        classes_instruction = f"DETECT ONLY THESE CLASSES: {', '.join(allowed_classes)}. Do not detect other objects."

    prompt_text = (
        "TASK: Object Detection. \n"
        "INSTRUCTIONS: Examine the image and list all gameplay objects. " + classes_instruction + " \n"
        "FORMAT: Return ONLY a valid JSON object. Do not write an introduction or description. \n"
        "JSON STRUCTURE: \n"
        "{\n"
        '  "objects": [\n'
        '    {"class_name": "fruit", "x": 100, "y": 200, "width": 50, "height": 50},\n'
        '    {"class_name": "bomb", "x": 300, "y": 400, "width": 60, "height": 60}\n'
        "  ]\n"
        "}\n\n"
        "REQUIREMENTS:\n"
        "1. 'x', 'y', 'width', 'height' must be integers (pixels).\n"
        "2. 'class_name' must be a short string.\n"
        "3. If no objects are found, return {\"objects\": []}.\n"
        "4. OUTPUT STRICT JSON ONLY."
    )

    response_input: Any = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
            ],
        }
    ]

    max_retries = 3
    for attempt in range(max_retries):
        try:
            temp = 0.2 + (0.1 * attempt) # Increase temp slightly on retries
            
            # Local models (like LM Studio with Moondream) often fail with response_format={"type": "json_object"}
            # The error explicitly said it supports 'text' or 'json_schema'.
            # We used strict prompt engineering, so 'text' is fine as a fallback.
            # To be safe, we omit response_format completely for local setups or if it fails.
            
            common_args = {
                "model": model,
                "messages": response_input,
                "max_tokens": 2048,
                "temperature": temp,
            }
            
            try:
                # Try with explicit JSON mode first (best for GPT-4 etc)
                completion = client.chat.completions.create(
                    **common_args,
                    response_format={"type": "json_object"}
                )
            except Exception:
                # Fallback to plain text if JSON mode is not supported by the provider
                # (e.g. some local endpoints or older models)
                completion = client.chat.completions.create(
                    **common_args
                )

            content = completion.choices[0].message.content or "{}"
            
            try:
                payload = extract_json_from_text(content)
            except Exception:
                # If extraction fails, maybe retry?
                if attempt < max_retries - 1:
                     continue
                payload = {}

            objects = payload.get("objects", [])
            if objects is None: objects = []

            boxes: list[BoundingBox] = []
            for obj in objects:
                try:
                    boxes.append(BoundingBox(
                        class_name=str(obj.get("class_name", "")).strip().lower().replace(" ", "_"),
                        x=float(obj.get("x", 0)),
                        y=float(obj.get("y", 0)),
                        width=float(obj.get("width", 0)),
                        height=float(obj.get("height", 0))
                    ))
                except (KeyError, ValueError, TypeError):
                    continue
            
            # If successful parse, return immediately
            return boxes

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"OpenAI error detecting objects in {frame_path.name}: {e}")
                return []
            time.sleep(1 + attempt)
    return []


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def validate_and_correct_boxes(boxes: list[BoundingBox], image_w: int, image_h: int) -> list[BoundingBox]:
    valid_boxes = []
    # Deduplication map (simple overlap check could be better but this covers exact dupes)
    seen = set()
    
    for box in boxes:
        # 1. Sanity checks on size
        if box.width <= 1 or box.height <= 1: # Require at least 2 pixels
            continue
        
        # 2. Check complete out of bounds
        if box.x >= image_w or box.y >= image_h: continue
        if box.x + box.width <= 0 or box.y + box.height <= 0: continue
        
        # 3. Clamp to image boundaries
        x1 = max(0, box.x)
        y1 = max(0, box.y)
        x2 = min(image_w, box.x + box.width)
        y2 = min(image_h, box.y + box.height)
        
        # 4. Check if clamped box is valid
        if x2 - x1 < 2 or y2 - y1 < 2: 
            continue

        # 5. Create new box with clamped values
        new_box = BoundingBox(
            class_name=box.class_name,
            x=x1,
            y=y1,
            width=x2 - x1,
            height=y2 - y1
        )
        
        # 6. Deduplicate based on rounded coordinates and class
        key = (new_box.class_name, round(new_box.x), round(new_box.y), round(new_box.width), round(new_box.height))
        if key in seen:
            continue
        seen.add(key)
        
        valid_boxes.append(new_box)

    return valid_boxes

def to_yolo_line(box: BoundingBox, class_id: int, img_w: int, img_h: int) -> str:
    # Use already clamped values from validation
    center_x = (box.x + (box.width / 2.0)) / float(img_w)
    center_y = (box.y + (box.height / 2.0)) / float(img_h)
    norm_w = box.width / float(img_w)
    norm_h = box.height / float(img_h)

    # Double check bounds just in case
    center_x = clamp(center_x, 0.0, 1.0)
    center_y = clamp(center_y, 0.0, 1.0)
    norm_w = clamp(norm_w, 0.0, 1.0)
    norm_h = clamp(norm_h, 0.0, 1.0)

    # YOLO format: <class_id> <x_center> <y_center> <width> <height>
    return f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}"


def write_yolo_labels(
    frame_path: Path,
    boxes: list[BoundingBox],
    class_to_id: dict[str, int],
) -> None:
    try:
        img_w, img_h = read_image_dimensions(frame_path)
    except Exception as e:
        print(f"Warning: Could not read dimensions for {frame_path}: {e}")
        return

    boxes = validate_and_correct_boxes(boxes, img_w, img_h)

    lines: list[str] = []
    for box in boxes:
        # Strict mode: Only write if class is known (from config)
        # If class_to_id was pre-populated, we should respect it.
        if box.class_name not in class_to_id:
             # Option: Skip unknown classes instead of adding them
             # This prevents the "23 classes" issue if filtering upstream failed
             continue
             
        class_id = class_to_id[box.class_name]
        lines.append(to_yolo_line(box, class_id, img_w, img_h))

    label_path = frame_path.with_suffix(".txt")
    label_path.write_text("\n".join(lines), encoding="utf-8")


def write_class_map(class_to_id: dict[str, int], output_path: Path) -> None:
    # YOLO usually stores class names by class index in a plain text names file.
    names = [name for name, _ in sorted(class_to_id.items(), key=lambda item: item[1])]
    output_path.write_text("\n".join(names), encoding="utf-8")


def run_pipeline(youtube_url: str, output_dir: Path, provider: str, model: str, phase: str = "all") -> None:
    provider = provider.strip().lower()
    openai_client: OpenAI | None = None
    gemini_model: Any = None

    # Initialize models only if needed (label phase or all)
    if phase in ["label", "all"]:
        if provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise PipelineError("GEMINI_API_KEY or GOOGLE_API_KEY is not set.")
            try:
                genai = importlib.import_module("google.generativeai")
            except ImportError as exc:
                raise PipelineError("google-generativeai is not installed. Run: uv sync") from exc

            configure = getattr(genai, "configure", None)
            generative_model_cls = getattr(genai, "GenerativeModel", None)
            if configure is None or generative_model_cls is None:
                raise PipelineError("google-generativeai API is unavailable in this environment.")

            configure(api_key=api_key)
            gemini_model = generative_model_cls(model)
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise PipelineError("OPENAI_API_KEY is not set.")
            openai_client = OpenAI(api_key=api_key)
        elif provider == "local":
             # Use OpenAI client but pointing to local endpoint
             base_url = os.getenv("LOCAL_VLM_URL", "http://localhost:1234/v1")
             api_key = os.getenv("LOCAL_VLM_API_KEY", "lm-studio") # dummy key often needed
             openai_client = OpenAI(base_url=base_url, api_key=api_key)
        else:
            raise PipelineError(f"Unsupported provider: {provider}. Use 'gemini', 'openai', or 'local'.")

    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "video.mp4"
    frames_dir = output_dir / "frames"

    # [Collect]
    if phase in ["collect", "all"]:
        print("[1/4] Downloading video with yt-dlp...")
        if not video_path.exists():
             video_path = download_video(youtube_url, video_path)
        else:
             print("Video already downloaded.")

        print("[2/4] Extracting frames at 1 FPS with ffmpeg...")
        if not frames_dir.exists() or not any(frames_dir.glob("*.jpg")):
             frames = extract_frames(video_path, frames_dir, fps=1)
        else:
             frames = sorted(frames_dir.glob("*.jpg"))
             print(f"Frames already extracted ({len(frames)}).")

    # [Label]
    if phase in ["label", "all"]:
        if not frames_dir.exists():
            # Try to find frames if skipping collect phase but label is requested
            # Assume frames are there? Or error out?
            # For now, if frames_dir doesn't exist, we can't label.
            raise PipelineError("Frames directory not found. Run 'collect' phase first.")

        frames = sorted(frames_dir.glob("*.jpg"))
        if not frames:
            raise PipelineError("No frames found to label.")

        # Load config to get classes
        config_path = Path(__file__).parent.parent / "config.json"
        allowed_classes: list[str] = []
        if config_path.exists():
            try:
                config_data = json.loads(config_path.read_text(encoding="utf-8"))
                allowed_classes = config_data.get("classes", [])
                print(f"Loaded allowed classes from config: {allowed_classes}")
            except Exception as e:
                print(f"Warning: Failed to load classes from config.json: {e}")
        else:
            print("Warning: config.json not found. Using open-vocab detection.")

        print(f"[3/4] Labeling {len(frames)} frames with {provider}:{model}...")
        
        # Initialize class map from config to ensure consistent IDs (0, 1, 2...)
        class_to_id: dict[str, int] = {}
        if allowed_classes:
            for idx, cls_name in enumerate(allowed_classes):
                # We do NOT rstrip("s") here because config defines the canonical names
                class_to_id[cls_name.strip().lower().replace(" ", "_")] = idx

        for idx, frame_path in enumerate(frames, start=1):
            # check if label already exists?
            label_file = frame_path.with_suffix(".txt")
            if label_file.exists():
                 print(f"  - Frame {idx}/{len(frames)}: {frame_path.name} (Skipping, labelled)")
                 continue

            print(f"  - Frame {idx}/{len(frames)}: {frame_path.name}")
            
            # Pass usage of allowed_classes to reduce hallucinations
            if provider == "gemini":
                boxes = detect_objects_gemini(gemini_model, frame_path, allowed_classes=allowed_classes)
            else:
                boxes = detect_objects(cast(OpenAI, openai_client), model, frame_path, allowed_classes=allowed_classes)
            
            # Additional filtering: if we have a strict class list, DISCARD anything not in it
            if allowed_classes:
                filtered_boxes = []
                for box in boxes:
                    raw_name = box.class_name.strip().lower().replace(" ", "_").rstrip("s") # handle 'fruits' -> 'fruit'
                    
                    # Try exact match against configured keys (which are also normalized)
                    # Note: We normalized keys in class_to_id when creating it
                    
                    matched_key = None
                    if raw_name in class_to_id:
                        matched_key = raw_name
                    elif raw_name + "s" in class_to_id:
                         matched_key = raw_name + "s"
                    elif raw_name.rstrip("s") in class_to_id:
                         matched_key = raw_name.rstrip("s")
                    
                    if matched_key:
                        box.class_name = matched_key
                        filtered_boxes.append(box)
                boxes = filtered_boxes

            write_yolo_labels(frame_path, boxes, class_to_id)

        write_class_map(class_to_id, output_dir / "classes.txt")

    # [Augment]
    if phase in ["augment", "all"]:
        # valid frames are expected in output_dir/frames
        if not frames_dir.exists():
             if phase == "augment":
                 raise PipelineError("Frames directory not found. Run 'collect' phase first.")
             print("Frames directory not found. Skipping augmentation.")
        else:
            labeled_frames = [f for f in frames_dir.glob("*.jpg") if f.with_suffix(".txt").exists()]
            
            if not labeled_frames:
                if phase == "augment":
                    raise PipelineError("No labeled frames found to augment.")
                # If phase is 'all', allow skipping silently if just starting? Or print warning.
                if phase == "all":
                     print("No labeled frames found. Skipping augmentation.")
            else:
                aug_dir = output_dir / "augmented"
                aug_dir.mkdir(parents=True, exist_ok=True)
                
                print(f"[augment] Augmenting {len(labeled_frames)} labeled frames...")
                
                for frame_path in labeled_frames:
                    try:
                        label_path = frame_path.with_suffix(".txt")
                        label_text = label_path.read_text(encoding="utf-8").strip()
                        if not label_text: continue
                        
                        label_lines = label_text.split("\n")
                        label_lines = [l for l in label_lines if l.strip()]

                        img = Image.open(frame_path).convert("RGB")
                        stem = frame_path.stem

                        # 1. Horizontal flip
                        flipped_img, flipped_labels = flip_horizontal(img, label_lines)
                        flipped_img.save(aug_dir / f"{stem}_flip.jpg", quality=95)
                        (aug_dir / f"{stem}_flip.txt").write_text("\n".join(flipped_labels), encoding="utf-8")

                        # 2. Brightness jitter (labels unchanged)
                        brightness_factor = random.uniform(0.6, 1.4)
                        bright_img = adjust_brightness(img, brightness_factor)
                        bright_img.save(aug_dir / f"{stem}_bright.jpg", quality=95)
                        (aug_dir / f"{stem}_bright.txt").write_text("\n".join(label_lines), encoding="utf-8")

                        # 3. Contrast jitter (labels unchanged)
                        contrast_factor = random.uniform(0.7, 1.3)
                        contrast_img = adjust_contrast(img, contrast_factor)
                        contrast_img.save(aug_dir / f"{stem}_contrast.jpg", quality=95)
                        (aug_dir / f"{stem}_contrast.txt").write_text("\n".join(label_lines), encoding="utf-8")
                        
                        # 4. Noise
                        noisy_img = add_noise(img)
                        noisy_img.save(aug_dir / f"{stem}_noise.jpg", quality=95)
                        (aug_dir / f"{stem}_noise.txt").write_text("\n".join(label_lines), encoding="utf-8")

                    except Exception as e:
                        print(f"Warning: Failed to augment {frame_path.name}: {e}")

    # [Train/Eval] - Basic implementation using ultralytics YOLO
    if phase in ["train", "eval"] and phase != "all":
        try:
            from ultralytics import YOLO
        except Exception:
            print("Warning: 'ultralytics' not installed; cannot run train/eval.")
            print(f"Phase '{phase}' requested but ultralytics is unavailable.")
        else:
            import shutil
            # Prepare dataset directories in YOLO format
            # We expect images + .txt labels in frames/ and augmented/ (optional)
            imgs_dir = frames_dir
            aug_dir = output_dir / "augmented"

            # Ensure class mapping exists
            classes_file = output_dir / "classes.txt"
            if not classes_file.exists():
                print("classes.txt not found; attempting to infer classes from labels.")
                # Try to build from labels by scanning class ids in .txt
                found = set()
                for t in list(imgs_dir.glob("*.txt")):
                    try:
                        for line in t.read_text(encoding="utf-8").splitlines():
                            parts = line.strip().split()
                            if parts:
                                found.add(int(parts[0]))
                    except Exception:
                        continue
                names = [f"class_{i}" for i in sorted(found)]
                classes_file.write_text("\n".join(names), encoding="utf-8")

            # Create a simple YAML dataset config for ultralytics (proper YAML list for names)
            data_yaml = output_dir / "data.yaml"
            # Use augmented images if present, else frames
            train_images = str((aug_dir if aug_dir.exists() and any(aug_dir.glob("*.jpg")) else imgs_dir).resolve())
            val_images = str(imgs_dir.resolve())
            class_names = [n.strip() for n in classes_file.read_text(encoding="utf-8").splitlines() if n.strip()]
            num_classes = len(class_names)
            names_block = "\n".join([f"  - '{n}'" for n in class_names]) if class_names else ""
            data_yaml.write_text(
                f"train: {train_images}\nval: {val_images}\nnc: {num_classes}\nnames:\n{names_block}\n",
                encoding="utf-8",
            )

            # Train
            if phase == "train":
                print("[train] Starting training with ultralytics YOLO...")
                try:
                    model = YOLO(str(Path(__file__).parent.parent / "yolodex_model.yaml"))
                except Exception:
                    # Fallback: use yolov8n backbone from ultralytics library
                    model = YOLO("yolov8n.pt")

                # Run training
                try:
                    # Ensure ultralytics saves outputs into the shared runs/detect folder
                    project_dir = output_dir / "runs" / "detect"
                    project_dir.mkdir(parents=True, exist_ok=True)
                    result = model.train(data=str(data_yaml), epochs=int(os.getenv("EPOCHS", "50")), imgsz=640, project=str(project_dir), name="train")
                    # Weights saved under project_dir / name / weights
                    weights_dir = project_dir / "train" / "weights"
                    weights_dir.mkdir(parents=True, exist_ok=True)
                    print("[train] Training finished. Weights saved to:", weights_dir)
                    
                    # Copy best weight to artifacts root for easy access
                    if (weights_dir / "best.pt").exists():
                         dest_weights = output_dir / "weights"
                         dest_weights.mkdir(parents=True, exist_ok=True)
                         shutil.copy2(weights_dir / "best.pt", dest_weights / "best.pt")

                except Exception as e:
                    print("[train] Training failed:", e)

            # Eval
            if phase == "eval":
                print("[eval] Running evaluation using ultralytics...")
                try:
                    # Find weights to evaluate
                    # 1. Check shared weights folder
                    candidate = output_dir / "weights" / "best.pt"
                    if not candidate.exists():
                         # 2. Check training runs
                         project_dir = output_dir / "runs" / "detect" / "train" / "weights"
                         if project_dir.exists():
                            pts = sorted(project_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
                            if pts:
                                candidate = pts[0]
                    
                    if not candidate or not candidate.exists():
                        # Try previously used default train runs location
                        default_weights = Path.cwd() / "runs" / "train" / "weights"
                        if default_weights.exists():
                            pts = sorted(default_weights.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
                            if pts:
                                candidate = pts[0]

                    if not candidate or not candidate.exists():
                        print("[eval] No weights found to evaluate.")
                    else:
                        print(f"[eval] Evaluating weights: {candidate}")
                        eval_model = YOLO(str(candidate))
                        metrics = eval_model.val(data=str(data_yaml))
                        out = output_dir / "eval_results.json"
                        import json as _json
                        out.write_text(_json.dumps(metrics, default=lambda o: str(o)), encoding="utf-8")
                        print("[eval] Evaluation complete. Results saved to:", out)
                except Exception as e:
                    print("[eval] Evaluation failed:", e)
    
    if phase == "all":
         # Original finish
         print("[4/4] Done.")
         print(f"Frames: {frames_dir}")
         print("YOLO labels saved next to each frame image.")
         print(f"Class mapping: {output_dir / 'classes.txt'}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YouTube -> frames -> Gemini/OpenAI vision labels -> YOLO format"
    )
    parser.add_argument("youtube_url", help="YouTube video URL")
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory (default: output)",
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "openai", "local"],
        default="gemini",
        help="Labeling provider (default: gemini)",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash-lite",
        help="Model name (default: gemini-2.5-flash-lite)",
    )
    parser.add_argument(
        "--phase",
        choices=["all", "collect", "label", "augment", "train", "eval"],
        default="all",
        help="Pipeline phase to run (default: all)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        run_pipeline(args.youtube_url, Path(args.output_dir), args.provider, args.model, args.phase)
        return 0
    except PipelineError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
