"""Coffee workflow variant that injects UI button context into Stage-1 prompts."""

import json
import os
from typing import Any, Dict, List

import run_sticky_evidence_actions_flush as pipeline
import run_sticky_evidence_actions_flush_multiproc as multiproc

# Configure the shared pipeline for the coffee workflow.
pipeline.TITLE = "Coffee Preparation and Cleanup"
pipeline.VIDEO_PATH = "./realwear-videos/coffee-full.mp4"
pipeline.HOWTO_JSON = "./realwear-videos/howto_coffee.json"
pipeline.OBJECTS_JSON = "./realwear-videos/objects.json"
pipeline.ACTION_VERBS_JSON = "./realwear-videos/action_verbs.json"
pipeline.OUTPUT_DIR = "realwear-videos/outputs/coffee"
pipeline.S1_MAX_FRAMES = 5  # Limit Stage-1 prompts to five frames per window.

_HERE = os.path.dirname(os.path.abspath(__file__))
_COFFEE_IMAGE_PATH = os.path.join(_HERE, "coffee.png")
_BUTTONS_JSON_PATH = os.path.join(_HERE, "buttons.json")
_REFERENCE_CONTEXT_TEXT = (
    "Reference context: The next image and JSON describe button locations only. "
    "Do not infer or describe any actions from this context; use it purely to recognize button positions in real frames."
)


def _read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _load_buttons_json_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return json.dumps(data, ensure_ascii=False, indent=2)


try:
    _COFFEE_IMAGE_BYTES = _read_file_bytes(_COFFEE_IMAGE_PATH)
except OSError:
    _COFFEE_IMAGE_BYTES = b""

try:
    _BUTTONS_JSON_TEXT = _load_buttons_json_text(_BUTTONS_JSON_PATH)
except (OSError, json.JSONDecodeError):
    _BUTTONS_JSON_TEXT = ""


if _BUTTONS_JSON_TEXT:
    _BUTTONS_PROMPT_PREFIX = (
        "Identification and locations of buttons (reference only; never infer actions from this data):\n"
        f"{_BUTTONS_JSON_TEXT}\n"
        "Use this solely to memorize button positions; ignore it when describing observed actions.\n\n"
    )
else:
    _BUTTONS_PROMPT_PREFIX = ""


def vlm_stage1_with_buttons(
    frames_jpg: List[bytes],
    t_start: float,
    t_end: float,
    title: str,
    object_knowledge: Dict[str, Any],
    steps: List[Dict[str, Any]],
    last_evidences: Dict[int, List[Dict[str, Any]]],
    action_verbs: List[str],
) -> List[Dict[str, Any]]:
    if not frames_jpg:
        return []

    frame_parts = [
        {"inline_data": {"data": b, "mime_type": "image/jpeg"}}
        for b in frames_jpg
    ]

    prompt = pipeline.build_s1_prompt(
        title,
        t_start,
        t_end,
        object_knowledge,
        steps,
        last_evidences,
        action_verbs,
    )
    if _BUTTONS_PROMPT_PREFIX:
        prompt_with_buttons = (
            f"{_BUTTONS_PROMPT_PREFIX}"
            "Reminder: Infer actions only from the live video frames; ignore the reference context when deciding actions.\n\n"
            f"{prompt}"
        )
    else:
        prompt_with_buttons = prompt

    contents: List[Dict[str, Any]] = [{"text": _REFERENCE_CONTEXT_TEXT}]
    if _COFFEE_IMAGE_BYTES:
        contents.append({"text": "Example image with labeled buttons"})
        contents.append(
            {"inline_data": {"data": _COFFEE_IMAGE_BYTES, "mime_type": "image/png"}}
        )
    contents.extend(frame_parts)
    contents.append({"text": prompt_with_buttons})

    data = pipeline.retry_json_call(
        pipeline.VLM_MODEL_S1,
        contents,
        temperature=0.0,
        thinking_config=pipeline._thinking_config_for_model(pipeline.VLM_MODEL_S1),
    )

    actions: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        actions = data.get("actions", [])

    out: List[Dict[str, Any]] = []
    allowed_verbs = {v.lower() for v in action_verbs}
    for a in actions:
        if not isinstance(a, dict):
            continue
        subject = str(a.get("subject", "")).strip()
        object_name = str(a.get("object", "")).strip()
        verb = str(a.get("verb", "")).strip()
        time_val = a.get("time")

        if allowed_verbs and verb.lower() not in allowed_verbs:
            continue

        if not verb or not object_name:
            continue
        if not subject:
            subject = "hands"

        if isinstance(time_val, (int, float)):
            action_time = pipeline.ts(float(time_val))
        else:
            action_time = str(time_val or "").strip()
            if not action_time:
                action_time = pipeline.ts(t_end)
            elif ":" not in action_time:
                try:
                    action_time = pipeline.ts(float(action_time))
                except ValueError:
                    action_time = pipeline.ts(t_end)

        action_time_sec = pipeline._sec(action_time)
        if action_time_sec == 0 and action_time != "00:00":
            action_time = pipeline.ts(t_end)

        detail_parts = [subject, verb, object_name]
        details = " ".join(p for p in detail_parts if p).strip()
        if not details:
            continue

        label = " ".join(p for p in [verb, object_name] if p).strip() or verb or object_name

        out.append(
            {
                "label": label,
                "details": details,
                "subject": subject,
                "object": object_name,
                "verb": verb,
                "time": action_time,
            }
        )
    return out


# Swap in our Stage-1 wrapper just for this script.
pipeline.vlm_stage1 = vlm_stage1_with_buttons


def main() -> None:
    os.makedirs(pipeline.OUTPUT_DIR, exist_ok=True)
    multiproc.main()


if __name__ == "__main__":
    main()
