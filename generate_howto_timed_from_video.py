import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from google import genai
from google.genai import types


DEFAULT_MODEL = "gemini-2.5-pro"
DEFAULT_OUTPUT = "howto_timed.json"

VIDEO_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".webm": "video/webm",
    ".m4v": "video/x-m4v",
}


def _ensure_env() -> str:
    """Load GOOGLE_API_KEY from environment or .env file."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY in environment or .env file.")
    return api_key


def _guess_mime_type(video_path: Path) -> str:
    return VIDEO_MIME_TYPES.get(video_path.suffix.lower(), "video/mp4")


def load_howto_steps(path: Path) -> List[Dict[str, Any]]:
    """Return normalized list of steps from howto JSON."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "steps" in data:
        steps = data["steps"]
    elif isinstance(data, list):
        steps = data
    else:
        raise ValueError("howto JSON must be a list or contain a top-level 'steps' list.")

    normalized: List[Dict[str, Any]] = []
    for idx, item in enumerate(steps, start=1):
        try:
            step_id = int(item.get("id", idx))
        except Exception as exc:
            raise ValueError(f"Step at index {idx} is missing a numeric 'id'.") from exc
        normalized.append(
            {
                "id": step_id,
                "process_title": item.get("process_title", ""),
                "step_text": item.get("step_text", ""),
                "how": item.get("how", ""),
                "question": item.get("question", ""),
            }
        )
    return normalized


def build_prompt(video_name: str, steps: List[Dict[str, Any]]) -> str:
    header = (
        "Analyze the full POV car service video to align each existing instructional step "
        "with when it happens on screen.\n"
        "The provided steps already describe the procedure; your task is ONLY to determine "
        "their execution times and note any useful observations.\n"
        "Keep steps in the given order and assume they occur sequentially unless clear overlaps are visible.\n\n"
        "For every step produce start_time and end_time in MM:SS, a confidence (0-100), and optional notes "
        "with concise evidence.\n"
        "If a step is never observed, set both start_time and end_time to null and explain why in notes.\n"
        "Do not invent new steps.\n\n"
        f"Video filename: {video_name}\n"
        "Steps:\n"
    )
    for step in steps:
        process = step["process_title"] or "General Task"
        header += (
            f"- Step {step['id']} ({process}): {step['step_text']}\n"
            f"  How: {step['how']}\n"
            f"  Question: {step['question']}\n"
        )

    example = {
        "video": video_name,
        "model": DEFAULT_MODEL,
        "steps": [
            {
                "id": 1,
                "start_time": "00:45",
                "end_time": "02:10",
                "confidence": 92,
                "notes": "Technician lifts hood and prepares tools; corresponds to Step 1 text."
            }
        ]
    }

    return (
        header
        + "\nReturn ONLY JSON with this structure:\n"
        + json.dumps(example, ensure_ascii=False, indent=2)
    )


def call_gemini(video_path: Path, prompt: str, model_name: str) -> Dict[str, Any]:
    api_key = _ensure_env()
    client = genai.Client(api_key=api_key)

    video_bytes = video_path.read_bytes()
    video_part = types.Part.from_bytes(data=video_bytes, mime_type=_guess_mime_type(video_path))

    cfg_kwargs = {
        "response_mime_type": "application/json",
        "temperature": 0.0,
    }

    try:
        thinking_config = types.ThinkingConfig(thinking_budget=256)
        cfg_kwargs["thinking_config"] = thinking_config
    except Exception:
        pass

    cfg = types.GenerateContentConfig(**cfg_kwargs)
    response = client.models.generate_content(
        model=model_name,
        contents=[video_part, {"text": prompt}],
        config=cfg,
    )

    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Model returned no text payload.")

    return json.loads(text)


def save_output(data: Dict[str, Any], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align existing how-to steps with timestamps using Gemini."
    )
    parser.add_argument("--video", required=True, help="Path to the full video file.")
    parser.add_argument(
        "--howto",
        default="howto.json",
        help="Path to the base howto JSON whose steps should be timestamped.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Path to save the timed howto JSON.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Gemini model to use (defaults to gemini-2.5-pro).",
    )

    args = parser.parse_args()

    video_path = Path(args.video).expanduser().resolve()
    howto_path = Path(args.howto).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not howto_path.exists():
        raise FileNotFoundError(f"How-to JSON not found: {howto_path}")

    steps = load_howto_steps(howto_path)
    prompt = build_prompt(video_path.name, steps)
    raw = call_gemini(video_path, prompt, args.model)
    if isinstance(raw, list):
        data: Dict[str, Any] = {"steps": raw}
    elif isinstance(raw, dict):
        data = raw
    else:
        raise ValueError("Model response must be a JSON object or list of steps.")

    if "steps" not in data or not isinstance(data["steps"], list):
        raise ValueError("Model response missing 'steps' list.")

    # Ensure the output includes identifiers for traceability
    data.setdefault("video", video_path.name)
    data.setdefault("model", args.model)

    save_output(data, output_path)
    print(f"Saved timed how-to JSON to {output_path}")


if __name__ == "__main__":
    main()
