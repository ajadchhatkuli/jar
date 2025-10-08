import argparse
import json
import os
import re
import sys
import time
from datetime import datetime


def load_steps(steps_path: str):
    with open(steps_path, "r", encoding="utf-8") as f:
        steps = json.load(f)
    # Basic normalization and validation
    normalized = []
    for item in steps:
        normalized.append(
            {
                "process_step_id": int(item.get("process_step_id")),
                "start_time": str(item.get("start_time")),
                "process_title": str(item.get("process_title")),
            }
        )
    return normalized


VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")


def find_video(path: str) -> str | None:
    """Return the first video file under path (non-recursive)."""
    for name in sorted(os.listdir(path)):
        if name.lower().endswith(VIDEO_EXTS) and os.path.isfile(os.path.join(path, name)):
            return os.path.join(path, name)
    return None


def build_prompt(steps: list[dict]) -> str:
    # Provide structured, unambiguous instructions to produce strict JSON only.
    lines = [
        "You are an expert automotive service video analyst.",
        "Use the given high-level process steps as guidance to structure a detailed action timeline.",
        "Analyze the video content and produce a precise, chronological list of actions.",
        "",
        "High-level steps (id, start_time, title):",
    ]
    for s in steps:
        lines.append(f"- {s['process_step_id']}: {s['start_time']} - {s['process_title']}")

    lines += [
        "",
        "Output strict JSON only with this schema (no markdown, no extra text):",
        "{",
        "  \"video_filename\": string,",
        "  \"model\": string,",
        "  \"generated_at\": ISO-8601 string,",
        "  \"actions\": [",
        "    {",
        "      \"index\": integer,",
        "      \"start_time\": string (mm:ss),",
        "      \"end_time\": string (mm:ss) or null,",
        "      \"label\": string,",
        "      \"description\": string (concise but specific),",
        "      \"related_high_level_step_id\": integer or null",
        "    }, ...",
        "  ]",
        "}",
        "",
        "Guidelines:",
        "- Align actions to visible changes or technician motions.",
        "- Prefer mm:ss timestamps; estimate when precise time is unclear.",
        "- Map each action to the nearest relevant high-level step id when possible.",
        "- Keep labels short (3-6 words); keep descriptions 1-2 sentences.",
        "- Do not add commentary outside the JSON object.",
    ]
    return "\n".join(lines)


def extract_json(text: str) -> dict:
    """Extract a JSON object from a model response text.

    Tries direct parse, then falls back to finding the first top-level JSON object.
    Raises ValueError if parsing fails.
    """
    # Direct attempt
    try:
        return json.loads(text)
    except Exception:
        pass

    # Remove markdown fences if present
    fence = re.compile(r"^```(?:json)?\n|\n```$", re.MULTILINE)
    cleaned = fence.sub("\n", text).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Fallback: find first balanced { ... }
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start : end + 1]
        return json.loads(candidate)

    raise ValueError("Unable to parse JSON from model response")


def upload_and_analyze_with_gemini(
    video_path: str,
    steps: list[dict],
    model_name: str,
    api_key: str,
    request_timeout: int = 600,
):
    try:
        import google.generativeai as genai
    except Exception as e:
        raise RuntimeError(
            "google-generativeai package is required. Install with: pip install google-generativeai"
        ) from e

    genai.configure(api_key=api_key)

    # Upload video file
    file_obj = genai.upload_file(path=video_path)

    # Wait for processing to complete
    # Some SDK versions use file_obj.state.name, others may expose state as string.
    def _get_state(f):
        try:
            return getattr(getattr(f, "state"), "name")
        except Exception:
            return getattr(f, "state", None)

    state = _get_state(file_obj)
    while state and str(state).upper() == "PROCESSING":
        time.sleep(5)
        file_obj = genai.get_file(file_obj.name)
        state = _get_state(file_obj)

    if state and str(state).upper() != "ACTIVE":
        raise RuntimeError(f"Upload failed or not active. State: {state}")

    # Build prompt and call the model
    model = genai.GenerativeModel(model_name)
    prompt = build_prompt(steps)

    response = model.generate_content(
        [file_obj, prompt],
        request_options={"timeout": request_timeout},
    )

    # Extract strict JSON
    text = getattr(response, "text", None) or "".join(
        part.text for cand in getattr(response, "candidates", []) for part in getattr(cand.content, "parts", []) if hasattr(part, "text")
    )
    if not text:
        raise RuntimeError("Empty response from model; no text content present")

    data = extract_json(text)

    # Ensure some top-level fields exist / overwrite to ensure correctness
    data.setdefault("video_filename", os.path.basename(video_path))
    data.setdefault("model", model_name)
    data.setdefault("generated_at", datetime.utcnow().isoformat() + "Z")

    # Basic sanity normalization of actions
    actions = data.get("actions", [])
    if not isinstance(actions, list):
        raise ValueError("Model response 'actions' must be a list")
    for i, a in enumerate(actions, start=1):
        if "index" not in a:
            a["index"] = i
        a.setdefault("start_time", None)
        a.setdefault("end_time", None)
        a.setdefault("label", "")
        a.setdefault("description", "")
        a.setdefault("related_high_level_step_id", None)

    return data


def main():
    # Load environment variables from .env if present
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        # If python-dotenv isn't installed, we continue; env vars may still be set.
        pass

    parser = argparse.ArgumentParser(description="Generate detailed action timeline from a service video using Gemini.")
    parser.add_argument("--video", type=str, default=None, help="Path to the video file. Defaults to first video in current folder.")
    parser.add_argument("--steps", type=str, default="video_process_steps.json", help="Path to high-level steps JSON.")
    parser.add_argument("--out", type=str, default="action_timeline.json", help="Output JSON filepath.")
    parser.add_argument("--model", type=str, default=os.environ.get("GEMINI_MODEL", "gemini-2.5-pro"), help="Gemini model name to use.")
    parser.add_argument("--timeout", type=int, default=600, help="Request timeout in seconds.")
    args = parser.parse_args()

    # Prefer GOOGLE_API_KEY from environment/.env as requested
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable is not set (load from .env or export it).", file=sys.stderr)
        sys.exit(2)

    if not os.path.exists(args.steps):
        print(f"Error: Steps file not found: {args.steps}", file=sys.stderr)
        sys.exit(2)

    steps = load_steps(args.steps)

    video_path = args.video
    if not video_path:
        video_path = find_video(os.getcwd())
        if not video_path:
            print("Error: No video file found in current directory.", file=sys.stderr)
            sys.exit(2)
    if not os.path.isfile(video_path):
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        sys.exit(2)

    try:
        result = upload_and_analyze_with_gemini(
            video_path=video_path,
            steps=steps,
            model_name=args.model,
            api_key=api_key,
            request_timeout=args.timeout,
        )
    except Exception as e:
        print(f"Gemini processing failed: {e}", file=sys.stderr)
        sys.exit(1)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Wrote detailed action timeline to: {args.out}")


if __name__ == "__main__":
    main()
