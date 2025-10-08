import os, json, time, random, subprocess
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from google import genai
from google.genai import types

# ---------------- Config ----------------
VIDEO_PATH = "../howto_clipped.mp4"
HOWTO_JSON = "./howto.json"

OUTPUT_DIR = "outputs/singleVLM"
LOOKBACK_SECONDS = 60
SHIFT_SECONDS = 10

TITLE = "iphone 4S battery replacement."

VLM_MODEL = "gemini-2.5-flash"
DEBUG = True
MAX_RETRIES = 3
CONF_THRESHOLD = 85
MAX_CONTEXT_ACTIONS = 50

# ---------------- Setup ----------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY in .env")

client = genai.Client(api_key=API_KEY)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- Utils ----------------
def ts_from_offset(sec: int) -> str:
    return f"{int(sec//60):02d}:{int(sec%60):02d}"

def load_steps(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "steps" in raw:
        raw = raw["steps"]
    steps = []
    for item in raw:
        try:
            sid = int(item["id"])
        except Exception:
            continue
        steps.append({
            "id": sid,
            "step_text": item.get("step_text", ""),
            "how": item.get("how", ""),
            "question": item.get("question", ""),
        })
    return steps

def get_video_chunk_bytes_ffmpeg(video_path: str, start_s: float, dur_s: float) -> Optional[bytes]:
    """Extract a chunk [start_s, start_s+dur_s] from video as mp4 bytes using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-ss", str(start_s),
        "-i", video_path,
        "-t", str(dur_s),
        "-vf", "scale=640:360",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-movflags", "frag_keyframe+empty_moov+default_base_moof",
        "-f", "mp4",
        "pipe:1"
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return proc.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ ffmpeg failed: {e.stderr.decode('utf-8')[:300]}...")
        return None

def retry_call_json(model: str, parts: List[types.Part], prompt: str, retries: int = MAX_RETRIES) -> Optional[Dict]:
    cfg = types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0)
    last_err = None
    for attempt in range(retries):
        try:
            res = client.models.generate_content(model=model, contents=[*parts, {"text": prompt}], config=cfg)
            text = getattr(res, "text", None)
            if text:
                return json.loads(text)
        except Exception as e:
            last_err = e
            wait = 2 ** attempt + random.random()
            print(f"⚠️ {model} error {attempt+1}/{retries}: {e}, retrying in {wait:.1f}s...")
            time.sleep(wait)
    print(f"❌ {model} failed after {retries} retries: {last_err}")
    return None

# ---------------- Main ----------------
def main():
    steps = load_steps(HOWTO_JSON)
    global_timeline: List[Dict[str, Any]] = []
    completed_steps: List[Dict[str, Any]] = []

    offset = 0
    cap_probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", VIDEO_PATH],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        total_dur = float(cap_probe.stdout.strip())
    except:
        raise RuntimeError("Could not determine video duration.")

    while offset < total_dur:
        start_s = max(0, offset - LOOKBACK_SECONDS)
        chunk_bytes = get_video_chunk_bytes_ffmpeg(VIDEO_PATH, start_s, LOOKBACK_SECONDS)
        if not chunk_bytes:
            offset += SHIFT_SECONDS
            continue

        video_part = types.Part.from_bytes(data=chunk_bytes, mime_type="video/mp4")

        # Limit prior context
        prior_actions = global_timeline[-MAX_CONTEXT_ACTIONS:]

        prompt = (
            f"Video title: {TITLE}\n"
            f"Time range analyzed: {ts_from_offset(start_s)}–{ts_from_offset(start_s+LOOKBACK_SECONDS)}\n\n"
            f"Context (previous actions, most recent limited):\n"
            f"{json.dumps(prior_actions, ensure_ascii=False)}\n\n"
            f"Candidate steps:\n"
            f"{json.dumps(steps, ensure_ascii=False)}\n\n"
            "Tasks:\n"
            "1. Describe the fine-grained actions visible in this video chunk.\n"
            "2. Indicate which of the candidate steps were newly completed recently (in this chunk).\n\n"
            "Return ONLY JSON of the form:\n"
            "{\n"
            "  \"actions\": [ {\"label\": \"...\", \"details\": \"...\"} ],\n"
            "  \"newly_completed_steps\": [ {\"id\": <int>, \"reason\": \"...\"} ]\n"
            "}"
        )

        res = retry_call_json(VLM_MODEL, [video_part], prompt)
        if not res:
            offset += SHIFT_SECONDS
            continue

        # merge into timeline
        if "actions" in res and isinstance(res["actions"], list):
            for act in res["actions"]:
                act["start"] = ts_from_offset(start_s)
                act["end"] = ts_from_offset(start_s + LOOKBACK_SECONDS)
            global_timeline.extend(res["actions"])

        # handle newly completed steps
        new_completions = res.get("newly_completed_steps", [])
        if new_completions:
            for step in new_completions:
                sid = step.get("id")
                step_def = next((s for s in steps if s["id"] == sid), None)
                if step_def:
                    print(f"✅ Step {sid} completed – {step_def['step_text']} | Reason: {step.get('reason','')}")
            completed_steps.extend(new_completions)

            # save completed steps separately
            with open(os.path.join(OUTPUT_DIR, "completed_steps.json"), "w", encoding="utf-8") as f:
                json.dump(completed_steps, f, indent=2, ensure_ascii=False)

        # save status
        status = {
            "timeline": global_timeline,
            "newly_completed_steps": new_completions
        }
        with open(os.path.join(OUTPUT_DIR, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2, ensure_ascii=False)

        print(f"▶️ Window {ts_from_offset(start_s)}–{ts_from_offset(start_s+LOOKBACK_SECONDS)} processed.")

        offset += SHIFT_SECONDS

    print("🏁 Done")

if __name__ == "__main__":
    main()