import os, json, time, random, io, av
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, Future

from google import genai
from google.genai import types
from google.genai.types import Part, Blob, VideoMetadata # <--- This will now correctly import from t

# ---------------- Config ----------------
HIER_JSON = "./process_hier.json"
VIDEO_PATH_OVERRIDE: Optional[str] = "./POV_car_repair.mp4"

OUTPUT_DIR = "outputs/twoVLM_hier_video"
WINDOW_SECONDS = 8
SHIFT_SECONDS = 4
FPS_TO_SEND = 3 
TITLE = "Car Repair POV, following a sequence of steps" 
# Models
VLM_MODEL = "gemini-2.5-flash"
CLEANUP_MODEL = "gemini-1.5-flash"

CONF_THRESHOLD = 90
MIN_MATCHES_TO_COMPLETE = 2
DEBUG = True
MAX_RETRIES = 3
CLEAN_EVERY_N_WINDOWS = 4
MAX_CLEAN_CONTEXT_ACTIONS = 50

# ---------------- Setup ----------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY in .env")
client = genai.Client(api_key=API_KEY)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- Utils ----------------
def ts_from_offset(sec: int) -> str:
    return f"{sec//60:02d}:{sec%60:02d}"

def load_hier_steps(path: str) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    source_video = data.get("source_video", "")
    processes = data.get("process_steps", [])
    title = TITLE #data.get("title") or os.path.splitext(os.path.basename(source_video))[0] or "Hierarchical Process"

    flat = []
    for p in processes:
        step_id = int(p.get("step_id"))
        process_title = p.get("process_title", f"Process {step_id}")
        for s in p.get("sub_steps", []):
            sub_id = int(s.get("sub_step_id"))
            flat.append({
                "step_id": step_id,
                "sub_step_id": sub_id,
                "process_title": process_title,
                "label": s.get("label", ""),
                "description": s.get("description", ""),
                "min_actions_to_complete": MIN_MATCHES_TO_COMPLETE
            })
    return title, processes, flat

def eligible_substeps(flat_substeps, completed, in_progress):
    completed_pairs = {(c["step_id"], c["sub_step_id"]) for c in completed}
    inprog_pairs = {(c["step_id"], c["sub_step_id"]) for c in in_progress}
    eligible = []
    for ss in flat_substeps:
        prereqs = ss.get("prerequisites", [])
        ok = True
        for pr in prereqs:
            if isinstance(pr, dict):
                pr_pair = (int(pr.get("step_id")), int(pr.get("sub_step_id")))
            elif isinstance(pr, (list, tuple)) and len(pr) == 2:
                pr_pair = (int(pr[0]), int(pr[1]))
            else:
                continue
            if not (pr_pair in completed_pairs or pr_pair in inprog_pairs):
                ok = False
                break
        if ok and (ss["step_id"], ss["sub_step_id"]) not in completed_pairs:
            eligible.append(ss)
    return eligible

# ---------------- Video Chunk Extraction ----------------
def get_video_chunk_bytes(container: av.container.InputContainer, start_sec: float, dur_sec: float) -> bytes:
    """Extracts [start_sec, start_sec+dur_sec) and re-encodes to safe H.264/MP4 in memory."""
    output_buf = io.BytesIO()
    output = av.open(output_buf, mode="w", format="mp4")

    in_v = container.streams.video[0]
    out_v = output.add_stream("h264", rate=in_v.average_rate or 30)
    out_v.width = 640
    out_v.height = 360
    out_v.pix_fmt = "yuv420p"

    start_pts = int(start_sec * in_v.time_base.denominator / in_v.time_base.numerator)
    end_pts = int((start_sec + dur_sec) * in_v.time_base.denominator / in_v.time_base.numerator)

    container.seek(start_pts, stream=in_v, any_frame=False, backward=True)
    for frame in container.decode(in_v):
        if frame.pts is None:
            continue
        if frame.pts > end_pts:
            break
        if frame.pts >= start_pts:
            frame = frame.reformat(width=640, height=360, format="yuv420p")
            for packet in out_v.encode(frame):
                output.mux(packet)

    # flush
    for packet in out_v.encode():
        output.mux(packet)

    output.close()
    return output_buf.getvalue()

# ---------------- Retry Helpers ----------------
def retry_call_video_json(model: str, video_bytes: bytes, prompt: str, retries: int = MAX_RETRIES):
    cfg = types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0)
    last_err = None
    
    video_blob = Blob(mime_type="video/mp4", data=video_bytes)

    # 2. Create the VideoMetadata object with your custom FPS
    # This tells Gemini how to sample the video data you are providing.
    custom_video_metadata = VideoMetadata(fps=FPS_TO_SEND)

    # 3. Combine them into a Part, using the video_metadata argument
    part = Part(
        inline_data=video_blob,             # Your video data
        video_metadata=custom_video_metadata # Your custom FPS metadata
    )

    for attempt in range(retries):
        try:
            res = client.models.generate_content(model=model, contents=[part, {"text": prompt}], config=cfg)
            return getattr(res, "text", None)
        except Exception as e:
            last_err = e
            wait = 2 ** attempt + random.random()
            print(f"⚠️ {model} error {attempt+1}/{retries}: {e}, retrying in {wait:.1f}s")
            time.sleep(wait)
    print(f"❌ {model} failed after {retries} retries: {last_err}")
    return None

def retry_call_text_json(model: str, text_blocks: List[Dict[str, str]], retries: int = MAX_RETRIES):
    cfg = types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0)
    last_err = None
    for attempt in range(retries):
        try:
            res = client.models.generate_content(model=model, contents=text_blocks, config=cfg)
            return getattr(res, "text", None)
        except Exception as e:
            last_err = e
            wait = 2 ** attempt + random.random()
            print(f"⚠️ {model} error {attempt+1}/{retries}: {e}, retrying in {wait:.1f}s")
            time.sleep(wait)
    print(f"❌ {model} failed after {retries} retries: {last_err}")
    return None

# ---------------- VLM Stages ----------------
def vlm_stage1_atomic(video_bytes: bytes, offset: int, title: str) -> Dict[str, Any]:
    prompt = (
        f"Video title: {title}\n"
        f"Analyze video chunk from {ts_from_offset(offset)}–{ts_from_offset(offset+WINDOW_SECONDS)}.\n"
        "Identify fine-grained human-object actions:\n"
        "- Touching, pointing, pouring, manipulating objects\n"
        "- Which object(s) are involved\n"
        "- Keep actions atomic\n\n"
        "Return ONLY JSON:\n"
        "{\"actions\":[{\"label\":\"...\",\"details\":\"...\",\"confidence\":\"high|medium|low\"}]}"
    )
    text = retry_call_video_json(VLM_MODEL, video_bytes, prompt)
    if not text:
        return {"actions": []}
    try:
        data = json.loads(text)
    except Exception:
        if DEBUG: print("⚠️ Stage1 non-JSON:", text[:200])
        data = {}
    actions = data.get("actions", []) if isinstance(data, dict) else []
    for i, act in enumerate(actions):
        act["start"] = ts_from_offset(offset)
        act["end"] = ts_from_offset(offset + WINDOW_SECONDS)
        act["id"] = hash(f"{offset}-{i}-{act.get('label','')}")
    return {"actions": actions}

def vlm_stage2_match(video_bytes: bytes,
                     current_actions: List[Dict[str, Any]],
                     eligible: List[Dict[str, Any]],
                     cleaned_timeline: Optional[Dict[str, Any]],
                     title: str) -> Dict[str, Any]:
    if not eligible or not current_actions:
        return {"step_id": "no_match", "sub_step_id": None, "confidence": 0}

    prior_actions = []
    if cleaned_timeline and isinstance(cleaned_timeline.get("actions"), list):
        prior_actions = cleaned_timeline["actions"][-MAX_CLEAN_CONTEXT_ACTIONS:]

    eligible_view = [
        {"step_id": e["step_id"], "sub_step_id": e["sub_step_id"], "label": e["label"]}
        for e in eligible
    ]

    prompt = (
        f"Video title: {title}\n"
        f"Time range: {current_actions[0].get('start')}–{current_actions[0].get('end')}\n\n"
        f"Previous cleaned actions: {json.dumps(prior_actions, ensure_ascii=False)}\n\n"
        f"Current actions: {json.dumps(current_actions, ensure_ascii=False)}\n\n"
        f"Candidate sub-steps: {json.dumps(eligible_view, ensure_ascii=False)}\n\n"
        "Interpretation:\n"
        "- Consider directionality and finegrain actions (insert vs remove, open vs close)\n"
        "- Pick best matching sub-step or 'no_match'\n\n"
        "Return ONLY JSON:\n"
        "{\"step_id\": <int|\"no_match\">, \"sub_step_id\": <int|null>, \"confidence\": <0-100>}"
    )

    text = retry_call_video_json(VLM_MODEL, video_bytes, prompt)
    if not text:
        return {"step_id": "no_match", "sub_step_id": None, "confidence": 0}
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {"step_id": "no_match", "sub_step_id": None, "confidence": 0}
    except Exception:
        if DEBUG: print("⚠️ Stage2 non-JSON:", text[:200])
        return {"step_id": "no_match", "sub_step_id": None, "confidence": 0}

# ---------------- Background Cleanup ----------------
def cleanup_timeline_llm(raw_timeline: Dict[str, Any]) -> Dict[str, Any]:
    prompt = (
        "You are given a raw action timeline from a video. Clean it by:\n"
        "- Deduplicating near-identical actions with overlapping time.\n"
        "- Merging micro-duplicates into a single representative entry.\n"
        "- Keeping chronological order and valid timestamps.\n"
        "- Preserve important details; do not invent new actions.\n\n"
        "Return ONLY JSON of the form: {\"actions\": [...]}.\n"
        f"Input timeline:\n{json.dumps(raw_timeline, ensure_ascii=False)}"
    )
    text = retry_call_text_json(CLEANUP_MODEL, [{"text": prompt}])
    if not text:
        return raw_timeline
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "actions" in data:
            return {"actions": data["actions"]}
    except Exception:
        if DEBUG: print("⚠️ Cleanup non-JSON:", text[:200])
    return raw_timeline

# ---------------- Promotion ----------------
def promote_substeps(in_progress, completed, working_timeline):
    still = []
    for sub in in_progress:
        need = sub.get("min_actions_to_complete", MIN_MATCHES_TO_COMPLETE)
        if len(sub.get("matched_actions", [])) >= need:
            completed.append(sub)
            matched_ids = {a["id"] for a in sub["matched_actions"]}
            working_timeline["actions"] = [a for a in working_timeline["actions"] if a["id"] not in matched_ids]
        else:
            still.append(sub)
    return still, completed, working_timeline

# ---------------- Main ----------------
def main():
    title, processes, flat_substeps = load_hier_steps(HIER_JSON)
    video_path = VIDEO_PATH_OVERRIDE or processes[0].get("source_video")
    if not video_path or not os.path.exists(video_path):
        raise RuntimeError(f"Video not found: {video_path}")

    container = av.open(video_path)

    global_timeline = {"actions": []}
    working_timeline = {"actions": []}
    cleaned_timeline = None
    completed, in_progress = [], []

    offset, windows_processed = 0, 0
    executor = ThreadPoolExecutor(max_workers=1)
    cleanup_future: Optional[Future] = None

    while True:
        try:
            chunk_bytes = get_video_chunk_bytes(container, offset, WINDOW_SECONDS)
        except Exception:
            break
        if not chunk_bytes:
            break
        print(f"▶️ Window {ts_from_offset(offset)}–{ts_from_offset(offset+WINDOW_SECONDS)}")

        s1 = vlm_stage1_atomic(chunk_bytes, offset, title)
        current_actions = s1.get("actions", [])
        global_timeline["actions"].extend(current_actions)
        working_timeline["actions"].extend(current_actions)

        elig = eligible_substeps(flat_substeps, completed, in_progress)
        sel = vlm_stage2_match(chunk_bytes, current_actions, elig, cleaned_timeline, title)

        if sel.get("step_id") != "no_match" and sel.get("confidence", 0) >= CONF_THRESHOLD and sel.get("sub_step_id"):
            sid, subid = sel["step_id"], sel["sub_step_id"]
            sub_def = next((e for e in flat_substeps if e["step_id"] == sid and e["sub_step_id"] == subid), None)
            if sub_def:
                target = next((x for x in in_progress if x["step_id"] == sid and x["sub_step_id"] == subid), None)
                if not target:
                    target = {**sub_def, "matched_actions": []}
                    in_progress.append(target)
                target["matched_actions"].extend(current_actions)
                print(f"✅ Matched {sid}.{subid} ({sel['confidence']}%) – {sub_def['label']}")

        in_progress, completed, working_timeline = promote_substeps(in_progress, completed, working_timeline)

        # Save timeline + status
        with open(os.path.join(OUTPUT_DIR, "timeline_live.json"), "w", encoding="utf-8") as f:
            json.dump(global_timeline, f, indent=2, ensure_ascii=False)

        def group_by_process(items):
            grouped = {}
            for it in items:
                grouped.setdefault(it["process_title"], []).append({
                    "step_id": it["step_id"],
                    "sub_step_id": it["sub_step_id"],
                    "label": it["label"],
                    "description": it["description"],
                    "matched_actions": it.get("matched_actions", [])
                })
            return grouped

        status = {
            "completed_sub_steps": group_by_process(completed),
            "in_progress_sub_steps": group_by_process(in_progress),
            "pending_sub_steps": [
                {
                    "process_title": ss["process_title"],
                    "step_id": ss["step_id"],
                    "sub_step_id": ss["sub_step_id"],
                    "label": ss["label"],
                    "description": ss["description"]
                }
                for ss in flat_substeps
                if (ss["step_id"], ss["sub_step_id"]) not in {(c["step_id"], c["sub_step_id"]) for c in completed}
                and (ss["step_id"], ss["sub_step_id"]) not in {(p["step_id"], p["sub_step_id"]) for p in in_progress}
            ]
        }
        with open(os.path.join(OUTPUT_DIR, "instruction_status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2, ensure_ascii=False)

        # Background cleanup every N windows
        windows_processed += 1
        if windows_processed % CLEAN_EVERY_N_WINDOWS == 0:
            if cleanup_future and cleanup_future.done():
                try:
                    cleaned_timeline = cleanup_future.result()
                    with open(os.path.join(OUTPUT_DIR, "timeline_clean.json"), "w", encoding="utf-8") as f:
                        json.dump(cleaned_timeline, f, indent=2, ensure_ascii=False)
                    print("🧹 Updated cleaned timeline snapshot.")
                except Exception as e:
                    print(f"⚠️ Cleanup failed: {e}")
            cleanup_future = executor.submit(cleanup_timeline_llm, dict(global_timeline))

        offset += SHIFT_SECONDS
        if offset >= container.duration / av.time_base:
            break

    # Final cleanup
    if cleanup_future:
        try:
            cleaned_timeline = cleanup_future.result()
            with open(os.path.join(OUTPUT_DIR, "timeline_clean.json"), "w", encoding="utf-8") as f:
                json.dump(cleaned_timeline, f, indent=2, ensure_ascii=False)
            print("🧹 Final cleaned timeline snapshot saved.")
        except Exception as e:
            print(f"⚠️ Final cleanup failed: {e}")

    print("🏁 Done")

if __name__ == "__main__":
    main()