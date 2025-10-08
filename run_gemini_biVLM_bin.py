import os, cv2, json, time, random
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, Future

from google import genai
from google.genai import types
from google.genai.errors import ServerError

# ---------------- Config ----------------
HIER_JSON = "./process_hier.json"   # path to the hierarchical JSON you provided
# If you want to override the video path in JSON, set VIDEO_PATH below; otherwise it uses source_video in the JSON
VIDEO_PATH_OVERRIDE = "./POV_car_repair.mp4"

OUTPUT_DIR = "outputs/twoVLM_hier"
WINDOW_SECONDS = 6
SHIFT_SECONDS = 4
FPS_TO_SEND = 3
MAX_FRAMES_PER_CHUNK = 32   # limit to avoid timeouts

# Models
VLM_MODEL = "gemini-2.5-flash"
CLEANUP_MODEL = "gemini-1.5-flash"  # background timeline cleaner

CONF_THRESHOLD = 90
MIN_MATCHES_TO_COMPLETE = 2
DEBUG = True
MAX_RETRIES = 3
CLEAN_EVERY_N_WINDOWS = 4
MAX_CLEAN_CONTEXT_ACTIONS = 50  # limit previous actions fed to Stage 2 prompt

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
    """
    Returns:
      title (str),
      processes (list of process dicts),
      flat_substeps (list of dict: step_id, process_title, sub_step_id, label, description)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    source_video = data.get("source_video", "")
    processes = data.get("process_steps", [])
    title = data.get("title") or os.path.splitext(os.path.basename(source_video))[0] or "Hierarchical Process"

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
                # optional future keys: "prerequisites": [...]
                "min_actions_to_complete": MIN_MATCHES_TO_COMPLETE
            })
    return title, processes, flat

def eligible_substeps(flat_substeps: List[Dict[str, Any]],
                      completed: List[Dict[str, Any]],
                      in_progress: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Basic eligibility: if sub-step defines `prerequisites` (list of tuples or dicts), enforce them;
    else, consider all sub-steps eligible. You can enhance this with your own logic later.
    """
    # Build sets for quick checks
    completed_pairs = {(c["step_id"], c["sub_step_id"]) for c in completed}
    inprog_pairs = {(c["step_id"], c["sub_step_id"]) for c in in_progress}

    eligible = []
    for ss in flat_substeps:
        prereqs = ss.get("prerequisites", [])
        ok = True
        for pr in prereqs:
            # support either dict {step_id, sub_step_id} or tuple (step_id, sub_step_id)
            if isinstance(pr, dict):
                pr_pair = (int(pr.get("step_id")), int(pr.get("sub_step_id")))
            elif isinstance(pr, (list, tuple)) and len(pr) == 2:
                pr_pair = (int(pr[0]), int(pr[1]))
            else:
                continue
            if not (pr_pair in completed_pairs or pr_pair in inprog_pairs):
                ok = False
                break
        if ok:
            # avoid offering already completed sub-steps
            if (ss["step_id"], ss["sub_step_id"]) not in completed_pairs:
                eligible.append(ss)
    return eligible

def retry_call_parts_json(model: str, parts: List[types.Part], prompt: str, temperature: float = 0.0, retries: int = MAX_RETRIES) -> Optional[str]:
    cfg = types.GenerateContentConfig(response_mime_type="application/json", temperature=temperature)
    last_err = None
    for attempt in range(retries):
        try:
            res = client.models.generate_content(model=model, contents=[*parts, {"text": prompt}], config=cfg)
            return getattr(res, "text", None)
        except Exception as e:
            last_err = e
            wait = 2 ** attempt + random.random()
            print(f"⚠️ {model} error ({type(e).__name__}) attempt {attempt+1}/{retries}: {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
    print(f"❌ {model} failed after {retries} retries. Last error: {last_err}")
    return None

def retry_call_text_json(model: str, text_blocks: List[Dict[str, str]], temperature: float = 0.0, retries: int = MAX_RETRIES) -> Optional[str]:
    cfg = types.GenerateContentConfig(response_mime_type="application/json", temperature=temperature)
    last_err = None
    for attempt in range(retries):
        try:
            res = client.models.generate_content(model=model, contents=text_blocks, config=cfg)
            return getattr(res, "text", None)
        except Exception as e:
            last_err = e
            wait = 2 ** attempt + random.random()
            print(f"⚠️ {model} error ({type(e).__name__}) attempt {attempt+1}/{retries}: {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
    print(f"❌ {model} failed after {retries} retries. Last error: {last_err}")
    return None

# ---------------- Stage 1 VLM: Atomic actions ----------------
def vlm_stage1_atomic(frames: List[bytes], offset: int, title: str) -> Dict[str, Any]:
    frames = frames[:MAX_FRAMES_PER_CHUNK]
    parts = [types.Part.from_bytes(data=f, mime_type="image/jpeg") for f in frames]

    prompt = (
        f"Video title: {title}\n"
        f"Analyze frames from {ts_from_offset(offset)}–{ts_from_offset(offset+WINDOW_SECONDS)}.\n"
        "Describe extremely accurately what is happening here:\n"
        "- Touching, pointing, pouring, manipulating objects\n"
        "- Which object(s) are involved.\n"
        "- Keep actions **atomic** (minimal units, avoid combining multiple moves).\n\n"
        "Return ONLY JSON:\n"
        "{\"actions\":[{\"label\":\"...\",\"details\":\"...\",\"confidence\":\"high|medium|low\"}]}"
    )

    text = retry_call_parts_json(VLM_MODEL, parts, prompt, temperature=0.0)
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

# ---------------- Background Cleanup LLM ----------------
def cleanup_timeline_llm(raw_timeline: Dict[str, Any]) -> Dict[str, Any]:
    prompt = (
        "You are given a raw action timeline from a video. Clean it by:\n"
        "- Deduplicating near-identical actions with overlapping time.\n"
        "- Merging micro-duplicates into a single representative entry.\n"
        "- Keeping chronological order and valid 'MM:SS' timestamps.\n"
        "- Preserve important details and labels; do not invent new actions.\n\n"
        "Return ONLY JSON of the form: {\"actions\": [...]}.\n"
        "Input timeline:\n"
        f"{json.dumps(raw_timeline, ensure_ascii=False)}"
    )
    text = retry_call_text_json(CLEANUP_MODEL, [{"text": prompt}], temperature=0.0)
    if not text:
        return raw_timeline
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "actions" in data and isinstance(data["actions"], list):
            return {"actions": data["actions"]}
    except Exception:
        if DEBUG: print("⚠️ Cleanup LLM non-JSON:", text[:200])
    return raw_timeline

# ---------------- Stage 2 VLM: Match current sub-step (directionality + cleaned context) ----------------
def vlm_stage2_match(frames: List[bytes],
                     current_actions: List[Dict[str, Any]],
                     eligible: List[Dict[str, Any]],
                     cleaned_timeline: Optional[Dict[str, Any]],
                     title: str) -> Dict[str, Any]:
    """
    Returns: {'step_id': int|'no_match', 'sub_step_id': int|None, 'confidence': int}
    """
    if not eligible or not current_actions:
        return {"step_id": "no_match", "sub_step_id": None, "confidence": 0}

    frames = frames[:MAX_FRAMES_PER_CHUNK]
    parts = [types.Part.from_bytes(data=f, mime_type="image/jpeg") for f in frames]

    prior_actions = []
    if cleaned_timeline and isinstance(cleaned_timeline.get("actions"), list):
        prior_actions = cleaned_timeline["actions"][-MAX_CLEAN_CONTEXT_ACTIONS:]

    # Present eligible as compact tuples to control verbosity
    eligible_view = [
        {"step_id": e["step_id"], "sub_step_id": e["sub_step_id"], "label": e["label"]}
        for e in eligible
    ]

    prompt = (
        f"Video title: {title}\n"
        f"Time range: {current_actions[0].get('start','??')}–{current_actions[0].get('end','??')}\n\n"
        "Previous actions (most recent limited):\n"
        f"{json.dumps(prior_actions, ensure_ascii=False)}\n\n"
        "Detected **atomic actions in the current window** (touching/pointing/manipulating + objects):\n"
        f"{json.dumps(current_actions, ensure_ascii=False)}\n\n"
        "Candidate hierarchical sub-steps (process step_id, sub_step_id, label):\n"
        f"{json.dumps(eligible_view, ensure_ascii=False)}\n\n"
        "Interpretation rules:\n"
        "- Consider **directionality**: inserting vs removing, opening vs closing, putting in vs taking out.\n"
        "- Choose the **single best matching sub-step** for the current window (step_id + sub_step_id) if evidence matches the description, "
        "or return no_match if evidence is insufficient.\n"
        "- Be conservative and consistent with prior context; do not claim progress without strong visual cues.\n\n"
        "Return ONLY JSON:\n"
        "{\"step_id\": <integer or \"no_match\">, \"sub_step_id\": <integer or null>, \"confidence\": <0-100>}"
    )

    text = retry_call_parts_json(VLM_MODEL, parts, prompt, temperature=0.0)
    if not text:
        return {"step_id": "no_match", "sub_step_id": None, "confidence": 0}
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "step_id" in data:
            # normalize possible string 'no_match'
            if str(data["step_id"]) == "no_match":
                return {"step_id": "no_match", "sub_step_id": None, "confidence": int(data.get("confidence", 0) or 0)}
            # cast numbers
            return {
                "step_id": int(data["step_id"]),
                "sub_step_id": int(data["sub_step_id"]) if data.get("sub_step_id") is not None else None,
                "confidence": int(data.get("confidence", 0) or 0)
            }
        if isinstance(data, list) and data and isinstance(data[0], dict):
            d = data[0]
            if str(d.get("step_id")) == "no_match":
                return {"step_id": "no_match", "sub_step_id": None, "confidence": int(d.get("confidence", 0) or 0)}
            return {
                "step_id": int(d.get("step_id")),
                "sub_step_id": int(d.get("sub_step_id")) if d.get("sub_step_id") is not None else None,
                "confidence": int(d.get("confidence", 0) or 0)
            }
    except Exception:
        if DEBUG: print("⚠️ Stage2 non-JSON:", text[:200])
    return {"step_id": "no_match", "sub_step_id": None, "confidence": 0}

# ---------------- Promotion / consumption ----------------
def promote_substeps(in_progress: List[Dict[str, Any]],
                     completed: List[Dict[str, Any]],
                     working_timeline: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    still = []
    for sub in in_progress:
        need = sub.get("min_actions_to_complete", MIN_MATCHES_TO_COMPLETE)
        if len(sub.get("matched_actions", [])) >= need:
            completed.append(sub)
            matched_ids = {a["id"] for a in sub["matched_actions"]}
            # consume matched actions from working copy (keep raw in global_timeline)
            working_timeline["actions"] = [a for a in working_timeline["actions"] if a["id"] not in matched_ids]
        else:
            still.append(sub)
    return still, completed, working_timeline

# ---------------- Main ----------------
def main():
    title, processes, flat_substeps = load_hier_steps(HIER_JSON)
    # pick video path
    video_path = VIDEO_PATH_OVERRIDE
    if not video_path:
        # from hier json
        with open(HIER_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        video_path = data.get("source_video")
    if not video_path:
        raise RuntimeError("No video path found. Set VIDEO_PATH_OVERRIDE or provide 'source_video' in JSON.")
    if not os.path.exists(video_path):
        raise RuntimeError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = int(round(fps / FPS_TO_SEND))

    global_timeline = {"actions": []}   # raw
    working_timeline = {"actions": []}  # consumable
    cleaned_timeline = None             # latest snapshot from background LLM

    completed_substeps: List[Dict[str, Any]] = []
    in_progress_substeps: List[Dict[str, Any]] = []

    frame_idx, offset = 0, 0
    buffer_frames: List[bytes] = []

    # background cleanup executor
    executor = ThreadPoolExecutor(max_workers=1)
    cleanup_future: Optional[Future] = None
    windows_processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # downscale for payload robustness
        small = cv2.resize(frame, (640, 360))
        if frame_idx % frame_interval == 0:
            ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if ok:
                buffer_frames.append(buf.tobytes())

        # collect window
        buffered_secs = (len(buffer_frames) * frame_interval) / fps
        if buffered_secs >= WINDOW_SECONDS:
            print(f"▶️ Window {offset:02d}s–{offset+WINDOW_SECONDS:02d}s | frames={len(buffer_frames)}")

            # --- Stage 1: atomic actions ---
            s1 = vlm_stage1_atomic(buffer_frames, offset, title)
            current_actions = s1.get("actions", [])
            global_timeline["actions"].extend(current_actions)
            working_timeline["actions"].extend(current_actions)

            # --- Stage 2: hierarchical match with cleaned context (if any) ---
            elig = eligible_substeps(flat_substeps, completed_substeps, in_progress_substeps)
            sel = vlm_stage2_match(buffer_frames, current_actions, elig, cleaned_timeline, title)

            if sel.get("step_id") != "no_match" and sel.get("confidence", 0) >= CONF_THRESHOLD and sel.get("sub_step_id") is not None:
                sid, subid = sel["step_id"], sel["sub_step_id"]
                # find the sub-step definition
                sub_def = next((e for e in flat_substeps if e["step_id"] == sid and e["sub_step_id"] == subid), None)
                if sub_def:
                    target = next((x for x in in_progress_substeps if x["step_id"] == sid and x["sub_step_id"] == subid), None)
                    if not target:
                        target = {
                            "step_id": sid,
                            "sub_step_id": subid,
                            "process_title": sub_def["process_title"],
                            "label": sub_def["label"],
                            "description": sub_def["description"],
                            "min_actions_to_complete": sub_def.get("min_actions_to_complete", MIN_MATCHES_TO_COMPLETE),
                            "matched_actions": []
                        }
                        in_progress_substeps.append(target)
                    # accumulate only current chunk actions into this sub-step
                    target["matched_actions"].extend(current_actions)
                    print(f"✅ Matched step {sid}.{subid} ({sel['confidence']}%) – {sub_def['label']}")

            # promote when enough evidence
            in_progress_substeps, completed_substeps, working_timeline = promote_substeps(
                in_progress_substeps, completed_substeps, working_timeline
            )

            # save raw + status
            with open(os.path.join(OUTPUT_DIR, "timeline_live.json"), "w", encoding="utf-8") as f:
                json.dump(global_timeline, f, indent=2, ensure_ascii=False)

            # group status by process
            def group_by_process(items: List[Dict[str, Any]]):
                grouped: Dict[str, List[Dict[str, Any]]] = {}
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
                "completed_sub_steps": group_by_process(completed_substeps),
                "in_progress_sub_steps": group_by_process(in_progress_substeps),
                "pending_sub_steps": [
                    {
                        "process_title": ss["process_title"],
                        "step_id": ss["step_id"],
                        "sub_step_id": ss["sub_step_id"],
                        "label": ss["label"],
                        "description": ss["description"]
                    }
                    for ss in flat_substeps
                    if (ss["step_id"], ss["sub_step_id"]) not in {(c["step_id"], c["sub_step_id"]) for c in completed_substeps}
                    and (ss["step_id"], ss["sub_step_id"]) not in {(p["step_id"], p["sub_step_id"]) for p in in_progress_substeps}
                ]
            }
            with open(os.path.join(OUTPUT_DIR, "instruction_status.json"), "w", encoding="utf-8") as f:
                json.dump(status, f, indent=2, ensure_ascii=False)

            # --- Schedule background cleanup every N windows ---
            windows_processed += 1
            if windows_processed % CLEAN_EVERY_N_WINDOWS == 0:
                if cleanup_future and cleanup_future.done():
                    try:
                        cleaned_timeline = cleanup_future.result()
                        with open(os.path.join(OUTPUT_DIR, "timeline_clean.json"), "w", encoding="utf-8") as f:
                            json.dump(cleaned_timeline, f, indent=2, ensure_ascii=False)
                        print("🧹 Updated cleaned timeline snapshot.")
                    except Exception as e:
                        print(f"⚠️ Cleanup future failed: {e}")
                cleanup_future = executor.submit(cleanup_timeline_llm, dict(global_timeline))

            # slide window
            drop_frames = int(SHIFT_SECONDS * fps / frame_interval)
            buffer_frames = buffer_frames[drop_frames:]
            offset += SHIFT_SECONDS

    # finalize cleanup
    if cleanup_future:
        try:
            cleaned_timeline = cleanup_future.result()
            with open(os.path.join(OUTPUT_DIR, "timeline_clean.json"), "w", encoding="utf-8") as f:
                json.dump(cleaned_timeline, f, indent=2, ensure_ascii=False)
            print("🧹 Final cleaned timeline snapshot saved.")
        except Exception as e:
            print(f"⚠️ Final cleanup failed: {e}")

    cap.release()
    print("🏁 Done")

if __name__ == "__main__":
    main()