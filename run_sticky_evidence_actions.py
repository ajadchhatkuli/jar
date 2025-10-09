# run_two_sticky_evidence.py

import os, cv2, json, time, random, subprocess
from collections import deque
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

from google import genai
from google.genai import types

# ====================== Config ======================

TITLE = "Hyundai Santro Car Service and Repair"
VIDEO_PATH    = "./POV_car_repair.mp4"
HOWTO_JSON    = "./howto.json"
OBJECTS_JSON = "./process_hier.json"
ACTION_VERBS_JSON = "./action_verbs.json"
OUTPUT_DIR    = "outputs/twoVLM"

# Sliding window
STRIDE_SECONDS   = 10
S1_WINDOW_SEC    = 10
S2_WINDOW_SEC    = 60

# Sampling fps
S1_FPS = 2
S2_FPS = 1

# Patience policy
PATIENCE_SECONDS = 20

# Models
VLM_MODEL_S1 = "gemini-2.5-flash"
VLM_MODEL_S2 = "gemini-2.5-pro"

# Thresholds
CONF_THRESHOLD                = 90
STRONG_EVIDENCE               = 100
MIN_EVIDENCE_PER_STEP         = 2
EVIDENCE_CONTINUE_THRESHOLD   = 85
SEQUENTIAL_BIAS_STEPS         = 2
MAX_STEP_JUMP                 = 2
MAX_S2_FRAMES                 = 200   # 🔑 cap for Stage-2

# Retry / robustness
DEBUG        = True
MAX_RETRIES  = 3

# Downscale frames before JPEG
DOWNSCALE = (640, 360)
JPEG_QUALITY = 75

# ====================== Setup ======================
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY in .env")
client = genai.Client(api_key=API_KEY)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====================== Utils ======================
def ts(sec: float) -> str:
    sec = max(0, int(round(sec)))
    return f"{sec//60:02d}:{sec%60:02d}"

def _sec(ts_str: str) -> float:
    try:
        mm, ss = ts_str.split(":")
        return int(mm) * 60 + int(ss)
    except Exception:
        return 0.0

def load_steps(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "steps" in data:
        steps = data["steps"]
    elif isinstance(data, list):
        steps = data
    else:
        raise ValueError("Invalid howto.json format")
    cleaned = []
    for s in steps:
        try:
            sid = int(s.get("id"))
        except Exception:
            continue
        cleaned.append({
            "id": sid,
            "step_text": s.get("step_text", ""),
            "question": s.get("question", ""),
            "how": s.get("how", ""),
            "keywords": s.get("keywords", []),
            "prerequisites": s.get("prerequisites", []),
            "ambiguity": s.get("ambiguity", []),
        })
    return cleaned

def load_objects(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_action_verbs(path: str) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        return []

    verbs = data.get("unambiguous", []) if isinstance(data, dict) else []
    cleaned: List[str] = []
    for v in verbs:
        if not isinstance(v, str):
            continue
        val = v.strip()
        if val:
            cleaned.append(val)
    return cleaned

def retry_json_call(model: str, contents: List[Any], temperature: float = 0.0) -> Optional[Any]:
    cfg = types.GenerateContentConfig(response_mime_type="application/json", temperature=temperature)
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            res = client.models.generate_content(model=model, contents=contents, config=cfg)
            text = getattr(res, "text", None)
            if not text:
                return None
            return json.loads(text)
        except Exception as e:
            last_err = e
            wait = 2 ** attempt + random.random()
            print(f"⚠️ {model} error {type(e).__name__} attempt {attempt+1}/{MAX_RETRIES}: {e}, retry {wait:.1f}s…")
            time.sleep(wait)
    print(f"❌ {model} failed after {MAX_RETRIES} retries: {last_err}")
    return None

# ====================== Frame Selection ======================
def select_s2_frames(s2_buf, evidence_history, global_fps=0.2):
    """
    Select frames for Stage-2:
    - Uniform sparse global sampling
    - Always include evidence-linked frames
    - Cap total frames at MAX_S2_FRAMES
    """
    if not s2_buf:
        return []

    # 1. Global sampling (sparse coverage)
    stride = int(round(1.0 / global_fps))  # seconds
    global_selected = {int(t): b for t, b in s2_buf if int(t) % stride == 0}

    # 2. Evidence-based selection
    evidence_selected = {}
    for sid, evidences in evidence_history.items():
        for ev in evidences:
            t = _sec(ev["evidence_time"])
            if not s2_buf:
                continue
            # pick closest frame in buffer
            closest = min(s2_buf, key=lambda x: abs(x[0]-t))
            evidence_selected[int(closest[0])] = closest[1]

    # 3. Merge + sort
    combined = {**global_selected, **evidence_selected}
    selected_frames = sorted(combined.items(), key=lambda x: x[0])

    # 4. Hard cap to last MAX_S2_FRAMES
    if len(selected_frames) > MAX_S2_FRAMES:
        selected_frames = selected_frames[-MAX_S2_FRAMES:]

    return [b for _, b in selected_frames]


def atomic_conf_to_num(c: str) -> int:
    """Convert string confidence levels to numeric scores."""
    s = (c or "").lower()
    if s == "high":
        return 90
    if s == "medium":
        return 60
    if s == "low":
        return 25
    try:
        return int(c)  # in case the model already outputs a number
    except Exception:
        return 0

def parts_from_jpegs(jpegs: List[bytes]) -> List[dict]:
    return [{"image": {"bytes": b}} for b in jpegs]

# ====================== Prompts ======================
def build_s1_prompt(
                    title: str,
                    t_start: float,
                    t_end: float,
                    object_knowledge: dict,
                    steps: List[Dict[str,Any]],
                    last_evidences: Dict[int, List[Dict[str,Any]]],
                    action_verbs: List[str]
                ) -> str:
    """
    Build Stage-1 prompt for atomic action detection using predefined action templates.
    """
    start_m, start_s = divmod(int(t_start), 60)
    end_m, end_s = divmod(int(t_end), 60)
    time_range = f"{start_m:02d}:{start_s:02d}–{end_m:02d}:{end_s:02d}"
    verbs_block = json.dumps(action_verbs, ensure_ascii=False)

    return (
        f"Video title: {title}\n"
        f"Analyze frames from {time_range}.\n\n"
        "Your task is to extract **fine-grained human–object actions** that are "
        "specifically relevant to this car service and repair task.\n\n"
        "### Object Knowledge (reference only)\n"
        f"{json.dumps(object_knowledge, ensure_ascii=False, indent=2)}\n\n"
        "### Action Template\n"
        "- Subject: the tool or body part performing the action (hands, torque wrench, pry bar, etc.).\n"
        "- Object: the exact vehicle component being acted upon.\n"
        "- Verb: choose exactly one verb from this approved list.\n"
        f"{verbs_block}\n"
        "- Time: the timestamp (MM:SS) within this window when the action is visible.\n\n"
        "### Guidelines:\n"
        "- Carefully distinguish what object is being worked on (see 'possible_confusions').\n"
        "- Use only verbs from the approved list; skip the action if none apply.\n"
        "- Prefer precise tool names in subject; default to 'hands' only when no tool is clear.\n"
        "- Keep each action **atomic** (single observable motion).\n"
        "- Ensure the provided timestamp matches the moment the action happens within this window.\n"
        "### Output JSON format (ONLY JSON):\n"
        "{\n"
        "  \"actions\": [\n"
        "    {\n"
        "      \"subject\": \"tool or body part\",\n"
        "      \"object\": \"targeted component\",\n"
        "      \"verb\": \"verb from approved list\",\n"
        "      \"time\": \"MM:SS\"\n"
        "    }\n"
        "  ]\n"
        "}"
    )

def build_s2_prompt(title: str,
                    window_start: float,
                    window_end: float,
                    evidence_history: Dict[int, List[Dict[str,Any]]],
                    atomic_actions_recent: List[Dict[str,Any]],
                    steps: List[Dict[str,Any]],
                    last_completed_id: Optional[int],
                    last_candidate_id: Optional[int]) -> str:
    steps_summary = [
        {"id": s.get("id"), "step_text": s.get("step_text", "")}
        for s in steps
    ]
    return (
        f"Video title: {title}\n"
        f"Time range: {ts(window_start)}–{ts(window_end)}\n\n"
        "- Match the recent atomic actions with an instruction step if possible.\n"
        f"- Highly prefer steps close to the last step with evidence (in evidence time) (±{SEQUENTIAL_BIAS_STEPS}).\n"
        f"- Do not jump more than {MAX_STEP_JUMP} steps forward unless very strong.\n"
        "- Only confirm a step if the recent actions clearly without ambiguity support the described step in the instructions.\n"
        "Recent atomic actions observed (ordered):\n"
        f"{json.dumps(atomic_actions_recent, ensure_ascii=False, indent=2)}\n\n"
        "Evidence history so far:\n"
        f"{json.dumps(evidence_history, ensure_ascii=False, indent=2)}\n\n"
        "Candidate steps:\n"
        f"{json.dumps(steps_summary, ensure_ascii=False)}\n\n"
        "Policy:\n"
        "- If the verb is ambiguous, wait for more actions to promote the verb to specific action."
        "- If the last step still appears, accumulate more evidence.\n"
        "- Promote only when there’s strong evidence the previous step finished.\n"
        "- If unclear, return no_match.\n\n"
        "Return ONLY JSON:\n"
        "{ \"step_id\": <int>, \"confidence\": <0-100>, \"reason\": \"...\", \"evidence_time\": \"MM:SS\" }\n"
        "OR { \"step_id\": \"no_match\", \"confidence\": 0 }"
    )

# ====================== VLM Wrappers ======================
def vlm_stage1(
                frames_jpg: List[bytes],
                t_start: float,
                t_end: float,
                title: str,
                object_knowledge: dict,
                steps: List[Dict[str,Any]],                # 👈 add steps
                last_evidences: Dict[int, List[Dict[str,Any]]],  # 👈 add evidence history
                action_verbs: List[str]
                ) -> List[Dict[str,Any]]:
    if not frames_jpg:
        return []

    # Wrap frames correctly for Gemini
    frame_parts = [
        {"inline_data": {"data": b, "mime_type": "image/jpeg"}}
        for b in frames_jpg
    ]

    prompt = build_s1_prompt(title, t_start, t_end, object_knowledge, steps, last_evidences, action_verbs)
    contents = frame_parts + [{"text": prompt}]

    data = retry_json_call(VLM_MODEL_S1, contents, temperature=0.0)

    actions = []
    if isinstance(data, dict):
        actions = data.get("actions", [])

    out = []
    allowed_verbs = {v.lower() for v in action_verbs}
    for i, a in enumerate(actions):
        if not isinstance(a, dict):
            continue
        subject = str(a.get("subject", "")).strip()
        object_name = str(a.get("object", "")).strip()
        verb = str(a.get("verb", "")).strip()
        time_val = a.get("time")

        if allowed_verbs and verb.lower() not in allowed_verbs:
            continue

        if not verb:
            continue
        if not object_name:
            continue
        if not subject:
            subject = "hands"

        if isinstance(time_val, (int, float)):
            action_time = ts(float(time_val))
        else:
            action_time = str(time_val or "").strip()
            if not action_time:
                action_time = ts(t_end)
            elif ":" not in action_time:
                try:
                    action_time = ts(float(action_time))
                except ValueError:
                    action_time = ts(t_end)

        action_time_sec = _sec(action_time)
        if action_time_sec == 0 and action_time != "00:00":
            # Give up and fall back to window end if parsing failed
            action_time = ts(t_end)

        detail_parts = [subject, verb, object_name]
        details = " ".join(p for p in detail_parts if p).strip()
        if not details:
            continue

        label = " ".join(p for p in [verb, object_name] if p).strip() or verb or object_name

        out.append({
            "label": label,
            "details": details,
            "subject": subject,
            "object": object_name,
            "verb": verb,
            "time": action_time
        })
    return out

def vlm_stage2(
        frames_cur: List[bytes],
        frames_jpg: List[bytes],
        window_start: float,
        window_end: float,
        evidence_history: Dict[int, List[Dict[str,Any]]],
        recent_atomic: List[Dict[str,Any]],   # 👈 NEW
        steps_for_prompt: List[Dict[str,Any]],
        title: str,
        last_completed_id: Optional[int],
        last_candidate_id: Optional[int]
    ) -> Dict[str,Any]:

    if not frames_jpg:
        return {"step_id": "no_match", "confidence": 0}


    new_frame_parts = [
        {"inline_data": {"data": b, "mime_type": "image/jpeg"}}
        for b in frames_cur
    ]
    # Wrap frames correctly
    frame_parts = [
        {"inline_data": {"data": b, "mime_type": "image/jpeg"}}
        for b in frames_jpg
    ]

    prompt = build_s2_prompt(
        title,
        window_start,
        window_end,
        evidence_history,
        recent_atomic,    # 👈 include in prompt
        steps_for_prompt,
        last_completed_id,
        last_candidate_id
    )

    contents = [{"new video frames:"}, new_frame_parts, {"video history:"}, frame_parts, {"text": prompt}]

    data = retry_json_call(VLM_MODEL_S2, contents, temperature=0.0)
    if not isinstance(data, dict):
        return {"step_id": "no_match", "confidence": 0}

    return {
        "step_id": data.get("step_id", "no_match"),
        "confidence": int(data.get("confidence", 0) or 0),
        "reason": data.get("reason", ""),
        "evidence_time": data.get("evidence_time", ts(window_end))
    }

# ====================== Helpers ======================
def _in_range(act: Dict[str,Any], start: float, end: float) -> bool:
    t = _sec(act.get("time", "00:00"))
    return start <= t <= end

def _save_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _candidates_with_bias(steps: List[Dict[str,Any]], last_completed_id: Optional[int]) -> List[Dict[str,Any]]:
    if last_completed_id is None: return steps
    idx = {s["id"]: i for i,s in enumerate(steps)}
    base = idx.get(last_completed_id, 0)
    def score(s): return abs(idx.get(s["id"], base)-base)
    return sorted(steps, key=score)

def _parse_evidence_time(ev_time: str) -> float:
    """Return seconds from evidence_time (use upper bound if range)."""
    try:
        if "-" in ev_time:
            end = ev_time.split("-")[-1]
            mm, ss = end.split(":")
        else:
            mm, ss = ev_time.split(":")
        return int(mm) * 60 + int(ss)
    except Exception:
        return 0.0

# ====================== Main ======================
def main():
    steps = load_steps(HOWTO_JSON)
    steps_by_id = {s["id"]: s for s in steps}

    objects = load_objects(OBJECTS_JSON)   # ✅ load curated objects
    action_verbs = load_action_verbs(ACTION_VERBS_JSON)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    s1_interval = max(1, int(round(fps / S1_FPS)))
    s2_interval = max(1, int(round(fps / S2_FPS)))

    s1_buf, s2_buf = deque(), deque()
    atomic_timeline = []
    evidence_history: Dict[int,List[Dict[str,Any]]] = {}
    completed_steps = []
    last_completed_id, last_candidate_id = None, None
    next_stride_time, frame_idx = 0.0, 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        t_now = (cap.get(cv2.CAP_PROP_POS_MSEC) or 0)/1000.0

        small = cv2.resize(frame, DOWNSCALE)
        ok,jpg = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok: continue
        jpg_b = jpg.tobytes()
        if frame_idx % s1_interval == 0: s1_buf.append((t_now,jpg_b))
        if frame_idx % s2_interval == 0: s2_buf.append((t_now,jpg_b))
        while s1_buf and (t_now-s1_buf[0][0]>S1_WINDOW_SEC): s1_buf.popleft()
        while s2_buf and (t_now-s2_buf[0][0]>S2_WINDOW_SEC): s2_buf.popleft()

        if t_now < next_stride_time: continue
        next_stride_time = (int(t_now//STRIDE_SECONDS)+1)*STRIDE_SECONDS
        s1_start, s2_start, s1_end, s2_end = max(0,t_now-S1_WINDOW_SEC), max(0,t_now-S2_WINDOW_SEC), t_now, t_now
        print(f"▶️ Window {ts(s2_start)}–{ts(s2_end)}")

        # Stage 1
        s1_frames = [b for (_t,b) in s1_buf if _t>=s1_start]
        s1_actions = vlm_stage1(s1_frames, s1_start, s1_end, TITLE, objects, steps,
                                evidence_history, action_verbs)
        if s1_actions: atomic_timeline.extend(s1_actions)

        # Stage 2
        s2_frames = select_s2_frames(s2_buf, evidence_history, global_fps=0.2)
        candidates = _candidates_with_bias(steps, last_completed_id)

        # ✅ define recent_atomic here (subject, verb, object, plus time string)
        recent_atomic = []
        for a in atomic_timeline:
            if not _in_range(a, s2_start, s2_end):
                continue
            recent_atomic.append({
                "subject": a.get("subject", ""),
                "verb": a.get("verb", ""),
                "object": a.get("object", ""),
                "time": f"at {a.get('time', ts(s2_end))}"
            })

        s2_result = vlm_stage2(
            s1_frames,
            s2_frames,
            s2_start,
            s2_end,
            evidence_history,
            recent_atomic,      # 👈 now works
            candidates,
            TITLE,
            last_completed_id,
            last_candidate_id
        )
        step_id, conf, reason, ev_time = s2_result.get("step_id"), s2_result.get("confidence",0), s2_result.get("reason",""), s2_result.get("evidence_time","")
        #print(f"Stage-2 result raw: {s2_result}")
        if isinstance(step_id, int) and conf >= CONF_THRESHOLD:
            # Always record evidence
            hist = evidence_history.setdefault(step_id, [])
            hist.append({
                "offset": ts(s2_start),
                "confidence": conf,
                "reason": reason,
                "evidence_time": ev_time or ts(s2_end)
            })
            last_candidate_id = step_id
            print(f"📌 Evidence added for step {step_id} (conf={conf}) at {ev_time}")

            if step_id == last_completed_id:
                # still accumulating for current step → wipe forward ambiguous
                for sid in list(evidence_history.keys()):
                    if sid > step_id and sid <= step_id + 2:
                        del evidence_history[sid]
            else:
                # forward candidate during patience → just block promotion, keep evidence
                prev_id = last_completed_id
                if prev_id is not None:
                    last_ev = evidence_history.get(prev_id, [])
                    if last_ev:
                        last_end_sec = _parse_evidence_time(
                            last_ev[-1].get("evidence_time", ts(s2_end))
                        )
                        if (s2_end - last_end_sec) < PATIENCE_SECONDS:
                            print(f"⏳ Patience active for step {prev_id} (ends @ {ts(last_end_sec)}), not promoting to {step_id} yet")
                            # do not promote yet, but evidence stays
                            continue
        # Finalize the current candidate if patience expired and evidence is sufficient
        if last_candidate_id is not None:
            cand_id = last_candidate_id
            cand_hist = evidence_history.get(cand_id, [])
            if cand_hist:
                last_end_sec = _parse_evidence_time(
                    cand_hist[-1].get("evidence_time", ts(s2_end))
                )
                patience_elapsed = (s2_end - last_end_sec) >= PATIENCE_SECONDS
                strong_enough = (
                    len(cand_hist) >= MIN_EVIDENCE_PER_STEP
                    or any(h.get("confidence", 0) >= STRONG_EVIDENCE for h in cand_hist)
                )
                if patience_elapsed and strong_enough:
                    if cand_id not in [c["id"] for c in completed_steps]:
                        best = max(cand_hist, key=lambda h: h.get("confidence", 0))
                        completed_steps.append({
                            "id": cand_id,
                            "step_text": steps_by_id.get(cand_id, {}).get("step_text", ""),
                            "reason": best.get("reason", ""),
                            "evidence_time": best.get("evidence_time", ts(s2_end)),
                            "evidence_count": len(cand_hist),
                        })
                        last_completed_id = cand_id
                        last_candidate_id = None
                        print(f"✅ Step {cand_id} completed (patience via evidence_time): {steps_by_id.get(cand_id,{}).get('step_text','')}")
        _save_json(os.path.join(OUTPUT_DIR,"timeline_live.json"),{"actions":atomic_timeline})
        _save_json(os.path.join(OUTPUT_DIR,"instruction_status.json"),{"completed_steps":completed_steps})
        _save_json(os.path.join(OUTPUT_DIR,"evidence_history.json"), evidence_history)

    cap.release()
    print("🏁 Done.")

if __name__=="__main__":
    main()
