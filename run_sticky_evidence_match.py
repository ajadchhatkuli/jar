import os
from collections import deque
from typing import Any, Dict, List

import cv2

import run_sticky_evidence_actions as base
from utils.matching import load_step_triplets, match_steps


def main() -> None:
    steps = base.load_steps(base.HOWTO_JSON)
    steps_by_id = {s["id"]: s for s in steps}
    step_triplets = load_step_triplets(base.HOWTO_JSON)

    objects = base.load_objects(base.OBJECTS_JSON)
    action_verbs = base.load_action_verbs(base.ACTION_VERBS_JSON)

    os.makedirs(base.OUTPUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(base.VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    s1_interval = max(1, int(round(fps / base.S1_FPS)))

    s1_buf: deque = deque()
    atomic_timeline: List[Dict[str, Any]] = []
    evidence_history: Dict[int, List[Dict[str, Any]]] = {}
    completed_steps: List[Dict[str, Any]] = []

    next_stride_time = 0.0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        t_now = (cap.get(cv2.CAP_PROP_POS_MSEC) or 0) / 1000.0

        small = cv2.resize(frame, base.DOWNSCALE)
        ok, jpg = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), base.JPEG_QUALITY])
        if not ok:
            continue
        jpg_b = jpg.tobytes()

        if frame_idx % s1_interval == 0:
            s1_buf.append((t_now, jpg_b))
        while s1_buf and (t_now - s1_buf[0][0] > base.S1_WINDOW_SEC):
            s1_buf.popleft()

        if t_now < next_stride_time:
            continue
        next_stride_time = (int(t_now // base.STRIDE_SECONDS) + 1) * base.STRIDE_SECONDS

        window_start = max(0.0, t_now - base.S1_WINDOW_SEC)
        window_end = t_now
        print(f"▶️ Window {base.ts(window_start)}–{base.ts(window_end)}")

        s1_frames = [b for (_t, b) in s1_buf if _t >= window_start]
        actions = base.vlm_stage1(
            s1_frames,
            window_start,
            window_end,
            base.TITLE,
            objects,
            steps,
            evidence_history,
            action_verbs,
        )
        if actions:
            atomic_timeline.extend(actions)

        matches = match_steps(step_triplets, atomic_timeline)
        for step_id, match_list in matches.items():
            if step_id in evidence_history:
                continue

            times = [m["action"].get("time") for m in match_list if m["action"].get("time")]
            evidence_time = times[-1] if times else base.ts(window_end)
            evidence_entry = {
                "offset": base.ts(window_start),
                "evidence_time": evidence_time,
                "confidence": 100,
                "reason": "Matched all subject–verb–object triplets for this step.",
                "matches": match_list,
            }
            evidence_history.setdefault(step_id, []).append(evidence_entry)
            completed_steps.append(
                {
                    "id": step_id,
                    "step_text": steps_by_id.get(step_id, {}).get("step_text", ""),
                    "reason": evidence_entry["reason"],
                    "evidence_time": evidence_time,
                    "evidence_count": len(match_list),
                }
            )
            print(f"✅ Step {step_id} matched via SVO: {steps_by_id.get(step_id, {}).get('step_text', '')}")

        base._save_json(os.path.join(base.OUTPUT_DIR, "timeline_live.json"), {"actions": atomic_timeline})
        base._save_json(os.path.join(base.OUTPUT_DIR, "instruction_status.json"), {"completed_steps": completed_steps})
        base._save_json(os.path.join(base.OUTPUT_DIR, "evidence_history.json"), evidence_history)

    cap.release()
    print("🏁 Done.")


if __name__ == "__main__":
    main()
