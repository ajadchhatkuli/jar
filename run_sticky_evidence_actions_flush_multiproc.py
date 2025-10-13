# Multiprocessing variant of run_sticky_evidence_actions_flush.py
# --------------------------------------------------------------
# This script keeps the original stage-1 / stage-2 logic intact, but
# splits frame acquisition and pipeline consumption into two processes.

import multiprocessing as mp
import os
import time
import copy
import cv2
import queue
import threading
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import run_sticky_evidence_actions_flush as base

PRODUCER_FPS = 1.0
FRAME_QUEUE_MAXSIZE = 8


def wall_clock_str(epoch: Optional[float] = None) -> str:
    """Return wall clock timestamp string."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch or time.time()))


def _emit_log(prefix: str, message: str, epoch: Optional[float] = None) -> None:
    """Print helper with wall-clock timestamp."""
    print(f"[{prefix} {wall_clock_str(epoch)}] {message}", flush=True)


def _encode_frame(frame) -> bytes:
    """Downscale and JPEG-encode frame according to base settings."""
    small = cv2.resize(frame, base.DOWNSCALE)
    ok, jpg = cv2.imencode(
        ".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), base.JPEG_QUALITY]
    )
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return jpg.tobytes()


def producer_process(frame_queue: mp.Queue, stop_event: mp.Event) -> None:
    """Read video frames and emit JPEGs at PRODUCER_FPS with real-time pacing."""
    cap = cv2.VideoCapture(base.VIDEO_PATH)
    if not cap.isOpened():
        _emit_log("Producer", f"Failed to open video: {base.VIDEO_PATH}")
        frame_queue.put({"type": "eof", "emitted_at": time.time(), "error": "open_failed"})
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    emit_period = 1.0 / PRODUCER_FPS if PRODUCER_FPS > 0 else 1.0
    last_emit_video_time = -emit_period
    start_wall = time.perf_counter()

    _emit_log(
        "Producer",
        f"Streaming {base.VIDEO_PATH} at {PRODUCER_FPS:.2f} fps (video fps ~{video_fps:.2f})",
    )

    frame_idx = 0
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            # Prefer CAP_PROP_POS_MSEC for accurate video time, fallback to idx / fps.
            video_time = (
                (cap.get(cv2.CAP_PROP_POS_MSEC) or (frame_idx / video_fps) * 1000.0)
                / 1000.0
            )
            # Emit at most one frame per emit_period seconds (absolute video time).
            if video_time - last_emit_video_time < emit_period - 1e-3:
                continue
            try:
                jpg_bytes = _encode_frame(frame)
            except Exception:
                _emit_log(
                    "Producer",
                    f"Skipped frame at video {base.ts(video_time)} due to encode error",
                )
                last_emit_video_time = video_time
                continue

            target_wall = start_wall + video_time
            delay = target_wall - time.perf_counter()
            if delay > 0:
                time.sleep(delay)

            payload = {
                "type": "frame",
                "video_time": video_time,
                "jpeg": jpg_bytes,
                "emitted_at": time.time(),
            }
            frame_queue.put(payload)
            _emit_log("Producer", f"Emitted frame at video {base.ts(video_time)}")
            last_emit_video_time = video_time

            if stop_event.is_set():
                break
    finally:
        frame_queue.put({"type": "eof", "emitted_at": time.time()})
        cap.release()
        _emit_log("Producer", "Producer finished.")


def consumer_process(
    frame_queue: mp.Queue, result_queue: mp.Queue, stop_event: mp.Event
) -> None:
    """Consume live frames and run the original pipeline logic."""
    steps = base.load_steps(base.HOWTO_JSON)
    steps_by_id = {s["id"]: s for s in steps}
    step_index_by_id = {s["id"]: idx for idx, s in enumerate(steps)}
    objects = base.load_objects(base.OBJECTS_JSON)
    action_verbs = base.load_action_verbs(base.ACTION_VERBS_JSON)

    s1_buf: deque[Tuple[float, bytes]] = deque()
    s2_buf: deque[Tuple[float, bytes]] = deque()
    state = base.PipelineState()

    s1_queue: "queue.Queue[base.WindowJob]" = queue.Queue(maxsize=4)
    s2_queue: "queue.Queue[base.WindowJob]" = queue.Queue(maxsize=4)
    print_lock = threading.Lock()

    def stage1_worker() -> None:
        while True:
            job = s1_queue.get()
            if job is base.SENTINEL:
                s2_queue.put(base.SENTINEL)
                break
            with state.lock:
                evidence_snapshot = copy.deepcopy(state.evidence_history)
            timer_start = time.perf_counter()
            actions: List[Dict[str, Any]] = []
            try:
                actions = base.vlm_stage1(
                    job.s1_frames,
                    job.s1_start,
                    job.s1_end,
                    base.TITLE,
                    objects,
                    steps,
                    evidence_snapshot,
                    action_verbs,
                )
                if actions:
                    with state.lock:
                        state.action_timeline.extend(actions)
            except Exception as exc:
                with print_lock:
                    _emit_log(
                        "Consumer",
                        f"Stage-1 worker error on window {job.window_label}: {exc}",
                    )
            finally:
                job.stage1_time = time.perf_counter() - timer_start
                job.stage1_actions = actions
                s2_queue.put(job)

    def stage2_worker() -> None:
        while True:
            job = s2_queue.get()
            if job is base.SENTINEL:
                break
            try:
                with state.lock:
                    evidence_snapshot = copy.deepcopy(state.evidence_history)
                    recent_atomic = state.action_timeline.recent(job.s2_start, job.s2_end)
                    last_completed = state.last_completed_id
                s2_timer_start = time.perf_counter()
                s2_frames = base.select_s2_frames(
                    job.s2_buffer, evidence_snapshot, global_fps=0.2
                )
                candidates = base._candidates_with_bias(steps, last_completed)
                s2_result = base.vlm_stage2(
                    job.s1_frames,
                    s2_frames,
                    job.s2_start,
                    job.s2_end,
                    evidence_snapshot,
                    recent_atomic,
                    candidates,
                    base.TITLE,
                    last_completed,
                )
                stage2_time = time.perf_counter() - s2_timer_start
                total_elapsed = job.stage1_time + stage2_time
                waiting_overhead = 0.0
                if base.DEBUG:
                    raw_elapsed = time.perf_counter() - job.window_timer_start
                    waiting_overhead = max(0.0, raw_elapsed - total_elapsed)

                log_lines: List[str] = []
                step_id = s2_result.get("step_id")
                conf = s2_result.get("confidence", 0)
                reason = s2_result.get("reason", "")
                ev_time = s2_result.get("evidence_time", base.ts(job.s2_end))

                timeline_for_save: Dict[str, Any] = {"actions": []}
                instruction_status_for_save: Dict[str, Any] = {"completed_steps": []}
                evidence_for_save: Dict[int, List[Dict[str, Any]]] = {}

                with state.lock:
                    if isinstance(step_id, int) and conf >= base.CONF_THRESHOLD:
                        hist = state.evidence_history.setdefault(step_id, [])
                        hist.append(
                            {
                                "offset": base.ts(job.s2_start),
                                "confidence": conf,
                                "reason": reason,
                                "evidence_time": ev_time or base.ts(job.s2_end),
                            }
                        )
                        consume_until_raw = base._parse_evidence_time(
                            ev_time or base.ts(job.s2_end)
                        )
                        if consume_until_raw <= 0:
                            consume_until_raw = job.s2_end
                        consume_until = min(
                            max(consume_until_raw, job.s2_start), job.s2_end
                        )
                        consumed_count = state.action_timeline.consume(
                            job.s2_start, consume_until
                        )
                        if base.DEBUG and consumed_count:
                            log_lines.append(
                                f"🧹 Flushed {consumed_count} actions up to {base.ts(consume_until)}"
                            )
                        log_lines.append(
                            f"📌 Evidence added for step {step_id} (conf={conf}) at {ev_time}"
                        )
                        promote_allowed = True
                        if state.last_completed_id is not None:
                            prev_idx = step_index_by_id.get(state.last_completed_id)
                            curr_idx = step_index_by_id.get(step_id)
                            if (
                                prev_idx is not None
                                and curr_idx is not None
                                and abs(curr_idx - prev_idx)
                                > base.MAX_PROMOTION_STEP_GAP
                            ):
                                promote_allowed = False
                        if promote_allowed:
                            already_completed = any(
                                c["id"] == step_id for c in state.completed_steps
                            )
                            if not already_completed:
                                best = max(hist, key=lambda h: h.get("confidence", 0))
                                completion = {
                                    "id": step_id,
                                    "step_text": steps_by_id.get(step_id, {}).get(
                                        "step_text", ""
                                    ),
                                    "reason": best.get("reason", ""),
                                    "evidence_time": best.get(
                                        "evidence_time", base.ts(job.s2_end)
                                    ),
                                    "evidence_count": len(hist),
                                }
                                state.completed_steps.append(completion)
                                log_lines.append(
                                    f"✅ Step {step_id} completed automatically: {completion['step_text']}"
                                )
                                result_queue.put(
                                    {
                                        "type": "step_completed",
                                        "step_id": step_id,
                                        "step_text": completion["step_text"],
                                        "confidence": conf,
                                        "reason": completion["reason"],
                                        "evidence_time": completion["evidence_time"],
                                        "wall_time": time.time(),
                                    }
                                )
                            state.last_completed_id = step_id
                        else:
                            log_lines.append(
                                f"🚫 Skipping promotion for step {step_id}; beyond {base.MAX_PROMOTION_STEP_GAP} step gap."
                            )
                    timeline_for_save = {
                        "actions": state.action_timeline.serializable()
                    }
                    instruction_status_for_save = {
                        "completed_steps": list(state.completed_steps)
                    }
                    evidence_for_save = copy.deepcopy(state.evidence_history)
                with print_lock:
                    _emit_log(
                        "Consumer",
                        f"▶️ Window {job.window_label}  (T: {total_elapsed:.2f}s, S1: {job.stage1_time:.2f}s, S2: {stage2_time:.2f}s)",
                    )
                    if base.DEBUG and waiting_overhead > 0.0:
                        _emit_log(
                            "Consumer",
                            f"  ⏱️ queue overhead: {waiting_overhead:.2f}s",
                        )
                    for line in log_lines:
                        _emit_log("Consumer", line)
                base._save_json(
                    os.path.join(base.OUTPUT_DIR, "timeline_live.json"),
                    timeline_for_save,
                )
                base._save_json(
                    os.path.join(base.OUTPUT_DIR, "instruction_status.json"),
                    instruction_status_for_save,
                )
                base._save_json(
                    os.path.join(base.OUTPUT_DIR, "evidence_history.json"),
                    evidence_for_save,
                )
            except Exception as exc:
                with print_lock:
                    _emit_log(
                        "Consumer",
                        f"Stage-2 worker error on window {job.window_label}: {exc}",
                    )

    stage1_thread = threading.Thread(
        target=stage1_worker, name="Stage1Worker", daemon=True
    )
    stage2_thread = threading.Thread(
        target=stage2_worker, name="Stage2Worker", daemon=True
    )
    stage1_thread.start()
    stage2_thread.start()

    next_stride_time = 0.0
    window_id = 0

    try:
        while True:
            if stop_event.is_set():
                break
            try:
                message = frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            msg_type = message.get("type")
            if msg_type == "frame":
                t_now = float(message["video_time"])
                jpg_b = message["jpeg"]
                _emit_log(
                    "Consumer",
                    f"Received frame at video {base.ts(t_now)}",
                    message.get("emitted_at"),
                )

                s1_buf.append((t_now, jpg_b))
                s2_buf.append((t_now, jpg_b))
                while s1_buf and (t_now - s1_buf[0][0] > base.S1_WINDOW_SEC):
                    s1_buf.popleft()
                while s2_buf and (t_now - s2_buf[0][0] > base.S2_WINDOW_SEC):
                    s2_buf.popleft()

                if t_now < next_stride_time:
                    continue
                next_stride_time = (int(t_now // base.STRIDE_SECONDS) + 1) * base.STRIDE_SECONDS

                s1_start = max(0.0, t_now - base.S1_WINDOW_SEC)
                s2_start = max(0.0, t_now - base.S2_WINDOW_SEC)
                s1_end = t_now
                s2_end = t_now
                window_label = f"{base.ts(s2_start)}–{base.ts(s2_end)}"

                job = base.WindowJob(
                    window_id=window_id,
                    s1_start=s1_start,
                    s1_end=s1_end,
                    s2_start=s2_start,
                    s2_end=s2_end,
                    window_label=window_label,
                    s1_frames=[b for (_t, b) in s1_buf if _t >= s1_start],
                    s2_buffer=list(s2_buf),
                    window_timer_start=time.perf_counter(),
                )
                window_id += 1
                s1_queue.put(job)
            elif msg_type in {"eof", "stop"}:
                _emit_log("Consumer", f"Received {msg_type} signal.")
                break
    finally:
        s1_queue.put(base.SENTINEL)
        stage1_thread.join()
        stage2_thread.join()
        result_queue.put({"type": "done", "finished_at": time.time()})
        _emit_log("Consumer", "Consumer finished.")


def main() -> None:
    frame_queue: mp.Queue = mp.Queue(maxsize=FRAME_QUEUE_MAXSIZE)
    result_queue: mp.Queue = mp.Queue()
    stop_event = mp.Event()

    producer = mp.Process(
        target=producer_process, args=(frame_queue, stop_event), name="FrameProducer"
    )
    consumer = mp.Process(
        target=consumer_process,
        args=(frame_queue, result_queue, stop_event),
        name="PipelineConsumer",
    )

    producer.start()
    consumer.start()

    try:
        done = False
        while not done:
            try:
                message = result_queue.get(timeout=1.0)
            except queue.Empty:
                if not producer.is_alive() and not consumer.is_alive():
                    break
                continue
            msg_type = message.get("type")
            if msg_type == "step_completed":
                _emit_log(
                    "Main",
                    f"Step {message['step_id']} matched (conf={message['confidence']}): {message['step_text']} @ {message['evidence_time']}",
                    message.get("wall_time"),
                )
            elif msg_type == "done":
                _emit_log("Main", "Consumer reported completion.", message.get("finished_at"))
                done = True
    except KeyboardInterrupt:
        _emit_log("Main", "Interrupted, signalling shutdown.")
    finally:
        stop_event.set()
        try:
            frame_queue.put_nowait({"type": "stop"})
        except Exception:
            pass

        producer.join(timeout=5)
        consumer.join(timeout=5)
        if producer.is_alive():
            _emit_log("Main", "Terminating producer.")
            producer.terminate()
        if consumer.is_alive():
            _emit_log("Main", "Terminating consumer.")
            consumer.terminate()

        frame_queue.close()
        frame_queue.join_thread()
        result_queue.close()
        result_queue.join_thread()


if __name__ == "__main__":
    main()
