"""Coffee workflow variant that swaps Stage-1/Stage-2 inference to Qwen3-VL-8B-Instruct."""

import io
import json
import os
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

import run_sticky_evidence_actions_flush as pipeline

# =============================================================================
# Pipeline configuration mirrors the base coffee wrapper.
# =============================================================================
pipeline.TITLE = "Coffee Preparation and Cleanup"
pipeline.VIDEO_PATH = "./realwear-videos/coffee-full.mp4"
pipeline.HOWTO_JSON = "./realwear-videos/howto_coffee.json"
pipeline.OBJECTS_JSON = "./realwear-videos/objects.json"
pipeline.ACTION_VERBS_JSON = "./realwear-videos/action_verbs.json"
pipeline.OUTPUT_DIR = "realwear-videos/outputs/coffee-qwen"
pipeline.S1_MAX_FRAMES = 5

# =============================================================================
# Qwen model configuration
# =============================================================================
_QWEN_VL_MODEL_ID = os.getenv("QWEN_VL_MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")
_QWEN_VL_DEVICE = os.getenv("QWEN_VL_DEVICE")
_QWEN_VL_DEVICE_MAP = os.getenv("QWEN_VL_DEVICE_MAP", "auto")
_QWEN_VL_DTYPE = os.getenv("QWEN_VL_DTYPE", "auto")
_QWEN_VL_ATTN_IMPL = os.getenv("QWEN_VL_ATTN_IMPL", "flash_attention_2")
_QWEN_VL_LOCAL_FILES_ONLY = os.getenv("QWEN_VL_LOCAL_FILES_ONLY", "0") == "1"
_QWEN_VL_MAX_NEW_TOKENS = int(os.getenv("QWEN_VL_MAX_NEW_TOKENS", "768"))
_QWEN_VL_TEMPERATURE = float(os.getenv("QWEN_VL_TEMPERATURE", "0.0"))
_QWEN_VL_TOP_P = os.getenv("QWEN_VL_TOP_P")
_QWEN_VL_DO_SAMPLE = os.getenv("QWEN_VL_DO_SAMPLE")
_QWEN_VL_TOP_P_VALUE: Optional[float] = float(_QWEN_VL_TOP_P) if _QWEN_VL_TOP_P else None
_QWEN_VL_FORCE_DO_SAMPLE = (
    _QWEN_VL_DO_SAMPLE == "1" or (_QWEN_VL_DO_SAMPLE is None and _QWEN_VL_TEMPERATURE > 0.0)
)

_QWEN_VL_MODEL: Optional[Qwen3VLForConditionalGeneration] = None
_QWEN_VL_PROCESSOR: Optional[AutoProcessor] = None

_SYSTEM_JSON_ONLY = (
    "You are an expert analyst for industrial procedures. "
    "Always respond with strict JSON using only double-quoted keys and values. "
    "Do not include extra commentary before or after the JSON."
)


def _resolve_torch_dtype(name: str) -> Any:
    key = (name or "").strip().lower()
    if not key or key == "auto" or key == "default":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(key, "auto")


def _ensure_qwen_loaded() -> None:
    """Load the Qwen3-VL model and processor on first use."""
    global _QWEN_VL_MODEL, _QWEN_VL_PROCESSOR
    if _QWEN_VL_MODEL is not None and _QWEN_VL_PROCESSOR is not None:
        return

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": _QWEN_VL_LOCAL_FILES_ONLY,
    }

    resolved_dtype = _resolve_torch_dtype(_QWEN_VL_DTYPE)
    if resolved_dtype is not None:
        model_kwargs["torch_dtype"] = resolved_dtype

    if _QWEN_VL_DEVICE and _QWEN_VL_DEVICE.strip():
        model_kwargs.pop("device_map", None)
    else:
        model_kwargs["device_map"] = _QWEN_VL_DEVICE_MAP

    if _QWEN_VL_ATTN_IMPL:
        model_kwargs["attn_implementation"] = _QWEN_VL_ATTN_IMPL

    _QWEN_VL_MODEL = Qwen3VLForConditionalGeneration.from_pretrained(
        _QWEN_VL_MODEL_ID,
        **model_kwargs,
    )

    if _QWEN_VL_DEVICE and _QWEN_VL_DEVICE.strip():
        _QWEN_VL_MODEL.to(_QWEN_VL_DEVICE)

    _QWEN_VL_MODEL.eval()

    _QWEN_VL_PROCESSOR = AutoProcessor.from_pretrained(
        _QWEN_VL_MODEL_ID,
        trust_remote_code=True,
        local_files_only=_QWEN_VL_LOCAL_FILES_ONLY,
    )

    gen_cfg = _QWEN_VL_MODEL.generation_config
    for attr in ("temperature", "top_p", "top_k"):
        if hasattr(gen_cfg, attr):
            setattr(gen_cfg, attr, None)

    pipeline.VLM_MODEL_S1 = _QWEN_VL_MODEL_ID
    pipeline.VLM_MODEL_S2 = _QWEN_VL_MODEL_ID


def _decode_images(frames: List[bytes]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for idx, payload in enumerate(frames):
        try:
            with Image.open(io.BytesIO(payload)) as img:
                images.append(img.convert("RGB"))
        except Exception as exc:
            if pipeline.DEBUG:
                print(f"⚠️ Qwen image decode failed on frame {idx}: {exc}")
    return images


def _prepare_inputs(messages: List[Dict[str, Any]], images: List[Image.Image]) -> Dict[str, Any]:
    assert _QWEN_VL_PROCESSOR is not None
    conversation = _QWEN_VL_PROCESSOR.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = _QWEN_VL_PROCESSOR(
        text=conversation,
        images=images if images else None,
        return_tensors="pt",
    )
    return {
        k: v.to(_QWEN_VL_MODEL.device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }


def _qwen_generate_json(messages: List[Dict[str, Any]], images: List[Image.Image]) -> Optional[Dict[str, Any]]:
    _ensure_qwen_loaded()
    inputs = _prepare_inputs(messages, images)
    generate_kwargs: Dict[str, Any] = {
        "max_new_tokens": _QWEN_VL_MAX_NEW_TOKENS,
        "pad_token_id": _QWEN_VL_PROCESSOR.tokenizer.pad_token_id,
        "eos_token_id": _QWEN_VL_PROCESSOR.tokenizer.eos_token_id,
    }
    if _QWEN_VL_FORCE_DO_SAMPLE:
        generate_kwargs["do_sample"] = True
        if _QWEN_VL_TEMPERATURE > 0.0:
            generate_kwargs["temperature"] = _QWEN_VL_TEMPERATURE
        if _QWEN_VL_TOP_P_VALUE is not None:
            generate_kwargs["top_p"] = _QWEN_VL_TOP_P_VALUE
    else:
        generate_kwargs["do_sample"] = False

    with torch.inference_mode():
        generated = _QWEN_VL_MODEL.generate(**inputs, **generate_kwargs)

    input_length = inputs["input_ids"].shape[-1]
    generated_ids = generated[:, input_length:]
    raw = _QWEN_VL_PROCESSOR.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if not raw:
        return None

    text = raw[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if pipeline.DEBUG:
            print("⚠️ Qwen response was not valid JSON:", text)
    return None


def vlm_stage1_qwen(
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

    prompt = pipeline.build_s1_prompt(
        title,
        t_start,
        t_end,
        object_knowledge,
        steps,
        last_evidences,
        action_verbs,
    )

    images = _decode_images(frames_jpg)
    if not images:
        return []

    user_content: List[Dict[str, Any]] = [{"type": "image"} for _ in images]
    user_content.append({"type": "text", "text": prompt})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": _SYSTEM_JSON_ONLY}]},
        {"role": "user", "content": user_content},
    ]

    data = None
    for attempt in range(pipeline.MAX_RETRIES):
        data = _qwen_generate_json(messages, images)
        if isinstance(data, dict):
            break
        if pipeline.DEBUG:
            print(f"⚠️ Qwen Stage-1 retry {attempt+1}/{pipeline.MAX_RETRIES}")
    if not isinstance(data, dict):
        return []

    actions = data.get("actions", [])
    if not isinstance(actions, list):
        return []

    results: List[Dict[str, Any]] = []
    allowed_verbs = {v.lower() for v in action_verbs}
    for entry in actions:
        if not isinstance(entry, dict):
            continue
        subject = str(entry.get("subject", "")).strip()
        object_name = str(entry.get("object", "")).strip()
        verb = str(entry.get("verb", "")).strip()
        time_val = entry.get("time")

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
        details = " ".join(part for part in detail_parts if part)
        if not details:
            continue

        label = " ".join(part for part in [verb, object_name] if part) or verb or object_name

        results.append(
            {
                "label": label,
                "details": details,
                "subject": subject,
                "object": object_name,
                "verb": verb,
                "time": action_time,
            }
        )
    return results


def vlm_stage2_qwen(
    frames_cur: List[bytes],
    frames_jpg: List[bytes],
    window_start: float,
    window_end: float,
    evidence_history: Dict[int, List[Dict[str, Any]]],
    recent_atomic: List[Dict[str, Any]],
    steps_for_prompt: List[Dict[str, Any]],
    title: str,
    last_completed_id: Optional[int],
) -> Dict[str, Any]:
    prompt = pipeline.build_s2_prompt(
        title,
        window_start,
        window_end,
        evidence_history,
        recent_atomic,
        steps_for_prompt,
        last_completed_id,
    )

    images = _decode_images(frames_jpg or frames_cur)
    user_content: List[Dict[str, Any]] = []
    if images:
        user_content.extend({"type": "image"} for _ in images)
    user_content.append({"type": "text", "text": prompt})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": _SYSTEM_JSON_ONLY}]},
        {"role": "user", "content": user_content},
    ]

    data = None
    for attempt in range(pipeline.MAX_RETRIES):
        data = _qwen_generate_json(messages, images)
        if isinstance(data, dict):
            break
        if pipeline.DEBUG:
            print(f"⚠️ Qwen Stage-2 retry {attempt+1}/{pipeline.MAX_RETRIES}")

    if not isinstance(data, dict):
        return {"step_id": "no_match", "confidence": 0}

    step_id = data.get("step_id", "no_match")
    try:
        confidence = int(data.get("confidence", 0) or 0)
    except (TypeError, ValueError):
        confidence = 0

    reason = data.get("reason", "")
    evidence_time = data.get("evidence_time", pipeline.ts(window_end))

    return {
        "step_id": step_id,
        "confidence": confidence,
        "reason": reason,
        "evidence_time": evidence_time,
    }


# Swap the inference functions for this script only.
pipeline.vlm_stage1 = vlm_stage1_qwen
pipeline.vlm_stage2 = vlm_stage2_qwen


def main() -> None:
    _ensure_qwen_loaded()
    os.makedirs(pipeline.OUTPUT_DIR, exist_ok=True)
    pipeline.main()


if __name__ == "__main__":
    main()
