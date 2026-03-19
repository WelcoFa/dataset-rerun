import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


# ============================================================
# Config
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
DATA_ROOT = REPO_ROOT / "data" / "gigahands"
GIGAHANDS_ROOT = DATA_ROOT / "gigahands_demo_all"

SCENE_NAME = "p36-tea-0010"
CAM_NAME = "brics-odroid-010_cam0"
VIDEO_STEM = "brics-odroid-010_cam0_1727030430697198"

VIDEO_PATH = (
    GIGAHANDS_ROOT
    / "hand_pose"
    / SCENE_NAME
    / "rgb_vid"
    / CAM_NAME
    / f"{VIDEO_STEM}.mp4"
)

OUTPUT_DIR = DATA_ROOT / "annotations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_STEPS_PATH = OUTPUT_DIR / f"pred_steps_{SCENE_NAME}.json"
OUTPUT_RAW_CLIPS_PATH = OUTPUT_DIR / f"pred_raw_clips_{SCENE_NAME}.json"

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# 速度优先参数
CLIP_LEN_FRAMES = 64
CLIP_STRIDE_FRAMES = 64
NUM_SAMPLED_FRAMES = 2
BATCH_SIZE = 1

# 只输出短 JSON
MAX_NEW_TOKENS = 24
DO_SAMPLE = False
TEMPERATURE = 0.2
TOP_P = 0.9

# 如果装了 flash-attn，可以设为 True
USE_FLASH_ATTN = False

# 如果只想先测前 N 个 clip，改成整数，比如 4；全部跑就设 None
MAX_CLIPS = None

CANONICAL_LABELS = [
    "approach",
    "touch",
    "grasp",
    "hold",
    "lift",
    "move",
    "place",
    "release",
    "manipulate",
    "other",
]


# ============================================================
# Data structures
# ============================================================

@dataclass
class ClipInfo:
    clip_id: int
    start_frame: int
    end_frame: int
    sampled_frame_indices: List[int]
    sampled_images: List[Image.Image]


# ============================================================
# Video utilities
# ============================================================

def get_video_info(video_path: Path) -> Tuple[int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if fps <= 0:
        fps = 30.0
    return total_frames, fps


def read_frame_bgr(cap: cv2.VideoCapture, frame_idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def bgr_to_pil(frame_bgr) -> Image.Image:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def linspace_int(start: int, end: int, n: int) -> List[int]:
    if n <= 1:
        return [start]
    if end < start:
        end = start

    vals = []
    for i in range(n):
        t = i / (n - 1)
        x = round(start * (1 - t) + end * t)
        vals.append(int(x))

    out = []
    seen = set()
    for v in vals:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def sample_video_clips_in_memory(
    video_path: Path,
    clip_len_frames: int,
    clip_stride_frames: int,
    num_sampled_frames: int,
    max_clips: Optional[int] = None,
) -> Tuple[List[ClipInfo], float, int]:
    total_frames, fps = get_video_info(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    clips: List[ClipInfo] = []
    clip_id = 0

    starts = list(range(0, max(total_frames - 1, 1), clip_stride_frames))
    for start_frame in starts:
        end_frame = min(start_frame + clip_len_frames - 1, total_frames - 1)
        if end_frame < start_frame:
            continue

        sampled_indices = linspace_int(start_frame, end_frame, num_sampled_frames)
        sampled_images: List[Image.Image] = []

        for frame_idx in sampled_indices:
            frame = read_frame_bgr(cap, frame_idx)
            if frame is None:
                continue
            sampled_images.append(bgr_to_pil(frame))

        if sampled_images:
            clips.append(
                ClipInfo(
                    clip_id=clip_id,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    sampled_frame_indices=sampled_indices[: len(sampled_images)],
                    sampled_images=sampled_images,
                )
            )
            clip_id += 1

        if max_clips is not None and len(clips) >= max_clips:
            break

        if end_frame >= total_frames - 1:
            break

    cap.release()
    return clips, fps, total_frames


# ============================================================
# Prompt / parsing
# ============================================================

def make_prompt(scene_name: str, start_frame: int, end_frame: int) -> str:
    labels_text = ", ".join(CANONICAL_LABELS)
    return f"""
You are annotating an egocentric hand-object interaction clip from the GigaHands dataset.

Scene: {scene_name}
Clip frame range: {start_frame} to {end_frame}

Task:
1. Treat the images as one short action clip.
2. Choose exactly ONE label from this closed vocabulary:
   [{labels_text}]
3. Write one short action description in plain English.
4. Focus on the dominant hand-object action in this clip.
5. If uncertain, choose the closest label. Use "other" only when necessary.

Return STRICT JSON only:
{{
  "label": "one_label_from_the_list",
  "text": "one short sentence"
}}
""".strip()


def extract_json_block(text: str) -> Optional[str]:
    text = text.strip()

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)

    plain = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if plain:
        return plain.group(1)

    return None


def normalize_label(label: str) -> str:
    if not label:
        return "other"

    x = label.strip().lower()

    direct_map = {
        "approach": "approach",
        "touch": "touch",
        "grasp": "grasp",
        "hold": "hold",
        "lift": "lift",
        "move": "move",
        "place": "place",
        "release": "release",
        "manipulate": "manipulate",
        "other": "other",
    }
    if x in direct_map:
        return direct_map[x]

    synonym_rules = [
        (["reach", "reaches", "reaching", "approach", "approaches"], "approach"),
        (["touch", "touches", "contact"], "touch"),
        (["grasp", "grasps", "grab", "grabs", "grabbing", "pick up"], "grasp"),
        (["hold", "holds", "holding"], "hold"),
        (["lift", "lifts", "lifting", "raise", "raises"], "lift"),
        (["move", "moves", "moving", "pull", "pulls", "slide", "slides"], "move"),
        (["place", "places", "set down", "put down", "puts down"], "place"),
        (["release", "releases", "let go"], "release"),
        (["manipulate", "rotates", "rotate", "adjust", "open", "close"], "manipulate"),
    ]

    for keys, target in synonym_rules:
        for k in keys:
            if k in x:
                return target

    return "other"


def parse_model_json(raw_text: str) -> Dict[str, Any]:
    raw_text = raw_text.strip()
    json_block = extract_json_block(raw_text)

    if json_block is None:
        return {
            "label": "other",
            "text": raw_text[:200] if raw_text else "unparsed response",
            "parse_ok": False,
        }

    try:
        data = json.loads(json_block)
    except Exception:
        return {
            "label": "other",
            "text": raw_text[:200] if raw_text else "json parse failed",
            "parse_ok": False,
        }

    label = normalize_label(str(data.get("label", "other")))
    text = str(data.get("text", "")).strip()
    if not text:
        text = label

    return {
        "label": label,
        "text": text,
        "parse_ok": True,
    }


# ============================================================
# Model loading
# ============================================================

def get_torch_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        # 4060 通常 fp16 更稳
        return torch.float16
    return torch.float32


def load_model_and_processor(model_id: str):
    print(f"Loading model: {model_id}")
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu name:", torch.cuda.get_device_name(0))

    kwargs = {
        "torch_dtype": get_torch_dtype(),
    }

    if USE_FLASH_ATTN:
        kwargs["attn_implementation"] = "flash_attention_2"

    if torch.cuda.is_available():
        # 明确放到 GPU，避免 auto 分配到 CPU
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            **kwargs,
        ).cuda()
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            **kwargs,
        )

    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"

    print("model device:", next(model.parameters()).device)
    return model, processor


# ============================================================
# Batched inference
# ============================================================

def build_messages_for_clip(clip: ClipInfo, scene_name: str) -> List[Dict[str, Any]]:
    prompt = make_prompt(scene_name, clip.start_frame, clip.end_frame)

    content = []
    for img in clip.sampled_images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt})

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]
    return messages


def run_qwen_batch(
    model,
    processor,
    batch_clips: List[ClipInfo],
    scene_name: str,
) -> List[Tuple[str, Dict[str, Any]]]:
    messages_list = [build_messages_for_clip(clip, scene_name) for clip in batch_clips]

    texts = [
        processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        for messages in messages_list
    ]

    image_inputs_all = []
    video_inputs_all = []

    for messages in messages_list:
        image_inputs, video_inputs = process_vision_info(messages)
        image_inputs_all.append(image_inputs)
        video_inputs_all.append(video_inputs)

    inputs = processor(
        text=texts,
        images=image_inputs_all,
        padding=True,
        return_tensors="pt",
    )

    if torch.cuda.is_available():
        inputs = {
            k: v.cuda(non_blocking=True) if hasattr(v, "cuda") else v
            for k, v in inputs.items()
        }

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE if DO_SAMPLE else None,
            top_p=TOP_P if DO_SAMPLE else None,
            use_cache=True,
        )

    input_lengths = (inputs["input_ids"] != processor.tokenizer.pad_token_id).sum(dim=1).tolist()

    outputs = []
    for i in range(len(batch_clips)):
        trimmed = generated_ids[i][input_lengths[i]:]
        decoded = processor.decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        parsed = parse_model_json(decoded)
        outputs.append((decoded, parsed))

    return outputs


# ============================================================
# Merge steps
# ============================================================

def merge_clip_predictions(raw_clip_preds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not raw_clip_preds:
        return []

    merged: List[Dict[str, Any]] = []

    current = {
        "start": raw_clip_preds[0]["start"],
        "end": raw_clip_preds[0]["end"],
        "label": raw_clip_preds[0]["label"],
        "text": raw_clip_preds[0]["text"],
    }

    for item in raw_clip_preds[1:]:
        same_label = item["label"] == current["label"]

        if same_label:
            current["end"] = item["end"]
            if len(item["text"]) > len(current["text"]):
                current["text"] = item["text"]
        else:
            merged.append(current)
            current = {
                "start": item["start"],
                "end": item["end"],
                "label": item["label"],
                "text": item["text"],
            }

    merged.append(current)
    return merged


# ============================================================
# Main
# ============================================================

def main():
    print("=== run_vlm_gigahands_fast.py ===")
    print("SCENE_NAME        =", SCENE_NAME)
    print("VIDEO_PATH        =", VIDEO_PATH)
    print("MODEL_ID          =", MODEL_ID)
    print("OUTPUT_STEPS      =", OUTPUT_STEPS_PATH)
    print("OUTPUT_RAW_CLIPS  =", OUTPUT_RAW_CLIPS_PATH)
    print("CLIP_LEN_FRAMES   =", CLIP_LEN_FRAMES)
    print("CLIP_STRIDE_FRAMES=", CLIP_STRIDE_FRAMES)
    print("NUM_SAMPLED_FRAMES=", NUM_SAMPLED_FRAMES)
    print("BATCH_SIZE        =", BATCH_SIZE)
    print("MAX_NEW_TOKENS    =", MAX_NEW_TOKENS)
    print()

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    clips, fps, total_frames = sample_video_clips_in_memory(
        video_path=VIDEO_PATH,
        clip_len_frames=CLIP_LEN_FRAMES,
        clip_stride_frames=CLIP_STRIDE_FRAMES,
        num_sampled_frames=NUM_SAMPLED_FRAMES,
        max_clips=MAX_CLIPS,
    )

    print(f"Total frames : {total_frames}")
    print(f"FPS          : {fps:.2f}")
    print(f"Num clips    : {len(clips)}")
    print()

    model, processor = load_model_and_processor(MODEL_ID)

    raw_clip_preds: List[Dict[str, Any]] = []
    total_start = time.time()

    for batch_start in range(0, len(clips), BATCH_SIZE):
        batch = clips[batch_start: batch_start + BATCH_SIZE]
        batch_end = min(batch_start + BATCH_SIZE, len(clips))
        print(f"Running batch {batch_start + 1}-{batch_end}/{len(clips)}")

        t0 = time.time()
        outputs = run_qwen_batch(
            model=model,
            processor=processor,
            batch_clips=batch,
            scene_name=SCENE_NAME,
        )
        dt = time.time() - t0
        print(f"  batch_time = {dt:.2f}s")
        print(f"  avg_per_clip = {dt / len(batch):.2f}s")

        for clip, (raw_text, parsed) in zip(batch, outputs):
            item = {
                "clip_id": clip.clip_id,
                "start": clip.start_frame,
                "end": clip.end_frame,
                "sampled_frames": clip.sampled_frame_indices,
                "label": parsed["label"],
                "text": parsed["text"],
                "parse_ok": parsed.get("parse_ok", False),
                "raw_response": raw_text,
            }
            raw_clip_preds.append(item)

            print(
                f"  clip {clip.clip_id:04d} | "
                f"frames {clip.start_frame}-{clip.end_frame} | "
                f"label={item['label']} | parse_ok={item['parse_ok']}"
            )
            print(f"    text={item['text']}")

        print()

    merged_steps = merge_clip_predictions(raw_clip_preds)

    with open(OUTPUT_RAW_CLIPS_PATH, "w", encoding="utf-8") as f:
        json.dump(raw_clip_preds, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_STEPS_PATH, "w", encoding="utf-8") as f:
        json.dump(merged_steps, f, ensure_ascii=False, indent=2)

    total_dt = time.time() - total_start

    print("Saved raw clip predictions to:")
    print(" ", OUTPUT_RAW_CLIPS_PATH)
    print("Saved merged step predictions to:")
    print(" ", OUTPUT_STEPS_PATH)
    print()
    print(f"Total inference time: {total_dt:.2f}s")
    print()

    print("Merged steps preview:")
    for i, step in enumerate(merged_steps, 1):
        print(
            f"  Step {i}: [{step['start']}, {step['end']}] "
            f"{step['label']} | {step['text']}"
        )


if __name__ == "__main__":
    main()
