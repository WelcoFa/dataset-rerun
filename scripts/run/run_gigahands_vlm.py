import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor


# ============================================================
# Config
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
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

MODEL_ID = os.environ.get("GIGAHANDS_VLM_MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct")

# 速度优先参数
CLIP_LEN_FRAMES = 64
CLIP_STRIDE_FRAMES = 64
NUM_SAMPLED_FRAMES = int(os.environ.get("GIGAHANDS_NUM_SAMPLED_FRAMES", "3"))
BATCH_SIZE = int(os.environ.get("GIGAHANDS_BATCH_SIZE", "2"))

# 只输出短 JSON
MAX_NEW_TOKENS = int(os.environ.get("GIGAHANDS_MAX_NEW_TOKENS", "96"))
DO_SAMPLE = False
TEMPERATURE = 0.2
TOP_P = 0.9

# 如果装了 flash-attn，可以设为 True
USE_FLASH_ATTN = False

# Resize sampled frames before VLM inference. Smaller images are much faster.
MAX_IMAGE_EDGE = int(os.environ.get("GIGAHANDS_MAX_IMAGE_EDGE", "448"))

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


def infer_main_task(scene_name: str) -> str:
    scene = scene_name.lower()
    if "tea" in scene:
        return "Preparing tea with a teapot"
    if "boxing" in scene:
        return "Interacting with a boxing bag"
    return "Hand-object manipulation"


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
    image = Image.fromarray(frame_rgb)
    if MAX_IMAGE_EDGE > 0:
        width, height = image.size
        longest_edge = max(width, height)
        if longest_edge > MAX_IMAGE_EDGE:
            scale = MAX_IMAGE_EDGE / float(longest_edge)
            resized_size = (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            )
            image = image.resize(resized_size, Image.BILINEAR)
    return image


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
2. Focus only on the clip-specific semantics. The overall scene task is known already.
3. Choose exactly ONE label from this closed vocabulary:
   [{labels_text}]
4. Focus on the dominant hand-object action in this clip.
5. sub_task should name the current phase in 3 to 8 words, such as "pouring tea into the mug".
6. interaction should be one concise sentence describing the hand-object roles if visible.
7. objects must be a JSON list of visible manipulated objects only. Avoid generic words like "scene", "hand", or "object".
8. current_action should be one concise sentence that clearly describes what is happening in this clip.
9. If uncertain, choose the closest label. Use "other" only when no label fits.

Return STRICT JSON only:
{{
  "sub_task": "specific clip-level phase",
  "interaction": "one sentence describing the interaction",
  "objects": ["object one", "object two"],
  "label": "one_label_from_the_list",
  "current_action": "one concise sentence"
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


def extract_step_text(value: Any) -> str:
    if isinstance(value, dict):
        if "text" in value:
            return extract_step_text(value["text"])
        if "label" in value:
            return extract_step_text(value["label"])
        return json.dumps(value, ensure_ascii=False)

    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return stripped
            return extract_step_text(parsed)
        return stripped

    if value is None:
        return ""

    return str(value)


def clean_semantic_text(value: Any) -> str:
    text = extract_step_text(value)
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip(" \n\r\t`")


def extract_embedded_step_fields(value: Any) -> Dict[str, str]:
    if isinstance(value, dict):
        return {
            "label": clean_semantic_text(value.get("label", "")),
            "text": clean_semantic_text(value.get("text", "")),
        }

    if not isinstance(value, str):
        return {"label": "", "text": clean_semantic_text(value)}

    stripped = value.strip()
    label_match = re.search(r'"label"\s*:\s*"([^"]+)', stripped)
    text_match = re.search(r'"text"\s*:\s*"([^"]+)', stripped, flags=re.DOTALL)

    return {
        "label": label_match.group(1).strip() if label_match else "",
        "text": text_match.group(1).strip() if text_match else "",
    }


def extract_partial_json_fields(text: str) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    for key in ("main_task", "sub_task", "interaction", "label", "current_action", "text"):
        exact_match = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', text, flags=re.DOTALL)
        if exact_match:
            try:
                fields[key] = json.loads(f'"{exact_match.group(1)}"')
            except json.JSONDecodeError:
                fields[key] = exact_match.group(1).strip()
            continue

        partial_match = re.search(rf'"{key}"\s*:\s*"([^"\r\n]*)', text, flags=re.DOTALL)
        if partial_match:
            fields[key] = partial_match.group(1).strip()

    objects_match = re.search(r'"objects"\s*:\s*\[(.*?)\]', text, flags=re.DOTALL)
    if objects_match:
        fields["objects"] = re.findall(r'"([^"]+)"', objects_match.group(1))

    return fields


def normalize_object_name(name: str) -> List[str]:
    name = name.lower().replace("-", "_").strip()

    banned_exact = {
        "transform",
        "mesh",
        "transform_mesh",
        "object",
        "world",
        "camera",
        "scene",
        "hand",
        "left hand",
        "right hand",
    }
    if name in banned_exact:
        return []

    tokens = [t for t in name.split("_") if t]
    banned_tokens = {"transform", "mesh", "world", "camera", "scene"}
    if tokens and all(t in banned_tokens for t in tokens):
        return []

    special_map = {
        "teapot_with_lid": ["teapot", "lid"],
        "boxing_bag_stand": ["boxing bag", "stand"],
    }
    if name in special_map:
        return special_map[name]

    drop_tokens = {"with", "and", "transform", "mesh", "object", "objects"}
    tokens = [t for t in tokens if t not in drop_tokens]

    if not tokens:
        return []

    return [" ".join(tokens)]


def discover_scene_objects(scene_name: str) -> List[str]:
    pose_dir = GIGAHANDS_ROOT / "object_pose" / scene_name / "pose"
    if not pose_dir.exists():
        return []

    object_names = []
    for path in pose_dir.iterdir():
        if path.suffix.lower() in {".obj", ".ply", ".glb", ".stl"}:
            object_names.append(path.stem)
    return sorted(set(object_names))


def build_scene_object_registry(scene_name: str) -> List[Dict[str, Any]]:
    registry: List[Dict[str, Any]] = []
    for raw_name in discover_scene_objects(scene_name):
        labels = normalize_object_name(raw_name)
        if not labels:
            continue
        registry.append({"raw_name": raw_name, "labels": labels})
    return registry


def get_scene_object_labels(scene_registry: Optional[List[Dict[str, Any]]]) -> List[str]:
    if not scene_registry:
        return []

    labels: List[str] = []
    for item in scene_registry:
        labels.extend(item["labels"])
    return sorted(set(labels))


def normalize_objects(
    value: Any,
    scene_registry: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    if isinstance(value, list):
        raw = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                raw = parsed
            else:
                raw = [text]
        else:
            raw = re.split(r"[,;/\n]+", text)
    else:
        raw = [value]

    cleaned: List[str] = []
    for item in raw:
        text = clean_semantic_text(item).strip().lower()
        if not text or text in {"none", "n/a", "unknown", "other"}:
            continue
        cleaned.extend(normalize_object_name(text))

    scene_objects = set(get_scene_object_labels(scene_registry))
    if scene_objects:
        cleaned = [item for item in cleaned if item in scene_objects]

    deduped = []
    seen = set()
    for item in cleaned:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def choose_best_text(*values: Any) -> str:
    invalid_values = {"", "other", "unknown", "none", "n/a", "null"}
    for value in values:
        text = clean_semantic_text(value)
        if text.lower() not in invalid_values:
            return text
    return ""


def parse_model_json(
    raw_text: str,
    scene_name: str,
    scene_registry: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    raw_text = raw_text.strip()
    json_block = extract_json_block(raw_text)
    partial_fields = extract_partial_json_fields(raw_text)
    embedded_fields = extract_embedded_step_fields(raw_text)
    data: Dict[str, Any] = {}
    parse_ok = False

    if json_block is not None:
        try:
            data = json.loads(json_block)
            parse_ok = True
        except Exception:
            data = {}

    label_source = choose_best_text(
        data.get("label", ""),
        partial_fields.get("label", ""),
        embedded_fields.get("label", ""),
    )
    label = normalize_label(label_source)

    current_action = choose_best_text(
        data.get("current_action", ""),
        data.get("text", ""),
        partial_fields.get("current_action", ""),
        partial_fields.get("text", ""),
        embedded_fields.get("text", ""),
    )
    if not current_action:
        current_action = label if label != "other" else "unparsed response"

    sub_task = choose_best_text(
        data.get("sub_task", ""),
        partial_fields.get("sub_task", ""),
        current_action,
        label,
    )
    interaction = choose_best_text(
        data.get("interaction", ""),
        partial_fields.get("interaction", ""),
        current_action,
        label,
    )
    main_task = choose_best_text(
        data.get("main_task", ""),
        partial_fields.get("main_task", ""),
        infer_main_task(scene_name),
    )

    objects = normalize_objects(
        data.get("objects", partial_fields.get("objects", [])),
        scene_registry=scene_registry,
    )
    if not objects:
        objects = get_scene_object_labels(scene_registry)

    return {
        "main_task": main_task,
        "sub_task": sub_task,
        "interaction": interaction,
        "objects": objects,
        "label": label,
        "current_action": current_action,
        "text": current_action,
        "parse_ok": parse_ok,
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
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "Qwen VLM inference requires transformers with `Qwen2_5_VLForConditionalGeneration` support."
        ) from exc

    print(f"Loading model: {model_id}")
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu name:", torch.cuda.get_device_name(0))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

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
    scene_registry: Optional[List[Dict[str, Any]]] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "Qwen VLM inference requires `qwen-vl-utils` and its vision dependencies. "
            "Install the VLM extras first with `uv sync --extra gigahands-vlm`."
        ) from exc

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

    for messages in messages_list:
        image_inputs, video_inputs = process_vision_info(messages)
        image_inputs_all.append(image_inputs)
        del video_inputs

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
        parsed = parse_model_json(
            decoded,
            scene_name=scene_name,
            scene_registry=scene_registry,
        )
        outputs.append((decoded, parsed))

    return outputs


# ============================================================
# Merge steps
# ============================================================

def merge_clip_predictions(raw_clip_preds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not raw_clip_preds:
        return []

    def keyword_set(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z]+", text.lower())
            if len(token) >= 4 and token not in {"with", "from", "into", "their", "while"}
        }

    def should_merge(prev_item: Dict[str, Any], next_item: Dict[str, Any]) -> bool:
        if prev_item["label"] != next_item["label"]:
            return False
        if prev_item["label"] != "other":
            return True

        shared_objects = set(prev_item["objects"]) & set(next_item["objects"])
        shared_keywords = keyword_set(prev_item["sub_task"]) & keyword_set(next_item["sub_task"])
        return bool(shared_objects or shared_keywords)

    merged: List[Dict[str, Any]] = []

    current = {
        "start": raw_clip_preds[0]["start"],
        "end": raw_clip_preds[0]["end"],
        "main_task": raw_clip_preds[0]["main_task"],
        "sub_task": raw_clip_preds[0]["sub_task"],
        "interaction": raw_clip_preds[0]["interaction"],
        "objects": list(raw_clip_preds[0]["objects"]),
        "label": raw_clip_preds[0]["label"],
        "current_action": raw_clip_preds[0]["current_action"],
        "text": raw_clip_preds[0]["text"],
    }

    for item in raw_clip_preds[1:]:
        if should_merge(current, item):
            current["end"] = item["end"]
            if len(item["main_task"]) > len(current["main_task"]):
                current["main_task"] = item["main_task"]
            if len(item["sub_task"]) > len(current["sub_task"]):
                current["sub_task"] = item["sub_task"]
            if len(item["interaction"]) > len(current["interaction"]):
                current["interaction"] = item["interaction"]
            if len(item["current_action"]) > len(current["current_action"]):
                current["current_action"] = item["current_action"]
            if len(item["text"]) > len(current["text"]):
                current["text"] = item["text"]
            current["objects"] = normalize_objects(current["objects"] + item["objects"])
        else:
            merged.append(current)
            current = {
                "start": item["start"],
                "end": item["end"],
                "main_task": item["main_task"],
                "sub_task": item["sub_task"],
                "interaction": item["interaction"],
                "objects": list(item["objects"]),
                "label": item["label"],
                "current_action": item["current_action"],
                "text": item["text"],
            }

    merged.append(current)
    return merged


# ============================================================
# Main
# ============================================================

def main():
    effective_batch_size = BATCH_SIZE if torch.cuda.is_available() else 1

    print("=== run_vlm_gigahands_fast.py ===")
    print("SCENE_NAME        =", SCENE_NAME)
    print("VIDEO_PATH        =", VIDEO_PATH)
    print("MODEL_ID          =", MODEL_ID)
    print("OUTPUT_STEPS      =", OUTPUT_STEPS_PATH)
    print("OUTPUT_RAW_CLIPS  =", OUTPUT_RAW_CLIPS_PATH)
    print("CLIP_LEN_FRAMES   =", CLIP_LEN_FRAMES)
    print("CLIP_STRIDE_FRAMES=", CLIP_STRIDE_FRAMES)
    print("NUM_SAMPLED_FRAMES=", NUM_SAMPLED_FRAMES)
    print("BATCH_SIZE        =", effective_batch_size)
    print("MAX_NEW_TOKENS    =", MAX_NEW_TOKENS)
    print("MAX_IMAGE_EDGE    =", MAX_IMAGE_EDGE)
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

    scene_registry = build_scene_object_registry(SCENE_NAME)
    print("Scene objects =", get_scene_object_labels(scene_registry))

    model, processor = load_model_and_processor(MODEL_ID)

    raw_clip_preds: List[Dict[str, Any]] = []
    total_start = time.time()

    for batch_start in range(0, len(clips), effective_batch_size):
        batch = clips[batch_start: batch_start + effective_batch_size]
        batch_end = min(batch_start + effective_batch_size, len(clips))
        print(f"Running batch {batch_start + 1}-{batch_end}/{len(clips)}")

        t0 = time.time()
        outputs = run_qwen_batch(
            model=model,
            processor=processor,
            batch_clips=batch,
            scene_name=SCENE_NAME,
            scene_registry=scene_registry,
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
                "main_task": parsed["main_task"],
                "sub_task": parsed["sub_task"],
                "interaction": parsed["interaction"],
                "objects": parsed["objects"],
                "label": parsed["label"],
                "current_action": parsed["current_action"],
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
            print(f"    sub_task={item['sub_task']}")
            print(f"    interaction={item['interaction']}")
            print(f"    objects={item['objects']}")
            print(f"    current_action={item['current_action']}")

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
            f"{step['label']} | {step['current_action']}"
        )


if __name__ == "__main__":
    main()
