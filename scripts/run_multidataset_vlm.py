import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import h5py
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rerun_viz.core import read_image_rgb_unicode_safe


DATA_ROOT = REPO_ROOT / "data"
MODEL_ID = os.environ.get("MULTIDATASET_VLM_MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct")
MAX_NEW_TOKENS = int(os.environ.get("MULTIDATASET_MAX_NEW_TOKENS", "128"))
MAX_IMAGE_EDGE = int(os.environ.get("MULTIDATASET_MAX_IMAGE_EDGE", "448"))
BATCH_SIZE = int(os.environ.get("MULTIDATASET_BATCH_SIZE", "1"))
DEFAULT_NUM_SAMPLED_FRAMES = int(os.environ.get("MULTIDATASET_NUM_SAMPLED_FRAMES", "3"))

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


@dataclass
class ClipInfo:
    dataset: str
    source_id: str
    clip_id: int
    start_frame: int
    end_frame: int
    sampled_frame_indices: List[int]
    sampled_images: List[Image.Image]
    prompt_context: Dict[str, Any]
    output_dir: Path


def parse_args():
    parser = argparse.ArgumentParser(description="Universal VLM semantic extractor for datasets under data/.")
    parser.add_argument("--input", type=Path, default=DATA_ROOT, help="Dataset root, item root, or the full data/ directory.")
    parser.add_argument(
        "--dataset",
        choices=["auto", "gigahands", "hot3d", "being-h0", "dexwild", "thermohands", "wiyh"],
        default="auto",
        help="Force a dataset type or auto-detect from --input.",
    )
    parser.add_argument("--list-jobs", action="store_true", help="Only show detected jobs.")
    parser.add_argument("--max-jobs", type=int, default=-1, help="Limit the number of detected jobs.")
    parser.add_argument("--max-clips", type=int, default=-1, help="Limit the number of clips per job.")
    return parser.parse_args()


def ensure_annotations_dir(dataset_root: Path) -> Path:
    output_dir = dataset_root / "annotations"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_csv(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def linspace_int(start: int, end: int, n: int) -> List[int]:
    if n <= 1:
        return [start]
    if end < start:
        end = start
    vals = []
    for i in range(n):
        t = i / (n - 1)
        vals.append(int(round(start * (1 - t) + end * t)))
    out = []
    seen = set()
    for value in vals:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def resize_image(image: Image.Image) -> Image.Image:
    if MAX_IMAGE_EDGE <= 0:
        return image
    width, height = image.size
    longest_edge = max(width, height)
    if longest_edge <= MAX_IMAGE_EDGE:
        return image
    scale = MAX_IMAGE_EDGE / float(longest_edge)
    resized_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return image.resize(resized_size, Image.BILINEAR)


def pil_from_rgb_array(array: np.ndarray) -> Image.Image:
    return resize_image(Image.fromarray(np.asarray(array, dtype=np.uint8)))


def pil_from_image_path(path: Path) -> Image.Image:
    return resize_image(Image.fromarray(read_image_rgb_unicode_safe(path)))


def normalize_label(label: str) -> str:
    x = (label or "").strip().lower()
    if x in CANONICAL_LABELS:
        return x
    rules = [
        (["reach", "approach"], "approach"),
        (["touch", "contact"], "touch"),
        (["grasp", "grab", "pick up"], "grasp"),
        (["hold", "holding"], "hold"),
        (["lift", "raise"], "lift"),
        (["move", "pull", "push", "carry"], "move"),
        (["place", "set down", "put down"], "place"),
        (["release", "let go"], "release"),
        (["manipulate", "adjust", "open", "close", "rotate", "pour"], "manipulate"),
    ]
    for keys, target in rules:
        if any(key in x for key in keys):
            return target
    return "other"


def extract_json_block(text: str) -> Optional[str]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)
    plain = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if plain:
        return plain.group(1)
    return None


def parse_response(raw_text: str, fallback_main_task: str) -> Dict[str, Any]:
    result = {
        "main_task": fallback_main_task,
        "sub_task": "other",
        "interaction": "other",
        "objects": [],
        "label": "other",
        "current_action": "other",
        "parse_ok": False,
    }
    json_block = extract_json_block(raw_text)
    if json_block is None:
        return result

    try:
        parsed = json.loads(json_block)
    except json.JSONDecodeError:
        return result

    if not isinstance(parsed, dict):
        return result

    result["parse_ok"] = True
    result["main_task"] = str(parsed.get("main_task", fallback_main_task)).strip() or fallback_main_task
    result["sub_task"] = str(parsed.get("sub_task", "other")).strip() or "other"
    result["interaction"] = str(parsed.get("interaction", "other")).strip() or "other"
    result["current_action"] = str(parsed.get("current_action", "other")).strip() or "other"
    result["label"] = normalize_label(str(parsed.get("label", "other")))
    objects = parsed.get("objects", [])
    if isinstance(objects, list):
        result["objects"] = [str(item).strip() for item in objects if str(item).strip()]
    return result


def choose_overall_task(dataset: str, context: Dict[str, Any]) -> str:
    if dataset == "gigahands":
        scene = str(context.get("scene_name", "")).lower()
        if "tea" in scene:
            return "Preparing tea with a teapot"
        if "boxing" in scene:
            return "Interacting with a boxing bag"
        return "Hand-object manipulation"
    if dataset == "wiyh":
        return str(context.get("task_description", "Robot manipulation task"))
    if dataset == "being-h0":
        return str(context.get("instruction", "Behavior cloning task"))
    if dataset == "dexwild":
        return f"DexWild manipulation episode {context.get('episode', 'unknown')}"
    if dataset == "thermohands":
        return f"ThermoHands scene {context.get('scene_name', 'unknown')}"
    if dataset == "hot3d":
        return f"HOT3D sequence {context.get('sequence_name', 'unknown')}"
    return "Manipulation task"


def make_prompt(clip: ClipInfo) -> str:
    labels_text = ", ".join(CANONICAL_LABELS)
    fallback_main_task = choose_overall_task(clip.dataset, clip.prompt_context)
    context_lines = [f"Dataset: {clip.dataset}", f"Source id: {clip.source_id}", f"Clip frame range: {clip.start_frame} to {clip.end_frame}"]
    for key, value in clip.prompt_context.items():
        if key == "images_note":
            continue
        context_lines.append(f"{key}: {value}")

    return f"""
You are annotating a short manipulation clip for a dashboard.

Context:
{chr(10).join(context_lines)}

Task:
1. Treat the inputs as one short clip or short temporal segment.
2. Return a concise dashboard-friendly semantic summary.
3. Choose exactly ONE label from this closed vocabulary:
   [{labels_text}]
4. main_task should describe the overall task in one sentence.
5. sub_task should name the current phase in 3 to 8 words.
6. interaction should be one concise sentence describing hand-object, robot-object, or person-object roles.
7. objects must be a JSON list of manipulated or task-relevant objects only.
8. current_action should be one concise sentence describing what happens in this clip.
9. If uncertain, choose the closest label. Use "other" only when no label fits.

Fallback overall task if unsure: {fallback_main_task}

Return STRICT JSON only:
{{
  "main_task": "one sentence overall task",
  "sub_task": "specific current phase",
  "interaction": "one sentence interaction summary",
  "objects": ["object one", "object two"],
  "label": "one_label_from_the_list",
  "current_action": "one concise sentence"
}}
""".strip()


def build_messages(clip: ClipInfo):
    content = []
    for image in clip.sampled_images:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": make_prompt(clip)})
    return [{"role": "user", "content": content}]


def merge_clips(raw_clip_preds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not raw_clip_preds:
        return []
    merged = [dict(raw_clip_preds[0])]
    for item in raw_clip_preds[1:]:
        current = merged[-1]
        shared_objects = set(current["objects"]) & set(item["objects"])
        shared_words = set(current["sub_task"].lower().split()) & set(item["sub_task"].lower().split())
        if current["label"] == item["label"] and (shared_objects or shared_words):
            current["end_frame"] = item["end_frame"]
            if len(item["sub_task"]) > len(current["sub_task"]):
                current["sub_task"] = item["sub_task"]
            if len(item["interaction"]) > len(current["interaction"]):
                current["interaction"] = item["interaction"]
            if len(item["current_action"]) > len(current["current_action"]):
                current["current_action"] = item["current_action"]
            current["objects"] = sorted(set(current["objects"]) | set(item["objects"]))
        else:
            merged.append(dict(item))
    return merged


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()
    return processor, model


def run_inference(clips: List[ClipInfo], processor, model) -> List[Dict[str, Any]]:
    raw_clip_preds: List[Dict[str, Any]] = []
    total_batches = (len(clips) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_index in range(total_batches):
        batch = clips[batch_index * BATCH_SIZE : (batch_index + 1) * BATCH_SIZE]
        batch_start = batch_index * BATCH_SIZE + 1
        batch_end = batch_index * BATCH_SIZE + len(batch)
        print(f"Running batch {batch_start}-{batch_end}/{len(clips)}")
        t0 = time.time()
        conversations = [build_messages(clip) for clip in batch]
        prompts = [processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True) for conv in conversations]
        image_inputs, video_inputs = process_vision_info(conversations)
        inputs = processor(text=prompts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        trimmed_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        outputs = processor.batch_decode(trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        dt = time.time() - t0
        print(f"  batch_time = {dt:.2f}s")
        print(f"  avg_per_clip = {dt / len(batch):.2f}s")
        for clip, raw_text in zip(batch, outputs):
            parsed = parse_response(raw_text, choose_overall_task(clip.dataset, clip.prompt_context))
            item = {
                "dataset": clip.dataset,
                "source_id": clip.source_id,
                "clip_id": clip.clip_id,
                "start_frame": clip.start_frame,
                "end_frame": clip.end_frame,
                "main_task": parsed["main_task"],
                "sub_task": parsed["sub_task"],
                "interaction": parsed["interaction"],
                "objects": parsed["objects"],
                "label": parsed["label"],
                "current_action": parsed["current_action"],
                "parse_ok": parsed["parse_ok"],
                "raw_response": raw_text.strip(),
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
    return raw_clip_preds


def detect_dataset(input_path: Path, requested_dataset: str) -> str:
    if requested_dataset != "auto":
        return requested_dataset
    parts = {part.lower() for part in input_path.parts}
    name = input_path.name.lower()
    if "gigahands" in parts or name == "gigahands":
        return "gigahands"
    if "hot3d" in parts or name == "hot3d":
        return "hot3d"
    if "being-h0" in parts or name == "being-h0":
        return "being-h0"
    if "dexwild" in parts or name == "dexwild":
        return "dexwild"
    if "thermohands" in parts or name == "thermohands":
        return "thermohands"
    if "wyih" in parts or name == "wyih":
        return "wiyh"
    raise ValueError(f"Could not auto-detect dataset from {input_path}")


def build_gigahands_jobs(root: Path) -> List[ClipInfo]:
    dataset_root = root / "gigahands_demo_all" if (root / "gigahands_demo_all").is_dir() else root
    annotations_dir = ensure_annotations_dir(root if root.name == "gigahands" else root.parent)
    hand_pose_root = dataset_root / "hand_pose"
    seq_dirs = sorted([p for p in hand_pose_root.iterdir() if p.is_dir()])
    clips: List[ClipInfo] = []
    for seq_dir in seq_dirs[:1]:
        cam_dirs = sorted([p for p in (seq_dir / "rgb_vid").iterdir() if p.is_dir()])
        if not cam_dirs:
            continue
        video_files = sorted(cam_dirs[0].glob("*.mp4"))
        if not video_files:
            continue
        video_path = video_files[0]
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        clip_starts = list(range(0, max(total_frames - 1, 1), 64))
        source_id = seq_dir.name
        clip_id = 0
        for start_frame in clip_starts:
            end_frame = min(start_frame + 63, total_frames - 1)
            sampled_indices = linspace_int(start_frame, end_frame, DEFAULT_NUM_SAMPLED_FRAMES)
            cap = cv2.VideoCapture(str(video_path))
            sampled_images: List[Image.Image] = []
            for frame_idx in sampled_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if ok and frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    sampled_images.append(pil_from_rgb_array(frame))
            cap.release()
            if sampled_images:
                clips.append(
                    ClipInfo(
                        dataset="gigahands",
                        source_id=source_id,
                        clip_id=clip_id,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        sampled_frame_indices=sampled_indices,
                        sampled_images=sampled_images,
                        prompt_context={"scene_name": seq_dir.name, "camera_name": cam_dirs[0].name},
                        output_dir=annotations_dir,
                    )
                )
                clip_id += 1
            if end_frame >= total_frames - 1:
                break
    return clips


def clean_instruction(text: str) -> str:
    text = (text or "").replace("<image>", "").replace("<PROP_CONTEXT>", "")
    text = " ".join(text.split()).strip()
    match = re.search(r"instruction\s+'([^']+)'", text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else text


def extract_instruction(conversations) -> str:
    if not isinstance(conversations, list):
        return ""
    for item in conversations:
        if isinstance(item, dict) and item.get("from", "").lower() == "human":
            return clean_instruction(item.get("value", ""))
    return clean_instruction(conversations[0].get("value", "")) if conversations and isinstance(conversations[0], dict) else ""


def build_beingh0_jobs(root: Path) -> List[ClipInfo]:
    base = root / "h0_post_train_db_2508" if (root / "h0_post_train_db_2508").is_dir() else root
    subset_dirs = sorted([p for p in base.iterdir() if p.is_dir() and (p / "images").is_dir()])
    clips: List[ClipInfo] = []
    output_dir = ensure_annotations_dir(root if root.name == "Being-h0" else root.parent)
    for subset_dir in subset_dirs[:1]:
        jsonl_path = sorted(subset_dir.glob("*_train.jsonl"))[0]
        samples = load_jsonl(jsonl_path)
        clip_id = 0
        for start in range(0, len(samples), 16):
            end = min(start + 15, len(samples) - 1)
            sampled_indices = linspace_int(start, end, DEFAULT_NUM_SAMPLED_FRAMES)
            sampled_images = []
            for idx in sampled_indices:
                image_rel = samples[idx].get("image", "")
                if image_rel:
                    sampled_images.append(pil_from_image_path(subset_dir / image_rel))
            instruction = extract_instruction(samples[start].get("conversations", []))
            if sampled_images:
                clips.append(
                    ClipInfo(
                        dataset="being-h0",
                        source_id=subset_dir.name,
                        clip_id=clip_id,
                        start_frame=start,
                        end_frame=end,
                        sampled_frame_indices=sampled_indices,
                        sampled_images=sampled_images,
                        prompt_context={"subset_name": subset_dir.name, "instruction": instruction},
                        output_dir=output_dir,
                    )
                )
                clip_id += 1
    return clips


def read_h5_any(ds):
    try:
        if getattr(ds, "shape", None) == ():
            return ds[()]
        return ds[:]
    except Exception:
        return ds[()]


def sorted_image_keys(group: h5py.Group) -> list[str]:
    return sorted(group.keys(), key=lambda x: int(x.split(".")[0]))


def build_dexwild_jobs(root: Path) -> List[ClipInfo]:
    hdf5_files = sorted(root.glob("*.hdf5")) + sorted(root.glob("*.h5")) if root.is_dir() else [root]
    if not hdf5_files:
        return []
    output_dir = ensure_annotations_dir(root if root.is_dir() else root.parent)
    clips: List[ClipInfo] = []
    with h5py.File(hdf5_files[0], "r") as f:
        episode_names = sorted(f.keys())
        for episode_name in episode_names[:1]:
            ep = f[episode_name]
            thumb_group = ep["right_thumb_cam"]
            pinky_group = ep["right_pinky_cam"]
            thumb_keys = sorted_image_keys(thumb_group)
            pinky_keys = sorted_image_keys(pinky_group)
            eef = np.asarray(read_h5_any(ep["right_arm_eef"]["right_arm_eef"]))
            n_frames = min(len(thumb_keys), len(pinky_keys), len(eef))
            clip_id = 0
            for start in range(0, n_frames, 16):
                end = min(start + 15, n_frames - 1)
                sampled_indices = linspace_int(start, end, DEFAULT_NUM_SAMPLED_FRAMES)
                sampled_images = []
                for idx in sampled_indices:
                    sampled_images.append(pil_from_rgb_array(np.asarray(thumb_group[thumb_keys[idx]][:])))
                    sampled_images.append(pil_from_rgb_array(np.asarray(pinky_group[pinky_keys[idx]][:])))
                xyz = eef[start, 1:4].tolist() if len(eef) > start else []
                clips.append(
                    ClipInfo(
                        dataset="dexwild",
                        source_id=episode_name,
                        clip_id=clip_id,
                        start_frame=start,
                        end_frame=end,
                        sampled_frame_indices=sampled_indices,
                        sampled_images=sampled_images,
                        prompt_context={"episode": episode_name, "eef_xyz_start": xyz},
                        output_dir=output_dir,
                    )
                )
                clip_id += 1
    return clips


def build_thermohands_jobs(root: Path) -> List[ClipInfo]:
    scene_dirs = [root] if (root / "rgb").is_dir() else sorted([p for p in root.iterdir() if p.is_dir() and (p / "rgb").is_dir()])
    output_dir = ensure_annotations_dir(root if root.name == "thermohands" else root.parent)
    clips: List[ClipInfo] = []
    for scene_dir in scene_dirs[:1]:
        rgb_files = sorted((scene_dir / "rgb").glob("*.png"))
        thermal_files = sorted((scene_dir / "thermal").glob("*.png"))
        ir_files = sorted((scene_dir / "ir").glob("*.png"))
        depth_files = sorted((scene_dir / "depth").glob("*.png"))
        gt_files = sorted((scene_dir / "gt_info").glob("*.json"))
        n_frames = min(len(rgb_files), len(thermal_files), len(ir_files), len(depth_files), len(gt_files))
        clip_id = 0
        for start in range(0, n_frames, 16):
            end = min(start + 15, n_frames - 1)
            sampled_indices = linspace_int(start, end, DEFAULT_NUM_SAMPLED_FRAMES)
            sampled_images = []
            for idx in sampled_indices:
                sampled_images.append(pil_from_image_path(rgb_files[idx]))
                sampled_images.append(pil_from_image_path(thermal_files[idx]))
                sampled_images.append(pil_from_image_path(ir_files[idx]))
                sampled_images.append(pil_from_image_path(depth_files[idx]))
            gt = read_json(gt_files[start])
            clips.append(
                ClipInfo(
                    dataset="thermohands",
                    source_id=scene_dir.name,
                    clip_id=clip_id,
                    start_frame=start,
                    end_frame=end,
                    sampled_frame_indices=sampled_indices,
                    sampled_images=sampled_images,
                    prompt_context={
                        "scene_name": scene_dir.name,
                        "left_hand_visible": "kps3D_L" in gt,
                        "right_hand_visible": "kps3D_R" in gt,
                    },
                    output_dir=output_dir,
                )
            )
            clip_id += 1
    return clips


def build_wiyh_jobs(root: Path) -> List[ClipInfo]:
    action_dirs = [root] if (root / "dataset.hdf5").exists() else sorted([p for p in root.iterdir() if p.is_dir() and (p / "dataset.hdf5").exists()])
    task_json = root / "task.json" if root.is_dir() else root.parent / "task.json"
    output_dir = ensure_annotations_dir(root if root.name == "wyih" else root.parent)
    clips: List[ClipInfo] = []
    for action_dir in action_dirs[:1]:
        task_entry = read_json(task_json) if task_json.exists() else []
        task_description = ""
        with h5py.File(action_dir / "dataset.hdf5", "r") as f:
            task_description = str(f["meta/task_description"][()].decode() if isinstance(f["meta/task_description"][()], bytes) else f["meta/task_description"][()])
            camera_streams = {
                name: read_h5_any(f[f"observation/camera/{name}/filepath"])
                for name in ["lf_chest_fisheye", "rf_chest_fisheye", "ldr_hand_fisheye", "rdr_hand_fisheye"]
            }
            total_frames = min(len(stream) for stream in camera_streams.values())
        clip_id = 0
        for start in range(0, total_frames, 12):
            end = min(start + 11, total_frames - 1)
            sampled_indices = linspace_int(start, end, DEFAULT_NUM_SAMPLED_FRAMES)
            sampled_images = []
            for idx in sampled_indices:
                for camera_name in camera_streams:
                    rel_path = camera_streams[camera_name][idx]
                    rel_path = rel_path.decode() if isinstance(rel_path, bytes) else str(rel_path)
                    sampled_images.append(pil_from_image_path(action_dir / Path(rel_path)))
            clips.append(
                ClipInfo(
                    dataset="wiyh",
                    source_id=action_dir.name,
                    clip_id=clip_id,
                    start_frame=start,
                    end_frame=end,
                    sampled_frame_indices=sampled_indices,
                    sampled_images=sampled_images,
                    prompt_context={"task_description": task_description, "action_name": action_dir.name},
                    output_dir=output_dir,
                )
            )
            clip_id += 1
    return clips


def build_hot3d_jobs(root: Path) -> List[ClipInfo]:
    hot3d_root = root / "hot3d_demo_full" if (root / "hot3d_demo_full").is_dir() else root
    seq_dirs = sorted([p for p in hot3d_root.iterdir() if p.is_dir() and (p / "ground_truth").is_dir() and (p / "hand_data").is_dir()])
    output_dir = ensure_annotations_dir(root if root.name == "HOT3D" else root.parent)
    clips: List[ClipInfo] = []
    for seq_dir in seq_dirs[:1]:
        metadata_path = seq_dir / "ground_truth" / "metadata.json"
        dynamic_path = seq_dir / "ground_truth" / "dynamic_objects.csv"
        hand_path = seq_dir / "hand_data" / "mano_hand_pose_trajectory.jsonl"
        metadata = read_json(metadata_path)
        dynamic_rows = load_csv(dynamic_path)
        hand_rows = load_jsonl(hand_path)
        timestamps = sorted({int(row["timestamp[ns]"]) for row in dynamic_rows})
        ts_to_objects: Dict[int, List[str]] = {}
        for row in dynamic_rows:
            ts = int(row["timestamp[ns]"])
            ts_to_objects.setdefault(ts, []).append(row["object_uid"])
        ts_to_hands = {int(row.get("timestamp_ns", row.get("timestamp[ns]", -1))): row for row in hand_rows if row.get("timestamp_ns", row.get("timestamp[ns]", None)) is not None}
        clip_id = 0
        for start in range(0, len(timestamps), 20):
            end = min(start + 19, len(timestamps) - 1)
            sampled_indices = linspace_int(start, end, min(DEFAULT_NUM_SAMPLED_FRAMES, end - start + 1))
            sampled_images: List[Image.Image] = []
            prompt_context = {
                "sequence_name": seq_dir.name,
                "recording_name": metadata.get("recording_name", "unknown"),
                "participant_id": metadata.get("participant_id", "unknown"),
                "objects_seen": sorted(set(obj for idx in sampled_indices for obj in ts_to_objects.get(timestamps[idx], [])))[:10],
                "hand_rows_present": sum(1 for idx in sampled_indices if timestamps[idx] in ts_to_hands),
            }
            clips.append(
                ClipInfo(
                    dataset="hot3d",
                    source_id=seq_dir.name,
                    clip_id=clip_id,
                    start_frame=start,
                    end_frame=end,
                    sampled_frame_indices=sampled_indices,
                    sampled_images=sampled_images,
                    prompt_context=prompt_context,
                    output_dir=output_dir,
                )
            )
            clip_id += 1
    return clips


def build_jobs(input_path: Path, dataset: str) -> List[ClipInfo]:
    input_path = input_path.resolve()
    if input_path == DATA_ROOT:
        jobs: List[ClipInfo] = []
        for child in sorted([p for p in input_path.iterdir() if p.is_dir()]):
            ds = detect_dataset(child, "auto")
            jobs.extend(build_jobs(child, ds))
        return jobs

    dataset = detect_dataset(input_path, dataset)
    if dataset == "gigahands":
        return build_gigahands_jobs(input_path)
    if dataset == "being-h0":
        return build_beingh0_jobs(input_path)
    if dataset == "dexwild":
        return build_dexwild_jobs(input_path)
    if dataset == "thermohands":
        return build_thermohands_jobs(input_path)
    if dataset == "wiyh":
        return build_wiyh_jobs(input_path)
    if dataset == "hot3d":
        return build_hot3d_jobs(input_path)
    raise ValueError(f"Unsupported dataset: {dataset}")


def group_jobs_by_source(clips: Iterable[ClipInfo]) -> Dict[tuple[str, str, Path], List[ClipInfo]]:
    grouped: Dict[tuple[str, str, Path], List[ClipInfo]] = {}
    for clip in clips:
        key = (clip.dataset, clip.source_id, clip.output_dir)
        grouped.setdefault(key, []).append(clip)
    return grouped


def save_outputs(grouped_preds: Dict[tuple[str, str, Path], List[Dict[str, Any]]]):
    for (dataset, source_id, output_dir), raw_clip_preds in grouped_preds.items():
        raw_path = output_dir / f"pred_raw_clips_{source_id}.json"
        steps_path = output_dir / f"pred_steps_{source_id}.json"
        merged_steps = merge_clips(raw_clip_preds)
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(raw_clip_preds, f, ensure_ascii=False, indent=2)
        with open(steps_path, "w", encoding="utf-8") as f:
            json.dump(merged_steps, f, ensure_ascii=False, indent=2)
        print(f"[{dataset}] Saved raw clips -> {raw_path}")
        print(f"[{dataset}] Saved merged steps -> {steps_path}")


def main():
    args = parse_args()
    start_time = time.time()
    clips = build_jobs(args.input, args.dataset)
    if args.max_jobs > 0:
        grouped = list(group_jobs_by_source(clips).values())[: args.max_jobs]
        clips = [clip for group in grouped for clip in group]
    if args.max_clips > 0:
        clips = clips[: args.max_clips]

    grouped = group_jobs_by_source(clips)
    print("Detected semantic jobs:")
    for (dataset, source_id, output_dir), items in grouped.items():
        print(f"  - dataset={dataset}, source_id={source_id}, clips={len(items)}, output_dir={output_dir}")

    if args.list_jobs:
        return

    if not clips:
        raise RuntimeError("No clips found to process.")

    processor, model = load_model()
    raw_preds = run_inference(clips, processor, model)

    preds_by_key: Dict[tuple[str, str, Path], List[Dict[str, Any]]] = {}
    for item in raw_preds:
        key = (item["dataset"], item["source_id"], next(clip.output_dir for clip in clips if clip.dataset == item["dataset"] and clip.source_id == item["source_id"]))
        preds_by_key.setdefault(key, []).append(item)

    save_outputs(preds_by_key)
    print(f"Total inference time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
