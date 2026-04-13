import argparse
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import rerun as rr


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Being-H0 subset with Rerun")
    parser.add_argument(
        "--subset-dir",
        type=str,
        default=".",
        help="Path to one subset folder. Default is current folder.",
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        default=None,
        help="Optional explicit path to *_train.jsonl. If omitted, auto-detect inside subset-dir.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start sample index",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Maximum number of samples to show. -1 means show all.",
    )
    parser.add_argument(
        "--step-sleep",
        type=float,
        default=0.08,
        help="Seconds to wait between samples",
    )
    parser.add_argument(
        "--spawn",
        action="store_true",
        help="Spawn the Rerun viewer automatically",
    )
    return parser.parse_args()


def find_jsonl(subset_dir: Path, explicit_jsonl: str | None) -> Path:
    if explicit_jsonl is not None:
        jsonl_path = Path(explicit_jsonl)
        if not jsonl_path.exists():
            raise FileNotFoundError(f"JSONL not found: {jsonl_path}")
        return jsonl_path

    matches = sorted(subset_dir.glob("*_train.jsonl"))
    if not matches:
        raise FileNotFoundError(f"No *_train.jsonl found in: {subset_dir}")
    return matches[0]


def load_jsonl(jsonl_path: Path):
    samples = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_idx} in {jsonl_path}: {e}") from e
    return samples


def clean_instruction(text: str) -> str:
    if not text:
        return ""

    text = text.replace("<image>", "")
    text = text.replace("<PROP_CONTEXT>", "")
    text = " ".join(text.split()).strip()

    m = re.search(r"instruction\s+'([^']+)'", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    return text


def extract_instruction(conversations) -> str:
    if not isinstance(conversations, list):
        return ""

    for item in conversations:
        if isinstance(item, dict) and item.get("from", "").lower() == "human":
            return clean_instruction(item.get("value", ""))

    if len(conversations) > 0 and isinstance(conversations[0], dict):
        return clean_instruction(conversations[0].get("value", ""))

    return ""


def parse_sample_id(sample_id: str):
    ep_match = re.search(r"ep(\d+)", sample_id or "")
    frame_match = re.search(r"frame(\d+)", sample_id or "")

    episode_idx = int(ep_match.group(1)) if ep_match else 0
    frame_idx = int(frame_match.group(1)) if frame_match else 0
    return episode_idx, frame_idx


def read_image_rgb(image_path: Path):
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def log_text(path: str, text: str):
    text = text if text else ""
    if hasattr(rr, "TextDocument"):
        try:
            rr.log(path, rr.TextDocument(text))
            return
        except Exception:
            pass
    if hasattr(rr, "TextLog"):
        rr.log(path, rr.TextLog(text))


def set_time_seq(timeline: str, value: int):
    if hasattr(rr, "set_time_sequence"):
        rr.set_time_sequence(timeline, value)
    elif hasattr(rr, "set_time"):
        rr.set_time(timeline, sequence=value)


def make_scalar(value: float):
    """
    Compatible scalar logging across rerun versions.
    """
    if hasattr(rr, "Scalar"):
        return rr.Scalar(float(value))
    if hasattr(rr, "Scalars"):
        return rr.Scalars(np.array([float(value)], dtype=np.float32))
    raise AttributeError("Your rerun version has neither Scalar nor Scalars.")


def log_vector(base_path: str, vec: np.ndarray):
    for i, value in enumerate(vec):
        rr.log(f"{base_path}/dim_{i:02d}", make_scalar(float(value)))


def log_action_chunk(action_chunk: np.ndarray):
    """
    Logs action chunk as time series on timeline 'future_step'.
    Expected shape: [T, D], commonly [16, 13].
    """
    if action_chunk.ndim != 2:
        return

    num_steps, num_dims = action_chunk.shape

    for t in range(num_steps):
        set_time_seq("future_step", t)
        for d in range(num_dims):
            rr.log(f"future_action/dim_{d:02d}", make_scalar(float(action_chunk[t, d])))

    set_time_seq("future_step", 0)


def send_blueprint():
    try:
        if hasattr(rr, "blueprint") and hasattr(rr, "send_blueprint"):
            rr.send_blueprint(
                rr.blueprint.Blueprint(
                    rr.blueprint.Horizontal(
                        rr.blueprint.Spatial2DView(origin="/world"),
                        rr.blueprint.Vertical(
                            rr.blueprint.TextDocumentView(origin="/sample"),
                            rr.blueprint.TimeSeriesView(origin="/state"),
                            rr.blueprint.TimeSeriesView(origin="/future_action"),
                        ),
                    )
                )
            )
    except Exception:
        pass


def main():
    args = parse_args()

    subset_dir = Path(args.subset_dir)
    if not subset_dir.exists():
        raise FileNotFoundError(f"Subset directory not found: {subset_dir}")

    jsonl_path = find_jsonl(subset_dir, args.jsonl)
    samples = load_jsonl(jsonl_path)

    if len(samples) == 0:
        raise RuntimeError(f"No samples found in {jsonl_path}")

    rr.init("being_h0_preview", spawn=args.spawn)
    send_blueprint()

    total = len(samples)
    start_idx = max(0, args.start)
    end_idx = total if args.max_samples < 0 else min(total, start_idx + args.max_samples)

    print(f"Subset dir : {subset_dir}")
    print(f"JSONL      : {jsonl_path}")
    print(f"Total      : {total}")
    print(f"Showing    : [{start_idx}, {end_idx})")

    if hasattr(rr, "ViewCoordinates"):
        try:
            rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        except Exception:
            pass

    for sample_idx in range(start_idx, end_idx):
        sample = samples[sample_idx]

        sample_id = sample.get("id", f"sample_{sample_idx:06d}")
        dataset_name = sample.get("dataset_name", subset_dir.name)
        image_rel = sample.get("image", "")
        instruction = extract_instruction(sample.get("conversations", []))

        proprio = np.asarray(sample.get("proprioception", []), dtype=np.float32)
        action_chunk = np.asarray(sample.get("action_chunk", []), dtype=np.float32)

        episode_idx, frame_idx = parse_sample_id(sample_id)

        set_time_seq("sample_idx", sample_idx)
        set_time_seq("episode", episode_idx)
        set_time_seq("frame", frame_idx)

        log_text("sample/id", sample_id)
        log_text("sample/dataset_name", dataset_name)
        log_text("sample/instruction", instruction)
        log_text(
            "sample/info",
            "\n".join(
                [
                    f"sample_idx: {sample_idx}",
                    f"episode: {episode_idx}",
                    f"frame: {frame_idx}",
                    f"image: {image_rel}",
                    f"proprio_shape: {tuple(proprio.shape)}",
                    f"action_chunk_shape: {tuple(action_chunk.shape)}",
                ]
            ),
        )

        if image_rel:
            image_path = subset_dir / image_rel
            try:
                img_rgb = read_image_rgb(image_path)
                rr.log("world/rgb", rr.Image(img_rgb))
            except Exception as e:
                log_text("sample/image_error", str(e))

        if proprio.ndim == 1 and proprio.size > 0:
            log_vector("state/proprioception", proprio)
        else:
            log_text("sample/proprio_warning", f"Unexpected proprio shape: {tuple(proprio.shape)}")

        if action_chunk.ndim == 2 and action_chunk.size > 0:
            log_action_chunk(action_chunk)
        else:
            log_text("sample/action_warning", f"Unexpected action_chunk shape: {tuple(action_chunk.shape)}")

        time.sleep(args.step_sleep)

    print("Done.")


if __name__ == "__main__":
    main()
