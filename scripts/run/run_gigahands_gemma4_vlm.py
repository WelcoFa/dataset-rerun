import json
import os
import time
from typing import Dict, List, Optional, Tuple

try:
    import torch
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "Missing dependency `torch`. Install the VLM extras first with "
        "`uv sync --extra gigahands-vlm`, or run this script with "
        "`uv run --extra gigahands-vlm python scripts/run/run_gigahands_gemma4_vlm.py`."
    ) from exc

try:
    from transformers import AutoProcessor
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "Missing dependency `transformers`. Install the VLM extras first with "
        "`uv sync --extra gigahands-vlm`, or run this script with "
        "`uv run --extra gigahands-vlm python scripts/run/run_gigahands_gemma4_vlm.py`."
    ) from exc

import run_gigahands_vlm as base

try:
    from transformers import AutoModelForImageTextToText as GemmaAutoModel
except ImportError:
    try:
        from transformers import AutoModelForMultimodalLM as GemmaAutoModel
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "Gemma 4 multimodal inference requires a newer transformers build with "
            "`AutoModelForImageTextToText` or `AutoModelForMultimodalLM`."
        ) from exc


MODEL_ID = os.environ.get("GIGAHANDS_GEMMA4_MODEL_ID", "google/gemma-4-E4B-it")
OUTPUT_TAG = os.environ.get("GIGAHANDS_GEMMA4_OUTPUT_TAG", "gemma4")
OUTPUT_STEPS_PATH = base.OUTPUT_DIR / f"pred_steps_{base.SCENE_NAME}_{OUTPUT_TAG}.json"
OUTPUT_RAW_CLIPS_PATH = base.OUTPUT_DIR / f"pred_raw_clips_{base.SCENE_NAME}_{OUTPUT_TAG}.json"
OUTPUT_RUN_META_PATH = base.OUTPUT_DIR / f"pred_run_meta_{base.SCENE_NAME}_{OUTPUT_TAG}.json"
BATCH_SIZE = int(os.environ.get("GIGAHANDS_GEMMA4_BATCH_SIZE", "1"))
MAX_NEW_TOKENS = int(os.environ.get("GIGAHANDS_GEMMA4_MAX_NEW_TOKENS", str(base.MAX_NEW_TOKENS)))
USE_SDPA = os.environ.get("GIGAHANDS_GEMMA4_USE_SDPA", "1") != "0"


def get_torch_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_model_and_processor(model_id: str) -> Tuple[torch.nn.Module, AutoProcessor]:
    print(f"Loading Gemma model: {model_id}")
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu name:", torch.cuda.get_device_name(0))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    kwargs: Dict[str, object] = {
        "torch_dtype": get_torch_dtype(),
        "trust_remote_code": True,
    }
    if USE_SDPA:
        kwargs["attn_implementation"] = "sdpa"

    model = GemmaAutoModel.from_pretrained(model_id, **kwargs)
    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    processor = AutoProcessor.from_pretrained(model_id, padding_side="left", trust_remote_code=True)
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"

    print("model device:", next(model.parameters()).device)
    return model, processor


def build_messages_for_clip(clip: base.ClipInfo, scene_name: str) -> List[Dict[str, object]]:
    prompt = base.make_prompt(scene_name, clip.start_frame, clip.end_frame)
    content: List[Dict[str, object]] = []
    for img in clip.sampled_images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def run_gemma_batch(
    model: torch.nn.Module,
    processor: AutoProcessor,
    batch_clips: List[base.ClipInfo],
    scene_name: str,
    scene_registry: Optional[List[Dict[str, object]]] = None,
) -> List[Tuple[str, Dict[str, object]]]:
    outputs: List[Tuple[str, Dict[str, object]]] = []

    for clip in batch_clips:
        messages = build_messages_for_clip(clip, scene_name)
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=base.DO_SAMPLE,
                temperature=base.TEMPERATURE if base.DO_SAMPLE else None,
                top_p=base.TOP_P if base.DO_SAMPLE else None,
                use_cache=True,
            )

        prompt_len = inputs["input_ids"].shape[-1]
        trimmed_ids = generated_ids[:, prompt_len:]
        decoded = processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        parsed = base.parse_model_json(
            decoded,
            scene_name=scene_name,
            scene_registry=scene_registry,
        )
        outputs.append((decoded, parsed))

    return outputs


def main() -> None:
    effective_batch_size = max(1, BATCH_SIZE)

    print("=== run_gigahands_gemma4_vlm.py ===")
    print("SCENE_NAME         =", base.SCENE_NAME)
    print("VIDEO_PATH         =", base.VIDEO_PATH)
    print("MODEL_ID           =", MODEL_ID)
    print("OUTPUT_STEPS       =", OUTPUT_STEPS_PATH)
    print("OUTPUT_RAW_CLIPS   =", OUTPUT_RAW_CLIPS_PATH)
    print("CLIP_LEN_FRAMES    =", base.CLIP_LEN_FRAMES)
    print("CLIP_STRIDE_FRAMES =", base.CLIP_STRIDE_FRAMES)
    print("NUM_SAMPLED_FRAMES =", base.NUM_SAMPLED_FRAMES)
    print("BATCH_SIZE         =", effective_batch_size)
    print("MAX_NEW_TOKENS     =", MAX_NEW_TOKENS)
    print("MAX_IMAGE_EDGE     =", base.MAX_IMAGE_EDGE)
    print()

    if not base.VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {base.VIDEO_PATH}")

    clips, fps, total_frames = base.sample_video_clips_in_memory(
        video_path=base.VIDEO_PATH,
        clip_len_frames=base.CLIP_LEN_FRAMES,
        clip_stride_frames=base.CLIP_STRIDE_FRAMES,
        num_sampled_frames=base.NUM_SAMPLED_FRAMES,
        max_clips=base.MAX_CLIPS,
    )

    print(f"Total frames : {total_frames}")
    print(f"FPS          : {fps:.2f}")
    print(f"Num clips    : {len(clips)}")
    print()

    scene_registry = base.build_scene_object_registry(base.SCENE_NAME)
    print("Scene objects =", base.get_scene_object_labels(scene_registry))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model, processor = load_model_and_processor(MODEL_ID)

    raw_clip_preds: List[Dict[str, object]] = []
    total_start = time.time()

    for batch_start in range(0, len(clips), effective_batch_size):
        batch = clips[batch_start : batch_start + effective_batch_size]
        batch_end = min(batch_start + effective_batch_size, len(clips))
        print(f"Running batch {batch_start + 1}-{batch_end}/{len(clips)}")

        t0 = time.time()
        outputs = run_gemma_batch(
            model=model,
            processor=processor,
            batch_clips=batch,
            scene_name=base.SCENE_NAME,
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
                "model_id": MODEL_ID,
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

    merged_steps = base.merge_clip_predictions(raw_clip_preds)

    with open(OUTPUT_RAW_CLIPS_PATH, "w", encoding="utf-8") as f:
        json.dump(raw_clip_preds, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_STEPS_PATH, "w", encoding="utf-8") as f:
        json.dump(merged_steps, f, ensure_ascii=False, indent=2)

    total_dt = time.time() - total_start
    run_meta = base.save_run_meta(
        model_id=MODEL_ID,
        batch_size=effective_batch_size,
        total_time_seconds=total_dt,
        total_frames=total_frames,
        fps=fps,
        num_clips=len(clips),
        output_path=OUTPUT_RUN_META_PATH,
        raw_output_path=OUTPUT_RAW_CLIPS_PATH,
        steps_output_path=OUTPUT_STEPS_PATH,
        runner_name="run_gigahands_gemma4_vlm",
    )

    print("Saved raw clip predictions to:")
    print(" ", OUTPUT_RAW_CLIPS_PATH)
    print("Saved merged step predictions to:")
    print(" ", OUTPUT_STEPS_PATH)
    print("Saved run metadata to:")
    print(" ", OUTPUT_RUN_META_PATH)
    print()
    print(f"Total inference time: {total_dt:.2f}s")
    if run_meta["peak_vram_allocated_gib"] is not None:
        print(f"Peak VRAM allocated: {run_meta['peak_vram_allocated_gib']:.2f} GiB")
        print(f"Peak VRAM reserved : {run_meta['peak_vram_reserved_gib']:.2f} GiB")
    print()

    print("Merged steps preview:")
    for i, step in enumerate(merged_steps, 1):
        print(
            f"  Step {i}: [{step['start']}, {step['end']}] "
            f"{step['label']} | {step['current_action']}"
        )


if __name__ == "__main__":
    main()
