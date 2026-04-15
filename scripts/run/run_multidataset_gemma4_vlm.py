from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import torch
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "Missing dependency `torch`. Install the VLM extras first, or run this script with "
        "`uv run --with torch --with torchvision --with transformers python scripts/run/run_multidataset_gemma4_vlm.py`."
    ) from exc

try:
    from transformers import AutoProcessor
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "Missing dependency `transformers`. Install the VLM extras first, or run this script with "
        "`uv run --with torch --with torchvision --with transformers python scripts/run/run_multidataset_gemma4_vlm.py`."
    ) from exc

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


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_multidataset_vlm as base
import universal_label_library as label_library_lib


MODEL_ID = os.environ.get("MULTIDATASET_GEMMA4_MODEL_ID", "google/gemma-4-E4B-it")
MAX_NEW_TOKENS = int(os.environ.get("MULTIDATASET_GEMMA4_MAX_NEW_TOKENS", str(base.MAX_NEW_TOKENS)))
BATCH_SIZE = int(os.environ.get("MULTIDATASET_GEMMA4_BATCH_SIZE", "1"))
USE_SDPA = os.environ.get("MULTIDATASET_GEMMA4_USE_SDPA", "1") != "0"
DEFAULT_OUTPUT_TAG = os.environ.get("MULTIDATASET_GEMMA4_OUTPUT_TAG", "gemma4").strip()
RUNNER_NAME = "run_multidataset_gemma4_vlm"


def get_torch_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_model():
    print(f"Loading Gemma model: {MODEL_ID}")
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu name:", torch.cuda.get_device_name(0))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    kwargs: Dict[str, object] = {
        "torch_dtype": get_torch_dtype(),
        "trust_remote_code": True,
    }
    if USE_SDPA:
        kwargs["attn_implementation"] = "sdpa"

    model = GemmaAutoModel.from_pretrained(MODEL_ID, **kwargs)
    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, padding_side="left", trust_remote_code=True)
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"

    print("model device:", next(model.parameters()).device)
    return processor, model


def build_messages(clip: base.ClipInfo):
    content = []
    for image in clip.sampled_images:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": base.make_prompt(clip)})
    return [{"role": "user", "content": content}]


def run_inference(
    clips: List[base.ClipInfo],
    processor,
    model,
    *,
    label_library: Optional[dict[str, Any]],
) -> List[Dict[str, Any]]:
    raw_clip_preds: List[Dict[str, Any]] = []
    label_vocab = (
        set(label_library_lib.get_canonical_labels(label_library))
        if label_library
        else set(base.CANONICAL_LABELS)
    )
    valid_clips = [clip for clip in clips if clip.sampled_images]
    skipped_clips = len(clips) - len(valid_clips)
    if skipped_clips:
        print(f"Skipping {skipped_clips} clip(s) with no sampled images.")

    effective_batch_size = max(1, BATCH_SIZE)
    total_batches = (len(valid_clips) + effective_batch_size - 1) // effective_batch_size
    for batch_index in range(total_batches):
        batch = valid_clips[batch_index * effective_batch_size : (batch_index + 1) * effective_batch_size]
        batch_start = batch_index * effective_batch_size + 1
        batch_end = batch_index * effective_batch_size + len(batch)
        print(f"Running batch {batch_start}-{batch_end}/{len(valid_clips)}")
        t0 = time.time()

        for clip in batch:
            clip.prompt_context["label_library"] = label_library
            messages = build_messages(clip)
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
                    do_sample=False,
                    use_cache=True,
                )

            prompt_len = inputs["input_ids"].shape[-1]
            trimmed_ids = generated_ids[:, prompt_len:]
            raw_text = processor.batch_decode(
                trimmed_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            parsed = base.parse_response(
                raw_text,
                base.choose_overall_task(clip.dataset, clip.prompt_context),
                label_library=label_library,
            )
            item = {
                "dataset": clip.dataset,
                "source_id": clip.source_id,
                "clip_id": clip.clip_id,
                "start_frame": clip.start_frame,
                "end_frame": clip.end_frame,
                "sampled_frames": clip.sampled_frame_indices,
                "main_task": parsed["main_task"],
                "sub_task": parsed["sub_task"],
                "interaction": parsed["interaction"],
                "objects": parsed["objects"],
                "label": parsed["label"],
                "raw_label": parsed.get("raw_label", ""),
                "label_match_kind": parsed.get("label_match_kind"),
                "label_match_score": parsed.get("label_match_score"),
                "label_id": parsed.get("label_id"),
                "current_action": parsed["current_action"],
                "parse_ok": parsed["parse_ok"],
                "model_id": MODEL_ID,
                "raw_response": raw_text,
            }
            item["validity"] = base.evaluate_validity(item, label_vocab=label_vocab)
            raw_clip_preds.append(item)
            print(
                f"  clip {clip.clip_id:04d} | "
                f"frames {clip.start_frame}-{clip.end_frame} | "
                f"label={item['label']} | parse_ok={item['parse_ok']} | validity={item['validity']['validity_score']:.2f}"
            )
            print(f"    sub_task={item['sub_task']}")
            print(f"    interaction={item['interaction']}")
            print(f"    objects={item['objects']}")
            print(f"    current_action={item['current_action']}")
            if item["validity"]["warnings"]:
                print(f"    warnings={item['validity']['warnings']}")

        dt = time.time() - t0
        print(f"  batch_time = {dt:.2f}s")
        print(f"  avg_per_clip = {dt / len(batch):.2f}s")
        print()

    return raw_clip_preds


def save_outputs(
    grouped_preds: Dict[tuple[str, str, Path], List[Dict[str, Any]]],
    *,
    label_library: Optional[dict[str, Any]],
    output_tag: str,
):
    for (dataset, source_id, output_dir), raw_clip_preds in grouped_preds.items():
        raw_path = base.tagged_output_path(output_dir, f"pred_raw_clips_{source_id}", output_tag)
        steps_path = base.tagged_output_path(output_dir, f"pred_steps_{source_id}", output_tag)
        eval_path = base.tagged_output_path(output_dir, f"pred_eval_{source_id}", output_tag)
        merged_steps = base.merge_clips(raw_clip_preds)
        validity_summary = base.summarize_validity(raw_clip_preds)
        for item in raw_clip_preds:
            label_library_lib.record_observation(
                label_library if label_library is not None else label_library_lib.default_label_library(),
                raw_label=item.get("raw_label", ""),
                canonical_label=item.get("label", "other"),
                run_name=RUNNER_NAME,
                clip_id=item.get("clip_id"),
                scene_name=item.get("source_id", source_id),
                raw_response=item.get("raw_response", ""),
                sub_task=item.get("sub_task", ""),
                interaction=item.get("interaction", ""),
                current_action=item.get("current_action", ""),
                match_kind=item.get("label_match_kind"),
                auto_append_new_labels=base.LABEL_LIBRARY_AUTO_APPEND,
            )
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(raw_clip_preds, f, ensure_ascii=False, indent=2)
        with open(steps_path, "w", encoding="utf-8") as f:
            json.dump(merged_steps, f, ensure_ascii=False, indent=2)
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset": dataset,
                    "source_id": source_id,
                    "model_id": MODEL_ID,
                    "output_tag": output_tag or None,
                    "label_library_path": str(base.LABEL_LIBRARY_PATH),
                    "label_library_summary": label_library_lib.summarize_library(label_library) if label_library is not None else None,
                    "summary": validity_summary,
                    "clips": [
                        {
                            "clip_id": item["clip_id"],
                            "start_frame": item["start_frame"],
                            "end_frame": item["end_frame"],
                            "label": item["label"],
                            "validity_score": item["validity"]["validity_score"],
                            "warnings": item["validity"]["warnings"],
                            "checks": item["validity"]["checks"],
                        }
                        for item in raw_clip_preds
                    ],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[{dataset}] Saved raw clips -> {raw_path}")
        print(f"[{dataset}] Saved merged steps -> {steps_path}")
        print(
            f"[{dataset}] Validity summary -> {eval_path} | "
            f"avg_validity={validity_summary['avg_validity_score']:.2f} | "
            f"parse_rate={validity_summary['rates'].get('parse_ok', 0.0):.2f}"
        )


def main():
    args = base.parse_args()
    start_time = time.time()
    payload = base.load_config_payload(args.config)
    dataset = base.normalize_dataset_name(payload.get("dataset", args.dataset))
    input_path = base.resolve_input_path(payload.get("input", args.input))
    selection = dict(payload.get("selection", {})) if isinstance(payload.get("selection", {}), dict) else {}
    dataset_options = dict(payload.get("dataset_options", {})) if isinstance(payload.get("dataset_options", {}), dict) else {}

    source_filter = args.source_id.strip()
    if not source_filter:
        source_filter = (
            str(selection.get("seq_name") or selection.get("sequence_name") or "")
            or Path(str(dataset_options.get("thermohands_scene_dir", "") or "")).name
            or str(dataset_options.get("dexwild_episode", "") or "")
            or Path(str(dataset_options.get("action_dir", "") or "")).name
            or Path(str(dataset_options.get("beingh0_subset_dir", "") or "")).name
        )

    output_tag = args.output_tag.strip() or DEFAULT_OUTPUT_TAG
    label_library = label_library_lib.load_label_library(base.LABEL_LIBRARY_PATH)
    clips = base.build_jobs(
        input_path,
        dataset,
        source_filter=source_filter,
        num_sampled_frames=max(1, args.num_sampled_frames),
        output_tag=output_tag,
    )
    if args.max_jobs > 0:
        grouped = list(base.group_jobs_by_source(clips).values())[: args.max_jobs]
        clips = [clip for group in grouped for clip in group]
    if args.max_clips > 0:
        clips = clips[: args.max_clips]

    grouped = base.group_jobs_by_source(clips)
    print("Detected semantic jobs:")
    for (dataset_name, source_id, output_dir), items in grouped.items():
        print(f"  - dataset={dataset_name}, source_id={source_id}, clips={len(items)}, output_dir={output_dir}")
    print("MODEL_ID           =", MODEL_ID)
    print("BATCH_SIZE         =", max(1, BATCH_SIZE))
    print("MAX_NEW_TOKENS     =", MAX_NEW_TOKENS)
    print("OUTPUT_TAG         =", output_tag)
    print("LABEL_LIBRARY      =", base.LABEL_LIBRARY_PATH)
    print("LABEL_LIBRARY_ENTRIES =", label_library_lib.library_size(label_library))

    if args.list_jobs:
        return
    if not clips:
        raise RuntimeError("No clips found to process.")

    processor, model = load_model()
    raw_preds = run_inference(clips, processor, model, label_library=label_library)

    preds_by_key: Dict[tuple[str, str, Path], List[Dict[str, Any]]] = {}
    for item in raw_preds:
        key = (
            item["dataset"],
            item["source_id"],
            next(
                clip.output_dir
                for clip in clips
                if clip.dataset == item["dataset"] and clip.source_id == item["source_id"]
            ),
        )
        preds_by_key.setdefault(key, []).append(item)

    save_outputs(preds_by_key, label_library=label_library, output_tag=output_tag)
    if base.LABEL_LIBRARY_AUTO_APPEND:
        label_library_lib.save_label_library(label_library, base.LABEL_LIBRARY_PATH)
    print(f"Total inference time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
