from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANNOTATIONS_DIR = REPO_ROOT / "data" / "gigahands" / "annotations"
CANONICAL_LABELS = {
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
}
PLACEHOLDER_TEXTS = {"", "other", "unknown", "n/a", "none", "null"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two GigaHands VLM runs without ground truth.")
    parser.add_argument("--scene-name", default="p36-tea-0010", help="Scene name used in the file stems.")
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=DEFAULT_ANNOTATIONS_DIR,
        help="Directory containing pred_raw_clips/pred_steps/pred_run_meta files.",
    )
    parser.add_argument(
        "--left-tag",
        default="",
        help="Suffix tag for the left run. Use empty string for the default Qwen outputs.",
    )
    parser.add_argument(
        "--right-tag",
        default="gemma4",
        help="Suffix tag for the right run. Default matches the Gemma 4 runner.",
    )
    parser.add_argument("--left-name", default="Qwen", help="Display name for the left run.")
    parser.add_argument("--right-name", default="Gemma", help="Display name for the right run.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path to save the comparison report.")
    parser.add_argument(
        "--label-library-out",
        type=Path,
        default=None,
        help="Optional path to save the discovered label library JSON.",
    )
    parser.add_argument(
        "--existing-label-library",
        type=Path,
        default=None,
        help="Optional existing label library JSON used to flag newly discovered labels.",
    )
    return parser.parse_args()


def file_suffix(tag: str) -> str:
    return f"_{tag}" if tag else ""


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_json_block(text: str) -> str | None:
    text = str(text or "").strip()
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def extract_raw_label(raw_response: str) -> str | None:
    json_block = extract_json_block(raw_response)
    if json_block is None:
        return None
    try:
        parsed = json.loads(json_block)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    value = str(parsed.get("label", "")).strip().lower()
    return value or None


def is_meaningful_text(value: str, min_words: int = 2) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if text.lower() in PLACEHOLDER_TEXTS:
        return False
    return len(text.split()) >= min_words


def normalize_object_name(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def evaluate_validity(item: dict[str, Any]) -> dict[str, Any]:
    objects = item.get("objects", [])
    normalized_objects = [normalize_object_name(obj) for obj in objects if normalize_object_name(obj)]
    unique_objects = list(dict.fromkeys(normalized_objects))
    checks = {
        "parse_ok": bool(item.get("parse_ok", False)),
        "label_in_vocab": item.get("label") in CANONICAL_LABELS,
        "main_task_ok": is_meaningful_text(item.get("main_task", ""), min_words=3),
        "sub_task_ok": is_meaningful_text(item.get("sub_task", ""), min_words=2),
        "interaction_ok": is_meaningful_text(item.get("interaction", ""), min_words=3),
        "current_action_ok": is_meaningful_text(item.get("current_action", ""), min_words=3),
        "objects_nonempty": len(unique_objects) > 0,
        "objects_unique": len(unique_objects) == len(normalized_objects),
        "raw_response_nonempty": bool(str(item.get("raw_response", "")).strip()),
    }
    passed = sum(1 for ok in checks.values() if ok)
    validity_score = passed / float(len(checks))
    warnings: list[str] = []
    if not checks["parse_ok"]:
        warnings.append("json_parse_failed")
    if item.get("label") == "other":
        warnings.append("label_is_other")
    if not checks["objects_nonempty"]:
        warnings.append("objects_empty")
    if not checks["sub_task_ok"]:
        warnings.append("sub_task_generic")
    if not checks["interaction_ok"]:
        warnings.append("interaction_generic")
    if not checks["current_action_ok"]:
        warnings.append("current_action_generic")
    return {
        "checks": checks,
        "validity_score": round(validity_score, 4),
        "warnings": warnings,
        "normalized_objects": unique_objects,
    }


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    result = 0.0
    for count in counter.values():
        p = count / total
        result -= p * math.log2(p)
    return result


def round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def summarize_run(raw_clips: list[dict[str, Any]], steps: list[dict[str, Any]], run_meta: dict[str, Any]) -> dict[str, Any]:
    enriched = []
    warning_counts: Counter[str] = Counter()
    label_distribution: Counter[str] = Counter()
    raw_label_distribution: Counter[str] = Counter()
    adjacent_label_flips = 0
    adjacent_raw_label_flips = 0
    adjacent_consistent_labels = 0
    adjacent_object_jaccards: list[float] = []
    duplicate_object_fractions: list[float] = []
    raw_label_alignment: list[float] = []

    previous_label: str | None = None
    previous_raw_label: str | None = None
    previous_objects: set[str] | None = None

    for item in raw_clips:
        validity = evaluate_validity(item)
        enriched.append(validity)
        canonical_label = str(item.get("label", "unknown"))
        label_distribution[canonical_label] += 1
        warning_counts.update(validity["warnings"])
        raw_label = extract_raw_label(str(item.get("raw_response", "")))
        if raw_label is not None:
            raw_label_distribution[raw_label] += 1
            raw_label_alignment.append(1.0 if raw_label == canonical_label else 0.0)

        normalized_objects = validity["normalized_objects"]
        total_objects = len(item.get("objects", []))
        if total_objects > 0:
            duplicate_fraction = 1.0 - (len(normalized_objects) / total_objects)
            duplicate_object_fractions.append(duplicate_fraction)

        current_label = canonical_label
        current_objects = set(normalized_objects)
        if previous_label is not None and current_label != previous_label:
            adjacent_label_flips += 1
        if previous_label is not None and current_label == previous_label:
            adjacent_consistent_labels += 1
        if previous_raw_label is not None and raw_label is not None and raw_label != previous_raw_label:
            adjacent_raw_label_flips += 1
        if previous_objects is not None:
            adjacent_object_jaccards.append(jaccard(previous_objects, current_objects))

        previous_label = current_label
        previous_raw_label = raw_label
        previous_objects = current_objects

    rates: dict[str, float] = {}
    if enriched:
        keys = list(enriched[0]["checks"].keys())
        rates = {
            key: round(sum(1 for item in enriched if item["checks"].get(key, False)) / len(enriched), 4)
            for key in keys
        }

    step_durations = [int(step["end"]) - int(step["start"]) + 1 for step in steps]
    step_labels = Counter(str(step.get("label", "unknown")) for step in steps)
    sec_per_clip = None
    if run_meta.get("num_clips"):
        sec_per_clip = float(run_meta.get("total_time_seconds", 0.0)) / float(run_meta["num_clips"])

    comparison = {
        "model_id": run_meta.get("model_id"),
        "runner_name": run_meta.get("runner_name"),
        "device": run_meta.get("device"),
        "device_name": run_meta.get("device_name"),
        "torch_dtype": run_meta.get("torch_dtype"),
        "batch_size": run_meta.get("batch_size"),
        "num_raw_clips": len(raw_clips),
        "num_steps": len(steps),
        "avg_validity_score": round_or_none(mean([float(item["validity_score"]) for item in enriched])),
        "parse_success_rate": rates.get("parse_ok"),
        "schema_completion_rate": round_or_none(
            mean(
                [
                    1.0
                    if all(
                        item["checks"][key]
                        for key in ("sub_task_ok", "interaction_ok", "current_action_ok", "objects_nonempty", "label_in_vocab")
                    )
                    else 0.0
                    for item in enriched
                ]
            )
        ),
        "canonical_label_rate": rates.get("label_in_vocab"),
        "raw_label_in_library_rate": round_or_none(
            sum(count for label, count in raw_label_distribution.items() if label in CANONICAL_LABELS) / sum(raw_label_distribution.values())
        ) if raw_label_distribution else None,
        "new_raw_label_rate": round_or_none(
            sum(count for label, count in raw_label_distribution.items() if label not in CANONICAL_LABELS) / sum(raw_label_distribution.values())
        ) if raw_label_distribution else None,
        "raw_to_canonical_alignment_rate": round_or_none(mean(raw_label_alignment)),
        "other_rate": round_or_none(label_distribution.get("other", 0) / len(raw_clips) if raw_clips else 0.0),
        "object_nonempty_rate": rates.get("objects_nonempty"),
        "object_dup_rate": round_or_none(mean(duplicate_object_fractions)),
        "adjacent_label_flip_rate": round_or_none(adjacent_label_flips / max(1, len(raw_clips) - 1)) if raw_clips else None,
        "adjacent_label_consistency_rate": round_or_none(adjacent_consistent_labels / max(1, len(raw_clips) - 1)) if raw_clips else None,
        "adjacent_raw_label_flip_rate": round_or_none(adjacent_raw_label_flips / max(1, len(raw_clips) - 1)) if raw_clips else None,
        "adjacent_object_jaccard": round_or_none(mean(adjacent_object_jaccards)),
        "step_compression_ratio": round_or_none(len(raw_clips) / len(steps)) if steps else None,
        "mean_step_length_frames": round_or_none(mean([float(duration) for duration in step_durations])),
        "short_step_rate": round_or_none(sum(1 for duration in step_durations if duration <= 64) / len(step_durations)) if step_durations else None,
        "step_label_entropy": round_or_none(entropy(step_labels)),
        "sec_per_clip": round_or_none(sec_per_clip),
        "total_time_seconds": run_meta.get("total_time_seconds"),
        "peak_vram_allocated_gib": run_meta.get("peak_vram_allocated_gib"),
        "peak_vram_reserved_gib": run_meta.get("peak_vram_reserved_gib"),
        "label_distribution": dict(sorted(label_distribution.items())),
        "raw_label_distribution": dict(sorted(raw_label_distribution.items())),
        "step_label_distribution": dict(sorted(step_labels.items())),
        "warning_counts": dict(sorted(warning_counts.items())),
    }
    return comparison


def build_label_library(
    *,
    left_name: str,
    right_name: str,
    left_raw_clips: list[dict[str, Any]],
    right_raw_clips: list[dict[str, Any]],
    existing_library: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entries: dict[str, dict[str, Any]] = {}
    known_labels = set(CANONICAL_LABELS)
    if existing_library:
        known_labels.update(existing_library.get("known_labels", []))
        known_labels.update(existing_library.get("canonical_labels", []))
        known_labels.update(existing_library.get("raw_labels", {}).keys())

    def ingest(run_name: str, raw_clips: list[dict[str, Any]]) -> None:
        for item in raw_clips:
            raw_label = extract_raw_label(str(item.get("raw_response", "")))
            if not raw_label:
                continue
            entry = entries.setdefault(
                raw_label,
                {
                    "raw_label": raw_label,
                    "canonical_label_counts": {},
                    "runs": {},
                    "example_clips": [],
                    "in_canonical_library": raw_label in CANONICAL_LABELS,
                    "is_new_vs_existing_library": raw_label not in known_labels,
                },
            )
            canonical_label = str(item.get("label", "unknown"))
            canonical_counts = Counter(entry["canonical_label_counts"])
            canonical_counts[canonical_label] += 1
            entry["canonical_label_counts"] = dict(sorted(canonical_counts.items()))

            runs = entry["runs"]
            run_entry = runs.setdefault(run_name, {"count": 0, "clip_ids": []})
            run_entry["count"] += 1
            run_entry["clip_ids"].append(item.get("clip_id"))

            if len(entry["example_clips"]) < 3:
                entry["example_clips"].append(
                    {
                        "run": run_name,
                        "clip_id": item.get("clip_id"),
                        "canonical_label": canonical_label,
                        "sub_task": item.get("sub_task"),
                    }
                )

    ingest(left_name, left_raw_clips)
    ingest(right_name, right_raw_clips)

    for entry in entries.values():
        counts = Counter(entry["canonical_label_counts"])
        dominant_label, dominant_count = counts.most_common(1)[0]
        total = sum(counts.values())
        entry["dominant_canonical_label"] = dominant_label
        entry["dominant_canonical_rate"] = round(dominant_count / total, 4)
        entry["total_count"] = total

    return {
        "canonical_labels": sorted(CANONICAL_LABELS),
        "known_labels": sorted(known_labels),
        "new_labels": sorted(label for label, entry in entries.items() if entry["is_new_vs_existing_library"]),
        "raw_labels": dict(sorted(entries.items())),
    }


def format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_summary_line(name: str, summary: dict[str, Any]) -> None:
    print(
        f"{name}: "
        f"parse={format_value(summary['parse_success_rate'])} "
        f"other={format_value(summary['other_rate'])} "
        f"obj_nonempty={format_value(summary['object_nonempty_rate'])} "
        f"steps={summary['num_steps']} "
        f"comp={format_value(summary['step_compression_ratio'])}x "
        f"sec/clip={format_value(summary['sec_per_clip'])} "
        f"vram={format_value(summary['peak_vram_allocated_gib'])}"
    )


def print_metric_table(left_name: str, right_name: str, left: dict[str, Any], right: dict[str, Any]) -> None:
    metrics = [
        "model_id",
        "runner_name",
        "device",
        "torch_dtype",
        "batch_size",
        "num_raw_clips",
        "num_steps",
        "avg_validity_score",
        "parse_success_rate",
        "schema_completion_rate",
        "canonical_label_rate",
        "raw_label_in_library_rate",
        "new_raw_label_rate",
        "raw_to_canonical_alignment_rate",
        "other_rate",
        "object_nonempty_rate",
        "object_dup_rate",
        "adjacent_label_flip_rate",
        "adjacent_label_consistency_rate",
        "adjacent_raw_label_flip_rate",
        "adjacent_object_jaccard",
        "step_compression_ratio",
        "mean_step_length_frames",
        "short_step_rate",
        "step_label_entropy",
        "sec_per_clip",
        "peak_vram_allocated_gib",
        "peak_vram_reserved_gib",
    ]
    metric_width = max(len(metric) for metric in metrics)
    left_width = max(len(left_name), 12)
    right_width = max(len(right_name), 12)

    print()
    print(f"{'metric':<{metric_width}}  {left_name:<{left_width}}  {right_name:<{right_width}}  delta")
    for metric in metrics:
        left_value = left.get(metric)
        right_value = right.get(metric)
        delta = "-"
        if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
            delta = format_value(right_value - left_value)
        print(
            f"{metric:<{metric_width}}  "
            f"{format_value(left_value):<{left_width}}  "
            f"{format_value(right_value):<{right_width}}  "
            f"{delta}"
        )


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    annotations_dir = args.annotations_dir
    scene_name = args.scene_name

    def paths_for(tag: str) -> dict[str, Path]:
        suffix = file_suffix(tag)
        return {
            "raw": annotations_dir / f"pred_raw_clips_{scene_name}{suffix}.json",
            "steps": annotations_dir / f"pred_steps_{scene_name}{suffix}.json",
            "meta": annotations_dir / f"pred_run_meta_{scene_name}{suffix}.json",
        }

    left_paths = paths_for(args.left_tag)
    right_paths = paths_for(args.right_tag)
    left_raw = load_json(left_paths["raw"])
    right_raw = load_json(right_paths["raw"])
    left_steps = load_json(left_paths["steps"])
    right_steps = load_json(right_paths["steps"])
    left_meta = load_json(left_paths["meta"])
    right_meta = load_json(right_paths["meta"])
    existing_library = load_json(args.existing_label_library) if args.existing_label_library else None

    report = {
        "scene_name": scene_name,
        "annotations_dir": str(annotations_dir),
        "left": {
            "name": args.left_name,
            "tag": args.left_tag,
            "paths": {k: str(v) for k, v in left_paths.items()},
            "summary": summarize_run(left_raw, left_steps, left_meta),
        },
        "right": {
            "name": args.right_name,
            "tag": args.right_tag,
            "paths": {k: str(v) for k, v in right_paths.items()},
            "summary": summarize_run(right_raw, right_steps, right_meta),
        },
        "label_library": build_label_library(
            left_name=args.left_name,
            right_name=args.right_name,
            left_raw_clips=left_raw,
            right_raw_clips=right_raw,
            existing_library=existing_library,
        ),
    }
    return report


def main() -> None:
    args = parse_args()
    report = build_report(args)

    left_name = report["left"]["name"]
    right_name = report["right"]["name"]
    left_summary = report["left"]["summary"]
    right_summary = report["right"]["summary"]

    print_summary_line(left_name, left_summary)
    print_summary_line(right_name, right_summary)
    print_metric_table(left_name, right_name, left_summary, right_summary)

    print()
    print("Label distribution:")
    print(f"{left_name}: {left_summary['label_distribution']}")
    print(f"{right_name}: {right_summary['label_distribution']}")

    print()
    print("Raw label distribution:")
    print(f"{left_name}: {left_summary['raw_label_distribution']}")
    print(f"{right_name}: {right_summary['raw_label_distribution']}")

    print()
    print("Warning counts:")
    print(f"{left_name}: {left_summary['warning_counts']}")
    print(f"{right_name}: {right_summary['warning_counts']}")

    print()
    print("Discovered label library:")
    print(report["label_library"]["raw_labels"].keys())
    if report["label_library"]["new_labels"]:
        print(f"New labels vs library: {report['label_library']['new_labels']}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print()
        print(f"Saved comparison report to: {args.json_out}")

    if args.label_library_out is not None:
        args.label_library_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.label_library_out, "w", encoding="utf-8") as f:
            json.dump(report["label_library"], f, ensure_ascii=False, indent=2)
        print(f"Saved label library to: {args.label_library_out}")


if __name__ == "__main__":
    main()
