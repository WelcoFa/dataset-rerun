from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import universal_label_library as label_library_lib


DEFAULT_LIBRARY_PATH = REPO_ROOT / "data" / "universal_label_library.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the universal label library from its accumulated observations.")
    parser.add_argument(
        "--library-path",
        type=Path,
        default=DEFAULT_LIBRARY_PATH,
        help="Path to the universal label library JSON.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to save the evaluation report as JSON.",
    )
    parser.add_argument(
        "--write-back",
        action="store_true",
        help="Persist the refreshed evaluation fields back into the library JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library = label_library_lib.load_label_library(args.library_path)
    evaluation = label_library_lib.recompute_library_evaluations(library)
    summary = label_library_lib.summarize_library(library)

    print("Library summary:")
    print(f"  name: {summary['library_name']}")
    print(f"  total labels: {summary['total_labels']}")
    print(f"  active labels: {summary['active_label_count']}")
    print(f"  unused labels: {summary['unused_label_count']}")
    print(f"  observations: {summary['observation_count']}")
    print(f"  needs review: {summary['labels_needing_review']}")
    print()
    print("Per-label evaluation:")

    rows = sorted(
        evaluation.get("label_rows", []),
        key=lambda row: (-int(row.get("usage_count", 0)), str(row.get("name", ""))),
    )
    for row in rows:
        print(
            f"  {row['name']}: "
            f"usage={row['usage_count']} "
            f"quality={row.get('label_quality_score')} "
            f"raw_variants={row['unique_raw_variant_count']} "
            f"exact={row.get('exact_match_rate')} "
            f"alias={row.get('alias_match_rate')} "
            f"similarity={row.get('similarity_match_rate')} "
            f"review={row.get('needs_review')}"
        )

    report = {
        "library_path": str(args.library_path),
        "summary": summary,
        "evaluation": evaluation,
    }

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print()
        print(f"Saved evaluation report to: {args.json_out}")

    if args.write_back:
        label_library_lib.save_label_library(library, args.library_path)
        print(f"Updated library with evaluation fields: {args.library_path}")


if __name__ == "__main__":
    main()
