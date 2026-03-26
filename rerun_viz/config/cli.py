from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Modular multi-dataset Rerun dashboard.")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON/TOML/YAML config file.")
    parser.add_argument("--input", type=Path, default=None, help="Dataset root, scene folder, subset folder, or source file.")
    parser.add_argument(
        "--dataset",
        choices=["auto", "gigahands", "hot3d", "being-h0", "dexwild", "thermohands", "wiyh", "generic"],
        default=None,
        help="Force dataset type or let the registry auto-detect it.",
    )
    parser.add_argument("--spawn", action=argparse.BooleanOptionalAction, default=None, help="Spawn the Rerun viewer.")
    parser.add_argument(
        "--view-mode",
        choices=["core", "enriched", "both"],
        default=None,
        help="Control whether to log raw visualization, enrichments, or both.",
    )
    parser.add_argument(
        "--enrichment",
        action="append",
        default=None,
        help="Enable one or more enrichment names. Repeatable.",
    )
    return parser.parse_args()

