from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from .schema import VizConfig


def _load_yaml_if_available(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            f"YAML config requested for {path}, but PyYAML is not installed. "
            "Use JSON/TOML for now, or install PyYAML."
        ) from exc

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_config_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix == ".toml":
        return tomllib.loads(path.read_text(encoding="utf-8"))
    if suffix in {".yaml", ".yml"}:
        return _load_yaml_if_available(path)
    raise ValueError(f"Unsupported config format: {path}")


def _coerce_path(value: Any) -> Path | None:
    if value is None or value == "":
        return None
    return Path(value)


def build_config(args) -> VizConfig:
    payload: dict[str, Any] = {}
    if args.config is not None:
        payload = load_config_file(args.config)

    view = payload.get("view", {})
    enrichment = payload.get("enrichment", {})

    config = VizConfig(
        dataset=payload.get("dataset", "auto"),
        input=_coerce_path(payload.get("input")),
        spawn=view.get("spawn", True),
        view_mode=view.get("mode", "both"),
        enrichments=list(enrichment.get("enabled", [])),
        selection=dict(payload.get("selection", {})),
        dataset_options=dict(payload.get("dataset_options", {})),
        config_path=args.config,
    )

    if args.dataset is not None:
        config.dataset = args.dataset
    if args.input is not None:
        config.input = args.input
    if args.spawn is not None:
        config.spawn = args.spawn
    if args.view_mode is not None:
        config.view_mode = args.view_mode
    if args.enrichment is not None:
        config.enrichments = list(args.enrichment)

    if config.input is None:
        config.input = Path("data")

    return config

