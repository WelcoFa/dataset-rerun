from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class VizConfig:
    dataset: str | None = "auto"
    input: Path | None = None
    spawn: bool = True
    view_mode: str = "both"
    enrichments: list[str] = field(default_factory=list)
    selection: dict[str, Any] = field(default_factory=dict)
    dataset_options: dict[str, Any] = field(default_factory=dict)
    config_path: Path | None = None

