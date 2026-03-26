from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DatasetSpec:
    dataset: str
    input: Path
    resolved_root: Path
    selection: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetContext:
    spec: DatasetSpec
    metadata: dict[str, Any] = field(default_factory=dict)
    resources: dict[str, Any] = field(default_factory=dict)


@dataclass
class FramePacket:
    frame_idx: int | None = None
    timestamp_ns: int | None = None
    scalars: dict[str, float] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)

