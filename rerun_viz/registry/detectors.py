from __future__ import annotations

import sys
from pathlib import Path


def _ensure_scripts_on_path():
    repo_root = Path(__file__).resolve().parents[2]
    scripts_dir = repo_root / "scripts"
    visualize_dir = scripts_dir / "visualize"
    for path in (scripts_dir, visualize_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def detect_with_legacy_plus(input_path: Path, dataset: str):
    _ensure_scripts_on_path()
    import visualize_universal_dashboard_plus as legacy_plus

    return legacy_plus.detect_dataset(input_path, dataset or "auto")
