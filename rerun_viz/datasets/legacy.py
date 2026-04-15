from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from rerun_viz.config.schema import VizConfig
from rerun_viz.core.panels import log_dashboard_panels
from rerun_viz.datasets.base import DatasetAdapter
from rerun_viz.registry.detectors import detect_with_legacy_plus


def _ensure_scripts_on_path():
    repo_root = Path(__file__).resolve().parents[2]
    scripts_dir = repo_root / "scripts"
    visualize_dir = scripts_dir / "visualize"
    experimental_dir = scripts_dir / "experimental"
    candidate_paths = [scripts_dir, visualize_dir, experimental_dir]
    if experimental_dir.exists():
        candidate_paths.extend(path for path in experimental_dir.iterdir() if path.is_dir())
    for path in candidate_paths:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


class LegacyUniversalAdapter(DatasetAdapter):
    def __init__(self, config: VizConfig):
        self.config = config
        self.legacy_adapter = None
        self.detection = None

    def _build_cli_like_namespace(self):
        selection = self.config.selection
        options = self.config.dataset_options
        return SimpleNamespace(
            gigahands_root=options.get("gigahands_root"),
            annotations_dir=options.get("annotations_dir"),
            seq_name=selection.get("seq_name"),
            cam_name=selection.get("cam_name"),
            frame_id=selection.get("frame_id"),
            sequence_name=selection.get("sequence_name"),
            frame_stride=options.get("frame_stride"),
            device=options.get("device"),
            beingh0_subset_dir=options.get("beingh0_subset_dir"),
            beingh0_jsonl=options.get("beingh0_jsonl"),
            beingh0_start=options.get("beingh0_start"),
            beingh0_max_samples=options.get("beingh0_max_samples"),
            dexwild_hdf5=options.get("dexwild_hdf5"),
            dexwild_episode=options.get("dexwild_episode"),
            dexwild_max_frames=options.get("dexwild_max_frames"),
            thermohands_scene_dir=options.get("thermohands_scene_dir"),
            thermohands_stride=options.get("thermohands_stride"),
            thermohands_max_frames=options.get("thermohands_max_frames"),
            generic_max_items=options.get("generic_max_items", 200),
        )

    def load(self):
        _ensure_scripts_on_path()

        requested_dataset = self.config.dataset or "auto"
        self.detection = detect_with_legacy_plus(self.config.input.resolve(), "auto")
        if requested_dataset not in {"auto", None} and self.detection.dataset != requested_dataset:
            raise ValueError(
                f"Input {self.config.input} auto-detected as '{self.detection.dataset}', "
                f"but config requested '{requested_dataset}'."
            )

        import visualize_universal_dashboard_plus as legacy_plus

        cli_like = self._build_cli_like_namespace()
        self.legacy_adapter = legacy_plus.create_adapter_from_detection(self.detection, cli_like)
        self.name = self.detection.dataset
        self.base = self.legacy_adapter.base
        self.viewer_name = self.legacy_adapter.viewer_name
        self.legacy_adapter.load()

    def log_static(self):
        self.legacy_adapter.log_static()

    def frames(self):
        return self.legacy_adapter.frames()

    def log_panels(self, panels):
        log_dashboard_panels(self.base, panels)

    def close(self):
        if self.legacy_adapter is not None:
            self.legacy_adapter.close()
