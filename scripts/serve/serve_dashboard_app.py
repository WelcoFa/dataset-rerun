from __future__ import annotations

import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
import mimetypes
import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rerun_viz.config.loader import load_config_file
from scripts.tools import compare_gigahands_vlm_runs as compare_tool
import universal_label_library as label_library_lib


APP_UI_DIR = REPO_ROOT / "app_ui"
SERVE_SCRIPT = REPO_ROOT / "scripts" / "serve" / "serve_rerun_dashboard.py"
RUN_GIGAHANDS_QWEN_SCRIPT = REPO_ROOT / "scripts" / "run" / "run_gigahands_vlm.py"
RUN_GIGAHANDS_GEMMA_SCRIPT = REPO_ROOT / "scripts" / "run" / "run_gigahands_gemma4_vlm.py"
RUN_MULTIDATASET_QWEN_SCRIPT = REPO_ROOT / "scripts" / "run" / "run_multidataset_vlm.py"
RUN_MULTIDATASET_GEMMA_SCRIPT = REPO_ROOT / "scripts" / "run" / "run_multidataset_gemma4_vlm.py"
UNIVERSAL_LABEL_LIBRARY_PATH = REPO_ROOT / "data" / "universal_label_library.json"
GIGAHANDS_ANNOTATIONS_DIR = REPO_ROOT / "data" / "gigahands" / "annotations"
ANNOTATION_RUN_SUPPORTED_DATASETS: set[str] = set()


@dataclass
class PlayableItem:
    item_id: str
    label: str
    path: Path
    dataset: str
    input_path: str
    description: str
    valid: bool
    selection: dict[str, Any]
    scenes: list["SceneOption"]
    default_scene_id: str | None = None
    error: str | None = None


@dataclass
class SceneOption:
    scene_id: str
    label: str
    description: str
    selection: dict[str, Any]
    dataset_options: dict[str, Any]
    details: list[dict[str, str]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("dashboard_app")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


class DashboardAppState:
    def __init__(
        self,
        config_dir: Path,
        outputs_dir: Path,
        viewer_port: int,
        grpc_port: int,
        logger: logging.Logger,
    ) -> None:
        self.config_dir = config_dir
        self.outputs_dir = outputs_dir
        self.viewer_port = viewer_port
        self.grpc_port = grpc_port
        self.logger = logger
        self.lock = threading.RLock()
        self.lifecycle_lock = threading.RLock()
        self.log_lines: deque[str] = deque(maxlen=300)
        self.annotation_log_lines: deque[str] = deque(maxlen=400)
        self.process: subprocess.Popen[str] | None = None
        self.annotation_process: subprocess.Popen[str] | None = None
        self.current_item_id: str | None = None
        self.current_scene_id: str | None = None
        self.current_recording: str | None = None
        self.status = "idle"
        self.annotation_status = "idle"
        self.last_error: str | None = None
        self.annotation_last_error: str | None = None
        self.started_at: str | None = None
        self.annotation_started_at: str | None = None
        self.annotation_runner: str | None = None
        self.annotation_item_id: str | None = None
        self.annotation_scene_id: str | None = None
        self.items = self._discover_items()

    def _is_port_in_use(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            return sock.connect_ex(("127.0.0.1", port)) == 0

    def _wait_for_ports_to_release(self, timeout: float = 10.0) -> None:
        deadline = time.time() + timeout
        ports = [self.viewer_port, self.grpc_port]
        while time.time() < deadline:
            busy_ports = [port for port in ports if self._is_port_in_use(port)]
            if not busy_ports:
                return
            time.sleep(0.2)
        busy_ports = [port for port in ports if self._is_port_in_use(port)]
        if busy_ports:
            raise RuntimeError(f"Ports still busy after shutdown: {', '.join(str(port) for port in busy_ports)}")

    def _discover_items(self) -> list[PlayableItem]:
        items: list[PlayableItem] = []
        if not self.config_dir.exists():
            return items

        for path in sorted(self.config_dir.iterdir()):
            if path.suffix.lower() not in {".json", ".toml", ".yaml", ".yml"}:
                continue
            if path.name == "docker-dashboard.toml":
                continue

            item_id = path.stem
            try:
                payload = load_config_file(path)
                dataset = str(payload.get("dataset", "auto"))
                input_path = str(payload.get("input", ""))
                selection = dict(payload.get("selection", {}))
                dataset_options = dict(payload.get("dataset_options", {}))
                scenes = self._discover_scenes(payload, selection, dataset_options)
                default_scene_id = self._select_default_scene_id(scenes, selection, dataset_options)
                description_parts = [dataset, input_path or "no input"]
                if len(scenes) > 1:
                    description_parts.append(f"{len(scenes)} scenes")
                description = " | ".join(description_parts)
                items.append(
                    PlayableItem(
                        item_id=item_id,
                        label=path.stem.replace("-", " ").replace("_", " "),
                        path=path,
                        dataset=dataset,
                        input_path=input_path,
                        description=description,
                        valid=True,
                        selection=selection,
                        scenes=scenes,
                        default_scene_id=default_scene_id,
                    )
                )
            except Exception as exc:
                items.append(
                    PlayableItem(
                        item_id=item_id,
                        label=path.stem,
                        path=path,
                        dataset="invalid",
                        input_path="",
                        description="Config parse failed",
                        valid=False,
                        selection={},
                        scenes=[],
                        error=str(exc),
                    )
                )
        return items

    def _resolve_input_path(self, value: Any) -> Path | None:
        if value in {None, ""}:
            return None
        raw_value = str(value)
        normalized = raw_value.replace("\\", "/")
        if normalized == "/data":
            docker_path = Path("/data")
            if docker_path.exists():
                return docker_path
            fallback = (REPO_ROOT / "data").resolve()
            return fallback if fallback.exists() else docker_path
        if normalized.startswith("/data/"):
            docker_path = Path(normalized)
            if docker_path.exists():
                return docker_path
            fallback = (REPO_ROOT / "data" / normalized.removeprefix("/data/")).resolve()
            return fallback if fallback.exists() else docker_path
        path = Path(raw_value)
        if path.is_absolute():
            if path.exists():
                return path
            return path
        return (REPO_ROOT / path).resolve()

    def _path_option_value(self, path: Path) -> str:
        return str(path)

    def _scene_detail(self, label: str, value: Any) -> dict[str, str]:
        return {"label": str(label), "value": str(value)}

    def _normalize_scene_entry(self, entry: Any) -> SceneOption | None:
        if not isinstance(entry, dict):
            return None

        raw_selection = entry.get("selection", {})
        selection = dict(raw_selection) if isinstance(raw_selection, dict) else {}
        for key in ("seq_name", "cam_name", "frame_id", "sequence_name"):
            if key in entry and key not in selection:
                selection[key] = entry[key]

        if not selection:
            return None

        raw_dataset_options = entry.get("dataset_options", {})
        dataset_options = dict(raw_dataset_options) if isinstance(raw_dataset_options, dict) else {}
        scene_id = str(entry.get("id") or selection.get("seq_name") or selection.get("sequence_name") or "").strip()
        if not scene_id:
            return None

        label = str(entry.get("label") or scene_id).strip() or scene_id
        description = str(entry.get("description") or "").strip()
        raw_details = entry.get("details", [])
        details = []
        if isinstance(raw_details, list):
            for detail in raw_details:
                if not isinstance(detail, dict):
                    continue
                detail_label = str(detail.get("label", "")).strip()
                detail_value = str(detail.get("value", "")).strip()
                if detail_label and detail_value:
                    details.append({"label": detail_label, "value": detail_value})
        return SceneOption(
            scene_id=scene_id,
            label=label,
            description=description,
            selection=selection,
            dataset_options=dataset_options,
            details=details,
        )

    def _looks_like_gigahands_root(self, path: Path) -> bool:
        return path.is_dir() and (path / "hand_pose").is_dir() and (path / "object_pose").is_dir()

    def _gigahands_object_id(self, seq_name: str) -> str:
        suffix = str(seq_name).rsplit("-", 1)[-1]
        if suffix.isdigit():
            return suffix[-3:].zfill(3)
        raise ValueError(f"Cannot infer GigaHands object id from sequence name: {seq_name}")

    def _scene_assets_exist(self, root: Path, seq_name: str, cam_name: str, frame_id: str) -> bool:
        object_id = self._gigahands_object_id(seq_name)
        hand_root = root / "hand_pose" / seq_name
        required = [
            hand_root / "rgb_vid" / cam_name / f"{cam_name}_{frame_id}.mp4",
            hand_root / "keypoints_2d" / "left" / object_id / f"{cam_name}_{frame_id}.jsonl",
            hand_root / "keypoints_2d" / "right" / object_id / f"{cam_name}_{frame_id}.jsonl",
            hand_root / "keypoints_3d" / object_id / "left.jsonl",
            hand_root / "keypoints_3d" / object_id / "right.jsonl",
        ]
        return all(path.exists() for path in required)

    def _find_gigahands_scene_selection(self, root: Path, seq_name: str) -> dict[str, str] | None:
        object_id = self._gigahands_object_id(seq_name)
        seq_dir = root / "hand_pose" / seq_name
        left_dir = seq_dir / "keypoints_2d" / "left" / object_id
        right_dir = seq_dir / "keypoints_2d" / "right" / object_id
        rgb_vid_dir = seq_dir / "rgb_vid"
        if not left_dir.is_dir() or not right_dir.is_dir() or not rgb_vid_dir.is_dir():
            return None

        left_stems = {path.stem for path in left_dir.glob("*.jsonl")}
        right_stems = {path.stem for path in right_dir.glob("*.jsonl")}
        common_stems = left_stems & right_stems
        if not common_stems:
            return None

        for cam_dir in sorted(rgb_vid_dir.iterdir()):
            if not cam_dir.is_dir():
                continue
            for video_path in sorted(cam_dir.glob("*.mp4")):
                if video_path.stem not in common_stems:
                    continue
                prefix = f"{cam_dir.name}_"
                frame_id = video_path.stem[len(prefix):] if video_path.stem.startswith(prefix) else video_path.stem
                if self._scene_assets_exist(root, seq_name, cam_dir.name, frame_id):
                    return {
                        "seq_name": seq_name,
                        "cam_name": cam_dir.name,
                        "frame_id": frame_id,
                    }

        for stem in sorted(common_stems):
            if "_cam" not in stem:
                continue
            cam_name, frame_id = stem.rsplit("_", 1)
            if self._scene_assets_exist(root, seq_name, cam_name, frame_id):
                return {
                    "seq_name": seq_name,
                    "cam_name": cam_name,
                    "frame_id": frame_id,
                }
        return None

    def _discover_gigahands_scenes(self, payload: dict[str, Any], selection: dict[str, Any]) -> list[SceneOption]:
        input_path = self._resolve_input_path(payload.get("input"))
        if input_path is None or not input_path.exists():
            return []

        root = None
        annotations_dir = None
        if input_path.name == "gigahands" and (input_path / "gigahands_demo_all").is_dir():
            root = input_path / "gigahands_demo_all"
            annotations_dir = input_path / "annotations"
        elif self._looks_like_gigahands_root(input_path):
            root = input_path
            annotations_dir = input_path.parent / "annotations"

        if root is None or annotations_dir is None:
            return []

        hand_pose_dir = root / "hand_pose"
        if not hand_pose_dir.is_dir():
            return []

        scenes: list[SceneOption] = []
        for seq_dir in sorted(hand_pose_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            chosen = None
            if seq_dir.name == str(selection.get("seq_name", "")):
                candidate_cam = str(selection.get("cam_name") or "")
                candidate_frame = str(selection.get("frame_id") or "")
                if candidate_cam and candidate_frame and self._scene_assets_exist(root, seq_dir.name, candidate_cam, candidate_frame):
                    chosen = {
                        "seq_name": seq_dir.name,
                        "cam_name": candidate_cam,
                        "frame_id": candidate_frame,
                    }
            if chosen is None:
                chosen = self._find_gigahands_scene_selection(root, seq_dir.name)
            if chosen is None:
                continue

            cam_name = chosen["cam_name"]
            frame_id = chosen["frame_id"]
            has_semantics = any(
                candidate.exists()
                for candidate in (
                    annotations_dir / f"pred_steps_{seq_dir.name}.json",
                    annotations_dir / f"pred_raw_clips_{seq_dir.name}.json",
                )
            )
            description = f"camera {cam_name}"
            description = f"{description} | semantic json ready" if has_semantics else f"{description} | no semantic json"
            scenes.append(
                SceneOption(
                    scene_id=seq_dir.name,
                    label=seq_dir.name,
                    description=description,
                    selection=chosen,
                    dataset_options={},
                    details=[
                        self._scene_detail("RGB Video", "Available"),
                        self._scene_detail("Keypoints 2D", "Left / Right"),
                        self._scene_detail("Keypoints 3D", "Left / Right"),
                        self._scene_detail("Object Pose", "Available"),
                        self._scene_detail("Semantic JSON", "Ready" if has_semantics else "Missing"),
                    ],
                )
            )
        return scenes

    def _discover_thermohands_scenes(self, payload: dict[str, Any], dataset_options: dict[str, Any]) -> list[SceneOption]:
        input_path = self._resolve_input_path(payload.get("input"))
        if input_path is None or not input_path.exists() or not input_path.is_dir():
            return []

        configured_scene_dir = str(dataset_options.get("thermohands_scene_dir", "")).replace("\\", "/")
        scenes: list[SceneOption] = []
        for child in sorted(input_path.iterdir()):
            if not child.is_dir():
                continue
            if not all((child / name).is_dir() for name in ("rgb", "thermal", "ir", "depth", "gt_info")):
                continue
            description = f"scene {child.name}"
            scenes.append(
                SceneOption(
                    scene_id=child.name,
                    label=child.name,
                    description=description,
                    selection={},
                    dataset_options={"thermohands_scene_dir": self._path_option_value(child)},
                    details=[
                        self._scene_detail("Scene Folder", child.name),
                        self._scene_detail("Modalities", "RGB / Thermal / IR / Depth"),
                        self._scene_detail("GT Info", "Available"),
                    ],
                )
            )

        if configured_scene_dir:
            for scene in scenes:
                configured_name = Path(configured_scene_dir).name
                if scene.scene_id == configured_name:
                    scene.description = f"{scene.description} | default"
                    break
        return scenes

    def _discover_hot3d_scenes(self, payload: dict[str, Any], selection: dict[str, Any]) -> list[SceneOption]:
        input_path = self._resolve_input_path(payload.get("input"))
        if input_path is None or not input_path.exists() or not input_path.is_dir():
            return []

        candidate_root = input_path / "hot3d_demo_full" if (input_path / "hot3d_demo_full").is_dir() else input_path
        required_root_dirs = ("object_models", "mano_models")
        if not all((candidate_root / name).is_dir() for name in required_root_dirs):
            return []

        configured_sequence = str(selection.get("sequence_name", ""))
        scenes: list[SceneOption] = []
        for child in sorted(candidate_root.iterdir()):
            if not child.is_dir():
                continue
            if not (child / "hand_data").is_dir() or not (child / "ground_truth").is_dir():
                continue
            description = f"sequence {child.name}"
            if child.name == configured_sequence:
                description = f"{description} | default"
            scenes.append(
                SceneOption(
                    scene_id=child.name,
                    label=child.name,
                    description=description,
                    selection={"sequence_name": child.name},
                    dataset_options={},
                    details=[
                        self._scene_detail("Hand Data", "Available"),
                        self._scene_detail("Ground Truth", "Available"),
                        self._scene_detail("Object Models", "Available"),
                        self._scene_detail("MANO Models", "Available"),
                    ],
                )
            )
        return scenes

    def _discover_beingh0_scenes(self, payload: dict[str, Any], dataset_options: dict[str, Any]) -> list[SceneOption]:
        input_path = self._resolve_input_path(payload.get("input"))
        if input_path is None or not input_path.exists() or not input_path.is_dir():
            return []

        base = input_path / "h0_post_train_db_2508" if (input_path / "h0_post_train_db_2508").is_dir() else input_path
        configured_subset = str(dataset_options.get("beingh0_subset_dir", "")).replace("\\", "/")
        scenes: list[SceneOption] = []
        for child in sorted(base.iterdir()):
            if not child.is_dir() or not (child / "images").is_dir():
                continue
            jsonl_path = next(iter(sorted(child.glob("*_train.jsonl"))), None)
            if jsonl_path is None:
                continue
            description = f"subset {child.name}"
            if child.name == Path(configured_subset).name:
                description = f"{description} | default"
            scenes.append(
                SceneOption(
                    scene_id=child.name,
                    label=child.name,
                    description=description,
                    selection={},
                    dataset_options={
                        "beingh0_subset_dir": self._path_option_value(child),
                        "beingh0_jsonl": self._path_option_value(jsonl_path),
                    },
                    details=[
                        self._scene_detail("Images", "Available"),
                        self._scene_detail("Train JSONL", jsonl_path.name),
                        self._scene_detail("Subset Format", "Image folder + annotations"),
                    ],
                )
            )
        return scenes

    def _discover_dexwild_scenes(self, payload: dict[str, Any], dataset_options: dict[str, Any]) -> list[SceneOption]:
        input_path = self._resolve_input_path(payload.get("input"))
        if input_path is None or not input_path.exists():
            return []

        hdf5_path = None
        if input_path.is_file() and input_path.suffix.lower() in {".hdf5", ".h5"}:
            hdf5_path = input_path
        elif input_path.is_dir():
            for candidate in sorted(input_path.glob("*.hdf5")) + sorted(input_path.glob("*.h5")):
                hdf5_path = candidate
                break
        if hdf5_path is None or not hdf5_path.exists():
            return []

        try:
            import h5py
        except ImportError:
            return []

        configured_episode = str(dataset_options.get("dexwild_episode", ""))
        scenes: list[SceneOption] = []
        with h5py.File(hdf5_path, "r") as f:
            for episode_name in sorted(f.keys()):
                description = f"episode {episode_name}"
                if episode_name == configured_episode:
                    description = f"{description} | default"
                scenes.append(
                    SceneOption(
                        scene_id=episode_name,
                        label=episode_name,
                        description=description,
                        selection={},
                        dataset_options={
                            "dexwild_hdf5": self._path_option_value(hdf5_path),
                            "dexwild_episode": episode_name,
                        },
                        details=[
                            self._scene_detail("Source Format", "HDF5 episode"),
                            self._scene_detail("RGB Frames", "Stored in HDF5"),
                            self._scene_detail("Hand Data", "Stored in HDF5"),
                            self._scene_detail("Source File", hdf5_path.name),
                        ],
                    )
                )
        return scenes

    def _discover_wiyh_scenes(self, payload: dict[str, Any], dataset_options: dict[str, Any]) -> list[SceneOption]:
        input_path = self._resolve_input_path(payload.get("input"))
        if input_path is None or not input_path.exists() or not input_path.is_dir():
            return []

        task_json_path = input_path / "task.json"
        configured_action_dir = str(dataset_options.get("action_dir", "")).replace("\\", "/")
        scenes: list[SceneOption] = []
        for child in sorted(input_path.iterdir()):
            if not child.is_dir() or not (child / "dataset.hdf5").exists():
                continue
            description = f"action {child.name}"
            if child.name == Path(configured_action_dir).name:
                description = f"{description} | default"
            scenes.append(
                SceneOption(
                    scene_id=child.name,
                    label=child.name,
                    description=description,
                    selection={},
                    dataset_options={
                        "action_dir": self._path_option_value(child),
                        "task_json": self._path_option_value(task_json_path),
                    },
                    details=[
                        self._scene_detail("dataset.hdf5", "Available"),
                        self._scene_detail("task.json", task_json_path.name),
                        self._scene_detail("Format", "Action folder + HDF5"),
                    ],
                )
            )
        return scenes

    def _discover_scenes(
        self,
        payload: dict[str, Any],
        selection: dict[str, Any],
        dataset_options: dict[str, Any],
    ) -> list[SceneOption]:
        raw_scenes = payload.get("scenes")
        if isinstance(raw_scenes, list):
            normalized = [scene for scene in (self._normalize_scene_entry(entry) for entry in raw_scenes) if scene is not None]
            if normalized:
                return normalized

        auto_requested = raw_scenes is None or raw_scenes == "" or raw_scenes == "auto"
        if isinstance(raw_scenes, dict):
            auto_requested = str(raw_scenes.get("source", "")).lower() in {"auto", "detect", "discover"}

        dataset = str(payload.get("dataset", "auto")).lower()
        if auto_requested and dataset == "gigahands":
            discovered = self._discover_gigahands_scenes(payload, selection)
            if discovered:
                return discovered
        if auto_requested and dataset == "thermohands":
            discovered = self._discover_thermohands_scenes(payload, dataset_options)
            if discovered:
                return discovered
        if auto_requested and dataset == "hot3d":
            discovered = self._discover_hot3d_scenes(payload, selection)
            if discovered:
                return discovered
        if auto_requested and dataset == "being-h0":
            discovered = self._discover_beingh0_scenes(payload, dataset_options)
            if discovered:
                return discovered
        if auto_requested and dataset == "dexwild":
            discovered = self._discover_dexwild_scenes(payload, dataset_options)
            if discovered:
                return discovered
        if auto_requested and dataset == "wiyh":
            discovered = self._discover_wiyh_scenes(payload, dataset_options)
            if discovered:
                return discovered

        normalized_selection = self._normalize_scene_entry({"id": selection.get("seq_name") or "default", "selection": selection})
        return [normalized_selection] if normalized_selection is not None else []

    def _select_default_scene_id(
        self,
        scenes: list[SceneOption],
        selection: dict[str, Any],
        dataset_options: dict[str, Any],
    ) -> str | None:
        if not scenes:
            return None
        for scene in scenes:
            if scene.selection == selection:
                return scene.scene_id
        target_seq = selection.get("seq_name") or selection.get("sequence_name")
        for scene in scenes:
            scene_seq = scene.selection.get("seq_name") or scene.selection.get("sequence_name")
            if scene_seq == target_seq:
                return scene.scene_id
        configured_scene_dir = str(dataset_options.get("thermohands_scene_dir", "")).replace("\\", "/")
        if configured_scene_dir:
            configured_name = Path(configured_scene_dir).name
            for scene in scenes:
                if scene.scene_id == configured_name:
                    return scene.scene_id
        configured_subset = str(dataset_options.get("beingh0_subset_dir", "")).replace("\\", "/")
        if configured_subset:
            configured_name = Path(configured_subset).name
            for scene in scenes:
                if scene.scene_id == configured_name:
                    return scene.scene_id
        configured_episode = str(dataset_options.get("dexwild_episode", ""))
        if configured_episode:
            for scene in scenes:
                if scene.scene_id == configured_episode:
                    return scene.scene_id
        configured_action_dir = str(dataset_options.get("action_dir", "")).replace("\\", "/")
        if configured_action_dir:
            configured_name = Path(configured_action_dir).name
            for scene in scenes:
                if scene.scene_id == configured_name:
                    return scene.scene_id
        return scenes[0].scene_id

    def _get_scene_for_item(self, item: PlayableItem, scene_id: str | None) -> SceneOption | None:
        chosen_id = scene_id or item.default_scene_id
        if chosen_id is None:
            return None
        return next((scene for scene in item.scenes if scene.scene_id == chosen_id), None)

    def list_items(self) -> list[dict[str, Any]]:
        with self.lock:
            return [
                {
                    "id": item.item_id,
                    "label": item.label,
                    "path": str(item.path),
                    "dataset": item.dataset,
                    "input": item.input_path,
                    "description": item.description,
                    "valid": item.valid,
                    "error": item.error,
                    "selection": item.selection,
                    "default_scene_id": item.default_scene_id,
                    "scenes": [
                        {
                            "id": scene.scene_id,
                            "label": scene.label,
                            "description": scene.description,
                            "selection": scene.selection,
                            "details": scene.details,
                        }
                        for scene in item.scenes
                    ],
                    "active": item.item_id == self.current_item_id and self.status in {"starting", "running"},
                    "active_scene_id": self.current_scene_id if item.item_id == self.current_item_id else None,
                }
                for item in self.items
            ]

    def _get_item(self, item_id: str) -> PlayableItem | None:
        return next((item for item in self.items if item.item_id == item_id), None)

    def _build_tree_node(
        self,
        path: Path,
        *,
        root: Path,
        max_depth: int,
        max_entries: int,
        counters: dict[str, int],
    ) -> dict[str, Any]:
        counters["entries"] += 1
        if counters["entries"] > max_entries:
            raise RuntimeError(f"Tree truncated after {max_entries} entries")

        try:
            rel_path = path.relative_to(root)
            display_path = "." if str(rel_path) == "." else str(rel_path)
        except ValueError:
            display_path = str(path)

        node: dict[str, Any] = {
            "name": path.name or str(path),
            "path": display_path,
            "kind": "directory" if path.is_dir() else "file",
        }

        if path.is_dir():
            children: list[dict[str, Any]] = []
            if max_depth > 0:
                for child in sorted(path.iterdir(), key=lambda entry: (not entry.is_dir(), entry.name.lower())):
                    children.append(
                        self._build_tree_node(
                            child,
                            root=root,
                            max_depth=max_depth - 1,
                            max_entries=max_entries,
                            counters=counters,
                        )
                    )
            node["children"] = children
            if max_depth == 0:
                node["truncated"] = True
        else:
            try:
                node["size_bytes"] = path.stat().st_size
            except OSError:
                node["size_bytes"] = None
        return node

    def get_dataset_tree(self, item_id: str, *, max_depth: int = 6, max_entries: int = 1200) -> dict[str, Any]:
        item = self._get_item(item_id)
        if item is None:
            raise ValueError(f"Unknown item: {item_id}")

        root_path = self._resolve_input_path(item.input_path)
        if root_path is None:
            raise ValueError(f"Item '{item_id}' has no input path configured")
        if not root_path.exists():
            raise ValueError(f"Dataset path does not exist: {root_path}")

        tree_root = root_path if root_path.is_dir() else root_path.parent
        counters = {"entries": 0}
        tree = self._build_tree_node(
            tree_root,
            root=tree_root,
            max_depth=max_depth,
            max_entries=max_entries,
            counters=counters,
        )
        return {
            "item_id": item.item_id,
            "label": item.label,
            "root_path": str(tree_root),
            "tree": tree,
            "entry_count": counters["entries"],
        }

    def list_recordings(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        if not self.outputs_dir.exists():
            return items
        for path in sorted(self.outputs_dir.rglob("*.rrd")):
            stat = path.stat()
            try:
                rel_path = path.resolve().relative_to(REPO_ROOT.resolve())
            except ValueError:
                rel_path = Path(os.path.relpath(path.resolve(), REPO_ROOT.resolve()))
            items.append(
                {
                    "name": path.name,
                    "path": str(rel_path),
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
                }
            )
        return items

    def _resolve_annotations_dir_for_item(self, item: PlayableItem) -> Path:
        input_path = self._resolve_input_path(item.input_path)
        if input_path is None:
            raise ValueError(f"Item '{item.item_id}' has no input path configured")
        if input_path.is_file():
            return input_path.parent / "annotations"
        direct = input_path / "annotations"
        if direct.exists() or input_path.name.lower() in {"gigahands", "hot3d", "being-h0", "dexwild", "thermohands", "wyih", "wiyh"}:
            return direct
        return input_path.parent / "annotations"

    def _annotation_source_id_for_scene(self, item: PlayableItem, scene: SceneOption) -> str:
        if item.dataset.lower() == "gigahands":
            return str(scene.selection.get("seq_name") or scene.scene_id)
        return scene.scene_id

    def _annotation_paths(self, annotations_dir: Path, source_id: str) -> dict[str, dict[str, Path]]:
        return {
            "qwen": {
                "raw": annotations_dir / f"pred_raw_clips_{source_id}.json",
                "steps": annotations_dir / f"pred_steps_{source_id}.json",
                "meta": annotations_dir / f"pred_run_meta_{source_id}.json",
                "eval": annotations_dir / f"pred_eval_{source_id}.json",
            },
            "gemma4": {
                "raw": annotations_dir / f"pred_raw_clips_{source_id}_gemma4.json",
                "steps": annotations_dir / f"pred_steps_{source_id}_gemma4.json",
                "meta": annotations_dir / f"pred_run_meta_{source_id}_gemma4.json",
                "eval": annotations_dir / f"pred_eval_{source_id}_gemma4.json",
            },
        }

    def _scene_context_for_annotation(self, item_id: str, scene_id: str | None) -> tuple[PlayableItem, SceneOption]:
        item = self._get_item(item_id)
        if item is None:
            raise ValueError(f"Unknown item: {item_id}")
        if not item.valid:
            raise ValueError(f"Config '{item_id}' is invalid: {item.error}")

        scene = self._get_scene_for_item(item, scene_id)
        if scene is None:
            raise ValueError(f"Unknown scene '{scene_id}' for config '{item_id}'")
        return item, scene

    def _annotation_can_run(self, item: PlayableItem) -> bool:
        return item.dataset.lower() in ANNOTATION_RUN_SUPPORTED_DATASETS

    def _annotation_env(
        self,
        item: PlayableItem,
        scene: SceneOption,
        runner: str,
        extra_env: dict[str, str] | None = None,
    ) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"

        dataset_name = item.dataset.lower()
        if dataset_name == "gigahands":
            env["GIGAHANDS_LABEL_LIBRARY_PATH"] = str(UNIVERSAL_LABEL_LIBRARY_PATH)
            env["MULTIDATASET_LABEL_LIBRARY_PATH"] = str(UNIVERSAL_LABEL_LIBRARY_PATH)
            env["GIGAHANDS_SCENE_NAME"] = str(scene.selection["seq_name"])
            env["GIGAHANDS_CAM_NAME"] = str(scene.selection["cam_name"])
            env["GIGAHANDS_FRAME_ID"] = str(scene.selection["frame_id"])
            if runner == "qwen":
                env.setdefault("GIGAHANDS_BATCH_SIZE", "1")
            if runner == "gemma4":
                env.setdefault("GIGAHANDS_GEMMA4_BATCH_SIZE", "1")
        else:
            env["MULTIDATASET_LABEL_LIBRARY_PATH"] = str(UNIVERSAL_LABEL_LIBRARY_PATH)
            env.setdefault("MULTIDATASET_BATCH_SIZE", "1")
            env.setdefault("MULTIDATASET_GEMMA4_BATCH_SIZE", "1")

        if extra_env:
            env.update(extra_env)
        return env

    def _annotation_process_args(
        self,
        item: PlayableItem,
        scene: SceneOption,
        runner: str,
    ) -> list[str]:
        dataset_name = item.dataset.lower()
        if dataset_name == "gigahands":
            required = ("seq_name", "cam_name", "frame_id")
            missing = [key for key in required if not str(scene.selection.get(key, "")).strip()]
            if missing:
                raise ValueError(
                    f"Scene '{scene.scene_id}' is missing required gigahands selection keys: {', '.join(missing)}"
                )
            if runner == "qwen":
                return [sys.executable, str(RUN_GIGAHANDS_QWEN_SCRIPT)]
            if runner == "gemma4":
                return [sys.executable, str(RUN_GIGAHANDS_GEMMA_SCRIPT)]
            raise ValueError(f"Unsupported runner: {runner}")

        source_id = self._annotation_source_id_for_scene(item, scene)
        script_path = RUN_MULTIDATASET_QWEN_SCRIPT if runner == "qwen" else RUN_MULTIDATASET_GEMMA_SCRIPT if runner == "gemma4" else None
        if script_path is None:
            raise ValueError(f"Unsupported runner: {runner}")
        return [
            sys.executable,
            str(script_path),
            "--config",
            str(item.path),
            "--source-id",
            source_id,
        ]

    def _annotation_runner_script_name(self, item: PlayableItem, runner: str) -> str:
        dataset_name = item.dataset.lower()
        if dataset_name == "gigahands":
            if runner == "qwen":
                return RUN_GIGAHANDS_QWEN_SCRIPT.name
            if runner == "gemma4":
                return RUN_GIGAHANDS_GEMMA_SCRIPT.name
        else:
            if runner == "qwen":
                return RUN_MULTIDATASET_QWEN_SCRIPT.name
            if runner == "gemma4":
                return RUN_MULTIDATASET_GEMMA_SCRIPT.name
        raise ValueError(f"Unsupported runner: {runner}")

    def get_annotation_status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "status": self.annotation_status,
                "runner": self.annotation_runner,
                "item_id": self.annotation_item_id,
                "scene_id": self.annotation_scene_id,
                "started_at": self.annotation_started_at,
                "process_running": self.annotation_process is not None and self.annotation_process.poll() is None,
                "last_error": self.annotation_last_error,
                "logs": list(self.annotation_log_lines),
            }

    def _append_annotation_log(self, line: str) -> None:
        line = line.rstrip()
        if not line:
            return
        with self.lock:
            self.annotation_log_lines.append(line)
        self.logger.info("[annotation] %s", line)

    def _read_annotation_output(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            self._append_annotation_log(line)

    def _watch_annotation_process(self, process: subprocess.Popen[str], runner: str) -> None:
        exit_code = process.wait()
        with self.lock:
            if self.annotation_process is process:
                self.annotation_status = "completed" if exit_code == 0 else "failed"
                if exit_code != 0 and self.annotation_last_error is None:
                    self.annotation_last_error = f"Annotation process exited with code {exit_code}"
                self.annotation_process = None
        self._append_annotation_log(f"{runner} annotation process exited with code {exit_code}")

    def stop_annotation(self) -> dict[str, Any]:
        with self.lifecycle_lock:
            with self.lock:
                process = self.annotation_process
            if process is None:
                return self.get_annotation_status()

            self._append_annotation_log("stopping current annotation process")
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._append_annotation_log("annotation process did not stop in time; killing it")
                process.kill()
                process.wait(timeout=5)
            finally:
                with self.lock:
                    if self.annotation_process is process:
                        self.annotation_process = None
                        self.annotation_status = "idle"
                        self.annotation_last_error = None
            return self.get_annotation_status()

    def start_annotation(
        self,
        item_id: str,
        *,
        scene_id: str | None,
        runner: str,
        model_id: str | None = None,
    ) -> dict[str, Any]:
        with self.lifecycle_lock:
            item, scene = self._scene_context_for_annotation(item_id, scene_id)
            if not self._annotation_can_run(item):
                raise ValueError(f"Automated annotation is not supported for dataset '{item.dataset}'")
            self.stop_annotation()

            extra_env: dict[str, str] = {}
            if item.dataset.lower() == "gigahands":
                if runner == "qwen" and model_id:
                    extra_env["GIGAHANDS_VLM_MODEL_ID"] = model_id
                if runner == "gemma4" and model_id:
                    extra_env["GIGAHANDS_GEMMA4_MODEL_ID"] = model_id
            else:
                if runner == "qwen" and model_id:
                    extra_env["MULTIDATASET_VLM_MODEL_ID"] = model_id
                if runner == "gemma4" and model_id:
                    extra_env["MULTIDATASET_GEMMA4_MODEL_ID"] = model_id

            process_args = self._annotation_process_args(item, scene, runner)
            process = subprocess.Popen(
                process_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(REPO_ROOT),
                env=self._annotation_env(item, scene, runner, extra_env=extra_env),
            )

            with self.lock:
                self.annotation_process = process
                self.annotation_status = "running"
                self.annotation_last_error = None
                self.annotation_started_at = utc_now()
                self.annotation_runner = runner
                self.annotation_item_id = item.item_id
                self.annotation_scene_id = scene.scene_id
                self.annotation_log_lines.clear()

        self._append_annotation_log(
            f"starting {runner} annotation for '{item.label}' scene '{scene.label}' using {self._annotation_runner_script_name(item, runner)}"
        )
        threading.Thread(target=self._read_annotation_output, args=(process,), daemon=True).start()
        threading.Thread(target=self._watch_annotation_process, args=(process, runner), daemon=True).start()
        return self.get_annotation_status()

    def _load_json_if_exists(self, path: Path) -> Any | None:
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _annotation_run_summary(
        self,
        raw_data: list[dict[str, Any]] | None,
        steps_data: list[dict[str, Any]] | None,
        meta_data: dict[str, Any] | None,
        eval_data: dict[str, Any] | None,
        *,
        label_vocab: set[str],
    ) -> dict[str, Any] | None:
        if raw_data is None or steps_data is None:
            return None
        if meta_data is None:
            num_clips = len(raw_data)
            meta_data = {
                "model_id": raw_data[0].get("model_id") if raw_data else None,
                "runner_name": "run_multidataset_vlm",
                "device": None,
                "device_name": None,
                "torch_dtype": None,
                "batch_size": None,
                "num_clips": num_clips,
                "total_time_seconds": None,
                "peak_vram_allocated_gib": None,
                "peak_vram_reserved_gib": None,
            }
        summary = compare_tool.summarize_run(raw_data, steps_data, meta_data, label_vocab=label_vocab)
        if eval_data and isinstance(eval_data, dict):
            summary["validity_report"] = eval_data.get("summary")
        return summary

    def get_annotation_data(self, item_id: str, *, scene_id: str | None) -> dict[str, Any]:
        item = self._get_item(item_id)
        if item is None:
            raise ValueError(f"Unknown item: {item_id}")
        if not item.valid:
            raise ValueError(f"Config '{item_id}' is invalid: {item.error}")
        scene = self._get_scene_for_item(item, scene_id)
        if scene is None:
            raise ValueError(f"Unknown scene '{scene_id}' for config '{item_id}'")

        source_id = self._annotation_source_id_for_scene(item, scene)
        annotations_dir = self._resolve_annotations_dir_for_item(item)
        paths = self._annotation_paths(annotations_dir, source_id)

        library = label_library_lib.load_label_library(UNIVERSAL_LABEL_LIBRARY_PATH)
        library_summary = label_library_lib.summarize_library(library)
        library_evaluation = label_library_lib.recompute_library_evaluations(library)
        label_vocab = compare_tool.label_vocab_from_library(library)

        runs: dict[str, Any] = {}
        loaded_runs: dict[str, dict[str, Any]] = {}
        for runner, runner_paths in paths.items():
            raw_data = self._load_json_if_exists(runner_paths["raw"])
            steps_data = self._load_json_if_exists(runner_paths["steps"])
            meta_data = self._load_json_if_exists(runner_paths["meta"])
            eval_data = self._load_json_if_exists(runner_paths["eval"])
            summary = self._annotation_run_summary(raw_data, steps_data, meta_data, eval_data, label_vocab=label_vocab)
            available = raw_data is not None and steps_data is not None
            runs[runner] = {
                "available": available,
                "paths": {key: str(value) for key, value in runner_paths.items()},
                "summary": summary,
                "raw_clip_count": len(raw_data) if isinstance(raw_data, list) else 0,
                "step_count": len(steps_data) if isinstance(steps_data, list) else 0,
                "meta": meta_data,
                "eval": eval_data,
                "recent_steps": (steps_data or [])[:6],
            }
            if available:
                loaded_runs[runner] = {
                    "raw": raw_data,
                    "steps": steps_data,
                    "meta": meta_data,
                }

        comparison = None
        if "qwen" in loaded_runs and "gemma4" in loaded_runs:
            comparison = {
                "left_name": "Qwen",
                "right_name": "Gemma 4",
                "left_summary": compare_tool.summarize_run(
                    loaded_runs["qwen"]["raw"],
                    loaded_runs["qwen"]["steps"],
                    loaded_runs["qwen"]["meta"],
                    label_vocab=label_vocab,
                ),
                "right_summary": compare_tool.summarize_run(
                    loaded_runs["gemma4"]["raw"],
                    loaded_runs["gemma4"]["steps"],
                    loaded_runs["gemma4"]["meta"],
                    label_vocab=label_vocab,
                ),
            }

        return {
            "supported": True,
            "item_id": item.item_id,
            "item_label": item.label,
            "scene_id": scene.scene_id,
            "scene_label": scene.label,
            "scene_name": source_id,
            "dataset": item.dataset,
            "can_run_automated": self._annotation_can_run(item),
            "selection": scene.selection,
            "annotations_dir": str(annotations_dir),
            "library_path": str(UNIVERSAL_LABEL_LIBRARY_PATH),
            "library_summary": library_summary,
            "library_evaluation": library_evaluation,
            "runs": runs,
            "comparison": comparison,
        }

    def get_status(self) -> dict[str, Any]:
        with self.lock:
            grpc_url = f"rerun+http://localhost:{self.grpc_port}/proxy"
            viewer_url = f"http://localhost:{self.viewer_port}/?url={quote(grpc_url, safe='')}"
            return {
                "status": self.status,
                "current_item_id": self.current_item_id,
                "current_scene_id": self.current_scene_id,
                "started_at": self.started_at,
                "viewer_url": viewer_url,
                "grpc_url": grpc_url,
                "recording_path": self.current_recording,
                "last_error": self.last_error,
                "process_running": self.process is not None and self.process.poll() is None,
            }

    def get_logs(self) -> list[str]:
        with self.lock:
            return list(self.log_lines)

    def _append_log(self, line: str) -> None:
        line = line.rstrip()
        if not line:
            return
        with self.lock:
            self.log_lines.append(line)
        self.logger.info(line)

    def _read_process_output(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            self._append_log(line)

    def _watch_process(self, process: subprocess.Popen[str], item_id: str) -> None:
        exit_code = process.wait()
        with self.lock:
                if self.process is process:
                    self.status = "stopped" if exit_code == 0 else "failed"
                    if exit_code != 0 and self.last_error is None:
                        self.last_error = f"Dashboard process exited with code {exit_code}"
                    self.process = None
                if exit_code == 0:
                    self.current_item_id = None
                    self.current_scene_id = None
        self._append_log(f"dashboard process for '{item_id}' exited with code {exit_code}")

    def _stop_current_locked(self) -> None:
        with self.lock:
            process = self.process
            if process is None:
                return

            self._append_log("stopping current dashboard process")
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._append_log("dashboard process did not stop in time; killing it")
                process.kill()
                process.wait(timeout=5)
            finally:
                    if self.process is process:
                        self.process = None
                        self.status = "idle"
                        self.current_item_id = None
                        self.current_scene_id = None
                        self.current_recording = None
                        self.started_at = None
            self._wait_for_ports_to_release()

    def stop_current(self) -> None:
        with self.lifecycle_lock:
            self._stop_current_locked()

    def start_item(self, item_id: str, *, scene_id: str | None = None, save_recording: bool = False) -> dict[str, Any]:
        with self.lifecycle_lock:
            item = next((item for item in self.items if item.item_id == item_id), None)
            if item is None:
                raise ValueError(f"Unknown item: {item_id}")
            if not item.valid:
                raise ValueError(f"Config '{item_id}' is invalid: {item.error}")
            scene = self._get_scene_for_item(item, scene_id)
            if scene_id is not None and scene is None:
                raise ValueError(f"Unknown scene '{scene_id}' for config '{item_id}'")

            self._stop_current_locked()

            recording_path = self.outputs_dir / f"{item.path.stem}.rrd" if save_recording else None
            cmd = [
                sys.executable,
                str(SERVE_SCRIPT),
                "--config",
                str(item.path),
                "--web-port",
                str(self.viewer_port),
                "--grpc-port",
                str(self.grpc_port),
                "--keep-alive",
            ]
            if scene is not None:
                cmd.extend(["--selection-json", json.dumps(scene.selection, ensure_ascii=False)])
                if scene.dataset_options:
                    cmd.extend(["--dataset-options-json", json.dumps(scene.dataset_options, ensure_ascii=False)])
            if recording_path is not None:
                cmd.extend(["--save-recording", str(recording_path)])

            env = os.environ.copy()
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(REPO_ROOT),
                env=env,
            )

            with self.lock:
                self.process = process
                self.current_item_id = item_id
                self.current_scene_id = scene.scene_id if scene is not None else None
                self.current_recording = str(recording_path) if recording_path is not None else None
                self.status = "starting"
                self.started_at = utc_now()
                self.last_error = None
                self.log_lines.clear()

        scene_suffix = f" (scene: {scene.label})" if scene is not None else ""
        self._append_log(f"starting dashboard for '{item_id}' using {item.path.name}{scene_suffix}")
        if recording_path is None:
            self._append_log("live viewer mode enabled; not saving a .rrd recording")
        else:
            self._append_log(f"recording output enabled at {recording_path}")
        threading.Thread(target=self._read_process_output, args=(process,), daemon=True).start()
        threading.Thread(target=self._watch_process, args=(process, item_id), daemon=True).start()

        def _mark_running() -> None:
            with self.lock:
                if self.process is process and process.poll() is None:
                    self.status = "running"

        threading.Timer(1.0, _mark_running).start()
        return self.get_status()


class DashboardAppHandler(BaseHTTPRequestHandler):
    server_version = "DashboardApp/0.1"

    @property
    def app_state(self) -> DashboardAppState:
        return self.server.app_state  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        self.app_state.logger.info("http %s - %s", self.address_string(), fmt % args)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return

        mime_type, _ = mimetypes.guess_type(path.name)
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        try:
            if self.path == "/" or self.path == "/index.html":
                self._send_file(APP_UI_DIR / "index.html")
                return
            if self.path == "/styles.css":
                self._send_file(APP_UI_DIR / "styles.css")
                return
            if self.path == "/app.js":
                self._send_file(APP_UI_DIR / "app.js")
                return
            if self.path.startswith("/assets/"):
                rel = self.path.removeprefix("/assets/")
                self._send_file(APP_UI_DIR / "assets" / rel)
                return
            if self.path == "/api/items":
                self._send_json({"items": self.app_state.list_items()})
                return
            if self.path == "/api/recordings":
                self._send_json({"items": self.app_state.list_recordings()})
                return
            if self.path == "/api/status":
                self._send_json(self.app_state.get_status())
                return
            if self.path == "/api/logs":
                self._send_json({"logs": self.app_state.get_logs()})
                return
            if self.path.startswith("/api/annotation/status"):
                self._send_json(self.app_state.get_annotation_status())
                return
            if self.path.startswith("/api/annotation/data?"):
                query = parse_qs(urlparse(self.path).query)
                item_id = query.get("item_id", [""])[0]
                scene_id = query.get("scene_id", [""])[0] or None
                if not item_id:
                    self._send_json({"error": "item_id is required"}, status=400)
                    return
                self._send_json(self.app_state.get_annotation_data(item_id, scene_id=scene_id))
                return
            if self.path.startswith("/api/download?"):
                rel_path = parse_qs(urlparse(self.path).query).get("path", [None])[0]
                if rel_path is None:
                    self._send_json({"error": "path is required"}, status=400)
                    return
                target = (REPO_ROOT / rel_path).resolve()
                if REPO_ROOT not in target.parents and target != REPO_ROOT:
                    self._send_json({"error": "invalid path"}, status=400)
                    return
                self._send_file(target)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
        except Exception as exc:
            self.app_state._append_log(f"http GET failed: {exc}")
            self._send_json({"error": str(exc)}, status=500)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body"}, status=400)
            return

        if self.path == "/api/open":
            item_id = str(payload.get("item_id", "")).strip()
            if not item_id:
                self._send_json({"error": "item_id is required"}, status=400)
                return
            scene_id = str(payload.get("scene_id", "")).strip() or None
            save_recording = bool(payload.get("save_recording", False))
            try:
                status = self.app_state.start_item(item_id, scene_id=scene_id, save_recording=save_recording)
                self._send_json(status)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
            return

        if self.path == "/api/stop":
            self.app_state.stop_current()
            self._send_json(self.app_state.get_status())
            return

        if self.path == "/api/annotation/run":
            item_id = str(payload.get("item_id", "")).strip()
            if not item_id:
                self._send_json({"error": "item_id is required"}, status=400)
                return
            scene_id = str(payload.get("scene_id", "")).strip() or None
            runner = str(payload.get("runner", "qwen")).strip().lower() or "qwen"
            model_id = str(payload.get("model_id", "")).strip() or None
            try:
                status = self.app_state.start_annotation(item_id, scene_id=scene_id, runner=runner, model_id=model_id)
                self._send_json(status)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
            return

        if self.path == "/api/annotation/stop":
            self._send_json(self.app_state.stop_annotation())
            return

        if self.path == "/api/annotation/evaluate-library":
            try:
                library = label_library_lib.load_label_library(UNIVERSAL_LABEL_LIBRARY_PATH)
                evaluation = label_library_lib.recompute_library_evaluations(library)
                label_library_lib.save_label_library(library, UNIVERSAL_LABEL_LIBRARY_PATH)
                self._send_json(
                    {
                        "ok": True,
                        "library_path": str(UNIVERSAL_LABEL_LIBRARY_PATH),
                        "summary": label_library_lib.summarize_library(library),
                        "evaluation": evaluation,
                    }
                )
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Basic web app in front of the Dockerized Rerun dashboard.")
    parser.add_argument("--app-port", type=int, default=8080, help="Port for the selection/logging UI.")
    parser.add_argument("--viewer-port", type=int, default=9090, help="Port for the Rerun web viewer.")
    parser.add_argument("--grpc-port", type=int, default=9876, help="Port for the Rerun gRPC server.")
    parser.add_argument("--config-dir", type=Path, default=REPO_ROOT / "configs", help="Directory of ready-to-play config files.")
    parser.add_argument("--outputs-dir", type=Path, default=REPO_ROOT / "outputs", help="Directory for logs and saved recordings.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.outputs_dir / "dashboard_app.log")
    state = DashboardAppState(
        config_dir=args.config_dir,
        outputs_dir=args.outputs_dir,
        viewer_port=args.viewer_port,
        grpc_port=args.grpc_port,
        logger=logger,
    )
    server = ThreadingHTTPServer(("0.0.0.0", args.app_port), DashboardAppHandler)
    server.app_state = state  # type: ignore[attr-defined]

    logger.info("dashboard app listening on http://0.0.0.0:%s", args.app_port)
    logger.info("configured viewer port=%s grpc port=%s", args.viewer_port, args.grpc_port)
    try:
        server.serve_forever()
    finally:
        state.stop_current()
        server.server_close()


if __name__ == "__main__":
    main()
