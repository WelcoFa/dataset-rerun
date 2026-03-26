from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import rerun as rr

from rerun_viz.config.schema import VizConfig
from rerun_viz.core import DashboardPanels, read_image_rgb_unicode_safe
from rerun_viz.datasets.base import DatasetAdapter


CAMERA_NAMES = [
    "lf_chest_fisheye",
    "rf_chest_fisheye",
    "ldl_hand_fisheye",
    "ldr_hand_fisheye",
    "rdl_hand_fisheye",
    "rdr_hand_fisheye",
]

HAND_CAMERA_POSE_DATASETS = {
    "ldl_hand_fisheye": "ldl_camera_pose_in_chest",
    "ldr_hand_fisheye": "ldr_camera_pose_in_chest",
    "rdl_hand_fisheye": "rdl_camera_pose_in_chest",
    "rdr_hand_fisheye": "rdr_camera_pose_in_chest",
}

LEFT_COLOR = np.array([0, 170, 255], dtype=np.uint8)
RIGHT_COLOR = np.array([255, 90, 90], dtype=np.uint8)
POINTCLOUD_FALLBACK_COLOR = np.array([180, 210, 255], dtype=np.uint8)


def decode_bytes(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return decode_bytes(value.item())
        if value.size == 1:
            return decode_bytes(value.reshape(-1)[0])
    return str(value)


def read_string_array(ds: h5py.Dataset) -> list[str]:
    return [decode_bytes(item) for item in ds[()]]


def read_scalar_string(ds: h5py.Dataset) -> str:
    return decode_bytes(ds[()])


def find_nearest_index(timestamps: np.ndarray, target: int) -> int:
    idx = int(np.searchsorted(timestamps, target))
    if idx <= 0:
        return 0
    if idx >= len(timestamps):
        return len(timestamps) - 1
    left = idx - 1
    return left if abs(int(timestamps[left]) - target) <= abs(int(timestamps[idx]) - target) else idx


def pose7_to_components(pose7: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pose7 = np.asarray(pose7, dtype=np.float32).reshape(7)
    translation = pose7[:3]
    quat_xyzw = pose7[3:7]
    norm = float(np.linalg.norm(quat_xyzw))
    if norm > 0:
        quat_xyzw = quat_xyzw / norm
    else:
        quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return translation, quat_xyzw


def matrix4x4_to_components(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mat = np.asarray(mat, dtype=np.float32).reshape(4, 4)
    rotation = mat[:3, :3]
    translation = mat[:3, 3]
    return translation, rotation


def try_extract_answer(cot_text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", cot_text, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def load_task_entry(task_json: Path, action_name: str) -> dict[str, Any] | None:
    if not task_json.exists():
        return None
    with open(task_json, "r", encoding="utf-8") as f:
        entries = json.load(f)
    for entry in entries:
        if entry.get("action_folder_path") == action_name:
            return entry
    return None


def load_laz_points(path: Path, max_points: int) -> tuple[np.ndarray, np.ndarray | None]:
    try:
        import laspy
    except ImportError as exc:
        raise RuntimeError(
            "Pointcloud loading needs `laspy[lazrs]`. Install it first, or re-run with --skip-pointcloud."
        ) from exc

    las = laspy.read(path)
    points = np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)]).astype(np.float32)
    if len(points) == 0:
        return points, None

    colors = None
    if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
        rgb = np.column_stack([np.asarray(las.red), np.asarray(las.green), np.asarray(las.blue)]).astype(np.float32)
        if rgb.max(initial=0.0) > 255.0:
            rgb = np.clip(rgb / 256.0, 0.0, 255.0)
        colors = rgb.astype(np.uint8)

    if max_points > 0 and len(points) > max_points:
        stride = max(1, len(points) // max_points)
        points = points[::stride][:max_points]
        if colors is not None:
            colors = colors[::stride][:max_points]

    return points, colors


def load_optional_steps(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def get_current_step(frame_idx: int, steps: list[dict[str, Any]]) -> dict[str, Any] | None:
    for step in steps:
        if int(step.get("start_frame", -1)) <= frame_idx <= int(step.get("end_frame", -1)):
            return step
    return None


def log_pose(path: str, translation: np.ndarray, quat_xyzw: np.ndarray, color: np.ndarray):
    rr.log(path, rr.Transform3D(translation=translation, rotation=rr.Quaternion(xyzw=quat_xyzw)))
    rr.log(f"{path}/origin", rr.Points3D([[0.0, 0.0, 0.0]], colors=[color], radii=0.015))


def log_matrix_pose(path: str, translation: np.ndarray, rotation: np.ndarray, color: np.ndarray):
    rr.log(path, rr.Transform3D(translation=translation, mat3x3=rotation))
    rr.log(f"{path}/origin", rr.Points3D([[0.0, 0.0, 0.0]], colors=[color], radii=0.012))


class WiyhAdapter(DatasetAdapter):
    name = "wiyh"

    def __init__(self, config: VizConfig):
        self.config = config
        self.base = "wiyh"
        self.viewer_name = "wiyh_action_viewer"
        self.last_frame_panels: DashboardPanels | None = None

    @staticmethod
    def detect(input_path: Path) -> bool:
        input_path = input_path.resolve()
        if input_path.is_dir() and (input_path / "dataset.hdf5").exists():
            return True
        if input_path.name.lower() == "wyih":
            return True
        return False

    def create_blueprint(self):
        import rerun.blueprint as rrb

        return rrb.Blueprint(
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.Grid(
                        rrb.Spatial2DView(origin=f"{self.base}/camera/lf_chest_fisheye", name="LF Chest"),
                        rrb.Spatial2DView(origin=f"{self.base}/camera/rf_chest_fisheye", name="RF Chest"),
                        rrb.Spatial2DView(origin=f"{self.base}/camera/ldl_hand_fisheye", name="LDL Hand"),
                        rrb.Spatial2DView(origin=f"{self.base}/camera/ldr_hand_fisheye", name="LDR Hand"),
                        rrb.Spatial2DView(origin=f"{self.base}/camera/rdl_hand_fisheye", name="RDL Hand"),
                        rrb.Spatial2DView(origin=f"{self.base}/camera/rdr_hand_fisheye", name="RDR Hand"),
                        grid_columns=2,
                    ),
                    rrb.Spatial3DView(origin=f"{self.base}/world", name="3D Scene"),
                ),
                rrb.Vertical(
                    rrb.TextDocumentView(origin=f"{self.base}/dashboard/summary/recording", name="Recording"),
                    rrb.TextDocumentView(origin=f"{self.base}/dashboard/summary/frame", name="Frame"),
                    rrb.TextDocumentView(origin=f"{self.base}/dashboard/semantic/main_task", name="Task"),
                    rrb.TextDocumentView(origin=f"{self.base}/dashboard/details/interaction", name="Interaction"),
                    rrb.TimeSeriesView(origin=f"{self.base}/plots", name="State Plots"),
                ),
            ),
            collapse_panels=True,
        )

    def load(self):
        options = self.config.dataset_options
        input_path = self.config.input.resolve()
        if (input_path / "dataset.hdf5").exists():
            self.action_dir = input_path
            self.task_json = Path(options.get("task_json", input_path.parent / "task.json"))
        else:
            self.action_dir = Path(options.get("action_dir", input_path / "action_000")).resolve()
            self.task_json = Path(options.get("task_json", input_path / "task.json")).resolve()

        self.stride = int(options.get("stride", 1))
        self.max_frames = int(options.get("max_frames", -1))
        self.max_points = int(options.get("max_points", 50000))
        self.skip_pointcloud = bool(options.get("skip_pointcloud", False))
        self.annotations_dir = Path(options.get("annotations_dir", self.action_dir.parent / "annotations"))
        self.pred_steps_path = self.annotations_dir / f"pred_steps_{self.action_dir.name}.json"
        self.pred_raw_clips_path = self.annotations_dir / f"pred_raw_clips_{self.action_dir.name}.json"

        if self.stride <= 0:
            raise ValueError("WIYH stride must be >= 1")

        dataset_path = self.action_dir / "dataset.hdf5"
        if not dataset_path.exists():
            raise FileNotFoundError(f"dataset.hdf5 not found: {dataset_path}")

        self.task_entry = load_task_entry(self.task_json, self.action_dir.name)

        with h5py.File(dataset_path, "r") as f:
            self.task_description = read_scalar_string(f["meta/task_description"])
            self.cot_text = read_scalar_string(f["meta/cot"])
            self.atomic_descriptions = read_string_array(f["annotation/atomic_task_description/atomic_task_description"])
            self.atomic_statuses = read_string_array(f["annotation/atomic_task_status/atomic_task_status"])

            self.camera_streams = {}
            for camera_name in CAMERA_NAMES:
                self.camera_streams[camera_name] = {
                    "filepaths": read_string_array(f[f"observation/camera/{camera_name}/filepath"]),
                    "timestamps": np.asarray(f[f"observation/camera/{camera_name}/timestamp"][()], dtype=np.int64),
                }

            self.pointcloud_paths = read_string_array(f["observation/pointcloud/chest/filepath"])
            self.pointcloud_timestamps = np.asarray(f["observation/pointcloud/chest/timestamp"][()], dtype=np.int64)

            self.arm_timestamps = np.asarray(f["action/arm_status_feedback/timestamp"][()], dtype=np.int64)
            self.left_eef_in_chest = np.asarray(f["action/arm_status_feedback/left_eef_pose_in_chest"][()], dtype=np.float32)
            self.right_eef_in_chest = np.asarray(f["action/arm_status_feedback/right_eef_pose_in_chest"][()], dtype=np.float32)
            self.left_eef_mask = np.asarray(f["action/arm_status_feedback/left_eef_pose_mask"][()], dtype=np.int32)
            self.right_eef_mask = np.asarray(f["action/arm_status_feedback/right_eef_pose_mask"][()], dtype=np.int32)

            self.hand_camera_static = {}
            for camera_name in CAMERA_NAMES:
                calib_path = f"meta/calibration/{camera_name}"
                if calib_path in f:
                    translation, rotation = matrix4x4_to_components(f[f"{calib_path}/extrinsic"][()])
                    self.hand_camera_static[camera_name] = (translation, rotation)

            self.hand_camera_dynamic = {}
            for camera_name, dataset_name in HAND_CAMERA_POSE_DATASETS.items():
                self.hand_camera_dynamic[camera_name] = np.asarray(
                    f[f"action/arm_status_feedback/{dataset_name}"][()], dtype=np.float32
                )

            self.hand_timestamps = np.asarray(f["action/hand_status_feedback/timestamp"][()], dtype=np.int64)
            self.left_hand_joints = np.asarray(f["action/hand_status_feedback/left_hand_joint_angle"][()], dtype=np.float32)
            self.right_hand_joints = np.asarray(f["action/hand_status_feedback/right_hand_joint_angle"][()], dtype=np.float32)
            self.left_hand_mask = np.asarray(f["action/hand_status_feedback/left_hand_joint_angle_mask"][()], dtype=np.int32)
            self.right_hand_mask = np.asarray(f["action/hand_status_feedback/right_hand_joint_angle_mask"][()], dtype=np.int32)

        self.total_frames = min(
            len(self.pointcloud_paths),
            len(self.pointcloud_timestamps),
            len(self.atomic_descriptions),
            len(self.atomic_statuses),
            *(len(stream["filepaths"]) for stream in self.camera_streams.values()),
        )
        if self.total_frames <= 0:
            raise RuntimeError(f"No WIYH frames found in {self.action_dir}")
        if self.max_frames > 0:
            self.total_frames = min(self.total_frames, self.max_frames)

        self.cot_answer = try_extract_answer(self.cot_text)
        self.pred_steps = load_optional_steps(self.pred_steps_path)
        self.pred_raw_clips = load_optional_steps(self.pred_raw_clips_path)
        self.left_traj: list[np.ndarray] = []
        self.right_traj: list[np.ndarray] = []
        self.pointcloud_warning_emitted = False
        self.recording_summary = "\n".join(
            [
                "World In Your Hands action viewer",
                f"action_dir: {self.action_dir}",
                f"frames: {self.total_frames}",
                f"stride: {self.stride}",
                "modalities: 6 fisheye cameras + chest pointcloud",
                "robot state: arm_status_feedback + hand_status_feedback",
                *((
                    f"task_name_en: {self.task_entry.get('task_name_en', '')}",
                    f"task_description_en: {self.task_entry.get('task_description_en', '')}",
                ) if self.task_entry is not None else ()),
            ]
        )

    def log_static(self):
        if hasattr(rr, "ViewCoordinates"):
            rr.log(f"{self.base}/world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        rr.log(
            f"{self.base}/dashboard/summary/recording",
            rr.TextDocument(self.recording_summary, media_type="text/plain"),
            static=True,
        )

        rr.log(f"{self.base}/world/robot/chest", rr.Points3D([[0.0, 0.0, 0.0]], colors=[[255, 255, 255]], radii=0.02), static=True)
        for camera_name, (translation, rotation) in self.hand_camera_static.items():
            log_matrix_pose(
                f"{self.base}/world/sensors_static/{camera_name}",
                translation,
                rotation,
                np.array([190, 190, 190], dtype=np.uint8),
            )

        if self.skip_pointcloud:
            rr.log(
                f"{self.base}/dashboard/details/objects",
                rr.TextDocument("Pointcloud loading disabled via skip_pointcloud.", media_type="text/plain"),
                static=True,
            )

    def frames(self):
        for frame_idx in range(0, self.total_frames, self.stride):
            obs_ts_us = int(self.pointcloud_timestamps[frame_idx])
            rr.set_time("frame", sequence=frame_idx)
            rr.set_time("time", timestamp=np.datetime64(obs_ts_us, "us"))

            for camera_name in CAMERA_NAMES:
                rel_path = self.camera_streams[camera_name]["filepaths"][frame_idx]
                image_path = self.action_dir / Path(rel_path)
                rr.log(f"{self.base}/camera/{camera_name}", rr.Image(read_image_rgb_unicode_safe(image_path)))

            arm_idx = find_nearest_index(self.arm_timestamps, obs_ts_us)
            hand_idx = find_nearest_index(self.hand_timestamps, obs_ts_us)

            left_translation = None
            right_translation = None
            if self.left_eef_mask[arm_idx]:
                left_translation, left_quat = pose7_to_components(self.left_eef_in_chest[arm_idx])
                log_pose(f"{self.base}/world/robot/left_eef", left_translation, left_quat, LEFT_COLOR)
                self.left_traj.append(left_translation.copy())
                rr.log(f"{self.base}/world/robot/left_eef_traj", rr.LineStrips3D([np.asarray(self.left_traj, dtype=np.float32)]))
                rr.log(f"{self.base}/plots/left_eef_z_m", rr.Scalars([float(left_translation[2])]))

            if self.right_eef_mask[arm_idx]:
                right_translation, right_quat = pose7_to_components(self.right_eef_in_chest[arm_idx])
                log_pose(f"{self.base}/world/robot/right_eef", right_translation, right_quat, RIGHT_COLOR)
                self.right_traj.append(right_translation.copy())
                rr.log(f"{self.base}/world/robot/right_eef_traj", rr.LineStrips3D([np.asarray(self.right_traj, dtype=np.float32)]))
                rr.log(f"{self.base}/plots/right_eef_z_m", rr.Scalars([float(right_translation[2])]))

            for camera_name, dataset in self.hand_camera_dynamic.items():
                translation, quat_xyzw = pose7_to_components(dataset[arm_idx])
                log_pose(f"{self.base}/world/sensors_dynamic/{camera_name}", translation, quat_xyzw, np.array([160, 220, 160], dtype=np.uint8))

            if self.left_hand_mask[hand_idx]:
                rr.log(f"{self.base}/plots/left_hand_mean_abs_joint", rr.Scalars([float(np.mean(np.abs(self.left_hand_joints[hand_idx])))]))
            if self.right_hand_mask[hand_idx]:
                rr.log(f"{self.base}/plots/right_hand_mean_abs_joint", rr.Scalars([float(np.mean(np.abs(self.right_hand_joints[hand_idx])))]))

            pointcloud_path = self.action_dir / Path(self.pointcloud_paths[frame_idx])
            if not self.skip_pointcloud:
                try:
                    points_xyz, colors_rgb = load_laz_points(pointcloud_path, self.max_points)
                    if len(points_xyz) > 0:
                        rr.log(
                            f"{self.base}/world/pointcloud/chest",
                            rr.Points3D(
                                points_xyz,
                                colors=colors_rgb if colors_rgb is not None else POINTCLOUD_FALLBACK_COLOR,
                                radii=0.0025,
                            ),
                        )
                except RuntimeError as exc:
                    if not self.pointcloud_warning_emitted:
                        rr.log(
                            f"{self.base}/dashboard/details/objects",
                            rr.TextDocument(str(exc), media_type="text/plain"),
                            static=True,
                        )
                        self.pointcloud_warning_emitted = True

            objects = [*CAMERA_NAMES, "robot left eef", "robot right eef", "chest pointcloud"]
            interaction = self.atomic_statuses[frame_idx]
            if self.cot_answer:
                interaction = f"{interaction}\npredicted_next_subtask: {self.cot_answer}"

            semantic_step = get_current_step(frame_idx, self.pred_raw_clips) or get_current_step(frame_idx, self.pred_steps)
            main_task = self.task_description
            sub_task = self.atomic_descriptions[frame_idx]
            current_action = self.atomic_statuses[frame_idx]
            if semantic_step is not None:
                main_task = str(semantic_step.get("main_task", main_task))
                sub_task = str(semantic_step.get("sub_task", sub_task))
                current_action = str(semantic_step.get("current_action", current_action))
                interaction = str(semantic_step.get("interaction", interaction))
                semantic_objects = semantic_step.get("objects", [])
                if isinstance(semantic_objects, list) and semantic_objects:
                    objects = [str(item) for item in semantic_objects if str(item).strip()]

            self.last_frame_panels = DashboardPanels(
                recording_summary=self.recording_summary,
                frame_summary="\n".join(
                    [
                        f"frame_index: {frame_idx}",
                        f"observation_timestamp_us: {obs_ts_us}",
                        f"atomic_task_description: {self.atomic_descriptions[frame_idx]}",
                        f"atomic_task_status: {self.atomic_statuses[frame_idx]}",
                        f"arm_state_index: {arm_idx}",
                        f"hand_state_index: {hand_idx}",
                        f"pointcloud_file: {pointcloud_path.name}",
                    ]
                ),
                main_task=main_task,
                sub_task=sub_task,
                current_action=current_action,
                interaction=interaction,
                objects=objects,
            )
            yield self.last_frame_panels

    def log_panels(self, panels):
        rr.log(f"{self.base}/dashboard/summary/recording", rr.TextDocument(panels.recording_summary))
        rr.log(f"{self.base}/dashboard/summary/frame", rr.TextDocument(panels.frame_summary))
        rr.log(f"{self.base}/dashboard/semantic/main_task", rr.TextDocument(panels.main_task))
        rr.log(f"{self.base}/dashboard/semantic/sub_task", rr.TextDocument(panels.sub_task))
        rr.log(f"{self.base}/dashboard/semantic/current_action", rr.TextDocument(panels.current_action))
        rr.log(f"{self.base}/dashboard/details/interaction", rr.TextDocument(panels.interaction))
        rr.log(f"{self.base}/dashboard/details/objects", rr.TextDocument("\n".join(f"- {item}" for item in panels.objects)))
