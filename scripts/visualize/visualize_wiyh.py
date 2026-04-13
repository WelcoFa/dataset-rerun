import argparse
import json
import re
from pathlib import Path
import sys
from typing import Any

import cv2
import h5py
import numpy as np
import rerun as rr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rerun_viz.core import read_image_rgb_unicode_safe


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ACTION_DIR = REPO_ROOT / "data" / "wyih" / "action_000"
DEFAULT_TASK_JSON = REPO_ROOT / "data" / "wyih" / "task.json"
BASE = "wiyh"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize one WIYH action folder in Rerun.")
    parser.add_argument(
        "--action-dir",
        type=Path,
        default=DEFAULT_ACTION_DIR,
        help="WIYH action directory containing dataset.hdf5, camera/, and pointcloud/.",
    )
    parser.add_argument(
        "--task-json",
        type=Path,
        default=DEFAULT_TASK_JSON,
        help="Path to WIYH task.json used for extra metadata.",
    )
    parser.add_argument(
        "--spawn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Spawn the Rerun viewer.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Log every Nth observation frame.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Maximum observation frames to log (-1 means all).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=50000,
        help="Maximum points to log from each LAZ frame after uniform subsampling.",
    )
    parser.add_argument(
        "--skip-pointcloud",
        action="store_true",
        help="Skip loading chest pointcloud frames.",
    )
    return parser.parse_args()


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


def log_pose(path: str, translation: np.ndarray, quat_xyzw: np.ndarray, color: np.ndarray):
    rr.log(path, rr.Transform3D(translation=translation, rotation=rr.Quaternion(xyzw=quat_xyzw)))
    rr.log(f"{path}/origin", rr.Points3D([[0.0, 0.0, 0.0]], colors=[color], radii=0.015))


def log_matrix_pose(path: str, translation: np.ndarray, rotation: np.ndarray, color: np.ndarray):
    rr.log(path, rr.Transform3D(translation=translation, mat3x3=rotation))
    rr.log(f"{path}/origin", rr.Points3D([[0.0, 0.0, 0.0]], colors=[color], radii=0.012))


def create_blueprint():
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Grid(
                    rrb.Spatial2DView(origin=f"{BASE}/camera/lf_chest_fisheye", name="LF Chest"),
                    rrb.Spatial2DView(origin=f"{BASE}/camera/rf_chest_fisheye", name="RF Chest"),
                    rrb.Spatial2DView(origin=f"{BASE}/camera/ldl_hand_fisheye", name="LDL Hand"),
                    rrb.Spatial2DView(origin=f"{BASE}/camera/ldr_hand_fisheye", name="LDR Hand"),
                    rrb.Spatial2DView(origin=f"{BASE}/camera/rdl_hand_fisheye", name="RDL Hand"),
                    rrb.Spatial2DView(origin=f"{BASE}/camera/rdr_hand_fisheye", name="RDR Hand"),
                    grid_columns=2,
                ),
                rrb.Spatial3DView(origin=f"{BASE}/world", name="3D Scene"),
            ),
            rrb.Vertical(
                rrb.TextDocumentView(origin=f"{BASE}/meta/recording", name="Recording"),
                rrb.TextDocumentView(origin=f"{BASE}/meta/task", name="Task"),
                rrb.TextDocumentView(origin=f"{BASE}/meta/frame", name="Frame"),
                rrb.TimeSeriesView(origin=f"{BASE}/plots", name="State Plots"),
            ),
        ),
        collapse_panels=True,
    )


def main():
    args = parse_args()
    if args.stride <= 0:
        raise ValueError("--stride must be >= 1")

    action_dir = args.action_dir.resolve()
    dataset_path = action_dir / "dataset.hdf5"
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset.hdf5 not found: {dataset_path}")

    task_entry = load_task_entry(args.task_json.resolve(), action_dir.name)

    with h5py.File(dataset_path, "r") as f:
        task_description = read_scalar_string(f["meta/task_description"])
        cot_text = read_scalar_string(f["meta/cot"])
        atomic_descriptions = read_string_array(f["annotation/atomic_task_description/atomic_task_description"])
        atomic_statuses = read_string_array(f["annotation/atomic_task_status/atomic_task_status"])

        camera_streams = {}
        for camera_name in CAMERA_NAMES:
            camera_streams[camera_name] = {
                "filepaths": read_string_array(f[f"observation/camera/{camera_name}/filepath"]),
                "timestamps": np.asarray(f[f"observation/camera/{camera_name}/timestamp"][()], dtype=np.int64),
            }

        pointcloud_paths = read_string_array(f["observation/pointcloud/chest/filepath"])
        pointcloud_timestamps = np.asarray(f["observation/pointcloud/chest/timestamp"][()], dtype=np.int64)

        arm_timestamps = np.asarray(f["action/arm_status_feedback/timestamp"][()], dtype=np.int64)
        left_eef_in_chest = np.asarray(f["action/arm_status_feedback/left_eef_pose_in_chest"][()], dtype=np.float32)
        right_eef_in_chest = np.asarray(f["action/arm_status_feedback/right_eef_pose_in_chest"][()], dtype=np.float32)
        left_eef_mask = np.asarray(f["action/arm_status_feedback/left_eef_pose_mask"][()], dtype=np.int32)
        right_eef_mask = np.asarray(f["action/arm_status_feedback/right_eef_pose_mask"][()], dtype=np.int32)

        hand_camera_static = {}
        for camera_name in CAMERA_NAMES:
            calib_path = f"meta/calibration/{camera_name}"
            if calib_path in f:
                translation, rotation = matrix4x4_to_components(f[f"{calib_path}/extrinsic"][()])
                hand_camera_static[camera_name] = (translation, rotation)

        hand_camera_dynamic = {}
        for camera_name, dataset_name in HAND_CAMERA_POSE_DATASETS.items():
            hand_camera_dynamic[camera_name] = np.asarray(
                f[f"action/arm_status_feedback/{dataset_name}"][()], dtype=np.float32
            )

        hand_timestamps = np.asarray(f["action/hand_status_feedback/timestamp"][()], dtype=np.int64)
        left_hand_joints = np.asarray(f["action/hand_status_feedback/left_hand_joint_angle"][()], dtype=np.float32)
        right_hand_joints = np.asarray(f["action/hand_status_feedback/right_hand_joint_angle"][()], dtype=np.float32)
        left_hand_mask = np.asarray(f["action/hand_status_feedback/left_hand_joint_angle_mask"][()], dtype=np.int32)
        right_hand_mask = np.asarray(f["action/hand_status_feedback/right_hand_joint_angle_mask"][()], dtype=np.int32)

    total_frames = min(
        len(pointcloud_paths),
        len(pointcloud_timestamps),
        len(atomic_descriptions),
        len(atomic_statuses),
        *(len(stream["filepaths"]) for stream in camera_streams.values()),
    )
    if total_frames <= 0:
        raise RuntimeError(f"No WIYH frames found in {action_dir}")
    if args.max_frames > 0:
        total_frames = min(total_frames, args.max_frames)

    rr.init("wiyh_action_viewer", spawn=args.spawn)
    rr.send_blueprint(create_blueprint())
    if hasattr(rr, "ViewCoordinates"):
        rr.log(f"{BASE}/world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    cot_answer = try_extract_answer(cot_text)
    recording_lines = [
        "World In Your Hands action viewer",
        f"action_dir: {action_dir}",
        f"frames: {total_frames}",
        f"stride: {args.stride}",
        "modalities: 6 fisheye cameras + chest pointcloud",
        "robot state: arm_status_feedback + hand_status_feedback",
    ]
    if task_entry is not None:
        recording_lines.extend(
            [
                f"task_name_en: {task_entry.get('task_name_en', '')}",
                f"task_description_en: {task_entry.get('task_description_en', '')}",
            ]
        )
    rr.log(f"{BASE}/meta/recording", rr.TextDocument("\n".join(recording_lines), media_type="text/plain"), static=True)

    task_lines = [f"task_description: {task_description}"]
    if task_entry is not None:
        task_lines.append(f"atomic_description: {task_entry.get('task_description_en', '')}")
    if cot_answer:
        task_lines.append(f"predicted_next_subtask: {cot_answer}")
    rr.log(f"{BASE}/meta/task", rr.TextDocument("\n".join(task_lines), media_type="text/plain"), static=True)

    if args.skip_pointcloud:
        rr.log(
            f"{BASE}/meta/pointcloud_note",
            rr.TextDocument("Pointcloud loading disabled via --skip-pointcloud.", media_type="text/plain"),
            static=True,
        )

    rr.log(f"{BASE}/world/robot/chest", rr.Points3D([[0.0, 0.0, 0.0]], colors=[[255, 255, 255]], radii=0.02), static=True)

    for camera_name, (translation, rotation) in hand_camera_static.items():
        log_matrix_pose(
            f"{BASE}/world/sensors_static/{camera_name}",
            translation,
            rotation,
            np.array([190, 190, 190], dtype=np.uint8),
        )

    left_traj: list[np.ndarray] = []
    right_traj: list[np.ndarray] = []
    pointcloud_warning_emitted = False

    for frame_idx in range(0, total_frames, args.stride):
        obs_ts_us = int(pointcloud_timestamps[frame_idx])
        rr.set_time("frame", sequence=frame_idx)
        rr.set_time("time", timestamp=np.datetime64(obs_ts_us, "us"))

        for camera_name in CAMERA_NAMES:
            rel_path = camera_streams[camera_name]["filepaths"][frame_idx]
            image_path = action_dir / Path(rel_path)
            rr.log(f"{BASE}/camera/{camera_name}", rr.Image(read_image_rgb_unicode_safe(image_path)))

        arm_idx = find_nearest_index(arm_timestamps, obs_ts_us)
        hand_idx = find_nearest_index(hand_timestamps, obs_ts_us)

        if left_eef_mask[arm_idx]:
            left_translation, left_quat = pose7_to_components(left_eef_in_chest[arm_idx])
            log_pose(f"{BASE}/world/robot/left_eef", left_translation, left_quat, LEFT_COLOR)
            left_traj.append(left_translation.copy())
            rr.log(
                f"{BASE}/world/robot/left_eef_traj",
                rr.LineStrips3D([np.asarray(left_traj, dtype=np.float32)]),
            )
            rr.log(f"{BASE}/plots/left_eef_z_m", rr.Scalars([float(left_translation[2])]))

        if right_eef_mask[arm_idx]:
            right_translation, right_quat = pose7_to_components(right_eef_in_chest[arm_idx])
            log_pose(f"{BASE}/world/robot/right_eef", right_translation, right_quat, RIGHT_COLOR)
            right_traj.append(right_translation.copy())
            rr.log(
                f"{BASE}/world/robot/right_eef_traj",
                rr.LineStrips3D([np.asarray(right_traj, dtype=np.float32)]),
            )
            rr.log(f"{BASE}/plots/right_eef_z_m", rr.Scalars([float(right_translation[2])]))

        for camera_name, dataset in hand_camera_dynamic.items():
            translation, quat_xyzw = pose7_to_components(dataset[arm_idx])
            log_pose(
                f"{BASE}/world/sensors_dynamic/{camera_name}",
                translation,
                quat_xyzw,
                np.array([160, 220, 160], dtype=np.uint8),
            )

        if left_hand_mask[hand_idx]:
            left_joint_mean = float(np.mean(np.abs(left_hand_joints[hand_idx])))
            rr.log(f"{BASE}/plots/left_hand_mean_abs_joint", rr.Scalars([left_joint_mean]))

        if right_hand_mask[hand_idx]:
            right_joint_mean = float(np.mean(np.abs(right_hand_joints[hand_idx])))
            rr.log(f"{BASE}/plots/right_hand_mean_abs_joint", rr.Scalars([right_joint_mean]))

        pointcloud_rel_path = pointcloud_paths[frame_idx]
        pointcloud_path = action_dir / Path(pointcloud_rel_path)
        if not args.skip_pointcloud:
            try:
                points_xyz, colors_rgb = load_laz_points(pointcloud_path, args.max_points)
                if len(points_xyz) > 0:
                    rr.log(
                        f"{BASE}/world/pointcloud/chest",
                        rr.Points3D(
                            points_xyz,
                            colors=colors_rgb if colors_rgb is not None else POINTCLOUD_FALLBACK_COLOR,
                            radii=0.0025,
                        ),
                    )
            except RuntimeError as exc:
                if not pointcloud_warning_emitted:
                    rr.log(
                        f"{BASE}/meta/pointcloud_note",
                        rr.TextDocument(str(exc), media_type="text/plain"),
                        static=True,
                    )
                    pointcloud_warning_emitted = True

        frame_lines = [
            f"frame_index: {frame_idx}",
            f"observation_timestamp_us: {obs_ts_us}",
            f"atomic_task_description: {atomic_descriptions[frame_idx]}",
            f"atomic_task_status: {atomic_statuses[frame_idx]}",
            f"arm_state_index: {arm_idx}",
            f"hand_state_index: {hand_idx}",
            f"pointcloud_file: {pointcloud_path.name}",
        ]
        for camera_name in CAMERA_NAMES:
            frame_lines.append(f"{camera_name}: {Path(camera_streams[camera_name]['filepaths'][frame_idx]).name}")
        rr.log(f"{BASE}/meta/frame", rr.TextDocument("\n".join(frame_lines), media_type="text/plain"))


if __name__ == "__main__":
    main()
