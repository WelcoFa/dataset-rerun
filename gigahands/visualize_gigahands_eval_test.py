import json
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import trimesh
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp


# =========================
# Config
# =========================
HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

USE_SMOOTHER = True
SMOOTH_WINDOW = 1

VIDEO_PATH = Path(
    r"C:\Users\WelcoFa\Desktop\相能\rerun\gigahands\gigahands_demo_all\hand_pose\p36-tea-0010\rgb_vid\brics-odroid-010_cam0\brics-odroid-010_cam0_1727030430697198.mp4"
)

LEFT_2D_PATH = Path(
    r"C:\Users\WelcoFa\Desktop\相能\rerun\gigahands\gigahands_demo_all\hand_pose\p36-tea-0010\keypoints_2d\left\010\brics-odroid-010_cam0_1727030430697198.jsonl"
)

RIGHT_2D_PATH = Path(
    r"C:\Users\WelcoFa\Desktop\相能\rerun\gigahands\gigahands_demo_all\hand_pose\p36-tea-0010\keypoints_2d\right\010\brics-odroid-010_cam0_1727030430697198.jsonl"
)

LEFT_3D_PATH = Path(
    r"C:\Users\WelcoFa\Desktop\相能\rerun\gigahands\gigahands_demo_all\hand_pose\p36-tea-0010\keypoints_3d\010\left.jsonl"
)

RIGHT_3D_PATH = Path(
    r"C:\Users\WelcoFa\Desktop\相能\rerun\gigahands\gigahands_demo_all\hand_pose\p36-tea-0010\keypoints_3d\010\right.jsonl"
)

MESH_PATH = Path(
    r"C:\Users\WelcoFa\Desktop\相能\dataset\gigahands\scans_publish\publish\0_tea\teapot_without_lid\teapot_without_lid.obj"
)

POSE_PATH = Path(
    r"C:\Users\WelcoFa\Desktop\相能\rerun\gigahands\gigahands_demo_all\object_pose\p36-tea-0010\pose\optimized_pose.json"
)

GT_STEPS_PATH = Path(
    r"C:\Users\WelcoFa\Desktop\相能\rerun\gigahands\annotations\gt_steps_p36-tea-0010.json"
)

PRED_STEPS_PATH = Path(
    r"C:\Users\WelcoFa\Desktop\相能\rerun\gigahands\annotations\pred_steps_p36-tea-0010.json"
)

SCENE_NAME = "scene_gigahands"


# =========================
# IO helpers
# =========================
def load_jsonl(path: Path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_2d(path: Path):
    raw = load_jsonl(path)
    out = []
    for x in raw:
        arr = np.asarray(x, dtype=np.float32).reshape(-1, 3)
        out.append(arr[:, :2])
    return out


def load_3d(path: Path):
    raw = load_jsonl(path)
    out = []
    for x in raw:
        arr = np.asarray(x, dtype=np.float32).reshape(-1, 4)
        out.append(arr[:, :3])
    return out


def load_mesh(path: Path):
    mesh = trimesh.load(path, process=False)

    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            raise RuntimeError(f"No geometry found in mesh scene: {path}")
        mesh = list(mesh.geometry.values())[0]

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    return vertices, faces


def load_steps(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Step file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list of step dicts.")

    for i, step in enumerate(data):
        for key in ["start", "end", "label", "text"]:
            if key not in step:
                raise ValueError(f"Step {i} in {path} missing key: {key}")

    return data


# =========================
# Math helpers
# =========================
def quat_to_rotmat(q_wxyz):
    q_wxyz = np.asarray(q_wxyz, dtype=np.float32).reshape(4,)
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float32)
    return Rotation.from_quat(q_xyzw).as_matrix().T


def moving_average_filter(arr: np.ndarray, window: int):
    if window <= 1 or len(arr) == 0:
        return arr.copy()

    out = np.zeros_like(arr, dtype=np.float32)
    half = window // 2

    for i in range(len(arr)):
        l = max(0, i - half)
        r = min(len(arr), i + half + 1)
        out[i] = np.mean(arr[l:r], axis=0)

    return out


def densify_poses(raw_poses: dict, total_frames: int):
    if not raw_poses:
        return {}

    frame_ids = sorted(int(k) for k in raw_poses.keys())

    ts = []
    rots_xyzw = []
    valid_ids = []

    for fid in frame_ids:
        pose = raw_poses[f"{fid:06d}"]
        t = np.asarray(pose["mesh_translation"], dtype=np.float32).reshape(3,)
        q_wxyz = np.asarray(pose["mesh_rotation"], dtype=np.float32).reshape(4,)
        q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float32)

        valid_ids.append(fid)
        ts.append(t)
        rots_xyzw.append(q_xyzw)

    valid_ids = np.asarray(valid_ids, dtype=np.int32)
    ts = np.asarray(ts, dtype=np.float32)
    rots_xyzw = np.asarray(rots_xyzw, dtype=np.float32)

    full_idx = np.arange(total_frames, dtype=np.int32)

    # -------------------------
    # Translation
    # -------------------------
    if len(valid_ids) == 1:
        interp_t = np.repeat(ts, total_frames, axis=0)
    else:
        interp_t = interp1d(
            valid_ids,
            ts,
            axis=0,
            kind="linear",
            fill_value="extrapolate",
            bounds_error=False,
        )(full_idx)

    # -------------------------
    # Rotation
    # Slerp cannot extrapolate, so clamp query times
    # -------------------------
    if len(valid_ids) == 1:
        interp_rot_xyzw = np.repeat(rots_xyzw, total_frames, axis=0)
    else:
        slerp = Slerp(valid_ids, Rotation.from_quat(rots_xyzw))

        query_idx = np.clip(full_idx, valid_ids[0], valid_ids[-1])
        interp_rot_xyzw = slerp(query_idx).as_quat()

    # -------------------------
    # Optional smoothing
    # -------------------------
    if USE_SMOOTHER and len(full_idx) >= SMOOTH_WINDOW and SMOOTH_WINDOW > 1:
        interp_t = moving_average_filter(interp_t, SMOOTH_WINDOW)
        interp_rot_xyzw = moving_average_filter(interp_rot_xyzw, SMOOTH_WINDOW)

        norms = np.linalg.norm(interp_rot_xyzw, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        interp_rot_xyzw = interp_rot_xyzw / norms

    result = {}
    for i, fid in enumerate(full_idx):
        xyzw = interp_rot_xyzw[i]
        wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)

        result[f"{fid:06d}"] = {
            "mesh_translation": np.asarray(interp_t[i], dtype=np.float32).tolist(),
            "mesh_rotation": wxyz.tolist(),
        }

    return result
# =========================
# Step helpers
# =========================
def get_current_step(frame_idx: int, steps):
    for i, step in enumerate(steps):
        if int(step["start"]) <= frame_idx <= int(step["end"]):
            return i, step
    return None, None


def temporal_iou(step_a, step_b):
    a_start = int(step_a["start"])
    a_end = int(step_a["end"])
    b_start = int(step_b["start"])
    b_end = int(step_b["end"])

    inter = max(0, min(a_end, b_end) - max(a_start, b_start) + 1)
    union = max(a_end, b_end) - min(a_start, b_start) + 1
    return inter / union if union > 0 else 0.0


def format_step_text(prefix, step_idx, step):
    if step is None:
        return f"{prefix}\nNo active step"

    return (
        f"{prefix} Step {step_idx + 1}\n"
        f"Frames: {step['start']} - {step['end']}\n"
        f"Label: {step['label']}\n"
        f"Text: {step['text']}"
    )


def format_compare_text(gt_idx, gt_step, pred_idx, pred_step):
    gt_text = "None"
    pred_text = "None"
    label_match = "N/A"
    iou_text = "N/A"

    if gt_step is not None:
        gt_text = (
            f"GT Step {gt_idx + 1}\n"
            f"Frames: {gt_step['start']} - {gt_step['end']}\n"
            f"Label: {gt_step['label']}\n"
            f"Text: {gt_step['text']}"
        )

    if pred_step is not None:
        pred_text = (
            f"Pred Step {pred_idx + 1}\n"
            f"Frames: {pred_step['start']} - {pred_step['end']}\n"
            f"Label: {pred_step['label']}\n"
            f"Text: {pred_step['text']}"
        )

    if gt_step is not None and pred_step is not None:
        label_match = str(gt_step["label"] == pred_step["label"])
        iou_text = f"{temporal_iou(gt_step, pred_step):.3f}"

    return (
        f"{gt_text}\n\n"
        f"{pred_text}\n\n"
        f"Label Match: {label_match}\n"
        f"Temporal IoU: {iou_text}"
    )


def log_step_summary(base: str, name: str, steps):
    for i, step in enumerate(steps):
        text = (
            f"Step {i + 1}\n"
            f"Frames: {step['start']} - {step['end']}\n"
            f"Label: {step['label']}\n"
            f"Text: {step['text']}"
        )
        rr.log(
            f"{base}/eval/{name}/step_{i + 1:02d}",
            rr.TextLog(text),
        )


# =========================
# Main
# =========================
def main():
    print("Loading GigaHands evaluation scene.")
    print("VIDEO_PATH      =", VIDEO_PATH)
    print("LEFT_2D_PATH    =", LEFT_2D_PATH)
    print("RIGHT_2D_PATH   =", RIGHT_2D_PATH)
    print("LEFT_3D_PATH    =", LEFT_3D_PATH)
    print("RIGHT_3D_PATH   =", RIGHT_3D_PATH)
    print("MESH_PATH       =", MESH_PATH)
    print("POSE_PATH       =", POSE_PATH)
    print("GT_STEPS_PATH   =", GT_STEPS_PATH)
    print("PRED_STEPS_PATH =", PRED_STEPS_PATH)

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")
    if not LEFT_2D_PATH.exists():
        raise FileNotFoundError(f"Left 2D path not found: {LEFT_2D_PATH}")
    if not RIGHT_2D_PATH.exists():
        raise FileNotFoundError(f"Right 2D path not found: {RIGHT_2D_PATH}")
    if not LEFT_3D_PATH.exists():
        raise FileNotFoundError(f"Left 3D path not found: {LEFT_3D_PATH}")
    if not RIGHT_3D_PATH.exists():
        raise FileNotFoundError(f"Right 3D path not found: {RIGHT_3D_PATH}")
    if not MESH_PATH.exists():
        raise FileNotFoundError(f"Mesh path not found: {MESH_PATH}")
    if not POSE_PATH.exists():
        raise FileNotFoundError(f"Pose path not found: {POSE_PATH}")
    if not GT_STEPS_PATH.exists():
        raise FileNotFoundError(f"GT steps path not found: {GT_STEPS_PATH}")
    if not PRED_STEPS_PATH.exists():
        raise FileNotFoundError(f"Pred steps path not found: {PRED_STEPS_PATH}")

    left_2d = load_2d(LEFT_2D_PATH)
    right_2d = load_2d(RIGHT_2D_PATH)
    left_3d = load_3d(LEFT_3D_PATH)
    right_3d = load_3d(RIGHT_3D_PATH)
    mesh_vertices, mesh_faces = load_mesh(MESH_PATH)
    gt_steps = load_steps(GT_STEPS_PATH)
    pred_steps = load_steps(PRED_STEPS_PATH)

    with open(POSE_PATH, "r", encoding="utf-8") as f:
        raw_poses = json.load(f)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = max(
            len(left_2d),
            len(right_2d),
            len(left_3d),
            len(right_3d),
        )

    print("FPS =", fps)
    print("TOTAL_FRAMES =", total_frames)

    poses = densify_poses(raw_poses, total_frames)

    rr.init("gigahands_eval", spawn=True)

    base = SCENE_NAME

    log_step_summary(base, "gt_all_steps", gt_steps)
    log_step_summary(base, "pred_all_steps", pred_steps)

    frame_idx = 0
    last_gt_idx = None
    last_pred_idx = None
    last_compare_key = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rr.set_time("frame", sequence=frame_idx)
            rr.set_time("video_time", duration=frame_idx / fps)


            rr.log(f"{base}/camera/rgb", rr.Image(frame))

            # -------------------------
            # 2D hands
            # -------------------------
            if frame_idx < len(left_2d):
                pts = left_2d[frame_idx]
                rr.log(f"{base}/camera/left_hand_2d/keypoints", rr.Points2D(pts))
                lines = [
                    np.stack([pts[a], pts[b]], axis=0)
                    for a, b in HAND_BONES
                    if a < len(pts) and b < len(pts)
                ]
                if lines:
                    rr.log(f"{base}/camera/left_hand_2d/bones", rr.LineStrips2D(lines))

            if frame_idx < len(right_2d):
                pts = right_2d[frame_idx]
                rr.log(f"{base}/camera/right_hand_2d/keypoints", rr.Points2D(pts))
                lines = [
                    np.stack([pts[a], pts[b]], axis=0)
                    for a, b in HAND_BONES
                    if a < len(pts) and b < len(pts)
                ]
                if lines:
                    rr.log(f"{base}/camera/right_hand_2d/bones", rr.LineStrips2D(lines))

            # -------------------------
            # 3D hands
            # -------------------------
            if frame_idx < len(left_3d):
                pts = left_3d[frame_idx]
                rr.log(f"{base}/world/left_hand_3d/keypoints", rr.Points3D(pts))
                lines = [
                    np.stack([pts[a], pts[b]], axis=0)
                    for a, b in HAND_BONES
                    if a < len(pts) and b < len(pts)
                ]
                if lines:
                    rr.log(f"{base}/world/left_hand_3d/bones", rr.LineStrips3D(lines))

            if frame_idx < len(right_3d):
                pts = right_3d[frame_idx]
                rr.log(f"{base}/world/right_hand_3d/keypoints", rr.Points3D(pts))
                lines = [
                    np.stack([pts[a], pts[b]], axis=0)
                    for a, b in HAND_BONES
                    if a < len(pts) and b < len(pts)
                ]
                if lines:
                    rr.log(f"{base}/world/right_hand_3d/bones", rr.LineStrips3D(lines))

            # -------------------------
            # Object pose + mesh
            # -------------------------
            pose_key = f"{frame_idx:06d}"
            if pose_key in poses:
                pose = poses[pose_key]
                t = np.asarray(pose["mesh_translation"], dtype=np.float32).reshape(3,)
                q = np.asarray(pose["mesh_rotation"], dtype=np.float32).reshape(4,)
                R = quat_to_rotmat(q)

                rr.log(
                    f"{base}/world/object",
                    rr.Transform3D(
                        translation=t,
                        mat3x3=R,
                    ),
                )

                rr.log(
                    f"{base}/world/object/mesh",
                    rr.Mesh3D(
                        vertex_positions=mesh_vertices,
                        triangle_indices=mesh_faces,
                    ),
                )

            # -------------------------
            # Evaluation: GT vs Pred
            # -------------------------
            gt_idx, gt_step = get_current_step(frame_idx, gt_steps)
            pred_idx, pred_step = get_current_step(frame_idx, pred_steps)

            if gt_idx != last_gt_idx:
                rr.log(
                    f"{base}/eval/gt_current_step",
                    rr.TextLog(format_step_text("GT", gt_idx, gt_step)),
                )
                rr.log(
                    f"{base}/eval/gt_step_id",
                    rr.TextLog(
                        f"gt_step_{gt_idx + 1:02d}" if gt_step is not None else "gt_none"
                    ),
                )
                last_gt_idx = gt_idx

            if pred_idx != last_pred_idx:
                rr.log(
                    f"{base}/eval/pred_current_step",
                    rr.TextLog(format_step_text("Pred", pred_idx, pred_step)),
                )
                rr.log(
                    f"{base}/eval/pred_step_id",
                    rr.TextLog(
                        f"pred_step_{pred_idx + 1:02d}" if pred_step is not None else "pred_none"
                    ),
                )
                last_pred_idx = pred_idx

            compare_key = (gt_idx, pred_idx)
            if compare_key != last_compare_key:
                rr.log(
                    f"{base}/eval/compare",
                    rr.TextLog(format_compare_text(gt_idx, gt_step, pred_idx, pred_step)),
                )
                last_compare_key = compare_key

            frame_idx += 1

    finally:
        cap.release()

    print("Done.")


if __name__ == "__main__":
    main()