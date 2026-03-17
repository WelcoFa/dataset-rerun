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
SMOOTH_WINDOW = 9

# -------------------------
# Change these paths
# -------------------------
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

VLM_STEPS_PATH = Path(
    r"C:\Users\WelcoFa\Desktop\相能\rerun\gigahands\gigahands_vlm_steps.json"
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


def load_vlm_steps(path: Path):
    if not path.exists():
        print(f"[WARN] VLM step file not found: {path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("VLM steps JSON must be a list of step dicts.")

    for i, step in enumerate(data):
        if not all(k in step for k in ("start", "end", "text")):
            raise ValueError(
                f"Step #{i} must contain keys: start, end, text"
            )

    return data


# =========================
# Math helpers
# =========================

def quat_to_rotmat(q):
    """
    Input q is assumed to be wxyz.
    Convert to scipy xyzw, then to rotation matrix.
    """
    q = np.asarray(q, dtype=np.float32).reshape(4,)
    q_xyzw = np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)
    return Rotation.from_quat(q_xyzw).as_matrix().T.astype(np.float32)


def moving_average_filter(signal, window_size=5):
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    pad_len = window_size // 2
    padded = np.pad(signal, ((pad_len, pad_len), (0, 0)), mode="edge")
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)

    smoothed = np.array([
        np.convolve(padded[:, i], kernel, mode="valid")
        for i in range(signal.shape[1])
    ]).T

    return smoothed.squeeze()


def interpolate_poses(poses):
    """
    poses: dict keyed by '000000', '000001', ...
    each item contains:
      - mesh_translation
      - mesh_rotation  (assumed wxyz)
    """
    tracked_frames = sorted(int(k) for k in poses.keys())
    if not tracked_frames:
        return {}

    trans = []
    rots = []

    for fid in tracked_frames:
        pose = poses[f"{fid:06d}"]
        trans.append(
            np.asarray(pose["mesh_translation"], dtype=np.float32).reshape(-1, 3)[0]
        )
        rots.append(
            np.asarray(pose["mesh_rotation"], dtype=np.float32).reshape(4,)
        )

    trans = np.asarray(trans, dtype=np.float32)
    rots = np.asarray(rots, dtype=np.float32)
    idxs = np.asarray(tracked_frames, dtype=np.int32)

    if len(idxs) == 1:
        return {
            f"{idxs[0]:06d}": {
                "mesh_translation": trans[0].tolist(),
                "mesh_rotation": rots[0].tolist(),
            }
        }

    full_idx = np.arange(idxs[0], idxs[-1] + 1)

    interp_t = interp1d(
        idxs,
        trans,
        axis=0,
        kind="linear",
        fill_value="extrapolate",
    )(full_idx)

    # scipy expects xyzw
    rots_xyzw = np.stack([rots[:, 1], rots[:, 2], rots[:, 3], rots[:, 0]], axis=1)
    rot_obj = Rotation.from_quat(rots_xyzw)
    interp_rot_xyzw = Slerp(idxs, rot_obj)(full_idx).as_quat()

    if USE_SMOOTHER and len(full_idx) >= SMOOTH_WINDOW:
        interp_t = moving_average_filter(interp_t, SMOOTH_WINDOW)
        interp_rot_xyzw = moving_average_filter(interp_rot_xyzw, SMOOTH_WINDOW)

        # normalize quaternions after smoothing
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
# VLM helpers
# =========================

def get_current_step(frame_idx: int, steps):
    for i, step in enumerate(steps):
        if int(step["start"]) <= frame_idx <= int(step["end"]):
            return i, step
    return None, None


def format_step_text(step_idx, step):
    if step is None:
        return "No active VLM step"

    return (
        f"Step {step_idx + 1}\n"
        f"Frames: {step['start']} - {step['end']}\n"
        f"Text: {step['text']}"
    )


# =========================
# Main
# =========================

def main():
    print("Loading GigaHands scene...")
    print("VIDEO_PATH      =", VIDEO_PATH)
    print("LEFT_2D_PATH    =", LEFT_2D_PATH)
    print("RIGHT_2D_PATH   =", RIGHT_2D_PATH)
    print("LEFT_3D_PATH    =", LEFT_3D_PATH)
    print("RIGHT_3D_PATH   =", RIGHT_3D_PATH)
    print("MESH_PATH       =", MESH_PATH)
    print("POSE_PATH       =", POSE_PATH)
    print("VLM_STEPS_PATH  =", VLM_STEPS_PATH)

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

    left_2d = load_2d(LEFT_2D_PATH)
    right_2d = load_2d(RIGHT_2D_PATH)
    left_3d = load_3d(LEFT_3D_PATH)
    right_3d = load_3d(RIGHT_3D_PATH)
    mesh_vertices, mesh_faces = load_mesh(MESH_PATH)
    vlm_steps = load_vlm_steps(VLM_STEPS_PATH)

    with open(POSE_PATH, "r", encoding="utf-8") as f:
        raw_poses = json.load(f)
    poses = interpolate_poses(raw_poses)

    print(f"Loaded left_2d frames:  {len(left_2d)}")
    print(f"Loaded right_2d frames: {len(right_2d)}")
    print(f"Loaded left_3d frames:  {len(left_3d)}")
    print(f"Loaded right_3d frames: {len(right_3d)}")
    print(f"Loaded VLM steps:       {len(vlm_steps)}")
    print(f"Mesh vertices:          {len(mesh_vertices)}")
    print(f"Mesh faces:             {len(mesh_faces)}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    rr.init("gigahands_vlm", spawn=True)

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rr.set_time("frame", sequence=frame_idx)

            base = SCENE_NAME

            # -------------------------
            # RGB
            # -------------------------
            rr.log(f"{base}/camera/rgb", rr.Image(frame))

            # -------------------------
            # 2D hands
            # -------------------------
            if frame_idx < len(left_2d):
                pts = left_2d[frame_idx]
                rr.log(
                    f"{base}/camera/left_hand_2d/keypoints",
                    rr.Points2D(pts),
                )
                lines = [
                    np.stack([pts[a], pts[b]], axis=0)
                    for a, b in HAND_BONES
                    if a < len(pts) and b < len(pts)
                ]
                if lines:
                    rr.log(
                        f"{base}/camera/left_hand_2d/bones",
                        rr.LineStrips2D(lines),
                    )

            if frame_idx < len(right_2d):
                pts = right_2d[frame_idx]
                rr.log(
                    f"{base}/camera/right_hand_2d/keypoints",
                    rr.Points2D(pts),
                )
                lines = [
                    np.stack([pts[a], pts[b]], axis=0)
                    for a, b in HAND_BONES
                    if a < len(pts) and b < len(pts)
                ]
                if lines:
                    rr.log(
                        f"{base}/camera/right_hand_2d/bones",
                        rr.LineStrips2D(lines),
                    )

            # -------------------------
            # 3D hands
            # -------------------------
            if frame_idx < len(left_3d):
                pts = left_3d[frame_idx]
                rr.log(
                    f"{base}/world/left_hand_3d/keypoints",
                    rr.Points3D(pts),
                )
                lines = [
                    np.stack([pts[a], pts[b]], axis=0)
                    for a, b in HAND_BONES
                    if a < len(pts) and b < len(pts)
                ]
                if lines:
                    rr.log(
                        f"{base}/world/left_hand_3d/bones",
                        rr.LineStrips3D(lines),
                    )

            if frame_idx < len(right_3d):
                pts = right_3d[frame_idx]
                rr.log(
                    f"{base}/world/right_hand_3d/keypoints",
                    rr.Points3D(pts),
                )
                lines = [
                    np.stack([pts[a], pts[b]], axis=0)
                    for a, b in HAND_BONES
                    if a < len(pts) and b < len(pts)
                ]
                if lines:
                    rr.log(
                        f"{base}/world/right_hand_3d/bones",
                        rr.LineStrips3D(lines),
                    )

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
            # VLM current step
            # -------------------------
            step_idx, step = get_current_step(frame_idx, vlm_steps)

            if step is not None:
                rr.log(
                    f"{base}/vlm/current_step",
                    rr.TextLog(format_step_text(step_idx, step)),
                )

                rr.log(
                    f"{base}/vlm/current_step_id",
                    rr.TextLog(f"step_{step_idx + 1:02d}"),
                )
            else:
                rr.log(
                    f"{base}/vlm/current_step",
                    rr.TextLog("No active VLM step"),
                )

            frame_idx += 1

    finally:
        cap.release()

    print("Done.")


if __name__ == "__main__":
    main()