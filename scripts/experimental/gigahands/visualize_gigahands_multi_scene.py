import json
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import trimesh
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp


HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

USE_SMOOTHER = True
SMOOTH_WINDOW = 9

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[2]
DATA_ROOT = REPO_ROOT / "data" / "gigahands"
GIGAHANDS_ROOT = DATA_ROOT / "gigahands_demo_all"
SUPPORTED_MESH_EXTENSIONS = (".obj", ".ply", ".glb", ".gltf", ".stl")


def resolve_mesh_path(mesh_root: Path) -> Path:
    if mesh_root.is_file():
        return mesh_root

    for ext in SUPPORTED_MESH_EXTENSIONS:
        matches = sorted(mesh_root.rglob(f"*{ext}"))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"No mesh file found under {mesh_root}. "
        f"Expected one of: {', '.join(SUPPORTED_MESH_EXTENSIONS)}"
    )


SCENES = [
    {
        "name": "scene_tea",
        "video_path": GIGAHANDS_ROOT / "hand_pose" / "p36-tea-0010" / "rgb_vid" / "brics-odroid-010_cam0" / "brics-odroid-010_cam0_1727030430697198.mp4",
        "left_2d_path": GIGAHANDS_ROOT / "hand_pose" / "p36-tea-0010" / "keypoints_2d" / "left" / "010" / "brics-odroid-010_cam0_1727030430697198.jsonl",
        "right_2d_path": GIGAHANDS_ROOT / "hand_pose" / "p36-tea-0010" / "keypoints_2d" / "right" / "010" / "brics-odroid-010_cam0_1727030430697198.jsonl",
        "left_3d_path": GIGAHANDS_ROOT / "hand_pose" / "p36-tea-0010" / "keypoints_3d" / "010" / "left.jsonl",
        "right_3d_path": GIGAHANDS_ROOT / "hand_pose" / "p36-tea-0010" / "keypoints_3d" / "010" / "right.jsonl",
        "mesh_path": GIGAHANDS_ROOT / "object_pose" / "p36-tea-0010" / "mesh",
        "pose_path": GIGAHANDS_ROOT / "object_pose" / "p36-tea-0010" / "pose" / "optimized_pose.json",
    },
    {
        "name": "scene_boxing",
        "video_path": GIGAHANDS_ROOT / "hand_pose" / "p41-boxing-0021" / "rgb_vid" / "brics-odroid-001_cam0" / "brics-odroid-001_cam0_1726962101790659.mp4",
        "left_2d_path": GIGAHANDS_ROOT / "hand_pose" / "p41-boxing-0021" / "keypoints_2d" / "left" / "021" / "brics-odroid-001_cam0_1726962101790659.jsonl",
        "right_2d_path": GIGAHANDS_ROOT / "hand_pose" / "p41-boxing-0021" / "keypoints_2d" / "right" / "021" / "brics-odroid-001_cam0_1726962101790659.jsonl",
        "left_3d_path": GIGAHANDS_ROOT / "hand_pose" / "p41-boxing-0021" / "keypoints_3d" / "021" / "left.jsonl",
        "right_3d_path": GIGAHANDS_ROOT / "hand_pose" / "p41-boxing-0021" / "keypoints_3d" / "021" / "right.jsonl",
        "mesh_path": GIGAHANDS_ROOT / "object_pose" / "p41-boxing-0021" / "mesh",
        "pose_path": GIGAHANDS_ROOT / "object_pose" / "p41-boxing-0021" / "pose" / "optimized_pose.json",
    },
]

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_2d(path):
    raw = load_jsonl(path)
    out = []
    for x in raw:
        arr = np.asarray(x, dtype=np.float32).reshape(-1, 3)
        out.append(arr[:, :2])
    return out


def load_3d(path):
    raw = load_jsonl(path)
    out = []
    for x in raw:
        arr = np.asarray(x, dtype=np.float32).reshape(-1, 4)
        out.append(arr[:, :3])
    return out


def load_mesh(path):
    mesh = trimesh.load(resolve_mesh_path(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = list(mesh.geometry.values())[0]
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    return vertices, faces


def quat_to_rotmat(q):
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
    tracked_frames = sorted(int(k) for k in poses.keys())
    if not tracked_frames:
        return {}

    trans = []
    rots = []
    for fid in tracked_frames:
        pose = poses[f"{fid:06d}"]
        trans.append(np.asarray(pose["mesh_translation"], dtype=np.float32).reshape(-1, 3)[0])
        rots.append(np.asarray(pose["mesh_rotation"], dtype=np.float32).reshape(4,))

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
        idxs, trans, axis=0, kind="linear", fill_value="extrapolate"
    )(full_idx)

    rots_xyzw = np.stack([rots[:, 1], rots[:, 2], rots[:, 3], rots[:, 0]], axis=1)
    rot_obj = Rotation.from_quat(rots_xyzw)
    interp_rot = Slerp(idxs, rot_obj)(full_idx).as_quat()

    if USE_SMOOTHER and len(full_idx) >= SMOOTH_WINDOW:
        interp_t = moving_average_filter(interp_t, SMOOTH_WINDOW)
        interp_rot = moving_average_filter(interp_rot, SMOOTH_WINDOW)

    result = {}
    for i, fid in enumerate(full_idx):
        xyzw = interp_rot[i]
        wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)
        result[f"{fid:06d}"] = {
            "mesh_translation": np.asarray(interp_t[i], dtype=np.float32).tolist(),
            "mesh_rotation": wxyz.tolist(),
        }
    return result


scenes = []
for cfg in SCENES:
    with open(cfg["pose_path"], "r", encoding="utf-8") as f:
        raw_poses = json.load(f)

    mesh_vertices, mesh_faces = load_mesh(cfg["mesh_path"])

    cap = cv2.VideoCapture(str(cfg["video_path"]))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {cfg['video_path']}")

    scenes.append({
        "name": cfg["name"],
        "cap": cap,
        "left_2d": load_2d(cfg["left_2d_path"]),
        "right_2d": load_2d(cfg["right_2d_path"]),
        "left_3d": load_3d(cfg["left_3d_path"]),
        "right_3d": load_3d(cfg["right_3d_path"]),
        "mesh_vertices": mesh_vertices,
        "mesh_faces": mesh_faces,
        "poses": interpolate_poses(raw_poses),

    })
    print("loaded:", cfg["name"])
    print(" video =", cfg["video_path"].name)
    print(" left2d =", cfg["left_2d_path"].name)
    print(" mesh =", cfg["mesh_path"].name)
    print(" pose =", cfg["pose_path"].name)


rr.init("gigahands_two_scenes", spawn=True)

frame_idx = 0
while True:
    any_valid = False
    rr.set_time("frame", sequence=frame_idx)

    for scene in scenes:
        ret, frame = scene["cap"].read()
        if not ret:
            continue

        any_valid = True
        base = scene["name"]

        rr.log(f"{base}/camera/rgb", rr.Image(frame))

        if frame_idx < len(scene["left_2d"]):
            pts = scene["left_2d"][frame_idx]
            rr.log(f"{base}/camera/left_hand_2d/keypoints", rr.Points2D(pts))
            lines = [np.stack([pts[a], pts[b]], axis=0) for a, b in HAND_BONES if a < len(pts) and b < len(pts)]
            if lines:
                rr.log(f"{base}/camera/left_hand_2d/bones", rr.LineStrips2D(lines))

        if frame_idx < len(scene["right_2d"]):
            pts = scene["right_2d"][frame_idx]
            rr.log(f"{base}/camera/right_hand_2d/keypoints", rr.Points2D(pts))
            lines = [np.stack([pts[a], pts[b]], axis=0) for a, b in HAND_BONES if a < len(pts) and b < len(pts)]
            if lines:
                rr.log(f"{base}/camera/right_hand_2d/bones", rr.LineStrips2D(lines))

        if frame_idx < len(scene["left_3d"]):
            pts = scene["left_3d"][frame_idx]
            rr.log(f"{base}/world/left_hand_3d/keypoints", rr.Points3D(pts))
            lines = [np.stack([pts[a], pts[b]], axis=0) for a, b in HAND_BONES if a < len(pts) and b < len(pts)]
            if lines:
                rr.log(f"{base}/world/left_hand_3d/bones", rr.LineStrips3D(lines))

        if frame_idx < len(scene["right_3d"]):
            pts = scene["right_3d"][frame_idx]
            rr.log(f"{base}/world/right_hand_3d/keypoints", rr.Points3D(pts))
            lines = [np.stack([pts[a], pts[b]], axis=0) for a, b in HAND_BONES if a < len(pts) and b < len(pts)]
            if lines:
                rr.log(f"{base}/world/right_hand_3d/bones", rr.LineStrips3D(lines))

        pose_key = f"{frame_idx:06d}"
        if pose_key in scene["poses"]:
            pose = scene["poses"][pose_key]
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
                    vertex_positions=scene["mesh_vertices"],
                    triangle_indices=scene["mesh_faces"],
                ),
            )

    if not any_valid:
        break

    frame_idx += 1

for scene in scenes:
    scene["cap"].release()
