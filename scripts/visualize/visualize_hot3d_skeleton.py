import csv
import json
from pathlib import Path

import numpy as np
import rerun as rr
import trimesh


ROOT = Path(__file__).resolve().parents[1]
SEQUENCE_NAME = "P0001_10a27bf7"

SEQUENCE_DIR = ROOT / SEQUENCE_NAME
HAND_DIR = SEQUENCE_DIR / "hand_data"
GT_DIR = SEQUENCE_DIR / "ground_truth"
OBJECT_MODELS_DIR = ROOT / "object_models"

FRAME_STRIDE = 5
OBJECT_SCALE = 0.001
WRIST_AXIS_LENGTH = 0.05
SHOW_LABELS = False

# 近似手指骨长（米）
FINGER_LENGTHS = {
    "thumb":  [0.035, 0.028, 0.022],
    "index":  [0.040, 0.025, 0.020],
    "middle": [0.045, 0.030, 0.022],
    "ring":   [0.043, 0.028, 0.020],
    "pinky":  [0.035, 0.022, 0.018],
}

# 近似 MCP 基座在手腕局部坐标系中的偏移
FINGER_BASE_OFFSETS = {
    "thumb":  np.array([0.020, -0.020, 0.000], dtype=np.float32),
    "index":  np.array([0.015, -0.008, 0.000], dtype=np.float32),
    "middle": np.array([0.015,  0.000, 0.000], dtype=np.float32),
    "ring":   np.array([0.015,  0.008, 0.000], dtype=np.float32),
    "pinky":  np.array([0.012,  0.016, 0.000], dtype=np.float32),
}

FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]

def maybe_labels(labels):
    if SHOW_LABELS:
        return labels
    return None

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_csv(path: Path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def quat_wxyz_to_rotmat(q):
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n

    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z

    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float32,
    )


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def transform_points(points: np.ndarray, q_wxyz, t_xyz):
    R = quat_wxyz_to_rotmat(q_wxyz)
    t = np.asarray(t_xyz, dtype=np.float32)
    return (points @ R.T) + t


def load_mesh_as_single_trimesh(mesh_file: Path):
    mesh = trimesh.load(mesh_file, force="scene")

    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            return None
        mesh = trimesh.util.concatenate(geoms)

    if not isinstance(mesh, trimesh.Trimesh):
        return None

    if mesh.vertices is None or mesh.faces is None:
        return None

    return mesh


def load_object_meshes(object_models_dir: Path, metadata: dict):
    object_uid_to_name = dict(zip(metadata["object_uids"], metadata["object_names"]))
    object_uid_to_bop = dict(zip(metadata["object_uids"], metadata["object_bop_uids"]))

    meshes = {}

    for obj_uid, obj_name in object_uid_to_name.items():
        bop_id = int(object_uid_to_bop[obj_uid])
        mesh_file = object_models_dir / f"obj_{bop_id:06d}.glb"
        if not mesh_file.exists():
            print(f"[WARN] Missing mesh for {obj_name}: {mesh_file.name}")
            continue

        mesh = load_mesh_as_single_trimesh(mesh_file)
        if mesh is None:
            print(f"[WARN] Invalid mesh: {mesh_file}")
            continue

        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.uint32)

        center = vertices.mean(axis=0)
        vertices = vertices - center

        meshes[obj_uid] = {
            "name": obj_name,
            "bop_id": bop_id,
            "vertices_local": vertices,
            "faces": faces,
        }

    return meshes


def index_dynamic_objects(rows):
    ts_to_objects = {}
    for row in rows:
        ts = int(row["timestamp[ns]"])
        ts_to_objects.setdefault(ts, []).append(row)
    return ts_to_objects


def index_headset_rows(rows):
    out = {}
    for row in rows:
        ts = int(row["timestamp[ns]"])
        out[ts] = row
    return out


def index_hand_rows(rows):
    out = {}
    for row in rows:
        ts = row.get("timestamp_ns", row.get("timestamp[ns]", None))
        if ts is None:
            continue
        out[int(ts)] = row
    return out


def log_headset_pose(path_prefix: str, row: dict):
    t_xyz = np.array(
        [
            float(row["t_wo_x[m]"]),
            float(row["t_wo_y[m]"]),
            float(row["t_wo_z[m]"]),
        ],
        dtype=np.float32,
    )
    rr.log(
        path_prefix,
        rr.Points3D(
            positions=[t_xyz],
            labels=["headset"],
        ),
    )


def log_object_mesh(path_prefix: str, mesh_info: dict, row: dict):
    t_xyz = np.array(
        [
            float(row["t_wo_x[m]"]),
            float(row["t_wo_y[m]"]),
            float(row["t_wo_z[m]"]),
        ],
        dtype=np.float32,
    )
    q_wxyz = np.array(
        [
            float(row["q_wo_w"]),
            float(row["q_wo_x"]),
            float(row["q_wo_y"]),
            float(row["q_wo_z"]),
        ],
        dtype=np.float32,
    )

    vertices_world = transform_points(mesh_info["vertices_local"] * OBJECT_SCALE, q_wxyz, t_xyz)

    rr.log(
        path_prefix,
        rr.Mesh3D(
            vertex_positions=vertices_world,
            triangle_indices=mesh_info["faces"],
        ),
    )

    rr.log(
        f"{path_prefix}_center",
        rr.Points3D(
            positions=[t_xyz],
            labels=maybe_labels([mesh_info["name"]]),
        ),
    )


def try_extract_hand_poses(row: dict):
    if "hand_poses" in row and isinstance(row["hand_poses"], dict):
        return row["hand_poses"]

    hand_poses = {}
    if "left_hand" in row:
        hand_poses["0"] = row["left_hand"]
    if "right_hand" in row:
        hand_poses["1"] = row["right_hand"]
    return hand_poses


def infer_hand_name(hand_key: str, hand_pose: dict):
    side = str(hand_pose.get("hand_side", hand_pose.get("side", hand_key))).lower()
    if "left" in side or side == "0":
        return "left_hand"
    if "right" in side or side == "1":
        return "right_hand"
    return f"hand_{hand_key}"


def normalize_joint_angles(joint_angles):
    arr = np.asarray(joint_angles, dtype=np.float32).reshape(-1)
    # 目标是 20 个值：5根手指 × 4个角
    if arr.size < 20:
        arr = np.pad(arr, (0, 20 - arr.size))
    return arr[:20]


def build_approx_hand_skeleton(hand_pose: dict):
    wrist = hand_pose.get("wrist_xform", {})
    q_wxyz = wrist.get("q_wxyz", None)
    t_xyz = wrist.get("t_xyz", None)
    joint_angles = hand_pose.get("joint_angles", None)

    if q_wxyz is None or t_xyz is None:
        return None

    if joint_angles is None:
        joint_angles = np.zeros(20, dtype=np.float32)
    joint_angles = normalize_joint_angles(joint_angles)

    wrist_pos = np.asarray(t_xyz, dtype=np.float32)
    wrist_R = quat_wxyz_to_rotmat(np.asarray(q_wxyz, dtype=np.float32))

    joints_world = [wrist_pos]
    lines = []

    cursor = 0
    for finger in FINGER_ORDER:
        base_local = FINGER_BASE_OFFSETS[finger]
        seg_lengths = FINGER_LENGTHS[finger]

        a0 = joint_angles[cursor + 0]
        a1 = joint_angles[cursor + 1]
        a2 = joint_angles[cursor + 2]
        a3 = joint_angles[cursor + 3]
        cursor += 4

        # finger base
        base_world = wrist_pos + wrist_R @ base_local

        # finger 基础展开方向
        if finger == "thumb":
            local_R = wrist_R @ rot_z(-0.6) @ rot_y(0.5)
        elif finger == "index":
            local_R = wrist_R @ rot_z(-0.15)
        elif finger == "middle":
            local_R = wrist_R
        elif finger == "ring":
            local_R = wrist_R @ rot_z(0.12)
        else:  # pinky
            local_R = wrist_R @ rot_z(0.25)

        # MCP / PIP / DIP / TIP
        p0 = base_world
        R0 = local_R @ rot_z(a0) @ rot_x(-a1)
        p1 = p0 + R0 @ np.array([seg_lengths[0], 0.0, 0.0], dtype=np.float32)

        R1 = R0 @ rot_x(-a2)
        p2 = p1 + R1 @ np.array([seg_lengths[1], 0.0, 0.0], dtype=np.float32)

        R2 = R1 @ rot_x(-a3)
        p3 = p2 + R2 @ np.array([seg_lengths[2], 0.0, 0.0], dtype=np.float32)

        finger_points = [wrist_pos, p0, p1, p2, p3]
        joints_world.extend([p0, p1, p2, p3])
        lines.append(np.stack(finger_points, axis=0))

    # wrist axes
    x_end = wrist_pos + wrist_R[:, 0] * WRIST_AXIS_LENGTH
    y_end = wrist_pos + wrist_R[:, 1] * WRIST_AXIS_LENGTH
    z_end = wrist_pos + wrist_R[:, 2] * WRIST_AXIS_LENGTH

    return {
        "wrist_pos": wrist_pos,
        "joints_world": np.asarray(joints_world, dtype=np.float32),
        "lines": lines,
        "x_axis": np.stack([wrist_pos, x_end], axis=0),
        "y_axis": np.stack([wrist_pos, y_end], axis=0),
        "z_axis": np.stack([wrist_pos, z_end], axis=0),
    }


def log_hand_skeleton(path_prefix: str, hand_name: str, skel: dict):
    rr.log(
        path_prefix,
        rr.Points3D(
            positions=[skel["wrist_pos"]],
            labels=maybe_labels(["headset"]),
        ),
    )

    for i, line in enumerate(skel["lines"]):
        rr.log(
            f"{path_prefix}/finger_{i}",
            rr.LineStrips3D([line]),
        )

    rr.log(
        f"{path_prefix}/x_axis",
        rr.LineStrips3D([skel["x_axis"]]),
    )
    rr.log(
        f"{path_prefix}/y_axis",
        rr.LineStrips3D([skel["y_axis"]]),
    )
    rr.log(
        f"{path_prefix}/z_axis",
        rr.LineStrips3D([skel["z_axis"]]),
    )

    rr.log(
        f"{path_prefix}/wrist_point",
        rr.Points3D(
            positions=[skel["wrist_pos"]],
            labels=maybe_labels(["headset"]),
        ),
    )


def main():
    rr.init("hot3d_skeleton_viewer", spawn=True)

    if not SEQUENCE_DIR.exists():
        raise FileNotFoundError(f"Missing sequence dir: {SEQUENCE_DIR}")
    if not HAND_DIR.exists():
        raise FileNotFoundError(f"Missing hand_data dir: {HAND_DIR}")
    if not GT_DIR.exists():
        raise FileNotFoundError(f"Missing ground_truth dir: {GT_DIR}")
    if not OBJECT_MODELS_DIR.exists():
        raise FileNotFoundError(f"Missing object_models dir: {OBJECT_MODELS_DIR}")

    metadata = load_json(GT_DIR / "metadata.json")
    dynamic_rows = load_csv(GT_DIR / "dynamic_objects.csv")
    headset_rows = load_csv(GT_DIR / "headset_trajectory.csv")

    hand_file = HAND_DIR / "umetrack_hand_pose_trajectory.jsonl"
    if not hand_file.exists():
        raise FileNotFoundError(f"Missing file: {hand_file}")

    hand_rows = load_jsonl(hand_file)

    ts_to_objects = index_dynamic_objects(dynamic_rows)
    ts_to_hands = index_hand_rows(hand_rows)
    ts_to_headset = index_headset_rows(headset_rows)

    object_meshes = load_object_meshes(OBJECT_MODELS_DIR, metadata)

    all_timestamps = sorted(ts_to_objects.keys())
    if not all_timestamps:
        raise RuntimeError("No timestamps found in dynamic_objects.csv")

    rr.log(
        "summary/recording",
        rr.TextDocument(
            "\n".join(
                [
                    "HOT3D approximate hand skeleton viewer",
                    "Note: skeleton is approximated from wrist_xform + joint_angles.",
                    "It is not guaranteed to match the official UmeTrack skeleton exactly.",
                    f"recording_name: {metadata.get('recording_name', 'unknown')}",
                    f"participant_id: {metadata.get('participant_id', 'unknown')}",
                    f"headset: {metadata.get('headset', 'unknown')}",
                    f"num_frames: {len(all_timestamps)}",
                    f"num_objects: {len(object_meshes)}",
                    f"hand_file: {hand_file.name}",
                    f"object_names: {metadata.get('object_names', [])}",
                ]
            )
        ),
    )

    # object trajectories
    object_traj = {}
    for row in dynamic_rows:
        uid = row["object_uid"]
        xyz = [
            float(row["t_wo_x[m]"]),
            float(row["t_wo_y[m]"]),
            float(row["t_wo_z[m]"]),
        ]
        object_traj.setdefault(uid, []).append(xyz)

    for uid, pts in object_traj.items():
        if uid not in object_meshes:
            continue
        rr.log(
            f"world/trajectories/objects/{object_meshes[uid]['name']}",
            rr.Points3D(positions=np.asarray(pts, dtype=np.float32)),
        )

    # headset trajectory
    headset_pts = []
    for row in headset_rows:
        headset_pts.append(
            [
                float(row["t_wo_x[m]"]),
                float(row["t_wo_y[m]"]),
                float(row["t_wo_z[m]"]),
            ]
        )
    if headset_pts:
        rr.log(
            "world/trajectories/headset",
            rr.Points3D(positions=np.asarray(headset_pts, dtype=np.float32)),
        )

    frame_counter = 0
    for ts in all_timestamps[::FRAME_STRIDE]:
        rr.set_time("time_sec", duration=float(ts) / 1e9)

        current_object_rows = ts_to_objects.get(ts, [])
        hand_row = ts_to_hands.get(ts)
        headset_row = ts_to_headset.get(ts)

        rr.log(
            "summary/current_frame",
            rr.TextDocument(
                "\n".join(
                    [
                        f"frame_index: {frame_counter}",
                        f"timestamp_ns: {ts}",
                        f"time_sec: {float(ts) / 1e9:.3f}",
                        f"num_dynamic_objects: {len(current_object_rows)}",
                        f"has_hand_row: {hand_row is not None}",
                        f"has_headset_row: {headset_row is not None}",
                    ]
                )
            ),
        )

        # objects
        for row in current_object_rows:
            uid = row["object_uid"]
            if uid not in object_meshes:
                continue
            mesh_info = object_meshes[uid]
            log_object_mesh(f"world/objects/{mesh_info['name']}", mesh_info, row)

        # hands
        if hand_row is not None:
            hand_poses = try_extract_hand_poses(hand_row)
            for hand_key, hand_pose in hand_poses.items():
                if hand_pose is None:
                    continue
                hand_name = infer_hand_name(hand_key, hand_pose)
                skel = build_approx_hand_skeleton(hand_pose)
                if skel is not None:
                    log_hand_skeleton(f"world/hands/{hand_name}", hand_name, skel)

        # headset
        if headset_row is not None:
            log_headset_pose("world/headset/current", headset_row)

        frame_counter += 1

    print("=" * 60)
    print("HOT3D approximate hand skeleton viewer loaded.")
    print(f"Sequence         : {SEQUENCE_NAME}")
    print(f"Frames loaded    : {frame_counter}")
    print(f"Objects loaded   : {len(object_meshes)}")
    print(f"Hand source      : {hand_file}")
    print("=" * 60)
    print("Note: this hand skeleton is approximate, not guaranteed exact.")
    print("=" * 60)


if __name__ == "__main__":
    main()
