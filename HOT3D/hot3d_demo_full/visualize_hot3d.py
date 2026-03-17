import csv
import json
from pathlib import Path

import numpy as np
import rerun as rr
import trimesh


ROOT = Path(__file__).resolve().parent
SEQUENCE_NAME = "P0001_10a27bf7"

SEQUENCE_DIR = ROOT / SEQUENCE_NAME
HAND_DIR = SEQUENCE_DIR / "hand_data"
GT_DIR = SEQUENCE_DIR / "ground_truth"
OBJECT_MODELS_DIR = ROOT / "object_models"

FRAME_STRIDE = 5


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


def index_hand_rows(rows):
    out = {}
    for row in rows:
        ts = int(row["timestamp_ns"])
        out[ts] = row
    return out


def index_headset_rows(rows):
    out = {}
    for row in rows:
        ts = int(row["timestamp[ns]"])
        out[ts] = row
    return out


def log_hand_wrist(path_prefix: str, hand_pose: dict):
    wrist = hand_pose["wrist_xform"]
    t_xyz = np.asarray(wrist["t_xyz"], dtype=np.float32)

    rr.log(
        path_prefix,
        rr.Points3D(
            positions=[t_xyz],
            labels=[path_prefix.split("/")[-1]],
        ),
    )


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

# def log_object_mesh(path_prefix: str, mesh_info: dict, row: dict):
#     t_xyz = np.array(
#         [
#             float(row["t_wo_x[m]"]),
#             float(row["t_wo_y[m]"]),
#             float(row["t_wo_z[m]"]),
#         ],
#         dtype=np.float32,
#     )
#     q_wxyz = np.array(
#         [
#             float(row["q_wo_w"]),
#             float(row["q_wo_x"]),
#             float(row["q_wo_y"]),
#             float(row["q_wo_z"]),
#         ],
#         dtype=np.float32,
#     )
#
#     vertices_world = transform_points(mesh_info["vertices_local"], q_wxyz, t_xyz)
#
#     rr.log(
#         path_prefix,
#         rr.Mesh3D(
#             vertex_positions=vertices_world,
#             triangle_indices=mesh_info["faces"],
#         ),
#     )
#
#     rr.log(
#         f"{path_prefix}_center",
#         rr.Points3D(
#             positions=[t_xyz],
#             labels=[mesh_info["name"]],
#         ),
#     )
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

    # 缩放系数：先试 0.001 / 0.01 / 0.1
    scale = 0.001

    vertices_local = mesh_info["vertices_local"] * scale
    vertices_world = transform_points(vertices_local, q_wxyz, t_xyz)

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
            labels=[mesh_info["name"]],
        ),
    )

def main():
    rr.init("hot3d_full_viewer", spawn=True)

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
    mano_rows = load_jsonl(HAND_DIR / "mano_hand_pose_trajectory.jsonl")

    ts_to_objects = index_dynamic_objects(dynamic_rows)
    ts_to_hands = index_hand_rows(mano_rows)
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
                    f"recording_name: {metadata.get('recording_name', 'unknown')}",
                    f"participant_id: {metadata.get('participant_id', 'unknown')}",
                    f"headset: {metadata.get('headset', 'unknown')}",
                    f"num_frames: {len(all_timestamps)}",
                    f"num_objects: {len(object_meshes)}",
                    f"object_names: {metadata.get('object_names', [])}",
                ]
            )
        ),
    )

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
        rr.set_time("timestamp_ns", duration=float(ts) / 1e9)

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

        for row in current_object_rows:
            uid = row["object_uid"]
            if uid not in object_meshes:
                continue
            mesh_info = object_meshes[uid]
            log_object_mesh(f"world/objects/{mesh_info['name']}", mesh_info, row)

        if hand_row is not None:
            hand_poses = hand_row.get("hand_poses", {})
            if "0" in hand_poses:
                log_hand_wrist("world/hands/hand_0_wrist", hand_poses["0"])
            if "1" in hand_poses:
                log_hand_wrist("world/hands/hand_1_wrist", hand_poses["1"])

        if headset_row is not None:
            log_headset_pose("world/headset/current", headset_row)

        frame_counter += 1

    print("=" * 60)
    print("HOT3D full viewer loaded.")
    print(f"Sequence         : {SEQUENCE_NAME}")
    print(f"Frames loaded    : {frame_counter}")
    print(f"Objects loaded   : {len(object_meshes)}")
    print(f"Sequence dir     : {SEQUENCE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()