import csv
import json
import inspect
from collections import namedtuple
from pathlib import Path

# =========================================================
# chumpy compatibility patch for Python 3.11+
# 必须放在 import smplx 之前
# =========================================================
if not hasattr(inspect, "getargspec"):
    ArgSpec = namedtuple("ArgSpec", ["args", "varargs", "keywords", "defaults"])

    def getargspec(func):
        spec = inspect.getfullargspec(func)
        return ArgSpec(
            spec.args,
            spec.varargs,
            spec.varkw,
            spec.defaults,
        )

    inspect.getargspec = getargspec

import numpy as np

# 旧 chumpy / smplx 兼容补丁
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "complex"):
    np.complex = complex
if not hasattr(np, "object"):
    np.object = object
if not hasattr(np, "str"):
    np.str = str
if not hasattr(np, "unicode"):
    np.unicode = str

import rerun as rr
import torch
import trimesh
import smplx


# =========================================================
# Config
# =========================================================
ROOT = Path(__file__).resolve().parents[1]
SEQUENCE_NAME = "P0001_10a27bf7"

SEQUENCE_DIR = ROOT / SEQUENCE_NAME
HAND_DIR = SEQUENCE_DIR / "hand_data"
GT_DIR = SEQUENCE_DIR / "ground_truth"
OBJECT_MODELS_DIR = ROOT / "object_models"
MANO_DIR = ROOT / "mano_models"

FRAME_STRIDE = 10
OBJECT_SCALE = 0.001
DEVICE = "cpu"
SHOW_LABELS = False


# =========================================================
# Helpers
# =========================================================
def maybe_labels(labels):
    return labels if SHOW_LABELS else None


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


def rotmat_to_axis_angle(R: np.ndarray) -> np.ndarray:
    trace = np.trace(R)
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-8:
        return np.zeros(3, dtype=np.float32)

    rx = R[2, 1] - R[1, 2]
    ry = R[0, 2] - R[2, 0]
    rz = R[1, 0] - R[0, 1]
    axis = np.array([rx, ry, rz], dtype=np.float32)
    axis = axis / (2.0 * np.sin(theta) + 1e-12)
    return axis * theta


def transform_points(points: np.ndarray, q_wxyz, t_xyz):
    R = quat_wxyz_to_rotmat(q_wxyz)
    t = np.asarray(t_xyz, dtype=np.float32)
    return (points @ R.T) + t


# =========================================================
# Mesh loading
# =========================================================
def load_mesh_as_single_trimesh(mesh_file: Path):
    mesh = trimesh.load(mesh_file, force="scene")

    # glb 很多时候读出来是 Scene，要先合并
    if isinstance(mesh, trimesh.Scene):
        geoms = []
        for g in mesh.geometry.values():
            if isinstance(g, trimesh.Trimesh):
                geoms.append(g)

        if not geoms:
            return None

        mesh = trimesh.util.concatenate(geoms)

    if not isinstance(mesh, trimesh.Trimesh):
        return None

    if mesh.vertices is None or mesh.faces is None:
        return None

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
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

        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.uint32)
        palette = [
            [1.0, 0.0, 0.0, 1.0],  # red
            [0.0, 1.0, 0.0, 1.0],  # green
            [0.0, 0.0, 1.0, 1.0],  # blue
            [1.0, 1.0, 0.0, 1.0],  # yellow
            [1.0, 0.0, 1.0, 1.0],  # magenta
            [0.0, 1.0, 1.0, 1.0],  # cyan
        ]

        albedo_factor = palette[len(meshes) % len(palette)]

        colors = None
        if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
            vc = np.asarray(mesh.visual.vertex_colors)
            print(mesh_file.name, "vertex_colors shape:", vc.shape, "dtype:", vc.dtype)
            if vc.ndim == 2 and vc.shape[0] == verts.shape[0] and vc.shape[1] >= 3:
                colors = vc[:, :4] if vc.shape[1] >= 4 else vc[:, :3]

                # 看看是不是几乎全白
                print("  min:", colors.min(axis=0), "max:", colors.max(axis=0))
        else:
            print(mesh_file.name, "no vertex_colors")

        # mesh 居中，后续再应用物体位姿
        center = verts.mean(axis=0)
        verts = verts - center

        meshes[obj_uid] = {
            "name": obj_name,
            "bop_id": bop_id,
            "vertices_local": verts,
            "faces": faces,
            "albedo_factor": albedo_factor,
        }

    return meshes


# =========================================================
# Index data
# =========================================================
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


# =========================================================
# MANO
# =========================================================
def create_mano_layer(is_rhand: bool):
    return smplx.create(
        model_path=str(MANO_DIR),
        model_type="mano",
        use_pca=False,
        is_rhand=is_rhand,
        flat_hand_mean=False,
        num_betas=10,
        batch_size=1,
    ).to(DEVICE)


def to_np(x):
    return np.asarray(x, dtype=np.float32)


def pick_first_existing(d: dict, keys):
    for k in keys:
        if k in d:
            return d[k]
    return None


def get_hand_pose_entry(hand_poses: dict, hand_key: str):
    if hand_key in hand_poses:
        return hand_poses[hand_key]
    return None


def extract_mano_params(hand_pose: dict):
    wrist = hand_pose.get("wrist_xform", {})
    q_wxyz = wrist.get("q_wxyz", [1, 0, 0, 0])
    t_xyz = wrist.get("t_xyz", [0, 0, 0])

    global_orient = rotmat_to_axis_angle(quat_wxyz_to_rotmat(to_np(q_wxyz)))

    betas = pick_first_existing(
        hand_pose,
        ["betas", "shape", "mano_shape", "shape_params"],
    )
    if betas is None:
        betas = np.zeros(10, dtype=np.float32)
    betas = to_np(betas).reshape(-1)
    if len(betas) < 10:
        betas = np.pad(betas, (0, 10 - len(betas)))
    betas = betas[:10]

    hand_pose_raw = pick_first_existing(
        hand_pose,
        ["hand_pose", "mano_pose", "pose", "pose_coeffs", "articulation"],
    )
    if hand_pose_raw is None:
        hand_pose_45 = np.zeros(45, dtype=np.float32)
    else:
        hand_pose_raw = to_np(hand_pose_raw).reshape(-1)
        if len(hand_pose_raw) == 45:
            hand_pose_45 = hand_pose_raw
        elif len(hand_pose_raw) == 48:
            global_orient = hand_pose_raw[:3]
            hand_pose_45 = hand_pose_raw[3:48]
        else:
            if len(hand_pose_raw) < 45:
                hand_pose_45 = np.pad(hand_pose_raw, (0, 45 - len(hand_pose_raw)))
            else:
                hand_pose_45 = hand_pose_raw[:45]

    transl = to_np(t_xyz).reshape(3)
    return betas, hand_pose_45, global_orient, transl


def generate_mano_mesh(mano_layer, hand_pose: dict):
    betas, hand_pose_45, global_orient, transl = extract_mano_params(hand_pose)

    betas_t = torch.tensor(betas[None], dtype=torch.float32, device=DEVICE)
    hand_pose_t = torch.tensor(hand_pose_45[None], dtype=torch.float32, device=DEVICE)
    global_orient_t = torch.tensor(global_orient[None], dtype=torch.float32, device=DEVICE)
    transl_t = torch.tensor(transl[None], dtype=torch.float32, device=DEVICE)

    out = mano_layer(
        betas=betas_t,
        hand_pose=hand_pose_t,
        global_orient=global_orient_t,
        transl=transl_t,
        return_verts=True,
    )

    verts = out.vertices[0].detach().cpu().numpy().astype(np.float32)
    joints = out.joints[0].detach().cpu().numpy().astype(np.float32)
    faces = mano_layer.faces.astype(np.uint32)

    return verts, joints, faces


# =========================================================
# Logging
# =========================================================
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
            labels=maybe_labels(["headset"]),
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

    vertices_world = transform_points(
        mesh_info["vertices_local"] * OBJECT_SCALE,
        q_wxyz,
        t_xyz,
    )

    rr.log(
        path_prefix,
        rr.Mesh3D(
            vertex_positions=vertices_world,
            triangle_indices=mesh_info["faces"],
            albedo_factor=mesh_info["albedo_factor"],
        ),
    )

def log_hand_mesh(path_prefix: str, verts: np.ndarray, joints: np.ndarray, faces: np.ndarray):
    rr.log(
        path_prefix,
        rr.Mesh3D(
            vertex_positions=verts,
            triangle_indices=faces,
        ),
    )

    rr.log(
        f"{path_prefix}_joints",
        rr.Points3D(
            positions=joints,
            labels=maybe_labels(["joint"] * len(joints)),
        ),
    )


# =========================================================
# Main
# =========================================================
def main():
    rr.init("hot3d_mano_viewer", spawn=True)

    if not SEQUENCE_DIR.exists():
        raise FileNotFoundError(f"Missing sequence dir: {SEQUENCE_DIR}")
    if not HAND_DIR.exists():
        raise FileNotFoundError(f"Missing hand_data dir: {HAND_DIR}")
    if not GT_DIR.exists():
        raise FileNotFoundError(f"Missing ground_truth dir: {GT_DIR}")
    if not OBJECT_MODELS_DIR.exists():
        raise FileNotFoundError(f"Missing object_models dir: {OBJECT_MODELS_DIR}")
    if not MANO_DIR.exists():
        raise FileNotFoundError(f"Missing mano_models dir: {MANO_DIR}")

    if not (MANO_DIR / "MANO_LEFT.pkl").exists():
        raise FileNotFoundError(f"Missing {(MANO_DIR / 'MANO_LEFT.pkl')}")
    if not (MANO_DIR / "MANO_RIGHT.pkl").exists():
        raise FileNotFoundError(f"Missing {(MANO_DIR / 'MANO_RIGHT.pkl')}")

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

    left_mano = create_mano_layer(is_rhand=False)
    right_mano = create_mano_layer(is_rhand=True)

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

    # 轨迹
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
            rr.Points3D(
                positions=np.asarray(pts, dtype=np.float32),
                labels=maybe_labels([object_meshes[uid]["name"]] * len(pts)),
            ),
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
            rr.Points3D(
                positions=np.asarray(headset_pts, dtype=np.float32),
                labels=maybe_labels(["headset"] * len(headset_pts)),
            ),
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

        # object meshes
        for row in current_object_rows:
            uid = row["object_uid"]
            if uid not in object_meshes:
                continue
            mesh_info = object_meshes[uid]
            log_object_mesh(f"world/objects/{mesh_info['name']}", mesh_info, row)

        # hands
        if hand_row is not None:
            hand_poses = hand_row.get("hand_poses", {})

            hand0 = get_hand_pose_entry(hand_poses, "0")
            hand1 = get_hand_pose_entry(hand_poses, "1")

            for idx, hp in [("0", hand0), ("1", hand1)]:
                if hp is None:
                    continue

                side = hp.get("hand_side", hp.get("side", "")).lower()
                if side in ["left", "l"]:
                    is_left = True
                elif side in ["right", "r"]:
                    is_left = False
                else:
                    is_left = idx == "0"

                try:
                    verts, joints, faces = generate_mano_mesh(
                        left_mano if is_left else right_mano,
                        hp,
                    )
                    log_hand_mesh(
                        f"world/hands/{'left' if is_left else 'right'}_hand",
                        verts,
                        joints,
                        faces,
                    )
                except Exception as e:
                    print(f"[WARN] Failed to generate MANO mesh for hand {idx} at ts={ts}: {e}")

        # headset
        if headset_row is not None:
            log_headset_pose("world/headset/current", headset_row)

        frame_counter += 1

    print("=" * 60)
    print("HOT3D + MANO viewer loaded.")
    print(f"Sequence         : {SEQUENCE_NAME}")
    print(f"Frames loaded    : {frame_counter}")
    print(f"Objects loaded   : {len(object_meshes)}")
    print(f"Sequence dir     : {SEQUENCE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
