import json
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import trimesh
from scipy.spatial.transform import Rotation


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
DATA_ROOT = REPO_ROOT / "data" / "gigahands"
GIGAHANDS_ROOT = DATA_ROOT / "gigahands_demo_all"

# =========================
# Paths
# =========================

HAND_SEQ_ROOT = (
    GIGAHANDS_ROOT
    / "hand_pose"
    / "p36-tea-0010"
)

OBJECT_POSE_ROOT = (
    GIGAHANDS_ROOT
    / "object_pose"
    / "p36-tea-0010"
)

OBJECT_MESH_PATH = (
    GIGAHANDS_ROOT
    / "object_pose"
    / "p36-tea-0010"
    / "pose"
    / "teapot_with_lid.obj"
)

CAM_NAME = "brics-odroid-010_cam0"
VIDEO_STEM = "brics-odroid-010_cam0_1727030430697198"
OBJECT_ID = "010"

VIDEO_PATH = HAND_SEQ_ROOT / "rgb_vid" / CAM_NAME / f"{VIDEO_STEM}.mp4"

KPTS2D_LEFT_PATH = HAND_SEQ_ROOT / "keypoints_2d" / "left" / OBJECT_ID / f"{VIDEO_STEM}.jsonl"
KPTS2D_RIGHT_PATH = HAND_SEQ_ROOT / "keypoints_2d" / "right" / OBJECT_ID / f"{VIDEO_STEM}.jsonl"

KPTS3D_LEFT_PATH = HAND_SEQ_ROOT / "keypoints_3d" / OBJECT_ID / "left.jsonl"
KPTS3D_RIGHT_PATH = HAND_SEQ_ROOT / "keypoints_3d" / OBJECT_ID / "right.jsonl"

OBJECT_POSE_PATH = OBJECT_POSE_ROOT / "pose" / "optimized_pose.json"


# ============================================================
# Hand skeleton
# ============================================================
HAND_BONES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]


# ============================================================
# Loaders
# ============================================================
def load_jsonl(path):
    data=[]
    with open(path,"r") as f:
        for l in f:
            data.append(json.loads(l))
    return data


def load_2d(path):
    raw=load_jsonl(path)
    pts=[]
    for r in raw:
        arr=np.asarray(r).reshape(-1,3)
        pts.append(arr[:,:2])
    return pts


def load_3d(path):
    raw=load_jsonl(path)
    pts=[]
    for r in raw:
        arr=np.asarray(r).reshape(-1,4)
        pts.append(arr[:,:3])
    return pts


def load_mesh(path):
    mesh=trimesh.load(path,process=False)

    if isinstance(mesh,trimesh.Scene):
        mesh=list(mesh.geometry.values())[0]

    return np.asarray(mesh.vertices),np.asarray(mesh.faces)


def load_object_pose(path):
    with open(path) as f:
        return json.load(f)


# ============================================================
# Math
# ============================================================
def quat_to_rotmat(q):
    q=np.asarray(q)
    quat_xyzw=[q[1],q[2],q[3],q[0]]
    return Rotation.from_quat(quat_xyzw).as_matrix().T


# ============================================================
# Rerun helpers
# ============================================================
def log_points2d(path,pts):

    rr.log(f"{path}/points",rr.Points2D(pts))

    lines=[]
    for a,b in HAND_BONES:
        if a<len(pts) and b<len(pts):
            lines.append(np.stack([pts[a],pts[b]]))

    if lines:
        rr.log(f"{path}/bones",rr.LineStrips2D(lines))


def log_points3d(path,pts):

    rr.log(f"{path}/points",rr.Points3D(pts))

    lines=[]
    for a,b in HAND_BONES:
        if a<len(pts) and b<len(pts):
            lines.append(np.stack([pts[a],pts[b]]))

    if lines:
        rr.log(f"{path}/bones",rr.LineStrips3D(lines))


def log_mesh(path,v,f,t,R):

    rr.log(
        path,
        rr.Transform3D(
            translation=t,
            mat3x3=R,
        ),
    )

    rr.log(
        f"{path}/mesh",
        rr.Mesh3D(
            vertex_positions=v,
            triangle_indices=f,
        ),
    )


# ============================================================
# Main
# ============================================================
def main():

    left2d=load_2d(KPTS2D_LEFT_PATH)
    right2d=load_2d(KPTS2D_RIGHT_PATH)

    left3d=load_3d(KPTS3D_LEFT_PATH)
    right3d=load_3d(KPTS3D_RIGHT_PATH)

    verts,faces=load_mesh(OBJECT_MESH_PATH)
    poses=load_object_pose(OBJECT_POSE_PATH)

    rr.init("gigahands_rerun",spawn=True)

    cap=cv2.VideoCapture(str(VIDEO_PATH))
    frame_idx=0

    while True:

        ret,frame=cap.read()
        if not ret:
            break

        rr.set_time("frame",sequence=frame_idx)

        rr.log("camera/rgb",rr.Image(frame))

        if frame_idx<len(left2d):
            log_points2d("camera/left_hand",left2d[frame_idx])

        if frame_idx<len(right2d):
            log_points2d("camera/right_hand",right2d[frame_idx])

        if frame_idx<len(left3d):
            log_points3d("world/left_hand",left3d[frame_idx])

        if frame_idx<len(right3d):
            log_points3d("world/right_hand",right3d[frame_idx])

        pose_key = f"{frame_idx:06d}"
        if pose_key in poses:
            pose = poses[pose_key]

            t = np.asarray(pose["mesh_translation"])
            q = np.asarray(pose["mesh_rotation"])
            R = quat_to_rotmat(q)

            log_mesh("world/object", verts, faces, t, R)

        frame_idx += 1

    cap.release()


if __name__=="__main__":
    main()
