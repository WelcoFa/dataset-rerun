import json
import re
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

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
DATA_ROOT = REPO_ROOT / "data" / "gigahands"
GIGAHANDS_ROOT = DATA_ROOT / "gigahands_demo_all"
ANNOTATIONS_DIR = DATA_ROOT / "annotations"

SEQ_NAME = "p36-tea-0010"
CAM_NAME = "brics-odroid-010_cam0"
FRAME_ID = "1727030430697198"
SCENE_NAME = "scene_gigahands"


# =========================
# Paths
# =========================

VIDEO_PATH = (
    GIGAHANDS_ROOT
    / "hand_pose"
    / SEQ_NAME
    / "rgb_vid"
    / CAM_NAME
    / f"{CAM_NAME}_{FRAME_ID}.mp4"
)

LEFT_2D_PATH = (
    GIGAHANDS_ROOT
    / "hand_pose"
    / SEQ_NAME
    / "keypoints_2d"
    / "left"
    / "010"
    / f"{CAM_NAME}_{FRAME_ID}.jsonl"
)

RIGHT_2D_PATH = (
    GIGAHANDS_ROOT
    / "hand_pose"
    / SEQ_NAME
    / "keypoints_2d"
    / "right"
    / "010"
    / f"{CAM_NAME}_{FRAME_ID}.jsonl"
)

LEFT_3D_PATH = (
    GIGAHANDS_ROOT
    / "hand_pose"
    / SEQ_NAME
    / "keypoints_3d"
    / "010"
    / "left.jsonl"
)

RIGHT_3D_PATH = (
    GIGAHANDS_ROOT
    / "hand_pose"
    / SEQ_NAME
    / "keypoints_3d"
    / "010"
    / "right.jsonl"
)

MESH_PATH = (
    GIGAHANDS_ROOT
    / "object_pose"
    / SEQ_NAME
    / "pose"
    / "teapot_with_lid.obj"
)

POSE_PATH = (
    GIGAHANDS_ROOT
    / "object_pose"
    / SEQ_NAME
    / "pose"
    / "optimized_pose.json"
)

GT_STEPS_PATH = ANNOTATIONS_DIR / f"gt_steps_{SEQ_NAME}.json"
PRED_RAW_CLIPS_PATH = ANNOTATIONS_DIR / f"pred_raw_clips_{SEQ_NAME}.json"
PRED_STEPS_PATH = ANNOTATIONS_DIR / f"pred_steps_{SEQ_NAME}.json"


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


def extract_step_text(value):
    if isinstance(value, dict):
        if "text" in value:
            return str(value["text"])
        if "label" in value:
            return str(value["label"])
        return json.dumps(value, ensure_ascii=False)

    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return stripped
            return extract_step_text(parsed)
        return stripped

    return str(value)


def extract_embedded_step_fields(value):
    if isinstance(value, dict):
        return {
            "label": extract_step_text(value.get("label", "")),
            "text": extract_step_text(value.get("text", "")),
        }

    if not isinstance(value, str):
        return {"label": "", "text": extract_step_text(value)}

    stripped = value.strip()
    label_match = re.search(r'"label"\s*:\s*"([^"]+)', stripped)
    text_match = re.search(r'"text"\s*:\s*"([^"]+)', stripped, flags=re.DOTALL)

    embedded_label = label_match.group(1).strip() if label_match else ""
    embedded_text = text_match.group(1).strip() if text_match else ""

    return {
        "label": embedded_label,
        "text": embedded_text,
    }


def load_steps(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Step file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list of step dicts.")

    required = ("start", "end", "text")
    for i, step in enumerate(data):
        for key in required:
            if key not in step:
                raise ValueError(f"{path} step #{i} missing key: {key}")

        step["start"] = int(step["start"])
        step["end"] = int(step["end"])
        embedded = extract_embedded_step_fields(step["text"])
        step["label"] = extract_step_text(step.get("label", step["text"]))
        step["text"] = extract_step_text(step["text"])
        step["_embedded_label"] = embedded["label"]
        step["_embedded_text"] = embedded["text"]

        if "sub_task" in step:
            step["sub_task"] = extract_step_text(step["sub_task"])
        if "interaction" in step:
            step["interaction"] = extract_step_text(step["interaction"])
        if "objects" in step and not isinstance(step["objects"], list):
            step["objects"] = [str(step["objects"])]

    return data


def load_optional_steps(path: Path):
    if not path.exists():
        return []
    return load_steps(path)


# =========================
# Math helpers
# =========================

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

    rots_xyzw = np.stack([rots[:, 1], rots[:, 2], rots[:, 3], rots[:, 0]], axis=1)
    rot_obj = Rotation.from_quat(rots_xyzw)
    interp_rot_xyzw = Slerp(idxs, rot_obj)(full_idx).as_quat()

    if USE_SMOOTHER and len(full_idx) >= SMOOTH_WINDOW:
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
        rr.log(f"{base}/eval/{name}/step_{i + 1:02d}", rr.TextLog(text))


# =========================
# Ropedia-style semantic helpers
# =========================

def infer_main_task(seq_name: str) -> str:
    seq = seq_name.lower()
    if "tea" in seq:
        return "Preparing tea with a teapot"
    if "boxing" in seq:
        return "Interacting with a boxing bag"
    return "Hand-object manipulation"


def discover_scene_objects(seq_name: str) -> list[str]:
    pose_dir = (
        GIGAHANDS_ROOT
        / "object_pose"
        / seq_name
        / "pose"
    )

    if not pose_dir.exists():
        return []

    object_names = []
    for p in pose_dir.iterdir():
        if p.suffix.lower() in {".obj", ".ply", ".glb", ".stl"}:
            object_names.append(p.stem)

    return sorted(set(object_names))


def normalize_object_name(name: str) -> list[str]:
    name = name.lower().replace("-", "_").strip()

    banned_exact = {
        "transform",
        "mesh",
        "transform_mesh",
        "object",
        "world",
        "camera",
        "scene",
    }
    if name in banned_exact:
        return []

    tokens = [t for t in name.split("_") if t]
    banned_tokens = {"transform", "mesh", "world", "camera", "scene"}
    if tokens and all(t in banned_tokens for t in tokens):
        return []

    special_map = {
        "teapot_with_lid": ["teapot", "lid"],
        "boxing_bag_stand": ["boxing bag", "stand"],
    }
    if name in special_map:
        return special_map[name]

    drop_tokens = {"with", "and", "transform", "mesh", "object", "objects"}
    tokens = [t for t in tokens if t not in drop_tokens]

    if not tokens:
        return []

    return [" ".join(tokens)]


def build_scene_object_registry(seq_name: str):
    raw_names = discover_scene_objects(seq_name)

    registry = []
    for raw in raw_names:
        labels = normalize_object_name(raw)
        if not labels:
            continue
        registry.append({
            "raw_name": raw,
            "labels": labels,
        })

    return registry


def get_active_objects_for_frame(frame_idx: int, registry, poses) -> list[str]:
    pose_key = f"{frame_idx:06d}"
    if pose_key not in poses:
        return []

    active = []
    for item in registry:
        active.extend(item["labels"])

    return sorted(set(active))


def infer_objects(frame_idx: int, seq_name: str, step, scene_registry, poses) -> list[str]:
    if step is not None and "objects" in step and isinstance(step["objects"], list):
        filtered = []
        for obj in step["objects"]:
            filtered.extend(normalize_object_name(str(obj)))
        filtered = sorted(set(filtered))
        if filtered:
            return filtered

    active = get_active_objects_for_frame(frame_idx, scene_registry, poses)
    if active:
        return active

    return []


def build_frame_info(frame_idx: int, step, seq_name: str, scene_registry, poses):
    if step is None:
        return {
            "sub_task": "No active sub task",
            "interaction": "None",
            "action_text": "No active action",
            "objects": get_active_objects_for_frame(frame_idx, scene_registry, poses),
        }

    fallback_label = str(step.get("_embedded_label", "")).strip()
    fallback_text = str(step.get("_embedded_text", "")).strip()

    main_task = str(step.get("main_task", infer_main_task(seq_name))).strip()
    if not main_task:
        main_task = infer_main_task(seq_name)

    sub_task = str(step.get("sub_task", step.get("label", "unknown"))).strip()
    interaction = str(step.get("interaction", step.get("label", "unknown"))).strip()
    action_text = str(step.get("current_action", step.get("text", ""))).strip()

    if sub_task.lower() == "other" and fallback_label:
        sub_task = fallback_label
    if interaction.lower() == "other" and fallback_label:
        interaction = fallback_label
    if (
        action_text.startswith("{")
        or action_text.startswith("```")
        or action_text.lower() == "other"
    ) and fallback_text:
        action_text = fallback_text

    return {
        "main_task": main_task,
        "sub_task": sub_task,
        "action_text": action_text,
        "interaction": interaction,
        "objects": infer_objects(frame_idx, seq_name, step, scene_registry, poses),
    }


def log_caption_panels(base: str, seq_name: str, frame_info, progress: float):
    rr.log(
        f"{base}/captions/Main_Task",
        rr.TextDocument(
            f"{frame_info['main_task']}\n\nProgress: {progress * 100:.1f}%"
        ),
    )

    rr.log(
        f"{base}/captions/Sub_Task",
        rr.TextDocument(frame_info["sub_task"]),
    )

    rr.log(
        f"{base}/captions/Current_Action",
        rr.TextDocument(frame_info["action_text"]),
    )

    rr.log(
        f"{base}/captions/details/interaction",
        rr.TextDocument(frame_info["interaction"]),
    )

    objects_text = "\n".join(f"- {obj}" for obj in frame_info["objects"])
    if not objects_text:
        objects_text = "None"

    rr.log(
        f"{base}/captions/details/objects",
        rr.TextDocument(objects_text),
    )


# =========================
# Timeline helpers
# =========================

def log_timeline_series(base: str, steps, name: str):
    del name

    def get_task_color(idx: int):
        colors = [
            [255, 50, 50], [50, 255, 50], [50, 50, 255], [255, 255, 50], [50, 255, 255],
            [255, 50, 255], [255, 128, 0], [128, 0, 128], [0, 128, 0], [0, 0, 128],
            [128, 128, 0], [128, 0, 0], [0, 128, 128],
        ]
        return colors[idx % len(colors)]

    runs = []
    for step in steps:
        task_name = step["label"]
        start_f = int(step["start"])
        end_f = int(step["end"])
        if runs and runs[-1]["task"] == task_name:
            runs[-1]["end"] = max(runs[-1]["end"], end_f)
        else:
            runs.append({"task": task_name, "start": start_f, "end": end_f})

    labeled_tasks = set()
    for run_idx, run in enumerate(runs):
        entity_path = f"{base}/timeline/run_{run_idx:03d}/line_0"
        label = ""
        if run["task"] not in labeled_tasks:
            label = run["task"]
            labeled_tasks.add(run["task"])
        try:
            rr.log(
                entity_path,
                rr.SeriesLines(colors=get_task_color(run_idx), widths=4, names=label),
                static=True,
            )
        except TypeError:
            rr.log(
                entity_path,
                rr.SeriesLines(colors=get_task_color(run_idx), widths=4, names=label),
            )

    return runs


def compute_progress(frame_idx: int, total_frames: int) -> float:
    if total_frames <= 1:
        return 1.0
    return frame_idx / float(total_frames - 1)


def log_timeline_state(base: str, frame_idx: int, total_frames: int, timeline_runs):
    progress = compute_progress(frame_idx, total_frames)
    rr.log(f"{base}/timeline/progress", rr.Scalars(progress))

    for run_idx, run in enumerate(timeline_runs):
        entity_path = f"{base}/timeline/run_{run_idx:03d}/line_0"
        if run["start"] <= frame_idx <= run["end"]:
            rr.log(entity_path, rr.Scalars(0.0))


# =========================
# Blueprint
# =========================

def create_blueprint():
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(origin="scene_gigahands/camera", name="RGB + 2D Hands"),
                rrb.Spatial3DView(origin="scene_gigahands/world", name="3D Scene"),
            ),
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Vertical(
                        rrb.TextDocumentView(origin="scene_gigahands/captions/Main_Task", name="Main Task"),
                        rrb.TextDocumentView(origin="scene_gigahands/captions/details/objects", name="Objects"),
                    ),
                    rrb.Vertical(
                        rrb.TextDocumentView(origin="scene_gigahands/captions/Sub_Task", name="Sub Task"),
                        rrb.TextDocumentView(origin="scene_gigahands/captions/details/interaction", name="Interaction"),
                        rrb.TextDocumentView(origin="scene_gigahands/captions/Current_Action", name="Current Action"),
                    ),
                ),
                rrb.TimeSeriesView(
                    origin="scene_gigahands/timeline",
                    name="Task Timeline",
                ),
            ),
        ),
        collapse_panels=True,
    )


# =========================
# Main
# =========================

def main():
    print("Loading GigaHands evaluation scene with Ropedia-style semantic panels...")

    required_paths = [
        VIDEO_PATH,
        LEFT_2D_PATH,
        RIGHT_2D_PATH,
        LEFT_3D_PATH,
        RIGHT_3D_PATH,
        MESH_PATH,
        POSE_PATH,
        PRED_STEPS_PATH,
    ]
    for p in required_paths:
        if not p.exists():
            raise FileNotFoundError(f"Path not found: {p}")

    left_2d = load_2d(LEFT_2D_PATH)
    right_2d = load_2d(RIGHT_2D_PATH)
    left_3d = load_3d(LEFT_3D_PATH)
    right_3d = load_3d(RIGHT_3D_PATH)
    mesh_vertices, mesh_faces = load_mesh(MESH_PATH)
    gt_steps = load_optional_steps(GT_STEPS_PATH)
    pred_raw_clips = load_optional_steps(PRED_RAW_CLIPS_PATH)
    pred_steps = load_steps(PRED_STEPS_PATH)

    with open(POSE_PATH, "r", encoding="utf-8") as f:
        raw_poses = json.load(f)
    poses = interpolate_poses(raw_poses)
    scene_registry = build_scene_object_registry(SEQ_NAME)
    print("Scene objects:", scene_registry)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = max(len(left_2d), len(right_2d), len(left_3d), len(right_3d), 1)

    rr.init("gigahands_eval_test", spawn=True)
    rr.send_blueprint(create_blueprint())

    base = SCENE_NAME

    log_step_summary(base, "gt_steps_summary", gt_steps)
    if pred_raw_clips:
        log_step_summary(base, "pred_raw_clips_summary", pred_raw_clips)
    log_step_summary(base, "pred_steps_summary", pred_steps)
    timeline_runs = log_timeline_series(base, pred_steps, "pred")

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rr.set_time("frame", sequence=frame_idx)

            # RGB
            rr.log(f"{base}/camera/rgb", rr.Image(frame))

            # 2D hands
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

            # 3D hands
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

            # Object mesh + pose
            pose_key = f"{frame_idx:06d}"
            if pose_key in poses:
                pose = poses[pose_key]
                t = np.asarray(pose["mesh_translation"], dtype=np.float32).reshape(3,)
                q = np.asarray(pose["mesh_rotation"], dtype=np.float32).reshape(4,)
                R = quat_to_rotmat(q)

                rr.log(
                    f"{base}/world/object",
                    rr.Transform3D(translation=t, mat3x3=R),
                )
                rr.log(
                    f"{base}/world/object/mesh",
                    rr.Mesh3D(
                        vertex_positions=mesh_vertices,
                        triangle_indices=mesh_faces,
                    ),
                )

            gt_idx, gt_step = get_current_step(frame_idx, gt_steps)
            pred_raw_idx, pred_raw_step = get_current_step(frame_idx, pred_raw_clips)
            pred_idx, pred_step = get_current_step(frame_idx, pred_steps)

            progress = compute_progress(frame_idx, total_frames)
            semantic_step = pred_raw_step if pred_raw_step is not None else pred_step
            frame_info = build_frame_info(frame_idx, semantic_step, SEQ_NAME, scene_registry, poses)
            log_caption_panels(base, SEQ_NAME, frame_info, progress)
            log_timeline_state(base, frame_idx, total_frames, timeline_runs)

            frame_idx += 1

    finally:
        cap.release()

    print("Done.")


if __name__ == "__main__":
    main()
