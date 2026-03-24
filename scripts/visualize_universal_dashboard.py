import argparse
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

import cv2
import numpy as np
import rerun as rr


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
HOT3D_DEFAULT_ROOT = REPO_ROOT / "data" / "HOT3D" / "hot3d_demo_full"
BEINGH0_DEFAULT_SUBSET_DIR = REPO_ROOT / "data" / "Being-h0" / "h0_post_train_db_2508" / "pick_duck_blue_lerobot"
DEXWILD_DEFAULT_HDF5 = REPO_ROOT / "data" / "dexwild" / "robot_pour_data.hdf5"

HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


@dataclass
class DashboardPanels:
    recording_summary: str
    frame_summary: str
    main_task: str
    sub_task: str
    current_action: str
    interaction: str
    objects: List[str]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def require_paths(paths: List[Path]):
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")


def summary_text(items: List[tuple[str, object]]) -> str:
    return "\n".join(f"{key}: {value}" for key, value in items)


def log_scalar(path: str, value: float):
    rr.log(path, rr.Scalars(float(value)))


def read_image_rgb_unicode_safe(image_path: Path) -> np.ndarray:
    data = np.fromfile(str(image_path), dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(f"Failed to read image bytes: {image_path}")
    image_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to decode image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def create_shared_blueprint(base: str):
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(origin=f"{base}/camera", name="2D View"),
                rrb.Spatial3DView(origin=f"{base}/world", name="3D View"),
            ),
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Vertical(
                        rrb.TextDocumentView(origin=f"{base}/dashboard/summary/recording", name="Recording"),
                        rrb.TextDocumentView(origin=f"{base}/dashboard/summary/frame", name="Frame"),
                        rrb.TextDocumentView(origin=f"{base}/dashboard/details/objects", name="Objects"),
                    ),
                    rrb.Vertical(
                        rrb.TextDocumentView(origin=f"{base}/dashboard/semantic/main_task", name="Main Task"),
                        rrb.TextDocumentView(origin=f"{base}/dashboard/semantic/sub_task", name="Sub Task"),
                        rrb.TextDocumentView(origin=f"{base}/dashboard/semantic/current_action", name="Current Action"),
                        rrb.TextDocumentView(origin=f"{base}/dashboard/details/interaction", name="Interaction"),
                    ),
                ),
                rrb.TimeSeriesView(origin=f"{base}/dashboard/timeline", name="Timeline"),
            ),
        ),
        collapse_panels=True,
    )


def log_dashboard_panels(base: str, panels: DashboardPanels):
    rr.log(f"{base}/dashboard/summary/recording", rr.TextDocument(panels.recording_summary))
    rr.log(f"{base}/dashboard/summary/frame", rr.TextDocument(panels.frame_summary))
    rr.log(f"{base}/dashboard/semantic/main_task", rr.TextDocument(panels.main_task))
    rr.log(f"{base}/dashboard/semantic/sub_task", rr.TextDocument(panels.sub_task))
    rr.log(f"{base}/dashboard/semantic/current_action", rr.TextDocument(panels.current_action))
    rr.log(f"{base}/dashboard/details/interaction", rr.TextDocument(panels.interaction))
    rr.log(
        f"{base}/dashboard/details/objects",
        rr.TextDocument("\n".join(f"- {item}" for item in panels.objects) if panels.objects else "None"),
    )


def log_hand_2d(base_path: str, pts: np.ndarray):
    rr.log(f"{base_path}/keypoints", rr.Points2D(pts))
    lines = [
        np.stack([pts[a], pts[b]], axis=0)
        for a, b in HAND_BONES
        if a < len(pts) and b < len(pts)
    ]
    if lines:
        rr.log(f"{base_path}/bones", rr.LineStrips2D(lines))


def log_hand_3d(base_path: str, pts: np.ndarray):
    rr.log(f"{base_path}/keypoints", rr.Points3D(pts))
    lines = [
        np.stack([pts[a], pts[b]], axis=0)
        for a, b in HAND_BONES
        if a < len(pts) and b < len(pts)
    ]
    if lines:
        rr.log(f"{base_path}/bones", rr.LineStrips3D(lines))


class GigahandsAdapter:
    def __init__(self, args):
        self.args = args
        self.mod = importlib.import_module("visualize_gigahands_eval")
        self.base = "universal/gigahands"
        self.viewer_name = "universal_gigahands_dashboard"

    def load(self):
        m = self.mod
        m.SEQ_NAME = self.args.seq_name
        m.CAM_NAME = self.args.cam_name
        m.FRAME_ID = self.args.frame_id
        m.VIDEO_PATH = m.GIGAHANDS_ROOT / "hand_pose" / m.SEQ_NAME / "rgb_vid" / m.CAM_NAME / f"{m.CAM_NAME}_{m.FRAME_ID}.mp4"
        m.LEFT_2D_PATH = m.GIGAHANDS_ROOT / "hand_pose" / m.SEQ_NAME / "keypoints_2d" / "left" / "010" / f"{m.CAM_NAME}_{m.FRAME_ID}.jsonl"
        m.RIGHT_2D_PATH = m.GIGAHANDS_ROOT / "hand_pose" / m.SEQ_NAME / "keypoints_2d" / "right" / "010" / f"{m.CAM_NAME}_{m.FRAME_ID}.jsonl"
        m.LEFT_3D_PATH = m.GIGAHANDS_ROOT / "hand_pose" / m.SEQ_NAME / "keypoints_3d" / "010" / "left.jsonl"
        m.RIGHT_3D_PATH = m.GIGAHANDS_ROOT / "hand_pose" / m.SEQ_NAME / "keypoints_3d" / "010" / "right.jsonl"
        m.MESH_PATH = m.GIGAHANDS_ROOT / "object_pose" / m.SEQ_NAME / "pose" / "teapot_with_lid.obj"
        m.POSE_PATH = m.GIGAHANDS_ROOT / "object_pose" / m.SEQ_NAME / "pose" / "optimized_pose.json"
        m.GT_STEPS_PATH = m.ANNOTATIONS_DIR / f"gt_steps_{m.SEQ_NAME}.json"
        m.PRED_RAW_CLIPS_PATH = m.ANNOTATIONS_DIR / f"pred_raw_clips_{m.SEQ_NAME}.json"
        m.PRED_STEPS_PATH = m.ANNOTATIONS_DIR / f"pred_steps_{m.SEQ_NAME}.json"

        require_paths([
            m.VIDEO_PATH,
            m.LEFT_2D_PATH,
            m.RIGHT_2D_PATH,
            m.LEFT_3D_PATH,
            m.RIGHT_3D_PATH,
            m.MESH_PATH,
            m.POSE_PATH,
            m.PRED_STEPS_PATH,
        ])

        self.left_2d = m.load_2d(m.LEFT_2D_PATH)
        self.right_2d = m.load_2d(m.RIGHT_2D_PATH)
        self.left_3d = m.load_3d(m.LEFT_3D_PATH)
        self.right_3d = m.load_3d(m.RIGHT_3D_PATH)
        self.mesh_vertices, self.mesh_faces = m.load_mesh(m.MESH_PATH)
        self.gt_steps = m.load_optional_steps(m.GT_STEPS_PATH)
        self.pred_raw_clips = m.load_optional_steps(m.PRED_RAW_CLIPS_PATH)
        self.pred_steps = m.load_steps(m.PRED_STEPS_PATH)
        self.poses = m.interpolate_poses(load_json(m.POSE_PATH))
        self.scene_registry = m.build_scene_object_registry(m.SEQ_NAME)
        self.timeline_runs = []

        self.cap = cv2.VideoCapture(str(m.VIDEO_PATH))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {m.VIDEO_PATH}")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.total_frames <= 0:
            self.total_frames = max(len(self.left_2d), len(self.right_2d), len(self.left_3d), len(self.right_3d), 1)

        self.recording_summary = summary_text(
            [
                ("dataset", "gigahands"),
                ("sequence", m.SEQ_NAME),
                ("camera", m.CAM_NAME),
                ("frame_id", m.FRAME_ID),
                ("total_frames", self.total_frames),
                ("scene_objects", ", ".join(sorted({label for item in self.scene_registry for label in item["labels"]})) or "None"),
                ("pred_steps", len(self.pred_steps)),
                ("pred_raw_clips", len(self.pred_raw_clips)),
            ]
        )

    def close(self):
        self.cap.release()

    def log_static(self):
        self.mod.log_step_summary(self.base, "gt_steps_summary", self.gt_steps)
        if self.pred_raw_clips:
            self.mod.log_step_summary(self.base, "pred_raw_clips_summary", self.pred_raw_clips)
        self.mod.log_step_summary(self.base, "pred_steps_summary", self.pred_steps)
        self.timeline_runs = self.mod.log_timeline_series(self.base, self.pred_steps, "pred")

    def frames(self) -> Iterator[DashboardPanels]:
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            rr.set_time("frame", sequence=frame_idx)
            rr.log(f"{self.base}/camera/rgb", rr.Image(frame))

            if frame_idx < len(self.left_2d):
                log_hand_2d(f"{self.base}/camera/left_hand_2d", self.left_2d[frame_idx])
            if frame_idx < len(self.right_2d):
                log_hand_2d(f"{self.base}/camera/right_hand_2d", self.right_2d[frame_idx])
            if frame_idx < len(self.left_3d):
                log_hand_3d(f"{self.base}/world/left_hand_3d", self.left_3d[frame_idx])
            if frame_idx < len(self.right_3d):
                log_hand_3d(f"{self.base}/world/right_hand_3d", self.right_3d[frame_idx])

            pose_key = f"{frame_idx:06d}"
            if pose_key in self.poses:
                pose = self.poses[pose_key]
                t = np.asarray(pose["mesh_translation"], dtype=np.float32).reshape(3,)
                q = np.asarray(pose["mesh_rotation"], dtype=np.float32).reshape(4,)
                rr.log(f"{self.base}/world/object", rr.Transform3D(translation=t, mat3x3=self.mod.quat_to_rotmat(q)))
                rr.log(f"{self.base}/world/object/mesh", rr.Mesh3D(vertex_positions=self.mesh_vertices, triangle_indices=self.mesh_faces))

            pred_raw_step = self.mod.get_current_step(frame_idx, self.pred_raw_clips)[1]
            pred_step = self.mod.get_current_step(frame_idx, self.pred_steps)[1]
            semantic_step = pred_raw_step if pred_raw_step is not None else pred_step
            progress = self.mod.compute_progress(frame_idx, self.total_frames)
            frame_info = self.mod.build_frame_info(frame_idx, semantic_step, self.mod.SEQ_NAME, self.scene_registry, self.poses)
            self.mod.log_timeline_state(self.base, frame_idx, self.total_frames, self.timeline_runs)

            yield DashboardPanels(
                recording_summary=self.recording_summary,
                frame_summary="\n".join(
                    [
                        f"frame_index: {frame_idx}",
                        f"progress: {progress * 100:.1f}%",
                        f"active_pred_step: {pred_step['label'] if pred_step is not None else 'None'}",
                        f"active_raw_clip: {pred_raw_step['label'] if pred_raw_step is not None else 'None'}",
                    ]
                ),
                main_task=frame_info["main_task"],
                sub_task=frame_info["sub_task"],
                current_action=frame_info["action_text"],
                interaction=frame_info["interaction"],
                objects=frame_info["objects"],
            )
            frame_idx += 1


class Hot3DManoAdapter:
    def __init__(self, args):
        self.args = args
        self.mod = None
        self.base = "universal/hot3d_mano"
        self.viewer_name = "universal_hot3d_mano_dashboard"

    def load(self):
        try:
            self.mod = importlib.import_module("visualize_hot3d_mano")
        except ModuleNotFoundError as exc:
            if exc.name == "smplx":
                raise ModuleNotFoundError(
                    "HOT3D-MANO support requires the `smplx` package in this environment."
                ) from exc
            raise

        m = self.mod
        m.ROOT = Path(self.args.hot3d_root)
        m.SEQUENCE_NAME = self.args.sequence_name
        m.SEQUENCE_DIR = m.ROOT / m.SEQUENCE_NAME
        m.HAND_DIR = m.SEQUENCE_DIR / "hand_data"
        m.GT_DIR = m.SEQUENCE_DIR / "ground_truth"
        m.OBJECT_MODELS_DIR = m.ROOT / "object_models"
        m.MANO_DIR = m.ROOT / "mano_models"
        m.FRAME_STRIDE = self.args.frame_stride
        m.DEVICE = self.args.device

        require_paths([
            m.SEQUENCE_DIR,
            m.HAND_DIR,
            m.GT_DIR,
            m.OBJECT_MODELS_DIR,
            m.MANO_DIR,
            m.MANO_DIR / "MANO_LEFT.pkl",
            m.MANO_DIR / "MANO_RIGHT.pkl",
        ])

        self.metadata = m.load_json(m.GT_DIR / "metadata.json")
        self.dynamic_rows = m.load_csv(m.GT_DIR / "dynamic_objects.csv")
        self.headset_rows = m.load_csv(m.GT_DIR / "headset_trajectory.csv")
        self.mano_rows = m.load_jsonl(m.HAND_DIR / "mano_hand_pose_trajectory.jsonl")
        self.ts_to_objects = m.index_dynamic_objects(self.dynamic_rows)
        self.ts_to_hands = m.index_hand_rows(self.mano_rows)
        self.ts_to_headset = m.index_headset_rows(self.headset_rows)
        self.object_meshes = m.load_object_meshes(m.OBJECT_MODELS_DIR, self.metadata)
        self.timestamps = sorted(self.ts_to_objects.keys())
        if not self.timestamps:
            raise RuntimeError("No timestamps found in HOT3D dynamic_objects.csv")

        self.left_mano = m.create_mano_layer(is_rhand=False)
        self.right_mano = m.create_mano_layer(is_rhand=True)
        self.recording_summary = summary_text(
            [
                ("dataset", "hot3d-mano"),
                ("recording_name", self.metadata.get("recording_name", "unknown")),
                ("sequence", m.SEQUENCE_NAME),
                ("participant_id", self.metadata.get("participant_id", "unknown")),
                ("headset", self.metadata.get("headset", "unknown")),
                ("timestamps", len(self.timestamps)),
                ("objects", len(self.object_meshes)),
            ]
        )

    def close(self):
        return None

    def log_static(self):
        headset_pts = [
            [float(row["t_wo_x[m]"]), float(row["t_wo_y[m]"]), float(row["t_wo_z[m]"])]
            for row in self.headset_rows
        ]
        if headset_pts:
            rr.log(f"{self.base}/world/trajectories/headset", rr.Points3D(positions=np.asarray(headset_pts, dtype=np.float32)))

    def frames(self) -> Iterator[DashboardPanels]:
        sampled_timestamps = self.timestamps[:: self.mod.FRAME_STRIDE]
        for frame_idx, ts in enumerate(sampled_timestamps):
            rr.set_time("time_sec", duration=float(ts) / 1e9)

            current_object_rows = self.ts_to_objects.get(ts, [])
            hand_row = self.ts_to_hands.get(ts)
            headset_row = self.ts_to_headset.get(ts)

            active_object_names = []
            for row in current_object_rows:
                uid = row["object_uid"]
                if uid not in self.object_meshes:
                    continue
                mesh_info = self.object_meshes[uid]
                active_object_names.append(mesh_info["name"])
                self.mod.log_object_mesh(f"{self.base}/world/objects/{mesh_info['name']}", mesh_info, row)

            active_hands = []
            if hand_row is not None:
                hand_poses = hand_row.get("hand_poses", {})
                for idx in ("0", "1"):
                    hand_pose = self.mod.get_hand_pose_entry(hand_poses, idx)
                    if hand_pose is None:
                        continue
                    side = hand_pose.get("hand_side", hand_pose.get("side", "")).lower()
                    is_left = True if side in {"left", "l"} else False if side in {"right", "r"} else idx == "0"
                    verts, joints, faces = self.mod.generate_mano_mesh(self.left_mano if is_left else self.right_mano, hand_pose)
                    hand_name = "left_hand" if is_left else "right_hand"
                    active_hands.append(hand_name.replace("_", " "))
                    self.mod.log_hand_mesh(f"{self.base}/world/hands/{hand_name}", verts, joints, faces)

            if headset_row is not None:
                self.mod.log_headset_pose(f"{self.base}/world/headset/current", headset_row)

            progress = frame_idx / float(max(1, len(sampled_timestamps) - 1))
            log_scalar(f"{self.base}/dashboard/timeline/progress", progress)
            log_scalar(f"{self.base}/dashboard/timeline/object_count", float(len(active_object_names)))
            log_scalar(f"{self.base}/dashboard/timeline/hand_count", float(len(active_hands)))
            log_scalar(f"{self.base}/dashboard/timeline/headset_visible", 1.0 if headset_row is not None else 0.0)

            yield DashboardPanels(
                recording_summary=self.recording_summary,
                frame_summary="\n".join(
                    [
                        f"frame_index: {frame_idx}",
                        f"timestamp_ns: {ts}",
                        f"time_sec: {float(ts) / 1e9:.3f}",
                        f"num_dynamic_objects: {len(active_object_names)}",
                        f"num_active_hands: {len(active_hands)}",
                        f"has_headset_row: {headset_row is not None}",
                    ]
                ),
                main_task="HOT3D object manipulation playback",
                sub_task=f"Inspecting frame {frame_idx}",
                current_action=f"Tracking {len(active_object_names)} object(s) and {len(active_hands)} hand mesh(es).",
                interaction="; ".join(active_hands) if active_hands else "No active hand mesh at this timestamp.",
                objects=sorted(set(active_object_names)),
            )


class BeingH0Adapter:
    def __init__(self, args):
        self.args = args
        self.mod = importlib.import_module("visualize_beingh0_subset")
        self.base = "universal/being_h0"
        self.viewer_name = "universal_being_h0_dashboard"

    def load(self):
        m = self.mod
        self.subset_dir = Path(self.args.beingh0_subset_dir)
        if not self.subset_dir.exists():
            raise FileNotFoundError(f"Subset directory not found: {self.subset_dir}")

        self.jsonl_path = m.find_jsonl(self.subset_dir, self.args.beingh0_jsonl)
        self.samples = m.load_jsonl(self.jsonl_path)
        if not self.samples:
            raise RuntimeError(f"No samples found in {self.jsonl_path}")

        self.start_idx = max(0, self.args.beingh0_start)
        self.end_idx = len(self.samples) if self.args.beingh0_max_samples < 0 else min(len(self.samples), self.start_idx + self.args.beingh0_max_samples)

        self.recording_summary = summary_text(
            [
                ("dataset", "being-h0"),
                ("subset_dir", self.subset_dir.name),
                ("jsonl", self.jsonl_path.name),
                ("total_samples", len(self.samples)),
                ("showing_range", f"[{self.start_idx}, {self.end_idx})"),
            ]
        )

    def close(self):
        return None

    def log_static(self):
        if hasattr(rr, "ViewCoordinates"):
            try:
                rr.log(f"{self.base}/world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
            except Exception:
                pass

    def frames(self) -> Iterator[DashboardPanels]:
        m = self.mod
        for sample_idx in range(self.start_idx, self.end_idx):
            sample = self.samples[sample_idx]
            sample_id = sample.get("id", f"sample_{sample_idx:06d}")
            dataset_name = sample.get("dataset_name", self.subset_dir.name)
            image_rel = sample.get("image", "")
            instruction = m.extract_instruction(sample.get("conversations", []))
            proprio = np.asarray(sample.get("proprioception", []), dtype=np.float32)
            action_chunk = np.asarray(sample.get("action_chunk", []), dtype=np.float32)
            episode_idx, frame_idx = m.parse_sample_id(sample_id)

            m.set_time_seq("sample_idx", sample_idx)
            m.set_time_seq("episode", episode_idx)
            m.set_time_seq("frame", frame_idx)

            image_status = image_rel or "None"
            if image_rel:
                try:
                    img_rgb = read_image_rgb_unicode_safe(self.subset_dir / image_rel)
                    rr.log(f"{self.base}/camera/rgb", rr.Image(img_rgb))
                except Exception as exc:
                    image_status = f"unavailable ({exc})"

            if proprio.ndim == 1 and proprio.size > 0:
                log_scalar(f"{self.base}/dashboard/timeline/proprio_norm", float(np.linalg.norm(proprio)))
                log_scalar(f"{self.base}/dashboard/timeline/proprio_dim", float(proprio.shape[0]))
            if action_chunk.ndim == 2 and action_chunk.size > 0:
                log_scalar(f"{self.base}/dashboard/timeline/action_abs_mean", float(np.abs(action_chunk).mean()))
                log_scalar(f"{self.base}/dashboard/timeline/action_horizon", float(action_chunk.shape[0]))
            log_scalar(
                f"{self.base}/dashboard/timeline/sample_progress",
                (sample_idx - self.start_idx) / float(max(1, self.end_idx - self.start_idx - 1)),
            )

            yield DashboardPanels(
                recording_summary=self.recording_summary,
                frame_summary=summary_text(
                    [
                        ("sample_idx", sample_idx),
                        ("sample_id", sample_id),
                        ("dataset_name", dataset_name),
                        ("episode", episode_idx),
                        ("frame", frame_idx),
                        ("image", image_status),
                    ]
                ),
                main_task=instruction or "Behavior cloning sample playback",
                sub_task=f"Episode {episode_idx}, frame {frame_idx}",
                current_action="Showing observation image, proprioception state, and predicted future action chunk.",
                interaction=f"proprio_shape={tuple(proprio.shape)}, action_chunk_shape={tuple(action_chunk.shape)}",
                objects=[],
            )


class DexWildAdapter:
    def __init__(self, args):
        self.args = args
        self.mod = importlib.import_module("visualize_dexwild_preview")
        self.base = "universal/dexwild"
        self.viewer_name = "universal_dexwild_dashboard"

    def load(self):
        m = self.mod
        self.hdf5_path = Path(self.args.dexwild_hdf5)
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")

        import h5py

        self.h5 = h5py.File(self.hdf5_path, "r")
        if self.args.dexwild_episode not in self.h5:
            raise KeyError(f"Episode '{self.args.dexwild_episode}' not found in {self.hdf5_path}")

        self.episode_name = self.args.dexwild_episode
        self.data = m.load_episode(self.h5[self.episode_name])
        self.n_frames = m.get_n_frames(self.data)
        if self.args.dexwild_max_frames > 0:
            self.n_frames = min(self.n_frames, self.args.dexwild_max_frames)

        self.recording_summary = summary_text(
            [
                ("dataset", "dexwild"),
                ("hdf5", self.hdf5_path.name),
                ("episode", self.episode_name),
                ("frames", self.n_frames),
                ("modalities", "right_thumb_cam, right_pinky_cam, right_arm_eef, right_leapv1, right_leapv2, right_manus"),
            ]
        )

    def close(self):
        if hasattr(self, "h5"):
            self.h5.close()

    def log_static(self):
        if hasattr(rr, "ViewCoordinates"):
            rr.log(f"{self.base}/world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    def frames(self) -> Iterator[DashboardPanels]:
        m = self.mod
        thumb_group = self.data["thumb_group"]
        pinky_group = self.data["pinky_group"]
        thumb_keys = self.data["thumb_keys"]
        pinky_keys = self.data["pinky_keys"]
        eef = self.data["eef"]
        leapv1 = self.data["leapv1"]
        leapv2 = self.data["leapv2"]
        manus = self.data["manus"]
        eef_traj = []

        for frame_idx in range(self.n_frames):
            ts_ns = m.get_timestamp_ns(self.data, frame_idx)
            rr.set_time("frame", sequence=frame_idx)
            rr.set_time("time", timestamp=np.datetime64(ts_ns, "ns"))

            rr.log(f"{self.base}/camera/right_thumb", rr.Image(np.asarray(thumb_group[thumb_keys[frame_idx]][:])))
            rr.log(f"{self.base}/camera/right_pinky", rr.Image(np.asarray(pinky_group[pinky_keys[frame_idx]][:])))

            eef_row = eef[frame_idx]
            xyz = eef_row[1:4].astype(np.float32)
            quat_xyzw = eef_row[4:8].astype(np.float32)
            rr.log(f"{self.base}/world/robot/eef/point", rr.Points3D([xyz], radii=0.01))
            rr.log(f"{self.base}/world/robot/eef/pose", rr.Transform3D(translation=xyz, rotation=rr.Quaternion(xyzw=quat_xyzw)))
            eef_traj.append(xyz.copy())
            rr.log(f"{self.base}/world/robot/eef/trajectory", rr.LineStrips3D([np.array(eef_traj, dtype=np.float32)]))

            log_scalar(f"{self.base}/dashboard/timeline/eef_x", float(xyz[0]))
            log_scalar(f"{self.base}/dashboard/timeline/eef_y", float(xyz[1]))
            log_scalar(f"{self.base}/dashboard/timeline/eef_z", float(xyz[2]))
            log_scalar(f"{self.base}/dashboard/timeline/eef_radius", float(np.linalg.norm(xyz)))
            if frame_idx < len(leapv1):
                log_scalar(f"{self.base}/dashboard/timeline/leapv1_mean", float(np.mean(leapv1[frame_idx][1:])))
            if frame_idx < len(leapv2):
                log_scalar(f"{self.base}/dashboard/timeline/leapv2_mean", float(np.mean(leapv2[frame_idx][1:])))
            if frame_idx < len(manus):
                log_scalar(f"{self.base}/dashboard/timeline/manus_mean", float(np.mean(manus[frame_idx][1:])))

            yield DashboardPanels(
                recording_summary=self.recording_summary,
                frame_summary=summary_text(
                    [
                        ("frame", frame_idx),
                        ("timestamp_ns", ts_ns),
                        ("thumb_key", thumb_keys[frame_idx]),
                        ("pinky_key", pinky_keys[frame_idx]),
                    ]
                ),
                main_task="DexWild episode preview",
                sub_task=self.episode_name,
                current_action="Showing thumb and pinky cameras with end-effector trajectory and hand-state streams.",
                interaction="right_thumb_cam + right_pinky_cam + right_arm_eef + right hand states",
                objects=["right_thumb_cam", "right_pinky_cam", "right_arm_eef", "right_leapv1", "right_leapv2", "right_manus"],
            )


def create_adapter(args):
    if args.dataset == "gigahands":
        return GigahandsAdapter(args)
    if args.dataset in {"hot3d", "hot3d-mano"}:
        return Hot3DManoAdapter(args)
    if args.dataset == "being-h0":
        return BeingH0Adapter(args)
    if args.dataset == "dexwild":
        return DexWildAdapter(args)
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def build_parser():
    parser = argparse.ArgumentParser(description="Universal Rerun dashboard for multiple datasets.")
    parser.add_argument("--dataset", choices=["gigahands", "hot3d", "hot3d-mano", "being-h0", "dexwild"], required=True)
    parser.add_argument("--seq-name", default="p36-tea-0010")
    parser.add_argument("--cam-name", default="brics-odroid-010_cam0")
    parser.add_argument("--frame-id", default="1727030430697198")
    parser.add_argument("--hot3d-root", default=str(HOT3D_DEFAULT_ROOT))
    parser.add_argument("--sequence-name", default="P0001_10a27bf7")
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--beingh0-subset-dir", default=str(BEINGH0_DEFAULT_SUBSET_DIR))
    parser.add_argument("--beingh0-jsonl", default=None)
    parser.add_argument("--beingh0-start", type=int, default=0)
    parser.add_argument("--beingh0-max-samples", type=int, default=-1)
    parser.add_argument("--dexwild-hdf5", default=str(DEXWILD_DEFAULT_HDF5))
    parser.add_argument("--dexwild-episode", default="ep_0000")
    parser.add_argument("--dexwild-max-frames", type=int, default=-1)
    return parser


def main():
    args = build_parser().parse_args()
    adapter = create_adapter(args)
    adapter.load()

    rr.init(adapter.viewer_name, spawn=True)
    rr.send_blueprint(create_shared_blueprint(adapter.base))
    adapter.log_static()

    try:
        for panels in adapter.frames():
            log_dashboard_panels(adapter.base, panels)
    finally:
        adapter.close()

if __name__ == "__main__":
    main()
