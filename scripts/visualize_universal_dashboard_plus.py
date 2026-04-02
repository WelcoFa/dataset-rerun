import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import cv2
import numpy as np
import rerun as rr

import visualize_universal_dashboard as ud


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT = REPO_ROOT / "data"


@dataclass
class DetectionResult:
    dataset: str
    resolved_input: Path
    reason: str
    config: dict[str, object]


def parser_with_default_spawn() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Universal-plus Rerun dashboard that auto-detects dataset structure from an input path.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Dataset root, scene folder, subset folder, or source file.")
    parser.add_argument(
        "--dataset",
        choices=["auto", "gigahands", "hot3d", "being-h0", "dexwild", "thermohands", "generic"],
        default="auto",
        help="Force a dataset type or let the script infer it from --input.",
    )
    parser.add_argument("--spawn", action=argparse.BooleanOptionalAction, default=True, help="Spawn the Rerun viewer.")
    parser.add_argument("--describe-only", action="store_true", help="Only print detection info without opening Rerun.")

    parser.add_argument("--seq-name", default=None)
    parser.add_argument("--cam-name", default=None)
    parser.add_argument("--frame-id", default=None)
    parser.add_argument("--sequence-name", default=None)
    parser.add_argument("--frame-stride", type=int, default=None)
    parser.add_argument("--device", default=None)

    parser.add_argument("--beingh0-subset-dir", default=None)
    parser.add_argument("--beingh0-jsonl", default=None)
    parser.add_argument("--beingh0-start", type=int, default=None)
    parser.add_argument("--beingh0-max-samples", type=int, default=None)

    parser.add_argument("--dexwild-hdf5", default=None)
    parser.add_argument("--dexwild-episode", default=None)
    parser.add_argument("--dexwild-max-frames", type=int, default=None)

    parser.add_argument("--thermohands-scene-dir", default=None)
    parser.add_argument("--thermohands-stride", type=int, default=None)
    parser.add_argument("--thermohands-max-frames", type=int, default=None)

    parser.add_argument("--generic-max-items", type=int, default=200, help="Safety cap for the generic fallback viewer.")
    return parser


def first_dir(path: Path, predicate) -> Path | None:
    for child in sorted(path.iterdir()):
        if child.is_dir() and predicate(child):
            return child
    return None


def first_file(path: Path, pattern: str) -> Path | None:
    matches = sorted(path.glob(pattern))
    return matches[0] if matches else None


def is_thermohands_scene(path: Path) -> bool:
    return path.is_dir() and all((path / name).is_dir() for name in ("rgb", "thermal", "ir", "depth", "gt_info"))


def is_beingh0_subset(path: Path) -> bool:
    return path.is_dir() and (path / "images").is_dir() and any(path.glob("*_train.jsonl"))


def is_hot3d_sequence(path: Path) -> bool:
    return path.is_dir() and (path / "hand_data").is_dir() and (path / "ground_truth").is_dir()


def is_hot3d_root(path: Path) -> bool:
    return path.is_dir() and (path / "object_models").is_dir() and (path / "mano_models").is_dir()


def is_gigahands_root(path: Path) -> bool:
    return path.is_dir() and (path / "hand_pose").is_dir() and (path / "object_pose").is_dir()


def infer_gigahands_config(input_path: Path) -> DetectionResult | None:
    root = None
    annotations_dir = None
    seq_name = None

    if input_path.name == "gigahands" and (input_path / "gigahands_demo_all").is_dir():
        root = input_path / "gigahands_demo_all"
        annotations_dir = input_path / "annotations"
    elif is_gigahands_root(input_path):
        root = input_path
        annotations_dir = input_path.parent / "annotations"
    elif input_path.parent.name == "hand_pose" and input_path.is_dir():
        root = input_path.parent.parent
        annotations_dir = root.parent / "annotations"
        seq_name = input_path.name

    if root is None or annotations_dir is None:
        return None

    defaults = ud.build_parser().parse_args(["--dataset", "gigahands"])
    default_seq = defaults.seq_name
    default_cam = defaults.cam_name
    default_frame = defaults.frame_id

    default_video = root / "hand_pose" / default_seq / "rgb_vid" / default_cam / f"{default_cam}_{default_frame}.mp4"
    default_pred = annotations_dir / f"pred_steps_{default_seq}.json"
    if seq_name is None and default_video.exists() and default_pred.exists():
        seq_name = default_seq
        cam_name = default_cam
        frame_id = default_frame
    else:
        hand_pose_dir = root / "hand_pose"
        if seq_name is None:
            seq_dir = first_dir(hand_pose_dir, lambda child: True)
            if seq_dir is None:
                raise FileNotFoundError(f"No sequence directory found in {hand_pose_dir}")
            seq_name = seq_dir.name
        else:
            seq_dir = hand_pose_dir / seq_name

        rgb_vid_dir = seq_dir / "rgb_vid"
        cam_dir = first_dir(rgb_vid_dir, lambda child: True)
        if cam_dir is None:
            raise FileNotFoundError(f"No camera directory found in {rgb_vid_dir}")
        video_path = first_file(cam_dir, "*.mp4")
        if video_path is None:
            raise FileNotFoundError(f"No MP4 found in {cam_dir}")

        cam_name = cam_dir.name
        prefix = f"{cam_name}_"
        stem = video_path.stem
        frame_id = stem[len(prefix):] if stem.startswith(prefix) else stem

    return DetectionResult(
        dataset="gigahands",
        resolved_input=root,
        reason=f"matched GigaHands layout under {root.name} and inferred sequence {seq_name}",
        config={
            "gigahands_root": str(root),
            "annotations_dir": str(annotations_dir),
            "seq_name": seq_name,
            "cam_name": cam_name,
            "frame_id": frame_id,
        },
    )


def infer_hot3d_config(input_path: Path) -> DetectionResult | None:
    if is_hot3d_sequence(input_path) and is_hot3d_root(input_path.parent):
        root = input_path.parent
        sequence_name = input_path.name
    elif is_hot3d_root(input_path):
        root = input_path
        sequence_dir = first_dir(root, is_hot3d_sequence)
        if sequence_dir is None:
            raise FileNotFoundError(f"No HOT3D sequence directory found in {root}")
        sequence_name = sequence_dir.name
    elif input_path.name == "HOT3D" and (input_path / "hot3d_demo_full").is_dir():
        return infer_hot3d_config(input_path / "hot3d_demo_full")
    else:
        return None

    return DetectionResult(
        dataset="hot3d",
        resolved_input=root,
        reason=f"matched HOT3D root and inferred sequence {sequence_name}",
        config={
            "hot3d_root": str(root),
            "sequence_name": sequence_name,
        },
    )


def infer_beingh0_config(input_path: Path) -> DetectionResult | None:
    subset_dir = None
    if is_beingh0_subset(input_path):
        subset_dir = input_path
    elif input_path.name == "Being-h0":
        base = input_path / "h0_post_train_db_2508"
        if base.is_dir():
            subset_dir = first_dir(base, is_beingh0_subset)
    elif (input_path / "h0_post_train_db_2508").is_dir():
        subset_dir = first_dir(input_path / "h0_post_train_db_2508", is_beingh0_subset)

    if subset_dir is None:
        return None

    jsonl_path = first_file(subset_dir, "*_train.jsonl")
    if jsonl_path is None:
        raise FileNotFoundError(f"No *_train.jsonl file found in {subset_dir}")

    return DetectionResult(
        dataset="being-h0",
        resolved_input=subset_dir,
        reason=f"matched Being-H0 subset layout in {subset_dir.name}",
        config={
            "beingh0_subset_dir": str(subset_dir),
            "beingh0_jsonl": str(jsonl_path),
        },
    )


def looks_like_dexwild_hdf5(path: Path) -> bool:
    if path.suffix.lower() not in {".hdf5", ".h5"}:
        return False
    try:
        import h5py

        with h5py.File(path, "r") as f:
            if not f.keys():
                return False
            first_key = sorted(f.keys())[0]
            ep = f[first_key]
            needed = {"right_thumb_cam", "right_pinky_cam", "right_arm_eef", "right_leapv1", "right_leapv2", "right_manus"}
            return needed.issubset(set(ep.keys()))
    except Exception:
        return False


def infer_dexwild_config(input_path: Path) -> DetectionResult | None:
    hdf5_path = None
    if input_path.is_file() and looks_like_dexwild_hdf5(input_path):
        hdf5_path = input_path
    elif input_path.is_dir():
        for candidate in sorted(input_path.glob("*.hdf5")) + sorted(input_path.glob("*.h5")):
            if looks_like_dexwild_hdf5(candidate):
                hdf5_path = candidate
                break
    if hdf5_path is None:
        return None

    import h5py

    with h5py.File(hdf5_path, "r") as f:
        episode_name = sorted(f.keys())[0]

    return DetectionResult(
        dataset="dexwild",
        resolved_input=hdf5_path,
        reason=f"matched DexWild HDF5 structure in {hdf5_path.name}",
        config={
            "dexwild_hdf5": str(hdf5_path),
            "dexwild_episode": episode_name,
        },
    )


def infer_thermohands_config(input_path: Path) -> DetectionResult | None:
    scene_dir = None
    if is_thermohands_scene(input_path):
        scene_dir = input_path
    elif input_path.name == "thermohands":
        scene_dir = first_dir(input_path, is_thermohands_scene)
    elif input_path.is_dir():
        scene_dir = first_dir(input_path, is_thermohands_scene)

    if scene_dir is None:
        return None

    return DetectionResult(
        dataset="thermohands",
        resolved_input=scene_dir,
        reason=f"matched ThermoHands scene layout in {scene_dir.name}",
        config={
            "thermohands_scene_dir": str(scene_dir),
        },
    )


def detect_dataset(input_path: Path, forced_dataset: str) -> DetectionResult:
    resolved = input_path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Input path not found: {resolved}")

    if forced_dataset != "auto":
        return DetectionResult(dataset=forced_dataset, resolved_input=resolved, reason="forced by --dataset", config={})

    detectors = [
        infer_dexwild_config,
        infer_thermohands_config,
        infer_beingh0_config,
        infer_hot3d_config,
        infer_gigahands_config,
    ]
    for detector in detectors:
        result = detector(resolved)
        if result is not None:
            return result

    return DetectionResult(
        dataset="generic",
        resolved_input=resolved,
        reason="no known dataset signature matched; using generic fallback viewer",
        config={},
    )


def summarize_jsonl(path: Path, max_lines: int = 3) -> str:
    lines = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= max_lines:
                break
            lines.append(line.strip())
    return "\n".join(lines) if lines else "(empty)"


def summarize_hdf5(path: Path, max_items: int = 40) -> str:
    try:
        import h5py
    except Exception as exc:
        return f"HDF5 summary unavailable: {exc}"

    lines: list[str] = []

    def walk(name: str, obj):
        if len(lines) >= max_items:
            return
        if isinstance(obj, h5py.Group):
            lines.append(f"[group] {name}")
        else:
            shape = getattr(obj, "shape", None)
            dtype = getattr(obj, "dtype", None)
            lines.append(f"[dataset] {name} shape={shape} dtype={dtype}")

    with h5py.File(path, "r") as f:
        f.visititems(walk)
    if len(lines) >= max_items:
        lines.append("...")
    return "\n".join(lines)


class GenericAutoAdapter:
    def __init__(self, input_path: Path, max_items: int):
        self.input_path = input_path
        self.max_items = max_items
        self.base = "universal_plus/generic"
        self.viewer_name = "universal_plus_generic_dashboard"
        self.mode = "summary"
        self.items: list[Path] = []
        self.video = None

    def load(self):
        path = self.input_path
        if path.is_file():
            suffix = path.suffix.lower()
            if suffix in {".png", ".jpg", ".jpeg", ".bmp"}:
                self.mode = "image"
            elif suffix in {".mp4", ".avi", ".mov", ".mkv"}:
                self.mode = "video"
                self.video = cv2.VideoCapture(str(path))
                if not self.video.isOpened():
                    raise RuntimeError(f"Cannot open video: {path}")
            elif suffix in {".json", ".jsonl", ".txt", ".md", ".csv", ".hdf5", ".h5"}:
                self.mode = "text"
            else:
                self.mode = "summary"
        elif path.is_dir():
            image_candidates = [p for p in sorted(path.rglob("*")) if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
            if image_candidates:
                self.mode = "image_sequence"
                self.items = image_candidates[: self.max_items]
            else:
                self.mode = "summary"

        entries = []
        if path.is_dir():
            children = sorted(path.iterdir())[: self.max_items]
            for child in children:
                kind = "dir" if child.is_dir() else child.suffix.lower() or "file"
                entries.append((child.name, kind))
        else:
            entries.append((path.name, path.suffix.lower() or "file"))

        self.recording_summary = ud.summary_text(
            [
                ("dataset", "generic"),
                ("input", str(path)),
                ("mode", self.mode),
                ("preview_items", len(self.items)),
            ]
        )
        self.frame_summary = ud.summary_text(entries) if entries else "No entries"

    def close(self):
        if self.video is not None:
            self.video.release()

    def log_static(self):
        return None

    def frames(self) -> Iterator[ud.DashboardPanels]:
        path = self.input_path
        if self.mode == "image":
            image = ud.read_image_rgb_unicode_safe(path)
            rr.set_time("frame", sequence=0)
            rr.log(f"{self.base}/camera/preview", rr.Image(image))
            yield ud.DashboardPanels(
                recording_summary=self.recording_summary,
                frame_summary=f"image_file: {path.name}",
                main_task="Generic image preview",
                sub_task=path.name,
                current_action="Showing the input image.",
                interaction="single image preview",
                objects=[path.suffix.lower()],
            )
            return

        if self.mode == "image_sequence":
            for idx, image_path in enumerate(self.items):
                rr.set_time("frame", sequence=idx)
                rr.log(f"{self.base}/camera/preview", rr.Image(ud.read_image_rgb_unicode_safe(image_path)))
                yield ud.DashboardPanels(
                    recording_summary=self.recording_summary,
                    frame_summary=ud.summary_text([("frame_index", idx), ("image", image_path.name)]),
                    main_task="Generic image-sequence preview",
                    sub_task=self.input_path.name,
                    current_action="Showing detected images from the input directory.",
                    interaction=f"image_count={len(self.items)}",
                    objects=["images"],
                )
            return

        if self.mode == "video":
            frame_idx = 0
            while frame_idx < self.max_items:
                ok, frame = self.video.read()
                if not ok:
                    break
                rr.set_time("frame", sequence=frame_idx)
                rr.log(f"{self.base}/camera/preview", rr.Image(frame))
                yield ud.DashboardPanels(
                    recording_summary=self.recording_summary,
                    frame_summary=ud.summary_text([("frame_index", frame_idx), ("video", path.name)]),
                    main_task="Generic video preview",
                    sub_task=path.name,
                    current_action="Showing frames from the input video.",
                    interaction=f"preview_limit={self.max_items}",
                    objects=[path.suffix.lower()],
                )
                frame_idx += 1
            return

        if self.mode == "text":
            if path.suffix.lower() == ".json":
                preview = json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2)[:8000]
            elif path.suffix.lower() == ".jsonl":
                preview = summarize_jsonl(path)
            elif path.suffix.lower() in {".hdf5", ".h5"}:
                preview = summarize_hdf5(path)
            else:
                preview = path.read_text(encoding="utf-8", errors="replace")[:8000]
            rr.set_time("frame", sequence=0)
            yield ud.DashboardPanels(
                recording_summary=self.recording_summary,
                frame_summary=self.frame_summary,
                main_task="Generic structured-data preview",
                sub_task=path.name,
                current_action="Summarizing the input file in text form.",
                interaction=preview,
                objects=[path.suffix.lower()],
            )
            return

        rr.set_time("frame", sequence=0)
        yield ud.DashboardPanels(
            recording_summary=self.recording_summary,
            frame_summary=self.frame_summary,
            main_task="Generic dataset summary",
            sub_task=self.input_path.name,
            current_action="Showing a structural summary because no known dataset signature matched.",
            interaction="Provide --dataset to override detection if needed.",
            objects=[],
        )


class PatchedGigahandsAdapter(ud.GigahandsAdapter):
    def __init__(self, args, gigahands_root: Path, annotations_dir: Path):
        super().__init__(args)
        self.gigahands_root = gigahands_root
        self.annotations_dir = annotations_dir
        self.base = "universal_plus/gigahands"
        self.viewer_name = "universal_plus_gigahands_dashboard"

    def load(self):
        self.mod.GIGAHANDS_ROOT = self.gigahands_root
        self.mod.ANNOTATIONS_DIR = self.annotations_dir
        if not bool(getattr(self.args, "skip_annotations", False)):
            super().load()
            return

        m = self.mod
        m.SEQ_NAME = self.args.seq_name
        m.CAM_NAME = self.args.cam_name
        m.FRAME_ID = self.args.frame_id
        object_id = ud.gigahands_object_id(m.SEQ_NAME)
        m.VIDEO_PATH = m.GIGAHANDS_ROOT / "hand_pose" / m.SEQ_NAME / "rgb_vid" / m.CAM_NAME / f"{m.CAM_NAME}_{m.FRAME_ID}.mp4"
        m.LEFT_2D_PATH = m.GIGAHANDS_ROOT / "hand_pose" / m.SEQ_NAME / "keypoints_2d" / "left" / object_id / f"{m.CAM_NAME}_{m.FRAME_ID}.jsonl"
        m.RIGHT_2D_PATH = m.GIGAHANDS_ROOT / "hand_pose" / m.SEQ_NAME / "keypoints_2d" / "right" / object_id / f"{m.CAM_NAME}_{m.FRAME_ID}.jsonl"
        m.LEFT_3D_PATH = m.GIGAHANDS_ROOT / "hand_pose" / m.SEQ_NAME / "keypoints_3d" / object_id / "left.jsonl"
        m.RIGHT_3D_PATH = m.GIGAHANDS_ROOT / "hand_pose" / m.SEQ_NAME / "keypoints_3d" / object_id / "right.jsonl"
        m.MESH_PATH = ud.resolve_gigahands_mesh_path(m.GIGAHANDS_ROOT, m.SEQ_NAME)
        m.POSE_PATH = m.GIGAHANDS_ROOT / "object_pose" / m.SEQ_NAME / "pose" / "optimized_pose.json"

        ud.require_paths([
            m.VIDEO_PATH,
            m.LEFT_2D_PATH,
            m.RIGHT_2D_PATH,
            m.LEFT_3D_PATH,
            m.RIGHT_3D_PATH,
            m.MESH_PATH,
            m.POSE_PATH,
        ])

        self.left_2d = m.load_2d(m.LEFT_2D_PATH)
        self.right_2d = m.load_2d(m.RIGHT_2D_PATH)
        self.left_3d = m.load_3d(m.LEFT_3D_PATH)
        self.right_3d = m.load_3d(m.RIGHT_3D_PATH)
        self.mesh_vertices, self.mesh_faces = m.load_mesh(m.MESH_PATH)
        self.gt_steps = []
        self.pred_raw_clips = []
        self.pred_steps = []
        self.poses = ud.load_json(m.POSE_PATH)
        self.poses = m.interpolate_poses(self.poses)
        self.scene_registry = m.build_scene_object_registry(m.SEQ_NAME)
        self.timeline_runs = []

        self.cap = cv2.VideoCapture(str(m.VIDEO_PATH))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {m.VIDEO_PATH}")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.total_frames <= 0:
            self.total_frames = max(len(self.left_2d), len(self.right_2d), len(self.left_3d), len(self.right_3d), 1)

        self.recording_summary = ud.summary_text(
            [
                ("dataset", "gigahands"),
                ("sequence", m.SEQ_NAME),
                ("camera", m.CAM_NAME),
                ("frame_id", m.FRAME_ID),
                ("total_frames", self.total_frames),
                ("scene_objects", ", ".join(sorted({label for item in self.scene_registry for label in item["labels"]})) or "None"),
                ("pred_steps", 0),
                ("pred_raw_clips", 0),
                ("annotations", "disabled"),
            ]
        )


def make_known_args(dataset: str) -> SimpleNamespace:
    defaults = ud.build_parser().parse_args(["--dataset", "hot3d" if dataset == "hot3d" else dataset])
    return SimpleNamespace(**vars(defaults))


def apply_overrides(config_args: SimpleNamespace, cli_args) -> SimpleNamespace:
    overrides = {
        "skip_annotations": getattr(cli_args, "skip_annotations", None),
        "seq_name": cli_args.seq_name,
        "cam_name": cli_args.cam_name,
        "frame_id": cli_args.frame_id,
        "sequence_name": cli_args.sequence_name,
        "frame_stride": cli_args.frame_stride,
        "device": cli_args.device,
        "beingh0_subset_dir": cli_args.beingh0_subset_dir,
        "beingh0_jsonl": cli_args.beingh0_jsonl,
        "beingh0_start": cli_args.beingh0_start,
        "beingh0_max_samples": cli_args.beingh0_max_samples,
        "dexwild_hdf5": cli_args.dexwild_hdf5,
        "dexwild_episode": cli_args.dexwild_episode,
        "dexwild_max_frames": cli_args.dexwild_max_frames,
        "thermohands_scene_dir": cli_args.thermohands_scene_dir,
        "thermohands_stride": cli_args.thermohands_stride,
        "thermohands_max_frames": cli_args.thermohands_max_frames,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(config_args, key, value)
    return config_args


def create_adapter_from_detection(detection: DetectionResult, cli_args):
    if detection.dataset == "generic":
        return GenericAutoAdapter(detection.resolved_input, cli_args.generic_max_items)

    args = apply_overrides(make_known_args(detection.dataset), cli_args)
    args.dataset = detection.dataset
    for key, value in detection.config.items():
        setattr(args, key, value)
    args = apply_overrides(args, cli_args)

    if detection.dataset == "gigahands":
        return PatchedGigahandsAdapter(args, Path(args.gigahands_root), Path(args.annotations_dir))
    if detection.dataset == "hot3d":
        adapter = ud.Hot3DManoAdapter(args)
        adapter.base = "universal_plus/hot3d"
        adapter.viewer_name = "universal_plus_hot3d_dashboard"
        return adapter
    if detection.dataset == "being-h0":
        adapter = ud.BeingH0Adapter(args)
        adapter.base = "universal_plus/being_h0"
        adapter.viewer_name = "universal_plus_being_h0_dashboard"
        return adapter
    if detection.dataset == "dexwild":
        adapter = ud.DexWildAdapter(args)
        adapter.base = "universal_plus/dexwild"
        adapter.viewer_name = "universal_plus_dexwild_dashboard"
        return adapter
    if detection.dataset == "thermohands":
        adapter = ud.ThermoHandsAdapter(args)
        adapter.base = "universal_plus/thermohands"
        adapter.viewer_name = "universal_plus_thermohands_dashboard"
        return adapter
    raise ValueError(f"Unsupported dataset: {detection.dataset}")


def main():
    args = parser_with_default_spawn().parse_args()
    detection = detect_dataset(args.input, args.dataset)

    print(f"Detected dataset: {detection.dataset}")
    print(f"Resolved input : {detection.resolved_input}")
    print(f"Reason         : {detection.reason}")
    if detection.config:
        for key, value in detection.config.items():
            print(f"{key:15}: {value}")

    if args.describe_only:
        return

    adapter = create_adapter_from_detection(detection, args)
    adapter.load()

    rr.init(adapter.viewer_name, spawn=args.spawn)
    rr.send_blueprint(ud.create_shared_blueprint(adapter.base))
    adapter.log_static()

    try:
        for panels in adapter.frames():
            ud.log_dashboard_panels(adapter.base, panels)
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
