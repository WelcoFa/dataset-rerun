import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import rerun as rr


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_SCENE_DIR = REPO_ROOT / "data" / "thermohands" / "cut_paper"

HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

LEFT_COLOR = np.array([0, 170, 255], dtype=np.uint8)
RIGHT_COLOR = np.array([255, 90, 90], dtype=np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize one ThermoHands scene in Rerun.")
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=DEFAULT_SCENE_DIR,
        help="ThermoHands scene directory containing rgb/depth/ir/thermal/gt_info.",
    )
    parser.add_argument(
        "--spawn",
        action="store_true",
        help="Spawn the Rerun viewer automatically.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Log every Nth frame.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Maximum number of frames to log (-1 means all).",
    )
    return parser.parse_args()


def require_dir(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")


def list_sorted_files(path: Path, suffix: str) -> list[Path]:
    files = sorted(p for p in path.glob(f"*{suffix}") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No '{suffix}' files found in: {path}")
    return files


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_image_unicode_safe(path: Path, flags: int) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(f"Failed to read bytes: {path}")
    image = cv2.imdecode(data, flags)
    if image is None:
        raise ValueError(f"Failed to decode image: {path}")
    return image


def read_rgb(path: Path) -> np.ndarray:
    image_bgr = read_image_unicode_safe(path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def normalize_to_u8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros(image.shape[:2], dtype=np.uint8)

    vals = image[finite].astype(np.float32)
    lo = float(vals.min())
    hi = float(vals.max())
    if hi <= lo:
        return np.zeros(image.shape[:2], dtype=np.uint8)

    scaled = (image.astype(np.float32) - lo) / (hi - lo)
    scaled = np.clip(scaled * 255.0, 0.0, 255.0)
    return scaled.astype(np.uint8)


def read_gray_preview(path: Path) -> np.ndarray:
    image = read_image_unicode_safe(path, cv2.IMREAD_UNCHANGED)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return normalize_to_u8(image)


def colorize_gray(gray_u8: np.ndarray, colormap: int) -> np.ndarray:
    bgr = cv2.applyColorMap(gray_u8, colormap)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def log_hand(path: str, points_xyz: np.ndarray, color_rgb: np.ndarray):
    color_batch = np.repeat(color_rgb[None, :], len(points_xyz), axis=0)
    rr.log(
        f"{path}/joints",
        rr.Points3D(points_xyz, colors=color_batch, radii=0.008),
    )

    lines = [
        np.stack([points_xyz[a], points_xyz[b]], axis=0)
        for a, b in HAND_BONES
        if a < len(points_xyz) and b < len(points_xyz)
    ]
    if lines:
        rr.log(
            f"{path}/bones",
            rr.LineStrips3D(lines, colors=[color_rgb], radii=0.003),
        )


def load_annotation(path: Path) -> dict[str, np.ndarray]:
    raw = load_json(path)
    out = {}
    for key in ["kps3D_L", "kps3D_R", "trans_L", "trans_R"]:
        if key in raw:
            out[key] = np.asarray(raw[key], dtype=np.float32).reshape(-1, 3)
    return out


def create_blueprint():
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial2DView(origin="thermohands/camera/rgb", name="RGB"),
                    rrb.Spatial2DView(origin="thermohands/camera/thermal", name="Thermal"),
                ),
                rrb.Horizontal(
                    rrb.Spatial2DView(origin="thermohands/camera/ir", name="IR"),
                    rrb.Spatial2DView(origin="thermohands/camera/depth", name="Depth"),
                ),
            ),
            rrb.Vertical(
                rrb.Spatial3DView(origin="thermohands/world", name="3D Hands"),
                rrb.TextDocumentView(origin="thermohands/meta/scene", name="Scene Info"),
                rrb.TextDocumentView(origin="thermohands/meta/frame", name="Frame Info"),
            ),
        ),
        collapse_panels=True,
    )


def main():
    args = parse_args()
    if args.stride <= 0:
        raise ValueError("--stride must be >= 1")

    scene_dir = args.scene_dir.resolve()
    require_dir(scene_dir)

    rgb_dir = scene_dir / "rgb"
    thermal_dir = scene_dir / "thermal"
    ir_dir = scene_dir / "ir"
    depth_dir = scene_dir / "depth"
    gt_dir = scene_dir / "gt_info"

    for directory in [rgb_dir, thermal_dir, ir_dir, depth_dir, gt_dir]:
        require_dir(directory)

    rgb_files = list_sorted_files(rgb_dir, ".png")
    thermal_files = list_sorted_files(thermal_dir, ".png")
    ir_files = list_sorted_files(ir_dir, ".png")
    depth_files = list_sorted_files(depth_dir, ".png")
    gt_files = list_sorted_files(gt_dir, ".json")

    total_frames = min(len(rgb_files), len(thermal_files), len(ir_files), len(depth_files), len(gt_files))
    if total_frames <= 0:
        raise RuntimeError("No aligned ThermoHands frames found.")

    if args.max_frames > 0:
        total_frames = min(total_frames, args.max_frames)

    rr.init("thermohands_scene_viewer", spawn=args.spawn)
    rr.send_blueprint(create_blueprint())

    rr.log("thermohands/world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    rr.log(
        "thermohands/meta/scene",
        rr.TextDocument(
            "\n".join(
                [
                    "ThermoHands scene viewer",
                    f"scene_dir: {scene_dir}",
                    f"frames: {total_frames}",
                    f"stride: {args.stride}",
                    "modalities: rgb, thermal, ir, depth",
                    "annotations: kps3D_L, kps3D_R, trans_L, trans_R",
                ]
            ),
            media_type="text/plain",
        ),
        static=True,
    )

    for frame_idx in range(0, total_frames, args.stride):
        rr.set_time("frame", sequence=frame_idx)

        rgb = read_rgb(rgb_files[frame_idx])
        thermal_gray = read_gray_preview(thermal_files[frame_idx])
        ir_gray = read_gray_preview(ir_files[frame_idx])
        depth_gray = read_gray_preview(depth_files[frame_idx])
        ann = load_annotation(gt_files[frame_idx])

        rr.log("thermohands/camera/rgb", rr.Image(rgb))
        rr.log("thermohands/camera/thermal", rr.Image(colorize_gray(thermal_gray, cv2.COLORMAP_INFERNO)))
        rr.log("thermohands/camera/ir", rr.Image(colorize_gray(ir_gray, cv2.COLORMAP_BONE)))
        rr.log("thermohands/camera/depth", rr.Image(colorize_gray(depth_gray, cv2.COLORMAP_TURBO)))

        if "kps3D_L" in ann:
            log_hand("thermohands/world/left_hand", ann["kps3D_L"], LEFT_COLOR)
        if "kps3D_R" in ann:
            log_hand("thermohands/world/right_hand", ann["kps3D_R"], RIGHT_COLOR)
        if "trans_L" in ann:
            rr.log(
                "thermohands/world/left_hand/root",
                rr.Points3D(ann["trans_L"], colors=[LEFT_COLOR], radii=0.012),
            )
        if "trans_R" in ann:
            rr.log(
                "thermohands/world/right_hand/root",
                rr.Points3D(ann["trans_R"], colors=[RIGHT_COLOR], radii=0.012),
            )

        left_depth = float(ann["trans_L"][0, 2]) if "trans_L" in ann else float("nan")
        right_depth = float(ann["trans_R"][0, 2]) if "trans_R" in ann else float("nan")
        rr.log("thermohands/plots/left_root_depth_m", rr.Scalars([left_depth]))
        rr.log("thermohands/plots/right_root_depth_m", rr.Scalars([right_depth]))

        rr.log(
            "thermohands/meta/frame",
            rr.TextDocument(
                "\n".join(
                    [
                        f"frame_index: {frame_idx}",
                        f"rgb_file: {rgb_files[frame_idx].name}",
                        f"thermal_file: {thermal_files[frame_idx].name}",
                        f"ir_file: {ir_files[frame_idx].name}",
                        f"depth_file: {depth_files[frame_idx].name}",
                        f"gt_file: {gt_files[frame_idx].name}",
                        f"left_root_depth_m: {left_depth:.4f}",
                        f"right_root_depth_m: {right_depth:.4f}",
                    ]
                ),
                media_type="text/plain",
            ),
        )

    print(f"[INFO] ThermoHands scene: {scene_dir}")
    print(f"[INFO] Logged frames: {len(range(0, total_frames, args.stride))}")
    print("[INFO] Done. Scrub the timeline in the Rerun viewer.")


if __name__ == "__main__":
    main()
