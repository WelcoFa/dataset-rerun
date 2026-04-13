import argparse
from pathlib import Path

import h5py
import numpy as np
import rerun as rr


def parse_args():
    parser = argparse.ArgumentParser(description="Preview DexWild HDF5 episode in Rerun.")
    parser.add_argument(
        "--hdf5",
        type=str,
        default="robot_pour_data.hdf5",
        help="Path to DexWild HDF5 file",
    )
    parser.add_argument(
        "--episode",
        type=str,
        default="ep_0000",
        help="Episode name inside HDF5, e.g. ep_0000",
    )
    parser.add_argument(
        "--spawn",
        action="store_true",
        help="Spawn Rerun viewer automatically",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Maximum number of frames to preview (-1 = all)",
    )
    return parser.parse_args()


def sorted_image_keys(group: h5py.Group) -> list[str]:
    return sorted(group.keys(), key=lambda x: int(x.split(".")[0]))


def read_h5_any(ds):
    """
    Robust reader:
    - scalar dataset -> ds[()]
    - array dataset  -> ds[:]
    """
    try:
        if getattr(ds, "shape", None) == ():
            return ds[()]
        return ds[:]
    except Exception:
        return ds[()]


def load_episode(ep: h5py.Group):
    eef = read_h5_any(ep["right_arm_eef"]["right_arm_eef"])    # (N, 8)
    leapv1 = read_h5_any(ep["right_leapv1"]["right_leapv1"])   # (N, 17)
    leapv2 = read_h5_any(ep["right_leapv2"]["right_leapv2"])   # (N, 18)
    manus = read_h5_any(ep["right_manus"]["right_manus"])      # (N, 71)

    # timesteps may be scalar or array
    timesteps = read_h5_any(ep["timesteps"]["timesteps"])

    thumb_group = ep["right_thumb_cam"]
    pinky_group = ep["right_pinky_cam"]

    thumb_keys = sorted_image_keys(thumb_group)
    pinky_keys = sorted_image_keys(pinky_group)

    return {
        "eef": np.asarray(eef),
        "leapv1": np.asarray(leapv1),
        "leapv2": np.asarray(leapv2),
        "manus": np.asarray(manus),
        "timesteps": timesteps,
        "thumb_group": thumb_group,
        "pinky_group": pinky_group,
        "thumb_keys": thumb_keys,
        "pinky_keys": pinky_keys,
    }


def get_n_frames(data: dict) -> int:
    counts = [
        len(data["eef"]),
        len(data["leapv1"]),
        len(data["leapv2"]),
        len(data["manus"]),
        len(data["thumb_keys"]),
        len(data["pinky_keys"]),
    ]
    return min(counts)


def get_timestamp_ns(data: dict, i: int) -> int:
    """
    Prefer explicit timesteps if they are a real array.
    Otherwise fall back to image filename timestamp.
    Finally fall back to first column of eef.
    """
    ts = data["timesteps"]

    # case 1: timesteps is array-like
    try:
        ts_arr = np.asarray(ts)
        if ts_arr.ndim > 0 and ts_arr.shape[0] > i:
            ts_i = ts_arr[i]
            if np.isscalar(ts_i):
                return int(ts_i)

            ts_i = np.asarray(ts_i).reshape(-1)
            if ts_i.size > 0:
                return int(ts_i[0])
    except Exception:
        pass

    # case 2: use image filename timestamp
    try:
        return int(data["thumb_keys"][i].split(".")[0])
    except Exception:
        pass

    # case 3: use first column of eef
    return int(data["eef"][i, 0])


def log_static(episode: str, n_frames: int):
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "meta/info",
        rr.TextDocument(
            f"DexWild HDF5 preview\n"
            f"episode: {episode}\n"
            f"frames: {n_frames}\n"
            f"- right_thumb_cam\n"
            f"- right_pinky_cam\n"
            f"- right_arm_eef\n"
            f"- right_leapv1\n"
            f"- right_leapv2\n"
            f"- right_manus",
            media_type="text/plain",
        ),
        static=True,
    )


def log_eef(eef_row: np.ndarray):
    # row = [timestamp, x, y, z, qx, qy, qz, qw]
    xyz = eef_row[1:4].astype(np.float32)
    quat_xyzw = eef_row[4:8].astype(np.float32)

    rr.log("robot/eef/point", rr.Points3D([xyz], radii=0.01))
    rr.log(
        "robot/eef/pose",
        rr.Transform3D(
            translation=xyz,
            rotation=rr.Quaternion(xyzw=quat_xyzw),
        ),
    )


def log_joint_series(path_prefix: str, row: np.ndarray):
    vals = row[1:]  # first value is timestamp
    for j, v in enumerate(vals):
        rr.log(f"{path_prefix}/j{j:02d}", rr.Scalars([float(v)]))


def main():
    args = parse_args()
    hdf5_path = Path(args.hdf5)

    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    rr.init("dexwild_hdf5_preview", spawn=args.spawn)

    with h5py.File(hdf5_path, "r") as f:
        if args.episode not in f:
            raise KeyError(
                f"Episode '{args.episode}' not found.\n"
                f"Available examples: {list(f.keys())[:10]}"
            )

        ep = f[args.episode]
        data = load_episode(ep)

        n_frames = get_n_frames(data)
        if args.max_frames > 0:
            n_frames = min(n_frames, args.max_frames)

        log_static(args.episode, n_frames)

        print(f"[INFO] HDF5: {hdf5_path}")
        print(f"[INFO] Episode: {args.episode}")
        print(f"[INFO] Frames: {n_frames}")

        thumb_group = data["thumb_group"]
        pinky_group = data["pinky_group"]
        thumb_keys = data["thumb_keys"]
        pinky_keys = data["pinky_keys"]

        eef = data["eef"]
        leapv1 = data["leapv1"]
        leapv2 = data["leapv2"]
        manus = data["manus"]

        eef_traj = []

        for i in range(n_frames):
            ts_ns = get_timestamp_ns(data, i)

            rr.set_time("frame", sequence=i)
            rr.set_time("time", timestamp=np.datetime64(ts_ns, "ns"))

            # images
            thumb_img = np.asarray(thumb_group[thumb_keys[i]][:])
            pinky_img = np.asarray(pinky_group[pinky_keys[i]][:])

            rr.log("camera/right_thumb", rr.Image(thumb_img))
            rr.log("camera/right_pinky", rr.Image(pinky_img))

            # eef
            eef_row = eef[i]
            log_eef(eef_row)

            xyz = eef_row[1:4].astype(np.float32)
            eef_traj.append(xyz.copy())
            rr.log("robot/eef/trajectory", rr.LineStrips3D([np.array(eef_traj, dtype=np.float32)]))

            # hand
            if i < len(leapv1):
                log_joint_series("hand/right_leapv1", leapv1[i])

            if i < len(leapv2):
                log_joint_series("hand/right_leapv2", leapv2[i])

            if i < len(manus):
                log_joint_series("hand/right_manus", manus[i])

            rr.log(
                "meta/frame_text",
                rr.TextLog(
                    f"frame={i}, ts_ns={ts_ns}, "
                    f"thumb={thumb_keys[i]}, pinky={pinky_keys[i]}"
                ),
            )

    print("[INFO] Done. Scrub timeline in Rerun Viewer.")


if __name__ == "__main__":
    main()
