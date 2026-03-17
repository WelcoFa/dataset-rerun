"""
Calibration and camera pose utilities for Xperience-10M.
"""

import numpy as np


def _read_calibration_from_hdf5_group(grp):
    """Recursively read calibration/ group from HDF5 into a dict."""
    out = {}
    for key in grp.keys():
        item = grp[key]
        if hasattr(item, "keys"):
            out[key] = _read_calibration_from_hdf5_group(item)
        else:
            arr = np.array(item[...])
            if arr.ndim == 0:
                scalar = arr.flat[0]
                if arr.dtype.kind in ("S", "U", "O") or (isinstance(scalar, (bytes, str))):
                    out[key] = scalar.decode("utf-8", errors="replace").strip("\x00") if isinstance(scalar, bytes) else str(scalar)
                else:
                    out[key] = float(scalar)
            else:
                out[key] = arr
    return out


def load_calibration_from_annotation_hdf5(f):
    """Load full calibration from annotation.hdf5 calibration/ group."""
    if "calibration" not in f:
        return None
    return _read_calibration_from_hdf5_group(f["calibration"])


def get_T_camera_body(calib_data, cam_id):
    """Get camera extrinsics T_c_b (camera-to-body) from calibration data."""
    if calib_data is None or cam_id not in calib_data:
        return None
    cam_data = calib_data[cam_id]
    if "T_c_b" in cam_data:
        return np.array(cam_data["T_c_b"], dtype=np.float32)
    if "T_cn_cnm1" in cam_data:
        T_cn_cnm1 = np.array(cam_data["T_cn_cnm1"], dtype=np.float32)
        cam_list = ["cam0", "cam1", "cam2", "cam3"]
        if cam_id in cam_list:
            idx = cam_list.index(cam_id)
            if idx > 0:
                prev_T_c_b = get_T_camera_body(calib_data, cam_list[idx - 1])
                if prev_T_c_b is not None:
                    T_cn_cnm1_inv = np.linalg.inv(T_cn_cnm1)
                    return T_cn_cnm1_inv @ prev_T_c_b
    return None


def get_fisheye_T_world_cam(calib_data, R_c2w_stereo, t_c2w_stereo):
    """Get fisheye camera poses T_world_cam (world-to-camera) in world coordinates."""
    fisheye_poses = {}
    cam01 = calib_data.get("cam01") if calib_data else None
    if cam01 is None or "T_c0_b" not in cam01:
        return fisheye_poses
    T_c0_b = np.array(cam01["T_c0_b"], dtype=np.float32)
    T_b_c_stereo = np.linalg.inv(T_c0_b)
    T_world_stereo = np.eye(4)
    T_world_stereo[:3, :3] = R_c2w_stereo
    T_world_stereo[:3, 3] = t_c2w_stereo
    T_world_body = T_world_stereo @ np.linalg.inv(T_b_c_stereo)
    T_c0_b = get_T_camera_body(calib_data, "cam0")
    if T_c0_b is not None:
        T_b_c0 = np.linalg.inv(T_c0_b)
        fisheye_poses["cam0"] = T_world_body @ T_b_c0
    cam_list = ["cam0", "cam1", "cam2", "cam3"]
    for i in range(1, len(cam_list)):
        cam_id = cam_list[i]
        prev_cam_id = cam_list[i - 1]
        if cam_id not in calib_data:
            continue
        cam_data = calib_data[cam_id]
        if "T_c_b" in cam_data:
            T_b_cam = np.linalg.inv(np.array(cam_data["T_c_b"], dtype=np.float32))
            fisheye_poses[cam_id] = T_world_body @ T_b_cam
            continue
        if prev_cam_id not in fisheye_poses or "T_cn_cnm1" not in cam_data:
            continue
        T_cnm1_cn = np.linalg.inv(np.array(cam_data["T_cn_cnm1"], dtype=np.float32))
        fisheye_poses[cam_id] = fisheye_poses[prev_cam_id] @ T_cnm1_cn
    return fisheye_poses


__all__ = [
    "load_calibration_from_annotation_hdf5",
    "get_T_camera_body",
    "get_fisheye_T_world_cam",
]
