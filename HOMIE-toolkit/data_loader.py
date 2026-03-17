"""
Xperience-10M: data loading API.

Read annotation.hdf5 and related assets.
"""

import cv2
import h5py
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from utils.calibration_utils import (
    load_calibration_from_annotation_hdf5,
    get_T_camera_body,
    get_fisheye_T_world_cam,
)
from utils.constants_utils import MANO_PARENT_INDICES, SMPL_H_BODY_PARENT_INDICES
from utils.video_utils import load_video_frame
from utils.caption_utils import load_caption_data_from_annotation_hdf5


def load_from_annotation_hdf5(annotation_path, start_idx, end_idx, slam_poses_are_world_to_body=True, load_slam_point_cloud=True, point_cloud_percentile=95.0):
    """
    Load visualization data from annotation.hdf5.

    Returns dict with: calib_data, R_c2w_all, t_c2w_all, img_names, depth_loader,
    depth_min, depth_max, depth_num_frames, hand_left_joints, hand_right_joints,
    smplh_body_joints, contacts, imu_ts, imu_accel_xyz, imu_gyro_xyz, imu_keyframe_indices,
    ground_height, slam_point_cloud, caption_main_task, caption_frame_info_map, caption_segment_boundaries, caption_task_to_id.
    """
    ann_path = str(annotation_path)
    imu_ts = None
    imu_accel_xyz = None
    imu_gyro_xyz = None
    imu_keyframe_indices = None
    with h5py.File(annotation_path, "r") as f:
        calib_data = load_calibration_from_annotation_hdf5(f)
        if calib_data is None:
            raise KeyError("annotation.hdf5 must contain calibration/ group.")

        if "slam/quat_wxyz" in f and "slam/trans_xyz" in f and "slam/frame_names" in f:
            quat_wxyz = np.array(f["slam/quat_wxyz"][...])
            trans_xyz = np.array(f["slam/trans_xyz"][...])
            frame_names_ds = f["slam/frame_names"]
            img_names = [np.array(frame_names_ds[i]).tobytes().decode("utf-8", errors="replace").strip("\x00") for i in range(frame_names_ds.shape[0])]
            N = len(img_names)
            if end_idx is None or end_idx == -1:
                end_idx = N
            end_idx = min(end_idx, N)
            quat_wxyz = quat_wxyz[start_idx:end_idx]
            trans_xyz = trans_xyz[start_idx:end_idx]
            img_names = img_names[start_idx:end_idx]
            T_c0_b = None
            if slam_poses_are_world_to_body:
                cam01 = calib_data.get("cam01") if calib_data else None
                if cam01 is not None and "T_c0_b" in cam01:
                    T_c0_b = np.array(cam01["T_c0_b"], dtype=np.float64)
            R_c2w_list = []
            t_c2w_list = []
            for i in range(len(quat_wxyz)):
                qw, qx, qy, qz = quat_wxyz[i]
                R_w2c = R.from_quat([qx, qy, qz, qw]).as_matrix()
                t_w2c = trans_xyz[i].copy()
                if T_c0_b is not None:
                    T_w2b = np.eye(4)
                    T_w2b[:3, :3] = R_w2c
                    T_w2b[:3, 3] = t_w2c
                    T_w2c = T_c0_b @ T_w2b
                    R_w2c = T_w2c[:3, :3]
                    t_w2c = T_w2c[:3, 3]
                R_c2w = R_w2c.T
                t_c2w = -R_c2w @ t_w2c
                R_c2w_list.append(R_c2w)
                t_c2w_list.append(t_c2w)
            R_c2w_all = np.stack(R_c2w_list)
            t_c2w_all = np.stack(t_c2w_list)
        else:
            N = None
            for key in ("hand_mocap/left_joints_3d", "depth/depth", "full_body_mocap/keypoints"):
                if key in f:
                    N = f[key].shape[0]
                    break
            if N is None:
                raise KeyError("annotation.hdf5 has no slam/ and no frame-length dataset to infer N.")
            if end_idx is None or end_idx == -1:
                end_idx = N
            end_idx = min(end_idx, N)
            n_slice = end_idx - start_idx
            img_names = [f"frame_{start_idx + i:06d}.jpg" for i in range(n_slice)]
            R_c2w_all = np.tile(np.eye(3), (n_slice, 1, 1))
            t_c2w_all = np.zeros((n_slice, 3))

        depth_num_frames = 0
        depth_min = 0.0
        depth_max = 1.0
        scale = 1.0
        upsample_ratio = 1.0
        if "depth/depth" in f:
            depth_ds = f["depth/depth"]
            depth_num_frames_raw = depth_ds.shape[0]
            end_idx_d = depth_num_frames_raw if (end_idx is None or end_idx == -1) else min(end_idx, depth_num_frames_raw)
            depth_num_frames = end_idx_d - start_idx
            if "depth/scale" in f:
                scale = float(np.array(f["depth/scale"][...]).flat[0])
                upsample_ratio = 1.0 / scale
            if "depth/depth_min" in f:
                depth_min = float(np.array(f["depth/depth_min"][...]).flat[0])
            if "depth/depth_max" in f:
                depth_max = float(np.array(f["depth/depth_max"][...]).flat[0])
            has_confidence = "depth/confidence" in f

            def depth_loader(frame_idx):
                if frame_idx < 0 or frame_idx >= depth_num_frames:
                    return None, None
                global_idx = start_idx + frame_idx
                with h5py.File(ann_path, "r") as h5f:
                    depth_frame = np.array(h5f["depth/depth"][global_idx], dtype=np.float32)
                    confidence_frame = np.array(h5f["depth/confidence"][global_idx], dtype=np.uint8) if has_confidence else None
                if scale < 1.0 and upsample_ratio > 1.0:
                    H, W = depth_frame.shape
                    new_H, new_W = int(H * upsample_ratio), int(W * upsample_ratio)
                    depth_frame = cv2.resize(depth_frame, (new_W, new_H), interpolation=cv2.INTER_NEAREST)
                    if confidence_frame is not None:
                        confidence_frame = cv2.resize(confidence_frame, (new_W, new_H), interpolation=cv2.INTER_NEAREST)
                return depth_frame, confidence_frame

            depth_result = (depth_loader, depth_min, depth_max, depth_num_frames)
        else:
            depth_result = (None, depth_min, depth_max, 0)

        hand_left_joints = None
        hand_right_joints = None
        if "hand_mocap/left_joints_3d" in f:
            left_j = np.array(f["hand_mocap/left_joints_3d"][...])
            right_j = np.array(f["hand_mocap/right_joints_3d"][...]) if "hand_mocap/right_joints_3d" in f else np.zeros_like(left_j)
            n = left_j.shape[0]
            e = min(end_idx, n) if end_idx != -1 else n
            hand_left_joints = left_j[start_idx:e]
            hand_right_joints = right_j[start_idx:e]

        smplh_body_joints = None
        contacts = None
        if "full_body_mocap/keypoints" in f:
            kp = np.array(f["full_body_mocap/keypoints"][...])
            if kp.ndim >= 1 and kp.shape[0] == 1:
                kp = kp.reshape(kp.shape[1:])
            n = kp.shape[0]
            e = min(end_idx, n) if end_idx != -1 else n
            smplh_body_joints = kp[start_idx:e]
        if "full_body_mocap/contacts" in f:
            c = np.array(f["full_body_mocap/contacts"][...])
            if c.ndim >= 1 and c.shape[0] == 1:
                c = c.reshape(c.shape[1:])
            n = c.shape[0]
            e = min(end_idx, n) if end_idx != -1 else n
            contacts = c[start_idx:e]

        if "imu/device_timestamp_ns" in f and "imu/accel_xyz" in f and "imu/gyro_xyz" in f:
            imu_ts = np.array(f["imu/device_timestamp_ns"][...]).flatten()
            imu_accel_xyz = np.array(f["imu/accel_xyz"][...])
            imu_gyro_xyz = np.array(f["imu/gyro_xyz"][...])
            if "imu/keyframe_indices" in f:
                imu_keyframe_indices = np.array(f["imu/keyframe_indices"][...]).flatten().astype(np.int64)

        ground_height = -1.75
        if "ground_height" in f:
            ground_height = float(np.asarray(f["ground_height"][...]).flat[0])
        elif "floor_z" in f:
            ground_height = float(np.asarray(f["floor_z"][...]).flat[0])
        elif "body_height" in f:
            ground_height = -float(np.asarray(f["body_height"][...]).flat[0])
        elif "metadata" in f:
            g = f["metadata"]
            if "ground_height" in g:
                ground_height = float(np.asarray(g["ground_height"][...]).flat[0])
            elif "floor_z" in g:
                ground_height = float(np.asarray(g["floor_z"][...]).flat[0])
            elif "body_height" in g:
                ground_height = -float(np.asarray(g["body_height"][...]).flat[0])

        slam_point_cloud = None
        if load_slam_point_cloud and "slam/point_cloud" in f:
            points = np.asarray(f["slam/point_cloud"][...], dtype=np.float32)
            if points.ndim == 2 and points.shape[1] == 3 and len(points) > 0:
                center = np.median(points, axis=0)
                dist = np.linalg.norm(points - center, axis=1)
                thresh = np.percentile(dist, point_cloud_percentile)
                slam_point_cloud = points[dist <= thresh]

    data_root = str(Path(annotation_path).parent)
    caption_main_task, caption_frame_info_map, caption_segment_boundaries, caption_task_to_id = load_caption_data_from_annotation_hdf5(
        annotation_path, data_root, img_names
    )

    return {
        "calib_data": calib_data,
        "R_c2w_all": R_c2w_all,
        "t_c2w_all": t_c2w_all,
        "img_names": img_names,
        "depth_loader": depth_result[0],
        "depth_min": depth_result[1],
        "depth_max": depth_result[2],
        "depth_num_frames": depth_result[3],
        "hand_left_joints": hand_left_joints,
        "hand_right_joints": hand_right_joints,
        "smplh_body_joints": smplh_body_joints,
        "contacts": contacts,
        "imu_ts": imu_ts,
        "imu_accel_xyz": imu_accel_xyz,
        "imu_gyro_xyz": imu_gyro_xyz,
        "imu_keyframe_indices": imu_keyframe_indices,
        "ground_height": ground_height,
        "slam_point_cloud": slam_point_cloud,
        "caption_main_task": caption_main_task,
        "caption_frame_info_map": caption_frame_info_map,
        "caption_segment_boundaries": caption_segment_boundaries,
        "caption_task_to_id": caption_task_to_id,
    }


def _format_scalar_for_list(val):
    """Format a single HDF5 scalar for display (number or short string)."""
    if isinstance(val, (np.ndarray, np.generic)):
        if val.size == 0:
            return "[]"
        raw = val.flat[0]
        if isinstance(raw, (bytes, str)):
            s = raw.decode("utf-8", errors="replace").strip("\x00") if isinstance(raw, bytes) else str(raw)
            return s[:60] + "..." if len(s) > 60 else s
        if np.issubdtype(getattr(val.dtype, "base", val.dtype), np.floating):
            return float(raw)
        if np.issubdtype(getattr(val.dtype, "base", val.dtype), np.integer):
            return int(raw)
        return str(raw)
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace").strip("\x00")[:60]
    return val


def list_annotation_contents(annotation_path):
    """List groups/datasets in annotation.hdf5: groups as 'group', arrays as shape, scalars as value."""
    out = {}

    def _visit(name, obj):
        if isinstance(obj, h5py.Group):
            out[name] = "group"
        else:
            try:
                shape = obj.shape
                if shape == ():
                    out[name] = _format_scalar_for_list(obj[()])
                else:
                    out[name] = shape
            except Exception:
                out[name] = "?"

    with h5py.File(annotation_path, "r") as f:
        f.visititems(_visit)
    return out


__all__ = [
    "load_from_annotation_hdf5",
    "load_calibration_from_annotation_hdf5",
    "load_video_frame",
    "list_annotation_contents",
    "get_T_camera_body",
    "get_fisheye_T_world_cam",
    "MANO_PARENT_INDICES",
    "SMPL_H_BODY_PARENT_INDICES",
]
