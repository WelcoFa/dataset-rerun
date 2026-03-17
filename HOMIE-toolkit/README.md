<p align="center">
  <a href="https://ropedia.com/">
    <img src="assets/logo.png" alt="HOMIE-toolkit logo" width="400" />
  </a>
  <br />
  <em>Interactive Intelligence from Human Xperience</em>
</p>

# HOMIE-toolkit

Tools for **reading** and **visualizing** [Xperience-10M](https://huggingface.co/datasets/ropedia-ai/xperience-10m) data.

- Load annotation and use the data in your own scripts (export, training, custom viz)
- Reuse visualization helpers (depth colormap, skeleton, point cloud) with Rerun

## 📁 Layout

| Path | Description |
|------|-------------|
| `data_loader.py` | Load `annotation.hdf5` (calibration, SLAM, hand/body mocap, depth, IMU, point cloud); list contents and load video frames. |
| `visualization.py` | Helpers: `create_blueprint`, `depth_to_colormap`, `depth_to_pointcloud`, `build_line3d_skeleton`. |
| `examples/example_load_annotation.py` | List HDF5 contents, load annotation, inspect calibration. |
| `examples/example_visualize_rrd.py` | Log skeleton + depth to a Rerun `.rrd` file; open with `rerun vis.rrd`. |

## 📦 Install

```bash
conda create -n homie python=3.12
conda activate homie
pip install -r requirements.txt
```

## 🚀 Getting Started

Download sample data [here](https://huggingface.co/datasets/ropedia-ai/xperience-10m-sample).

### 📋 List Annotations

```bash
python examples/example_load_annotation.py --data_root /path/to/episode
```

Example output (top-level structure + loaded summary):

```
--- annotation.hdf5 contents (top-level) ---
  calibration: group    (cam0, cam01, cam1, cam2, cam3: K, T_c_b, ...)
  depth: group           (depth, confidence, depth_min, depth_max, scale)
  full_body_mocap: group (keypoints, contacts, body_quats, ...)
  hand_mocap: group     (left_joints_3d, right_joints_3d, mano params)
  imu: group            (device_timestamp_ns, accel_xyz, gyro_xyz, keyframe_indices)
  slam: group           (quat_wxyz, trans_xyz, frame_names, point_cloud)
  caption: ...          metadata: ...

--- Loaded data summary ---
  Frames (img_names): N
  R_c2w_all: (N, 3, 3)   t_c2w_all: (N, 3)
  Hand left/right joints: (N, 21, 3)   Full-body keypoints: (N, 52, 3)
  Contacts: (N, 21)   Depth: lazy loader, N frames   IMU: M samples

--- Calibration ---
  cam01.K, cam0–cam3 T_c_b: available

Done. Use these arrays for your own processing or pass to example_visualize_rrd.py.
```

### 🎬 Visualize with Rerun

```bash
python examples/example_visualize_rrd.py --data_root /path/to/episode --output_rrd vis.rrd
```

Then open the Rerun viewer: `rerun vis.rrd`

![Rerun visualization](./assets/rerun.png)

