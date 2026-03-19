# CLI Arguments

This page is generated from the repo's maintained CLI registry.

## Public Scripts

### `visualize_beingh0_subset.py`

Visualize a Being-H0 subset with RGB frames, proprioception, and action chunks.

- `--data-root`: Base data directory. Supports a relative repo `data/` root or an absolute NAS path. Default: `None`.
- `--subset-dir`: Explicit subset directory. Overrides `--data-root`. Default: `None`.
- `--jsonl`: Explicit `*_train.jsonl` path. Default: `None`.
- `--start`: Start sample index. Default: `0`.
- `--max-samples`: Maximum number of samples to visualize. Default: `-1`.
- `--step-sleep`: Seconds to sleep between samples. Default: `0.08`.
- `--spawn`: Spawn the Rerun viewer. Default: `False`.

### `visualize_dexwild_preview.py`

Preview a DexWild HDF5 episode with synchronized robot state and camera images.

- `--data-root`: Base data directory. Supports a relative repo `data/` root or an absolute NAS path. Default: `None`.
- `--hdf5`: Explicit HDF5 file path. Overrides `--data-root`. Default: `None`.
- `--episode`: Episode name inside the HDF5 file. Default: `ep_0000`.
- `--max-frames`: Maximum number of frames to preview. Default: `-1`.
- `--spawn`: Spawn the Rerun viewer. Default: `False`.

### `visualize_gigahands_single_scene.py`

Visualize one GigaHands scene with RGB, 2D/3D hands, and object pose.

- `--data-root`: Base data directory. Supports a relative repo `data/` root or an absolute NAS path. Default: `None`.
- `--scene-name`: GigaHands scene name. Default: `p36-tea-0010`.
- `--cam-name`: Camera name. Default: `brics-odroid-010_cam0`.
- `--video-stem`: Video stem without extension. Default: `brics-odroid-010_cam0_1727030430697198`.
- `--object-id`: Object/frame id folder for annotation files. Default: `010`.
- `--mesh-path`: Explicit mesh path. Overrides automatic discovery. Default: `None`.
- `--spawn`: Spawn the Rerun viewer. Default: `False`.

### `visualize_hot3d_scene.py`

Visualize a HOT3D sequence with object and hand tracks.

- `--data-root`: Base data directory. Supports a relative repo `data/` root or an absolute NAS path. Default: `None`.
- `--sequence-name`: HOT3D sequence name. Default: `P0001_10a27bf7`.
- `--frame-stride`: Playback frame stride. Default: `5`.
- `--spawn`: Spawn the Rerun viewer. Default: `False`.

### `visualize_hot3d_mano.py`

Visualize a HOT3D sequence with MANO hand reconstruction.

- `--data-root`: Base data directory. Supports a relative repo `data/` root or an absolute NAS path. Default: `None`.
- `--sequence-name`: HOT3D sequence name. Default: `P0001_10a27bf7`.
- `--frame-stride`: Playback frame stride. Default: `10`.
- `--object-scale`: Scale applied to object meshes. Default: `0.001`.
- `--device`: Torch device for MANO inference. Default: `cpu`.
- `--show-labels`: Show labels in the viewer. Default: `False`.
- `--spawn`: Spawn the Rerun viewer. Default: `False`.

### `visualize_hot3d_mano_albedo.py`

Visualize a HOT3D sequence with MANO and albedo-aware mesh handling.

- `--data-root`: Base data directory. Supports a relative repo `data/` root or an absolute NAS path. Default: `None`.
- `--sequence-name`: HOT3D sequence name. Default: `P0001_10a27bf7`.
- `--frame-stride`: Playback frame stride. Default: `10`.
- `--object-scale`: Scale applied to object meshes. Default: `0.001`.
- `--device`: Torch device for MANO inference. Default: `cpu`.
- `--show-labels`: Show labels in the viewer. Default: `False`.
- `--spawn`: Spawn the Rerun viewer. Default: `False`.

### `visualize_hot3d_skeleton.py`

Visualize a HOT3D sequence with a lightweight skeleton layout.

- `--data-root`: Base data directory. Supports a relative repo `data/` root or an absolute NAS path. Default: `None`.
- `--sequence-name`: HOT3D sequence name. Default: `P0001_10a27bf7`.
- `--frame-stride`: Playback frame stride. Default: `5`.
- `--object-scale`: Scale applied to object meshes. Default: `0.001`.
- `--show-labels`: Show labels in the viewer. Default: `False`.
- `--spawn`: Spawn the Rerun viewer. Default: `False`.

## Experimental Scripts

The following scripts were moved into `scripts/` but still need CLI normalization before they are treated as stable public entrypoints:

- `visualize_gigahands_multi_scene.py`
- `visualize_gigahands_eval.py`
- `visualize_gigahands_eval_test.py`
- `visualize_gigahands_vlm.py`
- `visualize_gigahands_ropedia.py`
- `visualize_gigahands_ropedia_test.py`
- `run_gigahands_vlm.py`
- `inspect_gigahands_tree.py`
