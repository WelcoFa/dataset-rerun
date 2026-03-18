# Multi-Dataset Rerun Visualizer for Hand-Object Interaction

This project provides a unified visualization toolkit for multiple state-of-the-art hand-object interaction datasets using the Rerun framework. It is designed to help researchers and developers efficiently inspect, debug, and compare multimodal data across different datasets with varying formats and annotation standards.

We support visualization for five representative datasets, including:
- GigaHands
- HOT3D
- DexWild
- ThermoHands (unfinished)
- Being-H0

Each dataset has its own structure, annotation format, and available modalities (e.g., RGB video, 2D/3D hand keypoints, object pose, mesh, thermal data). This project standardizes the visualization pipeline across these datasets, enabling consistent rendering of temporal sequences, spatial annotations, and 3D information.

The visualizers are built on top of Rerun, allowing synchronized playback of:
- RGB frames
- 2D annotations (keypoints, bounding boxes)
- 3D hand joints
- Object pose and meshes (when available)
- Additional modalities such as thermal signals or robot states

## Data Layout

All dataset payloads now live under the top-level `data/` directory.

Examples:
- `data/gigahands/gigahands_demo_all/`
- `data/gigahands/annotations/`
- `data/HOT3D/hot3d_demo_full/`
- `data/dexwild/robot_pour_data.hdf5`
- `data/thermohands/Thermohands/`
- `data/Being-h0/h0_post_train_db_2508/`

## GigaHands

GigaHands is a large-scale dataset for hand-object interaction, providing multimodal annotations including RGB video, 2D/3D hand keypoints, object poses, and high-quality object meshes. It is designed for tasks such as hand tracking, interaction understanding, and vision-language modeling.

### Supported Modalities

The GigaHands tools in this repository are organized into two main stages:

1. **Dataset visualization**
2. **VLM inference + evaluation visualization**

The normal workflow is:

Raw GigaHands data
    ↓
visualize_gigahands.py / visualize_gigahands_single.py
    ↓
(optional) run_vlm_gigahands.py
    ↓
visualize_gigahands_eval.py

---

### Dataset Setup

Download GigaHands dataset and place it under:

data/gigahands/gigahands_demo_all/

Example structure:

data/gigahands/
├── gigahands_demo_all/
│   ├── hand_pose/
│   ├── object_pose/
│   └── scans_publish/
└── annotations/

### Visualizer Scripts

We provide multiple scripts for different use cases:
- `visualize_gigahands_single.py`
- `run_vlm_gigahands.py`
- `visualize_gigahands_eval.py`
---

### 1. `visualize_gigahands_single.py`

This is the starting point for inspecting a raw GigaHands sequence.

#### What it does

This script loads one GigaHands scene and visualizes the original dataset content in Rerun. Depending on what is available in the selected scene, it can display:

- RGB video frames
- 2D left/right hand keypoints
- 2D bounding boxes
- 3D hand joints
- object pose
- object mesh

It is mainly used to verify that:
- the dataset paths are correct
- the scene loads correctly
- annotations align with the video
- the object mesh and pose are reasonable
- playback timing looks normal

#### Input

Typical inputs are:
- one selected GigaHands scene
- the scene video
- hand annotations
- object pose files
- mesh files from `scans_publish`

#### Output

This script does not usually create a new result file.  
Its main output is the live visualization in the Rerun viewer.

#### Example role in the pipeline

Use this script to answer:

- Does this scene load correctly?
- Are the hands aligned with the video?
- Is the mesh path correct?
- Is this the scene I want to evaluate with the VLM?

### 2. `run_vlm_gigahands.py`

This script runs a Vision-Language Model on the selected GigaHands sequence.

#### What it does

This script reads the video from a GigaHands scene, splits it into temporal clips, samples a small number of frames from each clip, and sends those sampled frames to a VLM such as Qwen2.5-VL.

The model then produces semantic predictions for each clip, for example:
- interaction descriptions
- action summaries
- returned labels or text outputs
- clip-level reasoning results

The predictions are saved into a structured JSON file for later visualization.

#### Why it is separate from the visualizer

This file is the inference step, not the visualization step.  
Its job is to generate model outputs from the dataset.

#### Input

Typical inputs are:
- the scene video
- clip length settings
- frame sampling settings
- the selected VLM model
- prompt or label settings, depending on your version of the script

#### Processing procedure

A typical procedure inside this script is:

1. Open the target video
2. Divide the full sequence into clips
3. Sample a few representative frames from each clip
4. Send those frames to the VLM
5. Parse the model response
6. Save the response into a JSON result file

#### Output

The main output is a JSON file containing clip-level predictions.

This JSON is later used by `visualize_gigahands_eval.py`.

#### Example role in the pipeline

Use this script to answer:

- What does the VLM predict for this scene?
- What semantic results do I get clip by clip?
- Can I export model results into a format that Rerun can read?

---

### 3. `visualize_gigahands_eval.py`

This script visualizes the VLM output together with the original GigaHands sequence.

#### What it does

This script loads:
- the original GigaHands scene
- the exported JSON result from `run_vlm_gigahands.py`

It then replays the scene in Rerun while showing the VLM predictions only at the clips or frames where results exist.

For example, if the JSON contains only 4 returned predictions, the viewer should show those 4 results rather than repeating labels on every frame.

This script is used for evaluation-style viewing:
- compare model output with the scene
- inspect when each prediction appears
- check whether clip timing is correct
- debug alignment between inference output and video playback

#### Input

Typical inputs are:
- the original scene video
- the same scene annotations if needed
- the exported JSON from `run_vlm_gigahands.py`

#### Output
 
Its main output is the Rerun evaluation view.

#### Example role in the pipeline

Use this script to answer:

- Do the VLM results appear at the correct clip times?
- Are the predicted labels aligned with the video?
- Does the exported JSON look correct in a viewer?
- How does model output compare to the original scene?

---

## HOT3D

HOT3D is a high-quality hand-object interaction dataset that provides accurate 3D annotations of hands, objects, and camera motion. It includes:

- Object pose trajectories
- Headset (camera) trajectory
- MANO-based hand pose parameters
- Object meshes (with texture / UV)
- Timestamp-synchronized multi-modal data

This visualizer focuses on reconstructing the full 3D scene using MANO hand models and textured object meshes.

---

## Key Visualizer

### `visualize_hot3d_mano.py`

This is the main visualization script for HOT3D.

It reconstructs a full 3D scene by combining:
- object motion (dynamic_objects.csv)
- hand motion (MANO parameters)
- headset trajectory
- object meshes with texture

and renders everything in Rerun as a time-synchronized 3D scene.

---

## What the Script Does (Core Pipeline)

The script follows this pipeline:

### Step 1 — Load dataset files

It loads:

- `metadata.json` → object mapping and scene info
- `dynamic_objects.csv` → object pose per timestamp
- `headset_trajectory.csv` → camera motion
- `mano_hand_pose_trajectory.jsonl` → hand pose (MANO parameters)

These are indexed by timestamp for synchronized playback.

---

### Step 2 — Load object meshes

From:object_models/

Each object is:

- loaded as a `trimesh`
- processed to preserve:
  - UV coordinates
  - texture image (if available)
  - vertex colors (fallback)

Then:
- mesh is centered (local coordinates)
- stored for later transformation

---

### Step 3 — Initialize MANO hand model

The script uses:

- `MANO_LEFT.pkl`
- `MANO_RIGHT.pkl`

via `smplx`

Each frame:
- converts dataset pose → MANO parameters
- generates:
  - hand mesh (vertices)
  - hand joints

---

### Step 4 — Time synchronization

All data is aligned by: timestamp_ns → time_sec

In Rerun:

```python
rr.set_time("time_sec", ...)
```
### Step 5 — Per-frame rendering loop

For each timestamp:

## (1) Objects

apply:
- rotation (quaternion → rotation matrix)
 - translation
 - transform mesh vertices into world coordinates

- render using:
 - rr.Mesh3D(...)
 - Texture priority:
 - texture image (best)
 - vertex color
 - fallback color

## (2) Hands (MANO reconstruction)
- For each hand, extract:
 - shape (betas)
 - pose (45D articulation)
 - wrist transform
 - convert to MANO input
 - generate mesh + joints

- Render:
 - mesh → Mesh3D
 - joints → Points3D

## (3) Headset (camera trajectory)
- rendered as a 3D point
- shows camera movement in space

## (4) Summary panel
- Each frame logs:
 - frame index
 - timestamp
 - number of objects
 - whether hand/headset data exists

### Data Flow Summary
CSV / JSONL / JSON
        ↓
Index by timestamp
        ↓
MANO + Object Mesh Loading
        ↓
Per-frame transform (pose)
        ↓
Rerun logging
        ↓
3D interactive visualization

### Setup:
- Install dependencies:
 - pip install rerun-sdk numpy torch trimesh smplx

### Configure Paths:
- Inside the script:
 - SEQUENCE_NAME = "P0001_xxxxxxxx"

### Run
- python visualize_hot3d_mano.py
