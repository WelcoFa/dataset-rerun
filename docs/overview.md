# Overview

This repository is a collection of organized visualization scripts for public datasets, with Rerun as the default viewer.

The goal is simple:

- keep public dataset viewers in one place under `scripts/`
- keep dataset payloads under `data/`
- make common inspection workflows reproducible with `uv`
- keep visual layouts consistent across datasets, inspired by the HOMIE-toolkit approach

## Viewer Conventions

The default viewer is always Rerun.

Where possible, the scripts follow a shared layout:

- a primary 3D world panel
- a time panel
- a right-side camera or modality panel
- stable entity naming under `world/...` and `camera/...`

## Repo Layout

```text
scripts/   organized entrypoints by role, plus experimental and tools folders
docs/      install guide, CLI reference, and docs viewer
data/      dataset payloads, ignored by git
```

## Public Workflows

- `scripts/visualize/visualize_beingh0_subset.py`
- `scripts/visualize/visualize_dexwild_preview.py`
- `scripts/visualize/visualize_gigahands_single_scene.py`
- `scripts/visualize/visualize_hot3d_scene.py`
- `scripts/visualize/visualize_hot3d_mano.py`
- `scripts/visualize/visualize_hot3d_mano_albedo.py`
- `scripts/visualize/visualize_hot3d_skeleton.py`

## Experimental Workflows

Experimental GigaHands scripts are kept under `scripts/experimental/`, and utility helpers are under `scripts/tools/`.
