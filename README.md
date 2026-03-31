# Multi-Dataset Rerun Visualizers

This repository is a collection of organized scripts for visualizing public datasets with Rerun.

The purpose of the repo is to keep dataset viewers in one place, keep data layout predictable, and make it easy for other people to reproduce the same workflows with `uv` instead of hand-managed environments.

## What This Repo Contains

- public visualization entrypoints under `scripts/`
- documentation under `docs/`
- dataset payloads under `data/`
- a shared preference for Rerun as the default viewer

The viewer direction is inspired by HOMIE-toolkit:

- Rerun is the default viewer
- a consistent world-first layout is preferred
- camera views live alongside the main 3D scene
- stable path naming under `world/...` and `camera/...` is preferred where possible

## Start Here

Open the docs viewer:

- [`docs/index.html`](c:/Users/WelcoFa/Desktop/相能/rerun/docs/index.html)

Read the install guide:

- [`docs/install.md`](c:/Users/WelcoFa/Desktop/相能/rerun/docs/install.md)

Read the Docker/NAS guide:

- [`docs/docker.md`](c:/Users/WelcoFa/Desktop/相能/rerun/docs/docker.md)

The Docker dashboard now exposes a basic app shell with ready-to-play config selection and session logs on port `8080`.

Read the CLI reference:

- [`docs/cli-args.md`](c:/Users/WelcoFa/Desktop/相能/rerun/docs/cli-args.md)

Repo overview:

- [`docs/overview.md`](c:/Users/WelcoFa/Desktop/相能/rerun/docs/overview.md)

Dataset trees:

- `docs/dataset-trees.md`

## Layout

```text
scripts/   visualization entrypoints
docs/      install guide, overview, CLI docs, and local docs viewer
data/      dataset payloads, ignored by git
```

## Public Scripts

- `scripts/visualize_beingh0_subset.py`
- `scripts/visualize_dexwild_preview.py`
- `scripts/visualize_gigahands_single_scene.py`
- `scripts/visualize_hot3d_scene.py`
- `scripts/visualize_hot3d_mano.py`
- `scripts/visualize_hot3d_mano_albedo.py`
- `scripts/visualize_hot3d_skeleton.py`

## Universal Dashboard

The repo now also includes a shared multi-dataset dashboard entrypoint:

- `scripts/visualize_universal_dashboard.py`

It opens a common Rerun layout with:

- a 2D camera/modality panel
- a 3D world panel
- text panels for recording summary, frame summary, task fields, interaction, and objects
- a timeline panel for dataset-specific scalar streams

Supported datasets:

- `gigahands`
- `hot3d` / `hot3d-mano`
- `being-h0`
- `dexwild`
- `thermohands`

Example commands:

```powershell
uv run python scripts/visualize_universal_dashboard.py --dataset gigahands
uv run python scripts/visualize_universal_dashboard.py --dataset hot3d --sequence-name P0001_10a27bf7
uv run python scripts/visualize_universal_dashboard.py --dataset being-h0 --beingh0-max-samples 200
uv run python scripts/visualize_universal_dashboard.py --dataset dexwild --dexwild-episode ep_0000
uv run python scripts/visualize_universal_dashboard.py --dataset thermohands --thermohands-scene-dir data/thermohands/cut_paper
```

Useful dataset-specific flags include:

- GigaHands: `--seq-name`, `--cam-name`, `--frame-id`
- HOT3D: `--hot3d-root`, `--sequence-name`, `--frame-stride`, `--device`
- Being-H0: `--beingh0-subset-dir`, `--beingh0-jsonl`, `--beingh0-start`, `--beingh0-max-samples`
- DexWild: `--dexwild-hdf5`, `--dexwild-episode`, `--dexwild-max-frames`
- ThermoHands: `--thermohands-scene-dir`, `--thermohands-stride`, `--thermohands-max-frames`

## GigaHands Workflow

The current GigaHands workflow has three layers:

- `scripts/visualize_gigahands_single_scene.py` for scene inspection with RGB, 2D hands, 3D hands, and object pose
- `scripts/run_gigahands_vlm.py` for clip-level semantic prediction with fields such as `sub_task`, `interaction`, `objects`, `label`, and `current_action`
- `scripts/visualize_gigahands_eval_test.py` for a ROPedia-style semantic viewer that reads generated annotation JSONs and shows:
  - `Main Task`
  - `Sub Task`
  - `Interaction`
  - `Objects`
  - `Current Action`
  - `Task Timeline`

The generated GigaHands annotation files live under `data/gigahands/annotations/`, for example:

- `pred_raw_clips_p36-tea-0010.json`
- `pred_steps_p36-tea-0010.json`

The eval-test viewer uses:

- raw clip predictions for the live semantic panels
- merged step predictions for the task timeline
- GT steps only when they are present

The universal dashboard can also be used as a higher-level GigaHands viewer. It reuses the generated semantic JSONs and combines:

- RGB playback
- 2D hand keypoints
- 3D hand keypoints
- object pose and mesh
- semantic text panels
- a timeline view of predicted steps

For the VLM workflow, install the extra dependencies first:

```powershell
uv sync --extra gigahands-vlm
```

The current VLM runner is tuned for a rougher, faster pass by default:

- frame sampling count is configurable with `GIGAHANDS_NUM_SAMPLED_FRAMES`
- batch size is configurable with `GIGAHANDS_BATCH_SIZE`
- generated token budget is configurable with `GIGAHANDS_MAX_NEW_TOKENS`
- sampled frame resize is configurable with `GIGAHANDS_MAX_IMAGE_EDGE`

If you want GPU inference with PyTorch on Windows, install a CUDA-enabled PyTorch build instead of the CPU-only default wheel.

For HOT3D MANO playback, install the extra dependencies first:

```powershell
uv sync --extra hot3d-mano
```

## Notes

- The repo defaults to relative paths under `data/`.
- The maintained public scripts also support absolute dataset roots through CLI options such as `--data-root`, which is useful when your payloads live on a local NAS.
- Some moved GigaHands variants are still present in `scripts/`, but they should be treated as experimental until their CLIs are normalized the same way as the public entrypoints.
- The GigaHands evaluation viewer currently expects annotation-style semantic JSONs rather than running object detection online inside the viewer.
