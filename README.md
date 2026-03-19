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

Read the CLI reference:

- [`docs/cli-args.md`](c:/Users/WelcoFa/Desktop/相能/rerun/docs/cli-args.md)

Repo overview:

- [`docs/overview.md`](c:/Users/WelcoFa/Desktop/相能/rerun/docs/overview.md)

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

## Notes

- The repo defaults to relative paths under `data/`.
- The maintained public scripts also support absolute dataset roots through CLI options such as `--data-root`, which is useful when your payloads live on a local NAS.
- Some moved GigaHands variants are still present in `scripts/`, but they should be treated as experimental until their CLIs are normalized the same way as the public entrypoints.
