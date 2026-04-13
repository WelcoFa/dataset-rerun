# Install

This project uses `uv` so other people can recreate the environment quickly and consistently.

## 1. Install uv

Follow the official `uv` install instructions for your platform, then verify:

```bash
uv --version
```

## 2. Create the environment

From the repo root:

```bash
uv venv
```

Activate it:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

## 3. Sync the base dependencies

```bash
uv sync
```

This installs the shared visualization stack used by the repo.

## 4. Install optional extras when needed

HOT3D MANO scripts:

```bash
uv sync --extra hot3d-mano
```

GigaHands VLM scripts:

```bash
uv sync --extra gigahands-vlm
```

Developer tooling:

```bash
uv sync --extra dev
```

## 5. Put dataset payloads under data/

The default relative layout is:

```text
data/
  Being-h0/
  HOT3D/
  dexwild/
  gigahands/
  thermohands/
```

You can also keep the payloads on a local NAS and point the scripts at that absolute location with `--data-root`.

## 6. Run the docs viewer

Open:

```text
docs/index.html
```

The docs viewer includes a night mode toggle and links to all repo documentation.

## Dataset Setup

## Being-H0

```bash
uv run python scripts/visualize/visualize_beingh0_subset.py --spawn
```

NAS example:

```bash
uv run python scripts/visualize/visualize_beingh0_subset.py --data-root Z:\datasets --spawn
```

## DexWild

```bash
uv run python scripts/visualize/visualize_dexwild_preview.py --spawn
```

NAS example:

```bash
uv run python scripts/visualize/visualize_dexwild_preview.py --data-root Z:\datasets --episode ep_0000 --spawn
```

## GigaHands

Single-scene viewer:

```bash
uv run python scripts/visualize/visualize_gigahands_single_scene.py --spawn
```

NAS example:

```bash
uv run python scripts/visualize/visualize_gigahands_single_scene.py --data-root Z:\datasets --scene-name p36-tea-0010 --spawn
```

Experimental GigaHands scripts are available under `scripts/experimental/`, but the single-scene viewer is the stable starting point.

## HOT3D

Basic scene:

```bash
uv run python scripts/visualize/visualize_hot3d_scene.py --spawn
```

MANO viewer:

```bash
uv run python scripts/visualize/visualize_hot3d_mano.py --spawn
```

Albedo-aware MANO viewer:

```bash
uv run python scripts/visualize/visualize_hot3d_mano_albedo.py --spawn
```

Skeleton view:

```bash
uv run python scripts/visualize/visualize_hot3d_skeleton.py --spawn
```

NAS example:

```bash
uv run python scripts/visualize/visualize_hot3d_scene.py --data-root Z:\datasets --sequence-name P0001_10a27bf7 --spawn
```
