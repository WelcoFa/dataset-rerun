# Docker Deployment

This repo now includes a first Dockerized setup with two modes:

- `rerun-dashboard`: basic app shell plus headless Rerun dashboard service for NAS/browser access
- `multidataset-vlm`: batch semantic prediction worker

## What This First Version Does

The dashboard service now runs a small app entrypoint:

- `scripts/serve/serve_dashboard_app.py`

It uses the existing adapters in `rerun_viz/` and:

- serves a simple selection UI on port `8080`
- lists ready-to-play config files from `configs/`
- starts one dashboard session at a time
- captures session logs and status
- logs the dataset into a Rerun recording
- serves the Rerun gRPC endpoint on port `9876`
- serves the Rerun web viewer on port `9090`
- optionally saves an `.rrd` recording under `/outputs`

This works better on a NAS than `spawn=True`, which expects a local desktop viewer.

## Files

- `app_ui/`
- `Dockerfile`
- `.dockerignore`
- `docker-compose.yml`
- `scripts/serve/serve_dashboard_app.py`
- `scripts/serve/serve_rerun_dashboard.py`

## Quick Start

Build and run the dashboard:

```bash
docker compose up --build rerun-dashboard
```

Open:

```text
http://<your-nas-host>:8080
```

The compose file mounts:

- `./data` -> `/data`
- `./outputs` -> `/outputs`

Useful overrides:

- `DATA_ROOT=/absolute/path/to/datasets`
- `OUTPUT_ROOT=/absolute/path/to/outputs`
- `APP_PORT=8080`
- `RERUN_WEB_PORT=9090`
- `RERUN_GRPC_PORT=9876`

## Batch VLM Worker

Build and run the VLM worker only when needed:

```bash
docker compose --profile vlm up --build multidataset-vlm
```

The VLM worker writes annotation JSONs into the dataset tree, matching the current script behavior.

Useful VLM overrides:

- `MODEL_CACHE_ROOT=/absolute/path/to/model-cache`
- `MULTIDATASET_VLM_MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct`
- `MULTIDATASET_BATCH_SIZE=1`
- `MULTIDATASET_NUM_SAMPLED_FRAMES=3`

## Notes

- The first version assumes your datasets are mounted into `./data`.
- The app shell uses ready-to-play config files from `configs/` and launches `scripts/serve/serve_rerun_dashboard.py` behind the scenes.
- WIYH pointcloud loading is disabled in `configs/docker-dashboard.toml` for the initial container setup because it needs extra `laspy[lazrs]` dependencies.
- HOT3D MANO still needs the `hot3d-mano` extra and MANO assets if you want that mode in the container.
- The VLM worker may need network access the first time it downloads model weights into `/models/huggingface`.
