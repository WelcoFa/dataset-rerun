from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import rerun as rr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rerun_viz.config import build_config
from rerun_viz.core import create_shared_blueprint
from rerun_viz.registry import resolve_adapter


def parse_web_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a Rerun dashboard over gRPC + web viewer for headless/container use.")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON/TOML/YAML config file.")
    parser.add_argument("--input", type=Path, default=None, help="Dataset root, scene folder, subset folder, or source file.")
    parser.add_argument(
        "--dataset",
        choices=["auto", "gigahands", "hot3d", "being-h0", "dexwild", "thermohands", "wiyh", "generic"],
        default=None,
        help="Force dataset type or let the registry auto-detect it.",
    )
    parser.add_argument("--spawn", action=argparse.BooleanOptionalAction, default=None, help="Ignored here; always forced off.")
    parser.add_argument(
        "--view-mode",
        choices=["core", "enriched", "both"],
        default=None,
        help="Control whether to log raw visualization, enrichments, or both.",
    )
    parser.add_argument(
        "--enrichment",
        action="append",
        default=None,
        help="Enable one or more enrichment names. Repeatable.",
    )
    parser.add_argument("--grpc-port", type=int, default=9876, help="Port for the Rerun gRPC log server.")
    parser.add_argument("--web-port", type=int, default=9090, help="Port for the hosted Rerun web viewer.")
    parser.add_argument(
        "--keep-alive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the process alive after all frames are logged so the web viewer stays available.",
    )
    parser.add_argument(
        "--save-recording",
        type=Path,
        default=None,
        help="Optional .rrd output path to save alongside the live web session.",
    )
    parser.add_argument("--selection-json", type=str, default=None, help="Optional JSON object overriding config selection.")
    parser.add_argument(
        "--dataset-options-json",
        type=str,
        default=None,
        help="Optional JSON object overriding config dataset_options.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_web_args()
    config = build_config(
        SimpleNamespace(
            config=args.config,
            input=args.input,
            dataset=args.dataset,
            spawn=False,
            view_mode=args.view_mode,
            enrichment=args.enrichment,
        )
    )
    config.spawn = False
    if args.selection_json is not None:
        selection_override = json.loads(args.selection_json)
        if not isinstance(selection_override, dict):
            raise ValueError("--selection-json must decode to a JSON object")
        config.selection.update(selection_override)
    if args.dataset_options_json is not None:
        dataset_options_override = json.loads(args.dataset_options_json)
        if not isinstance(dataset_options_override, dict):
            raise ValueError("--dataset-options-json must decode to a JSON object")
        config.dataset_options.update(dataset_options_override)

    adapter = resolve_adapter(config)
    adapter.load()

    rr.init(adapter.viewer_name, spawn=False)
    blueprint = adapter.create_blueprint()
    if blueprint is None:
        blueprint = create_shared_blueprint(adapter.base)

    server_uri = rr.serve_grpc(grpc_port=args.grpc_port, default_blueprint=blueprint)
    rr.serve_web_viewer(web_port=args.web_port, open_browser=False, connect_to=server_uri)

    if args.save_recording is not None:
        args.save_recording.parent.mkdir(parents=True, exist_ok=True)
        rr.save(args.save_recording, default_blueprint=blueprint)

    rr.send_blueprint(blueprint)
    adapter.log_static()

    frame_count = 0
    try:
        for panels in adapter.frames():
            adapter.log_panels(panels)
            frame_count += 1

        print(f"Dashboard stream ready at http://0.0.0.0:{args.web_port}")
        print(f"Viewer connected to {server_uri}")
        print(f"Logged {frame_count} frame(s)")
        if args.save_recording is not None:
            print(f"Saved Rerun recording to {args.save_recording}")

        if args.keep_alive:
            stop = False

            def _stop_handler(signum, _frame):
                nonlocal stop
                stop = True
                print(f"Received signal {signum}, shutting down.")

            signal.signal(signal.SIGINT, _stop_handler)
            signal.signal(signal.SIGTERM, _stop_handler)

            while not stop:
                time.sleep(1.0)
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
