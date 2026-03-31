from __future__ import annotations

import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
import mimetypes
import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rerun_viz.config.loader import load_config_file


APP_UI_DIR = REPO_ROOT / "app_ui"
SERVE_SCRIPT = REPO_ROOT / "scripts" / "serve_rerun_dashboard.py"


@dataclass
class PlayableItem:
    item_id: str
    label: str
    path: Path
    dataset: str
    input_path: str
    description: str
    valid: bool
    error: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("dashboard_app")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


class DashboardAppState:
    def __init__(
        self,
        config_dir: Path,
        outputs_dir: Path,
        viewer_port: int,
        grpc_port: int,
        logger: logging.Logger,
    ) -> None:
        self.config_dir = config_dir
        self.outputs_dir = outputs_dir
        self.viewer_port = viewer_port
        self.grpc_port = grpc_port
        self.logger = logger
        self.lock = threading.RLock()
        self.log_lines: deque[str] = deque(maxlen=300)
        self.process: subprocess.Popen[str] | None = None
        self.current_item_id: str | None = None
        self.current_recording: str | None = None
        self.status = "idle"
        self.last_error: str | None = None
        self.started_at: str | None = None
        self.items = self._discover_items()

    def _is_port_in_use(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            return sock.connect_ex(("127.0.0.1", port)) == 0

    def _wait_for_ports_to_release(self, timeout: float = 10.0) -> None:
        deadline = time.time() + timeout
        ports = [self.viewer_port, self.grpc_port]
        while time.time() < deadline:
            busy_ports = [port for port in ports if self._is_port_in_use(port)]
            if not busy_ports:
                return
            time.sleep(0.2)
        busy_ports = [port for port in ports if self._is_port_in_use(port)]
        if busy_ports:
            raise RuntimeError(f"Ports still busy after shutdown: {', '.join(str(port) for port in busy_ports)}")

    def _discover_items(self) -> list[PlayableItem]:
        items: list[PlayableItem] = []
        if not self.config_dir.exists():
            return items

        for path in sorted(self.config_dir.iterdir()):
            if path.suffix.lower() not in {".json", ".toml", ".yaml", ".yml"}:
                continue
            if path.name == "docker-dashboard.toml":
                continue

            item_id = path.stem
            try:
                payload = load_config_file(path)
                dataset = str(payload.get("dataset", "auto"))
                input_path = str(payload.get("input", ""))
                description = f"{dataset} | {input_path or 'no input'}"
                items.append(
                    PlayableItem(
                        item_id=item_id,
                        label=path.stem.replace("-", " ").replace("_", " "),
                        path=path,
                        dataset=dataset,
                        input_path=input_path,
                        description=description,
                        valid=True,
                    )
                )
            except Exception as exc:
                items.append(
                    PlayableItem(
                        item_id=item_id,
                        label=path.stem,
                        path=path,
                        dataset="invalid",
                        input_path="",
                        description="Config parse failed",
                        valid=False,
                        error=str(exc),
                    )
                )
        return items

    def list_items(self) -> list[dict[str, Any]]:
        with self.lock:
            return [
                {
                    "id": item.item_id,
                    "label": item.label,
                    "path": str(item.path),
                    "dataset": item.dataset,
                    "input": item.input_path,
                    "description": item.description,
                    "valid": item.valid,
                    "error": item.error,
                    "active": item.item_id == self.current_item_id and self.status in {"starting", "running"},
                }
                for item in self.items
            ]

    def list_recordings(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        if not self.outputs_dir.exists():
            return items
        for path in sorted(self.outputs_dir.rglob("*.rrd")):
            stat = path.stat()
            try:
                rel_path = path.resolve().relative_to(REPO_ROOT.resolve())
            except ValueError:
                rel_path = Path(os.path.relpath(path.resolve(), REPO_ROOT.resolve()))
            items.append(
                {
                    "name": path.name,
                    "path": str(rel_path),
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
                }
            )
        return items

    def get_status(self) -> dict[str, Any]:
        with self.lock:
            grpc_url = f"rerun+http://localhost:{self.grpc_port}/proxy"
            viewer_url = f"http://localhost:{self.viewer_port}/?url={quote(grpc_url, safe='')}"
            return {
                "status": self.status,
                "current_item_id": self.current_item_id,
                "started_at": self.started_at,
                "viewer_url": viewer_url,
                "grpc_url": grpc_url,
                "recording_path": self.current_recording,
                "last_error": self.last_error,
                "process_running": self.process is not None and self.process.poll() is None,
            }

    def get_logs(self) -> list[str]:
        with self.lock:
            return list(self.log_lines)

    def _append_log(self, line: str) -> None:
        line = line.rstrip()
        if not line:
            return
        with self.lock:
            self.log_lines.append(line)
        self.logger.info(line)

    def _read_process_output(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            self._append_log(line)

    def _watch_process(self, process: subprocess.Popen[str], item_id: str) -> None:
        exit_code = process.wait()
        with self.lock:
            if self.process is process:
                self.status = "stopped" if exit_code == 0 else "failed"
                if exit_code != 0 and self.last_error is None:
                    self.last_error = f"Dashboard process exited with code {exit_code}"
                self.process = None
                if exit_code == 0:
                    self.current_item_id = None
        self._append_log(f"dashboard process for '{item_id}' exited with code {exit_code}")

    def stop_current(self) -> None:
        with self.lock:
            process = self.process
        if process is None:
            return

        self._append_log("stopping current dashboard process")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._append_log("dashboard process did not stop in time; killing it")
            process.kill()
            process.wait(timeout=5)
        finally:
            with self.lock:
                if self.process is process:
                    self.process = None
                    self.status = "idle"
                    self.current_item_id = None
                    self.current_recording = None
                    self.started_at = None
        self._wait_for_ports_to_release()

    def start_item(self, item_id: str, *, save_recording: bool = False) -> dict[str, Any]:
        item = next((item for item in self.items if item.item_id == item_id), None)
        if item is None:
            raise ValueError(f"Unknown item: {item_id}")
        if not item.valid:
            raise ValueError(f"Config '{item_id}' is invalid: {item.error}")

        self.stop_current()
        self._wait_for_ports_to_release()
        recording_path = self.outputs_dir / f"{item.path.stem}.rrd" if save_recording else None
        cmd = [
            sys.executable,
            str(SERVE_SCRIPT),
            "--config",
            str(item.path),
            "--web-port",
            str(self.viewer_port),
            "--grpc-port",
            str(self.grpc_port),
            "--keep-alive",
        ]
        if recording_path is not None:
            cmd.extend(["--save-recording", str(recording_path)])

        env = os.environ.copy()
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(REPO_ROOT),
            env=env,
        )

        with self.lock:
            self.process = process
            self.current_item_id = item_id
            self.current_recording = str(recording_path) if recording_path is not None else None
            self.status = "starting"
            self.started_at = utc_now()
            self.last_error = None
            self.log_lines.clear()

        self._append_log(f"starting dashboard for '{item_id}' using {item.path.name}")
        if recording_path is None:
            self._append_log("live viewer mode enabled; not saving a .rrd recording")
        else:
            self._append_log(f"recording output enabled at {recording_path}")
        threading.Thread(target=self._read_process_output, args=(process,), daemon=True).start()
        threading.Thread(target=self._watch_process, args=(process, item_id), daemon=True).start()

        def _mark_running() -> None:
            with self.lock:
                if self.process is process and process.poll() is None:
                    self.status = "running"

        threading.Timer(1.0, _mark_running).start()
        return self.get_status()


class DashboardAppHandler(BaseHTTPRequestHandler):
    server_version = "DashboardApp/0.1"

    @property
    def app_state(self) -> DashboardAppState:
        return self.server.app_state  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        self.app_state.logger.info("http %s - %s", self.address_string(), fmt % args)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return

        mime_type, _ = mimetypes.guess_type(path.name)
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        try:
            if self.path == "/" or self.path == "/index.html":
                self._send_file(APP_UI_DIR / "index.html")
                return
            if self.path == "/styles.css":
                self._send_file(APP_UI_DIR / "styles.css")
                return
            if self.path == "/app.js":
                self._send_file(APP_UI_DIR / "app.js")
                return
            if self.path.startswith("/assets/"):
                rel = self.path.removeprefix("/assets/")
                self._send_file(APP_UI_DIR / "assets" / rel)
                return
            if self.path == "/api/items":
                self._send_json({"items": self.app_state.list_items()})
                return
            if self.path == "/api/recordings":
                self._send_json({"items": self.app_state.list_recordings()})
                return
            if self.path == "/api/status":
                self._send_json(self.app_state.get_status())
                return
            if self.path == "/api/logs":
                self._send_json({"logs": self.app_state.get_logs()})
                return
            if self.path.startswith("/api/download?"):
                from urllib.parse import parse_qs, urlparse

                rel_path = parse_qs(urlparse(self.path).query).get("path", [None])[0]
                if rel_path is None:
                    self._send_json({"error": "path is required"}, status=400)
                    return
                target = (REPO_ROOT / rel_path).resolve()
                if REPO_ROOT not in target.parents and target != REPO_ROOT:
                    self._send_json({"error": "invalid path"}, status=400)
                    return
                self._send_file(target)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
        except Exception as exc:
            self.app_state._append_log(f"http GET failed: {exc}")
            self._send_json({"error": str(exc)}, status=500)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body"}, status=400)
            return

        if self.path == "/api/open":
            item_id = str(payload.get("item_id", "")).strip()
            if not item_id:
                self._send_json({"error": "item_id is required"}, status=400)
                return
            save_recording = bool(payload.get("save_recording", False))
            try:
                status = self.app_state.start_item(item_id, save_recording=save_recording)
                self._send_json(status)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
            return

        if self.path == "/api/stop":
            self.app_state.stop_current()
            self._send_json(self.app_state.get_status())
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Basic web app in front of the Dockerized Rerun dashboard.")
    parser.add_argument("--app-port", type=int, default=8080, help="Port for the selection/logging UI.")
    parser.add_argument("--viewer-port", type=int, default=9090, help="Port for the Rerun web viewer.")
    parser.add_argument("--grpc-port", type=int, default=9876, help="Port for the Rerun gRPC server.")
    parser.add_argument("--config-dir", type=Path, default=REPO_ROOT / "configs", help="Directory of ready-to-play config files.")
    parser.add_argument("--outputs-dir", type=Path, default=REPO_ROOT / "outputs", help="Directory for logs and saved recordings.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.outputs_dir / "dashboard_app.log")
    state = DashboardAppState(
        config_dir=args.config_dir,
        outputs_dir=args.outputs_dir,
        viewer_port=args.viewer_port,
        grpc_port=args.grpc_port,
        logger=logger,
    )
    server = ThreadingHTTPServer(("0.0.0.0", args.app_port), DashboardAppHandler)
    server.app_state = state  # type: ignore[attr-defined]

    logger.info("dashboard app listening on http://0.0.0.0:%s", args.app_port)
    logger.info("configured viewer port=%s grpc port=%s", args.viewer_port, args.grpc_port)
    try:
        server.serve_forever()
    finally:
        state.stop_current()
        server.server_close()


if __name__ == "__main__":
    main()
