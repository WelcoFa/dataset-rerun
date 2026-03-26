from __future__ import annotations

import rerun as rr

from .blueprints import create_shared_blueprint


def run_adapter_session(adapter, config):
    adapter.load()

    rr.init(adapter.viewer_name, spawn=config.spawn)
    blueprint = adapter.create_blueprint()
    if blueprint is None:
        blueprint = create_shared_blueprint(adapter.base)
    rr.send_blueprint(blueprint)
    adapter.log_static()

    try:
        for panels in adapter.frames():
            adapter.log_panels(panels)
    finally:
        adapter.close()
