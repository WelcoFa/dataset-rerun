from __future__ import annotations

from rerun_viz.config.schema import VizConfig
from rerun_viz.datasets.legacy import LegacyUniversalAdapter
from rerun_viz.datasets.wiyh import WiyhAdapter


def resolve_adapter(config: VizConfig):
    if config.dataset == "wiyh":
        return WiyhAdapter(config)
    if WiyhAdapter.detect(config.input.resolve()):
        return WiyhAdapter(config)
    return LegacyUniversalAdapter(config)
