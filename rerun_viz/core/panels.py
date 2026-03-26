from __future__ import annotations

from dataclasses import dataclass

import rerun as rr


@dataclass
class DashboardPanels:
    recording_summary: str
    frame_summary: str
    main_task: str
    sub_task: str
    current_action: str
    interaction: str
    objects: list[str]


def log_dashboard_panels(base: str, panels: DashboardPanels):
    rr.log(f"{base}/dashboard/summary/recording", rr.TextDocument(panels.recording_summary))
    rr.log(f"{base}/dashboard/summary/frame", rr.TextDocument(panels.frame_summary))
    rr.log(f"{base}/dashboard/semantic/main_task", rr.TextDocument(panels.main_task))
    rr.log(f"{base}/dashboard/semantic/sub_task", rr.TextDocument(panels.sub_task))
    rr.log(f"{base}/dashboard/semantic/current_action", rr.TextDocument(panels.current_action))
    rr.log(f"{base}/dashboard/details/interaction", rr.TextDocument(panels.interaction))
    rr.log(
        f"{base}/dashboard/details/objects",
        rr.TextDocument("\n".join(f"- {item}" for item in panels.objects) if panels.objects else "None"),
    )

