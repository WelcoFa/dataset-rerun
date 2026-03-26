from __future__ import annotations


def create_shared_blueprint(base: str):
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(origin=f"{base}/camera", name="2D View"),
                rrb.Spatial3DView(origin=f"{base}/world", name="3D View"),
            ),
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Vertical(
                        rrb.TextDocumentView(origin=f"{base}/dashboard/summary/recording", name="Recording"),
                        rrb.TextDocumentView(origin=f"{base}/dashboard/summary/frame", name="Frame"),
                        rrb.TextDocumentView(origin=f"{base}/dashboard/details/objects", name="Objects"),
                    ),
                    rrb.Vertical(
                        rrb.TextDocumentView(origin=f"{base}/dashboard/semantic/main_task", name="Main Task"),
                        rrb.TextDocumentView(origin=f"{base}/dashboard/semantic/sub_task", name="Sub Task"),
                        rrb.TextDocumentView(origin=f"{base}/dashboard/semantic/current_action", name="Current Action"),
                        rrb.TextDocumentView(origin=f"{base}/dashboard/details/interaction", name="Interaction"),
                    ),
                ),
                rrb.TimeSeriesView(origin=f"{base}/dashboard/timeline", name="Timeline"),
            ),
        ),
        collapse_panels=True,
    )

