from __future__ import annotations

import numpy as np
import rerun as rr


HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def log_hand_2d(base_path: str, pts: np.ndarray):
    rr.log(f"{base_path}/keypoints", rr.Points2D(pts))
    lines = [
        np.stack([pts[a], pts[b]], axis=0)
        for a, b in HAND_BONES
        if a < len(pts) and b < len(pts)
    ]
    if lines:
        rr.log(f"{base_path}/bones", rr.LineStrips2D(lines))


def log_hand_3d(base_path: str, pts: np.ndarray):
    rr.log(f"{base_path}/keypoints", rr.Points3D(pts))
    lines = [
        np.stack([pts[a], pts[b]], axis=0)
        for a, b in HAND_BONES
        if a < len(pts) and b < len(pts)
    ]
    if lines:
        rr.log(f"{base_path}/bones", rr.LineStrips3D(lines))

