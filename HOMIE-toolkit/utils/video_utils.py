"""
Video frame loading for Xperience-10M.
"""

import os
import cv2


def load_video_frame(video_path, frame_idx, log_image_scale=1.0):
    """Load a single frame from a video. Returns (H, W, 3) RGB or None."""
    if not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if log_image_scale != 1.0:
        frame_rgb = cv2.resize(frame_rgb, (int(frame_rgb.shape[1] * log_image_scale), int(frame_rgb.shape[0] * log_image_scale)))
    return frame_rgb


__all__ = ["load_video_frame"]
