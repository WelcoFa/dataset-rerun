from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image_any_unicode_safe(image_path: Path, flags: int) -> np.ndarray:
    data = np.fromfile(str(image_path), dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(f"Failed to read image bytes: {image_path}")
    image = cv2.imdecode(data, flags)
    if image is None:
        raise FileNotFoundError(f"Failed to decode image: {image_path}")
    return image


def read_image_rgb_unicode_safe(image_path: Path) -> np.ndarray:
    image_bgr = read_image_any_unicode_safe(image_path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def normalize_to_u8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros(image.shape[:2], dtype=np.uint8)

    vals = image[finite].astype(np.float32)
    lo = float(vals.min())
    hi = float(vals.max())
    if hi <= lo:
        return np.zeros(image.shape[:2], dtype=np.uint8)

    scaled = (image.astype(np.float32) - lo) / (hi - lo)
    scaled = np.clip(scaled * 255.0, 0.0, 255.0)
    return scaled.astype(np.uint8)


def read_gray_preview_unicode_safe(image_path: Path) -> np.ndarray:
    image = read_image_any_unicode_safe(image_path, cv2.IMREAD_UNCHANGED)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return normalize_to_u8(image)


def colorize_gray(gray_u8: np.ndarray, colormap: int) -> np.ndarray:
    image_bgr = cv2.applyColorMap(gray_u8, colormap)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

