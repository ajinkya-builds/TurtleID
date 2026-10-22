"""Crop a field photograph down to the animal.

BioCLIP resizes every photo and then takes the centre. On a seizure photo the
centre is often the paper, the ruler, or the water, and the shell sits off to
one side. Cutting to the animal first is what makes the centre of the frame
the turtle.

The background is taken from the border of the frame. The largest region that
does not match that border is treated as the animal. A tight head shot, where
the animal already fills the frame, is returned unchanged.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def crop_to_animal(image: Image.Image, pad: float = 0.12) -> Image.Image:
    """Return the photo cropped to the animal, or the original if no crop is safe."""
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    if height < 32 or width < 32:
        return image

    border = max(4, int(0.05 * min(height, width)))
    edges = np.concatenate([
        rgb[:border].reshape(-1, 3),
        rgb[-border:].reshape(-1, 3),
        rgb[:, :border].reshape(-1, 3),
        rgb[:, -border:].reshape(-1, 3),
    ]).astype(np.float32)
    background = np.median(edges, axis=0)
    distance = np.linalg.norm(rgb.astype(np.float32) - background, axis=2)
    border_distance = np.linalg.norm(edges - background, axis=1)
    threshold = max(32.0, float(np.percentile(border_distance, 90)) + 24.0)
    mask = distance > threshold

    fraction = float(mask.mean())
    if fraction < 0.02 or fraction > 0.85:
        return image

    mask = _largest_region(mask)
    if mask is None or float(mask.mean()) < 0.02:
        return image

    rows, cols = np.where(mask)
    top, bottom = int(rows.min()), int(rows.max())
    left, right = int(cols.min()), int(cols.max())
    box_h, box_w = bottom - top + 1, right - left + 1
    if box_h < 0.12 * height or box_w < 0.12 * width:
        return image

    pad_y, pad_x = int(pad * box_h), int(pad * box_w)
    top = max(0, top - pad_y)
    left = max(0, left - pad_x)
    bottom = min(height - 1, bottom + pad_y)
    right = min(width - 1, right + pad_x)
    return image.crop((left, top, right + 1, bottom + 1))


def _largest_region(mask: np.ndarray) -> np.ndarray | None:
    """Keep the largest connected blob. A ruler or a shadow should not win."""
    try:
        import cv2
    except ImportError:
        return mask

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    chosen = 1 + int(np.argmax(areas))
    return labels == chosen
