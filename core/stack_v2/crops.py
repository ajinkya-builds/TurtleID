"""Animal / part crops: GroundingDINO (+ optional SAM) with border-crop fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image

# Ensure HF cache is on this volume before transformers downloads.
import config  # noqa: F401
from core.foreground import crop_to_animal

logger = logging.getLogger(__name__)

_DETECTOR = None
_DETECTOR_NAME = "none"


def release_detector() -> None:
    """Free GroundingDINO weights before loading BioCLIP / DINOv2."""
    global _DETECTOR, _DETECTOR_NAME
    if _DETECTOR is None:
        return
    try:
        import torch
        processor, model, _ = _DETECTOR
        del processor, model
        _DETECTOR = None
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        _DETECTOR = None
    _DETECTOR_NAME = "none"
    logger.info("Released animal detector from memory")



@dataclass
class CropBundle:
    full: Image.Image
    animal: Image.Image
    head: Image.Image | None
    method: str


def _device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_grounding_dino():
    global _DETECTOR, _DETECTOR_NAME
    if _DETECTOR is not None or _DETECTOR_NAME == "failed":
        return _DETECTOR
    try:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        name = "IDEA-Research/grounding-dino-tiny"
        processor = AutoProcessor.from_pretrained(name)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(name)
        model = model.to(_device()).eval()
        _DETECTOR = (processor, model, torch)
        _DETECTOR_NAME = name
        logger.info("Loaded detector %s", name)
    except Exception as exc:
        logger.warning("GroundingDINO unavailable (%s); using border crop", exc)
        _DETECTOR_NAME = "failed"
        _DETECTOR = None
    return _DETECTOR


def _largest_box(boxes, scores) -> tuple[float, float, float, float] | None:
    if boxes is None or len(boxes) == 0:
        return None
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    index = int(areas.argmax())
    if float(scores[index]) < 0.15:
        return None
    return tuple(float(v) for v in boxes[index])


def _detect_box(image: Image.Image) -> tuple[float, float, float, float] | None:
    packed = _load_grounding_dino()
    if packed is None:
        return None
    processor, model, torch = packed
    text = "turtle. tortoise. softshell turtle."
    inputs = processor(images=image, text=text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    try:
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=0.2,
            text_threshold=0.2,
            target_sizes=[(image.height, image.width)],
        )[0]
    except TypeError:
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=0.2,
            text_threshold=0.2,
            target_sizes=[(image.height, image.width)],
        )[0]
    boxes = results.get("boxes")
    scores = results.get("scores")
    if boxes is None or len(boxes) == 0:
        return None
    if hasattr(boxes, "cpu"):
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
    return _largest_box(np.asarray(boxes), np.asarray(scores))


def _pad_crop(image: Image.Image, box, pad: float = 0.08) -> Image.Image:
    left, top, right, bottom = box
    width, height = image.size
    bw, bh = right - left, bottom - top
    left = max(0, int(left - pad * bw))
    top = max(0, int(top - pad * bh))
    right = min(width, int(right + pad * bw))
    bottom = min(height, int(bottom + pad * bh))
    if right - left < 16 or bottom - top < 16:
        return image
    return image.crop((left, top, right, bottom))


def _head_from_animal(animal: Image.Image) -> Image.Image:
    """Heuristic head band: top third of the animal crop (common dorsal phone shot)."""
    width, height = animal.size
    band = max(32, int(0.38 * height))
    return animal.crop((0, 0, width, band))


def prepare_crops(image: Image.Image) -> CropBundle:
    """Return full RGB, animal crop, and optional head crop."""
    try:
        rgb = ImageOps_exif(image).convert("RGB")
    except Exception:
        rgb = image.convert("RGB")

    box = _detect_box(rgb)
    if box is not None:
        animal = _pad_crop(rgb, box)
        method = "grounding_dino"
    else:
        animal = crop_to_animal(rgb)
        method = "border_crop"

    head = _head_from_animal(animal)
    return CropBundle(full=rgb, animal=animal, head=head, method=method)


def ImageOps_exif(image: Image.Image) -> Image.Image:
    from PIL import ImageOps

    try:
        return ImageOps.exif_transpose(image)
    except (ZeroDivisionError, OSError, ValueError):
        return image
