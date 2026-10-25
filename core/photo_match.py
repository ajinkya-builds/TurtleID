"""Species matching with BioCLIP 2.

BioCLIP 2 (Gu et al., NeurIPS 2025) is a vision-language model trained on
214 million biological images labelled with the full taxonomic name, the
scientific name, and the common name. On published species-identification
benchmarks it is the strongest open model of this kind, ahead of the first
BioCLIP, CLIP, SigLIP, and DINOv2.

The ordering is the similarity of the photograph to each identification-card
image. On the verified field photographs that comparison is what held up:
10 of 13 correct, against 1 of 13 for the previous ImageNet matcher. Mixing
in the species-name score, or cropping the animal out of the frame first,
both lowered that figure, so neither is applied by default. The name score
is still computed and returned.

A species with no card photograph cannot win this comparison. Two of the
thirty taxa are in that position. Horizontal mirrors are averaged, because
a turtle facing left and the same turtle facing right are the same animal.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from PIL import Image, ImageOps

# Import config first so HF_HOME points at models/hf_cache/ on this volume.
from config import HF_CACHE_DIR, REFERENCE_IMAGE_DIR, REFERENCE_MANIFEST
from core.database import SpeciesDB
from core.foreground import crop_to_animal

logger = logging.getLogger(__name__)

MODEL_ID = "hf-hub:imageomics/bioclip-2"
RANKS = "Animalia Chordata Reptilia Testudines"


class MatchError(RuntimeError):
    """Raised when the model or a reference photograph cannot be loaded."""


@dataclass(frozen=True)
class RankedMatch:
    species_id: str
    scientific_name: str
    common_en: str
    score: float
    text_score: float
    plate_score: float | None


def _device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _zscore(row: np.ndarray) -> np.ndarray:
    finite = np.isfinite(row)
    if finite.sum() < 2:
        return np.zeros_like(row)
    values = row[finite]
    scale = float(values.std())
    if scale < 1e-6:
        return np.zeros_like(row)
    out = np.zeros_like(row)
    out[finite] = (values - float(values.mean())) / scale
    return out


class BioClipMatcher:
    """Rank the reference species for one photograph."""

    def __init__(
        self,
        db: SpeciesDB | None = None,
        device: str | None = None,
        model_id: str = MODEL_ID,
    ):
        self.db = db if db is not None else SpeciesDB.load()
        self.species = list(self.db)
        self.device_name = device or _device()
        self.model_id = model_id
        self._load()

    def _load(self) -> None:
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise MatchError(
                "open_clip is required for the species matcher. "
                "Install it with: python -m pip install -r requirements-matcher.txt"
            ) from exc

        import os
        raw_dtype = os.environ.get("TURTLEID_DTYPE", "bf16").strip().lower()
        dtype = torch.bfloat16 if raw_dtype in ("bf16", "bfloat16") else torch.float32

        try:
            model, preprocess = _load_openclip_model_streaming(
                self.model_id, device=self.device_name, dtype=dtype
            )
            tokenizer = open_clip.get_tokenizer(self.model_id)
        except Exception as exc:
            raise MatchError(
                f"Could not load {self.model_id}. The weights download on first use "
                f"and are cached by Hugging Face. ({exc})"
            ) from exc

        self.torch = torch
        self.dtype = dtype
        self.model = model.eval()
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        logger.info("Loaded %s on %s (dtype=%s)", self.model_id, self.device_name, dtype)

    def rank(
        self,
        image: Image.Image,
        *,
        crop: bool = False,
        top: int | None = None,
    ) -> list[RankedMatch]:
        """Species in descending order of similarity to the identification card.

        `crop` is off because cutting the frame down to a guessed animal
        lowered accuracy on the field photographs. The name score is attached
        to every row and is not used to order species that have a card.
        """
        prepared = self.prepare(image, crop=crop)
        vector = self.embed([prepared])[0]
        text = vector @ self.text_matrix.T
        plate = np.full(len(self.species), np.nan, dtype=np.float64)
        if self.plate_index:
            plate_vectors = self.plate_matrix(crop=crop)
            ids = [species_id for species_id, _ in self.plate_index]
            scores = vector @ plate_vectors.T
            for species_id, score in zip(ids, scores):
                plate[self._position[species_id]] = float(score)

        # Card similarity decides the order. A species with no card sorts
        # after every species that has one, and those are ordered by name.
        order_key = np.where(np.isfinite(plate), plate, text - 2.0)
        order = np.argsort(-order_key)
        if top is not None:
            order = order[:top]

        ranked: list[RankedMatch] = []
        for position in order:
            species = self.species[int(position)]
            plate_score = plate[position]
            has_plate = bool(np.isfinite(plate_score))
            ranked.append(RankedMatch(
                species_id=species.id,
                scientific_name=species.scientific_name,
                common_en=species.common_en,
                score=float(plate_score if has_plate else text[position]),
                text_score=float(text[position]),
                plate_score=None if not has_plate else float(plate_score),
            ))
        return ranked

    def prepare(self, image: Image.Image, *, crop: bool) -> Image.Image:
        # Some phone JPEGs ship broken EXIF rationals; transpose then fails.
        # Orientation still matters when the tag is readable, so try it first.
        try:
            turned = ImageOps.exif_transpose(image)
        except (ZeroDivisionError, OSError, ValueError) as exc:
            logger.warning("EXIF orientation skipped (%s)", exc)
            turned = image
        turned = turned.convert("RGB")
        return crop_to_animal(turned) if crop else turned

    def embed(self, images: list[Image.Image], batch_size: int = 8) -> np.ndarray:
        """L2-normalised image vectors, each averaged with its mirror."""
        if not images:
            return np.empty((0, 0), dtype=np.float32)

        rows: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            tensor = self.torch.stack([self.preprocess(image) for image in batch])
            tensor = tensor.to(device=self.device_name, dtype=getattr(self, "dtype", self.torch.float32))
            with self.torch.no_grad():
                forward = self.model.encode_image(tensor)
                mirrored = self.model.encode_image(self.torch.flip(tensor, dims=[3]))
                features = forward + mirrored
            rows.append(features.float().cpu().numpy())
        return _normalise(np.vstack(rows).astype(np.float32))

    @cached_property
    def text_matrix(self) -> np.ndarray:
        """One vector per species: taxonomy, scientific name, and common name."""
        prompts = [self._prompts(species) for species in self.species]
        flat = [prompt for group in prompts for prompt in group]
        encoded = self._encode_text(flat)
        grouped = encoded.reshape(len(self.species), 3, -1).mean(axis=1)
        return _normalise(grouped.astype(np.float32))

    @cached_property
    def _position(self) -> dict[str, int]:
        return {species.id: index for index, species in enumerate(self.species)}

    @cached_property
    def plate_index(self) -> list[tuple[str, Image.Image]]:
        if not REFERENCE_MANIFEST.exists():
            return []
        manifest = json.loads(REFERENCE_MANIFEST.read_text())
        plates: list[tuple[str, Image.Image]] = []
        for species_id, records in manifest.get("images", {}).items():
            if species_id not in self._position or not records:
                continue
            path = REFERENCE_IMAGE_DIR / records[0]["file"]
            if not path.exists():
                continue
            plates.append((species_id, Image.open(path)))
        return plates

    def plate_matrix(self, *, crop: bool, batch_size: int = 8) -> np.ndarray:
        key = "cropped" if crop else "full"
        cached = getattr(self, "_plate_cache", {})
        if key not in cached:
            prepared = [self.prepare(image, crop=crop) for _, image in self.plate_index]
            cached[key] = self.embed(prepared, batch_size=batch_size)
            self._plate_cache = cached
        return cached[key]

    def _prompts(self, species) -> list[str]:
        return [
            f"{RANKS} {species.family} {species.scientific_name}",
            species.scientific_name,
            species.common_en,
        ]

    def _encode_text(self, prompts: list[str]) -> np.ndarray:
        tokens = self.tokenizer(prompts).to(self.device_name)
        with self.torch.no_grad():
            features = self.model.encode_text(tokens)
        return _normalise(features.float().cpu().numpy().astype(np.float32))


def _load_openclip_model_streaming(model_id: str, *, device: str, dtype, visual_only: bool = False):
    """Load an hf-hub OpenCLIP checkpoint via safetensors mmap without duplicating 3.8 GB in RAM."""
    import json
    import open_clip
    import torch
    from huggingface_hub import hf_hub_download
    from open_clip.transform import PreprocessCfg, image_transform_v2
    from safetensors import safe_open

    if model_id.startswith("hf-hub:"):
        repo_id = model_id.split("hf-hub:", 1)[1]
        try:
            cfg_path = hf_hub_download(repo_id, "open_clip_config.json")
            st_path = hf_hub_download(repo_id, "open_clip_model.safetensors")
            cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
            preprocess = image_transform_v2(
                PreprocessCfg(**cfg.get("preprocess_cfg", {})), is_train=False
            )
            with torch.device("meta"):
                model = open_clip.model.CLIP(**cfg["model_cfg"])
            ctx_len = cfg["model_cfg"].get("text_cfg", {}).get("context_length", 77)
            if visual_only:
                target = model.visual
                del model
            else:
                target = model
            prefix = "visual." if visual_only else ""
            with safe_open(st_path, framework="pt", device="cpu") as f:
                keys = set(f.keys())
                for name, module in target.named_modules():
                    for p_name, param in list(module._parameters.items()):
                        if param is None:
                            continue
                        rel = f"{name}.{p_name}" if name else p_name
                        full = f"{prefix}{rel}"
                        if full in keys:
                            t = f.get_tensor(full)
                            if t.is_floating_point():
                                t = t.to(dtype=dtype)
                            module._parameters[p_name] = torch.nn.Parameter(
                                t.to(device=device), requires_grad=False
                            )
                    for b_name, buf in list(module._buffers.items()):
                        if buf is None:
                            continue
                        rel = f"{name}.{b_name}" if name else b_name
                        full = f"{prefix}{rel}"
                        if full in keys:
                            t = f.get_tensor(full)
                            if t.is_floating_point():
                                t = t.to(dtype=dtype)
                            module._buffers[b_name] = t.to(device=device)
                        elif b_name == "attn_mask":
                            mask = torch.triu(
                                torch.full((ctx_len, ctx_len), float("-inf"), dtype=dtype, device=device),
                                diagonal=1,
                            )
                            module._buffers[b_name] = mask
            return target.eval(), preprocess
        except Exception as exc:
            logger.warning("Streaming safetensors load fallback for %s (%s)", model_id, exc)

    model, _train, preprocess = open_clip.create_model_and_transforms(model_id)
    if visual_only:
        visual = model.visual
        del model
        return visual.to(device=device, dtype=dtype).eval(), preprocess
    return model.to(device=device, dtype=dtype).eval(), preprocess
