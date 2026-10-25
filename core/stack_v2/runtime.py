"""Online Stack v2. Vision towers only; the text towers stay in the bundle.

Query time loads BioCLIP 2.5's visual tower, DINOv2-with-registers-large, and
optionally BioCAP's visual tower. Family, genus, species, and trait text
embeddings are read from ``data/stack_v2_bundle.npz``.
"""

from __future__ import annotations

import gc
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

import config  # noqa: F401  (HF_HOME before any hub download)
from config import (
    DATA_DIR,
    DEFAULT_ENERGY_THRESHOLD,
    DEFAULT_TEMPERATURE,
    GALLERY_SIMILARITY_FLOOR,
    MAX_NORMALISED_ENTROPY,
    REFERENCE_IMAGE_DIR,
    REFERENCE_MANIFEST,
    TIER_CONFIRMED,
    TIER_PROBABLE,
    TIER_TENTATIVE,
)
from core.database import SpeciesDB
from core.inference import free_energy, normalised_entropy, softmax
from core.morphkey import verification_questions
from core.stack_v2.bundle import (
    BIOCAP_DIM,
    BIOCLIP_DIM,
    BUNDLE_VERSION,
    DINO_DIM,
    FUSED_DIM,
    BundleError,
    bundle_meta,
    load_bundle,
)
from core.stack_v2.crops import crop_animal_and_head, release_detector
from core.stack_v2.scoring import (
    combine_channels,
    is_confusion_pair,
    l2_normalise,
    soft_hierarchy_bonus,
    tip_adapter_scores,
    cl2n,
)
from core.stack_v2.traits import prompts_for

logger = logging.getLogger(__name__)

BUNDLE_PATH = DATA_DIR / "stack_v2_bundle.npz"
BIOCLIP_ID = "hf-hub:imageomics/bioclip-2.5-vith14"
BIOCAP_ID = "hf-hub:imageomics/biocap"
DINO_ID = "facebook/dinov2-with-registers-large"

# Same channel weights as scripts/score_turtles_v2.py.
W_TIP = 1.25
W_TIP_DINO = 0.9
W_TEXT = 0.55
W_HIER = 0.35
W_PLATE = 0.35
W_BIOCAP = 0.45
W_PROBE = 1.05
MARGIN_SPECIALIST = 0.15
PLATE_WEIGHT = W_PLATE

_TIER_ACTION = {
    "CONFIRMED": (
        "Record the species. Retain the photographs as a voucher and "
        "attach them to the field register entry."
    ),
    "PROBABLE": (
        "Record as a probable determination. Verify the discriminating "
        "character before it goes on any official form."
    ),
    "TENTATIVE": (
        "Insufficient for a species-level record. Report to genus, work "
        "through the checklist, and retake photographs if a character is hidden."
    ),
    "INDETERMINATE": (
        "No candidate reaches the reporting threshold. Refer to a chelonian "
        "specialist. Do not enter a species on any form."
    ),
    "REJECTED": (
        "The photograph does not resemble the reference set. Retake it "
        "(dorsal, ventral, lateral, head) or use the morphological key."
    ),
}


def env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def detector_mode(explicit: str | None = None) -> str:
    raw = explicit or os.environ.get("TURTLEID_DETECTOR") or "grounding_dino"
    return str(raw).strip().lower() or "grounding_dino"


def use_biocap(explicit: bool | None = None) -> bool:
    if explicit is not None:
        return bool(explicit)
    return env_flag("TURTLEID_USE_BIOCAP", True)


def _strings(array) -> list[str]:
    return [str(item) for item in np.asarray(array).reshape(-1).tolist()]


def _l2_row(vector: np.ndarray) -> np.ndarray:
    row = np.asarray(vector, dtype=np.float32).reshape(1, -1)
    return l2_normalise(row)[0]


def center_query(vector: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Match export: L2, subtract the bundle mean, L2 again."""
    row = _l2_row(vector)
    centered, _ = cl2n(row.reshape(1, -1), np.asarray(mean, dtype=np.float32))
    return centered[0]


def probe_probabilities(
    fused: np.ndarray,
    coef: np.ndarray,
    intercept: np.ndarray,
    classes: list[str],
    species_ids: list[str],
) -> np.ndarray:
    """softmax((fused @ coef.T) + intercept), scattered into species order."""
    out = np.zeros(len(species_ids), dtype=np.float64)
    if coef.size == 0 or not classes:
        return out
    logits = np.asarray(fused, dtype=np.float64).reshape(1, -1) @ np.asarray(coef, dtype=np.float64).T
    logits = logits.reshape(-1) + np.asarray(intercept, dtype=np.float64).reshape(-1)
    probs = softmax(logits, temperature=1.0)
    index = {sid: i for i, sid in enumerate(species_ids)}
    for prob, name in zip(probs, classes):
        pos = index.get(str(name))
        if pos is not None:
            out[pos] = float(prob)
    return out


def plate_scores(query: np.ndarray, plate_c: np.ndarray, plate_ids: list[str], species_ids: list[str]) -> np.ndarray:
    scores = np.full(len(species_ids), np.nan, dtype=np.float64)
    if plate_c.shape[0] == 0:
        return scores
    sims = np.asarray(query, dtype=np.float64) @ np.asarray(plate_c, dtype=np.float64).T
    index = {sid: i for i, sid in enumerate(species_ids)}
    for sid, sim in zip(plate_ids, sims):
        pos = index.get(sid)
        if pos is None:
            continue
        value = float(sim)
        if not np.isfinite(scores[pos]) or value > scores[pos]:
            scores[pos] = value
    return scores


def max_visual_similarity(query: np.ndarray, *matrices: np.ndarray) -> float | None:
    best: float | None = None
    for matrix in matrices:
        if matrix is None or np.asarray(matrix).size == 0:
            continue
        sims = np.asarray(matrix, dtype=np.float64) @ np.asarray(query, dtype=np.float64)
        peak = float(np.max(sims))
        best = peak if best is None else max(best, peak)
    return best


def fuse_channels(
    *,
    bioclip_q: np.ndarray,
    dino_q: np.ndarray,
    biocap_q: np.ndarray | None,
    cache_mat: np.ndarray,
    cache_labels: list[str],
    local_dino: np.ndarray,
    local_labels: list[str],
    text_matrix: np.ndarray,
    fam_vecs: np.ndarray,
    fam_names: list[str],
    gen_vecs: np.ndarray,
    gen_names: list[str],
    species_families: list[str],
    species_genera: list[str],
    plate_c: np.ndarray,
    plate_ids: list[str],
    biocap_text: np.ndarray,
    probe_coef: np.ndarray,
    probe_intercept: np.ndarray,
    probe_classes: list[str],
    species_ids: list[str],
    use_biocap_channel: bool,
) -> np.ndarray:
    """Seven-channel Stack v2 score, one column per ``species_ids``."""
    n_species = len(species_ids)
    tip = tip_adapter_scores(
        bioclip_q, cache_mat, cache_labels, species_ids, alpha=1.15, beta=5.0
    )
    if local_dino.shape[0]:
        tip_dino = tip_adapter_scores(
            dino_q, local_dino, local_labels, species_ids, alpha=1.0, beta=5.0
        )
        dino_weight = W_TIP_DINO
    else:
        tip_dino = np.zeros(n_species, dtype=np.float64)
        dino_weight = 0.0

    text = np.asarray(bioclip_q, dtype=np.float64) @ np.asarray(text_matrix, dtype=np.float64).T
    hier = soft_hierarchy_bonus(
        np.asarray(bioclip_q, dtype=np.float64),
        np.asarray(fam_vecs, dtype=np.float64),
        np.asarray(gen_vecs, dtype=np.float64),
        species_families,
        species_genera,
        fam_names,
        gen_names,
    )
    plate = plate_scores(bioclip_q, plate_c, plate_ids, species_ids)
    if np.isfinite(plate).any():
        filled = float(np.nanmean(plate))
    else:
        filled = 0.0
    plate_row = np.nan_to_num(plate, nan=filled)

    if use_biocap_channel and biocap_q is not None and biocap_text.shape[0]:
        biocap_row = np.asarray(biocap_q, dtype=np.float64) @ np.asarray(biocap_text, dtype=np.float64).T
        biocap_weight = W_BIOCAP
    else:
        biocap_row = np.zeros(n_species, dtype=np.float64)
        biocap_weight = 0.0

    fused = np.concatenate(
        [
            np.asarray(bioclip_q, dtype=np.float32).reshape(-1),
            np.asarray(dino_q, dtype=np.float32).reshape(-1),
        ]
    )
    if fused.shape[0] != FUSED_DIM:
        raise BundleError(f"fused query is {fused.shape[0]}-d, bundle probe is {FUSED_DIM}-d")
    log_probs = probe_probabilities(
        fused, probe_coef, probe_intercept, probe_classes, species_ids
    )
    return combine_channels(
        [
            (tip, W_TIP),
            (tip_dino, dino_weight),
            (text, W_TEXT),
            (hier, W_HIER),
            (plate_row, PLATE_WEIGHT),
            (biocap_row, biocap_weight),
            (log_probs, W_PROBE),
        ]
    )


def apply_head_specialist(
    scores: np.ndarray,
    head_query: np.ndarray,
    cache_mat: np.ndarray,
    cache_labels: list[str],
    text_matrix: np.ndarray,
    species_ids: list[str],
) -> tuple[np.ndarray, bool]:
    """Re-rank inside the top 5 when #1/#2 is a known lookalike pair and close."""
    order = np.argsort(-scores)
    if len(order) < 2:
        return scores, False
    margin = float(scores[order[0]] - scores[order[1]])
    top1 = species_ids[int(order[0])]
    top2 = species_ids[int(order[1])]
    if margin >= MARGIN_SPECIALIST or not is_confusion_pair(top1, top2):
        return scores, False

    head_tip = tip_adapter_scores(
        head_query, cache_mat, cache_labels, species_ids, alpha=1.4, beta=6.0
    )
    head_text = np.asarray(head_query, dtype=np.float64) @ np.asarray(text_matrix, dtype=np.float64).T
    boost = combine_channels(
        [(head_tip, 1.2), (head_text, 0.8), (scores, 0.5)]
    )
    allowed = {species_ids[int(i)] for i in order[:5]}
    for i, sid in enumerate(species_ids):
        if sid not in allowed:
            boost[i] = -1e9
    return boost, True


def apply_geo_prior(
    probabilities: np.ndarray,
    species_ids: list[str],
    db: SpeciesDB,
    state: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (posterior, per-species multiplier). No state leaves both flat."""
    multipliers = np.ones(len(species_ids), dtype=np.float64)
    if not state:
        return probabilities.astype(np.float64), multipliers
    for i, sid in enumerate(species_ids):
        multipliers[i] = db.get(sid).geo_prior(state)
    posterior = probabilities.astype(np.float64) * multipliers
    total = float(posterior.sum())
    if total <= 0:
        return probabilities.astype(np.float64), multipliers
    return posterior / total, multipliers


def reference_image_urls() -> dict[str, str]:
    if not REFERENCE_MANIFEST.exists():
        return {}
    import json

    try:
        manifest = json.loads(REFERENCE_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    urls: dict[str, str] = {}
    for sid, records in (manifest.get("images") or {}).items():
        if not records:
            continue
        rel = str(records[0].get("file") or "").replace("\\", "/")
        if not rel:
            continue
        if not (REFERENCE_IMAGE_DIR / rel).exists():
            continue
        urls[str(sid)] = "/reference_images/" + rel
    return urls


def _tier_for(top_posterior: float, entropy: float) -> str:
    if entropy > MAX_NORMALISED_ENTROPY and top_posterior < TIER_CONFIRMED:
        return "TENTATIVE"
    if top_posterior >= TIER_CONFIRMED:
        return "CONFIRMED"
    if top_posterior >= TIER_PROBABLE:
        return "PROBABLE"
    if top_posterior >= TIER_TENTATIVE:
        return "TENTATIVE"
    return "INDETERMINATE"


@dataclass
class StackPrediction:
    engine: str
    tier: str
    abstain: bool
    warnings: list[str]
    pairwise_discriminator: str | None
    verification_questions: list[dict]
    top_matches: list[dict]
    margin: float = 0.0
    specialist: bool = False
    crop_method: str = ""
    max_similarity: float | None = None
    message: str = ""
    extra: dict = field(default_factory=dict)

    def as_response(self) -> dict:
        top = self.top_matches[0] if self.top_matches else None
        return {
            "status": "success",
            "engine": self.engine,
            "tier": self.tier,
            "abstain": self.abstain,
            "warnings": list(self.warnings),
            "pairwise_discriminator": self.pairwise_discriminator,
            "verification_questions": self.verification_questions,
            "top_match": top,
            "top_matches": self.top_matches,
            "matches": self.top_matches,
            "message": self.message or _TIER_ACTION.get(self.tier, ""),
            "margin": self.margin,
            "specialist": self.specialist,
            "crop_method": self.crop_method,
        }


class StackV2Predictor:
    """Rank one photograph with the precomputed Stack v2 bundle."""

    def __init__(
        self,
        bundle_path: Path | str | None = None,
        db: SpeciesDB | None = None,
        vision=None,
        *,
        use_biocap_channel: bool | None = None,
        detector: str | None = None,
        temperature: float | None = None,
    ):
        self.bundle_path = Path(bundle_path) if bundle_path else BUNDLE_PATH
        self.arrays = load_bundle(self.bundle_path)
        self.meta = bundle_meta(self.arrays)
        self.db = db if db is not None else SpeciesDB.load()
        self.species_ids = _strings(self.arrays["species_ids"])
        self.cache_labels = _strings(self.arrays["cache_labels"])
        self.local_labels = _strings(self.arrays["local_labels"])
        self.fam_names = _strings(self.arrays["fam_names"])
        self.gen_names = _strings(self.arrays["gen_names"])
        self.species_families = _strings(self.arrays["species_families"])
        self.species_genera = _strings(self.arrays["species_genera"])
        self.plate_ids = _strings(self.arrays["plate_ids"])
        self.probe_classes = _strings(self.arrays["probe_classes"])
        self.use_biocap_channel = use_biocap(use_biocap_channel)
        self.detector = detector_mode(detector)
        self.temperature = float(temperature if temperature is not None else _temperature())
        self.calibrated = env_flag("TURTLEID_CALIBRATED", False)
        self.vision = vision
        self.reference_urls = reference_image_urls()
        for sid in self.species_ids:
            self.db.get(sid)

    def load(self) -> None:
        """Download / map vision weights. Safe to call more than once."""
        if self.vision is None:
            self.vision = VisionTowers.load(use_biocap_channel=self.use_biocap_channel)

    def predict(self, image: Image.Image, state: str | None = None) -> StackPrediction:
        self.load()
        crops = crop_animal_and_head(image, detector=self.detector)
        # The detector is only needed for the box. Drop it before the ViT-H
        # forward when this process is memory constrained and the caller asked
        # for a one-shot load (TURTLEID_RELEASE_DETECTOR=1).
        if env_flag("TURTLEID_RELEASE_DETECTOR", False):
            release_detector()

        bioclip_animal = self._bioclip(crops.animal)
        dino_animal = self._dino(crops.animal)
        biocap_animal = self._biocap(crops.animal) if self.use_biocap_channel else None
        scores = self.score_embeddings(bioclip_animal, dino_animal, biocap_animal)

        specialist = False
        order = np.argsort(-scores)
        if len(order) >= 2:
            margin = float(scores[order[0]] - scores[order[1]])
            pair = (
                self.species_ids[int(order[0])],
                self.species_ids[int(order[1])],
            )
            if margin < MARGIN_SPECIALIST and is_confusion_pair(*pair) and crops.head is not None:
                head = self._bioclip(crops.head)
                scores, specialist = apply_head_specialist(
                    scores,
                    head,
                    self.arrays["cache_mat"],
                    self.cache_labels,
                    self.arrays["text_matrix"],
                    self.species_ids,
                )

        similarity = max_visual_similarity(
            bioclip_animal, self.arrays["cache_mat"], self.arrays["plate_c"]
        )
        return self._package(
            scores,
            state=state,
            specialist=specialist,
            crop_method=crops.method,
            max_similarity=similarity,
            engine="stack_v2",
        )

    def score_embeddings(
        self,
        bioclip_q: np.ndarray,
        dino_q: np.ndarray,
        biocap_q: np.ndarray | None,
    ) -> np.ndarray:
        """Score already-centered BioCLIP / DINOv2 queries (BioCAP is L2 only)."""
        return fuse_channels(
            bioclip_q=bioclip_q,
            dino_q=dino_q,
            biocap_q=biocap_q,
            cache_mat=self.arrays["cache_mat"],
            cache_labels=self.cache_labels,
            local_dino=self.arrays["local_dino"],
            local_labels=self.local_labels,
            text_matrix=self.arrays["text_matrix"],
            fam_vecs=self.arrays["fam_vecs"],
            fam_names=self.fam_names,
            gen_vecs=self.arrays["gen_vecs"],
            gen_names=self.gen_names,
            species_families=self.species_families,
            species_genera=self.species_genera,
            plate_c=self.arrays["plate_c"],
            plate_ids=self.plate_ids,
            biocap_text=self.arrays["biocap_text_matrix"],
            probe_coef=self.arrays["probe_coef"],
            probe_intercept=self.arrays["probe_intercept"],
            probe_classes=self.probe_classes,
            species_ids=self.species_ids,
            use_biocap_channel=self.use_biocap_channel,
        )

    def _bioclip(self, image: Image.Image) -> np.ndarray:
        raw = self.vision.embed_bioclip([image])[0]
        if np.asarray(raw).reshape(-1).shape[0] != BIOCLIP_DIM:
            raise BundleError("BioCLIP vision embedding is not 1024-d")
        return center_query(raw, self.arrays["bioclip_mean"])

    def _dino(self, image: Image.Image) -> np.ndarray:
        raw = self.vision.embed_dino([image])[0]
        if np.asarray(raw).reshape(-1).shape[0] != DINO_DIM:
            raise BundleError("DINOv2 embedding is not 1024-d")
        return center_query(raw, self.arrays["dino_mean"])

    def _biocap(self, image: Image.Image) -> np.ndarray | None:
        embed = getattr(self.vision, "embed_biocap", None)
        if embed is None:
            return None
        raw = embed([image])[0]
        if np.asarray(raw).reshape(-1).shape[0] != BIOCAP_DIM:
            raise BundleError("BioCAP vision embedding is not 512-d")
        return _l2_row(raw)

    def _package(
        self,
        scores: np.ndarray,
        *,
        state: str | None,
        specialist: bool,
        crop_method: str,
        max_similarity: float | None,
        engine: str,
        extra_warnings: list[str] | None = None,
    ) -> StackPrediction:
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        probabilities = softmax(scores, self.temperature)
        posterior, multipliers = apply_geo_prior(
            probabilities, self.species_ids, self.db, state
        )
        order = np.argsort(-posterior)
        score_order = np.argsort(-scores)
        margin = 0.0
        if len(score_order) >= 2:
            margin = float(scores[score_order[0]] - scores[score_order[1]])

        rejected = False
        floor = _similarity_floor()
        if (
            max_similarity is not None
            and floor is not None
            and max_similarity < floor
            and env_flag("TURTLEID_SIM_GATE", True)
        ):
            rejected = True
        if env_flag("TURTLEID_ENERGY_GATE", False):
            energy = free_energy(scores, self.temperature)
            if energy > DEFAULT_ENERGY_THRESHOLD:
                rejected = True
        else:
            energy = free_energy(scores, self.temperature)

        entropy = float(normalised_entropy(posterior))
        top_posterior = float(posterior[order[0]]) if len(order) else 0.0
        tier = "REJECTED" if rejected else _tier_for(top_posterior, entropy)

        warnings: list[str] = []
        if extra_warnings:
            warnings.extend(extra_warnings)
        if engine == "stack_v2" and not self.calibrated:
            warnings.append(
                "Model probabilities are UNCALIBRATED (softmax of fused Stack v2 "
                "scores). Treat them as a ranking aid and confirm every "
                "determination against the morphological checklist."
            )
        elif engine != "stack_v2" and not self.calibrated:
            warnings.append(
                "Model is UNCALIBRATED. Treat all probabilities as upper bounds "
                "and confirm every determination against the morphological key."
            )

        top_ids = [self.species_ids[int(i)] for i in order[:3]]
        pairwise = None
        if len(top_ids) >= 2:
            pairwise = self.db.discriminator(top_ids[0], top_ids[1])
            top = self.db.get(top_ids[0])
            runner = self.db.get(top_ids[1])
            runner_p = float(posterior[order[1]])
            if pairwise and (runner_p > 0.15 or margin < MARGIN_SPECIALIST):
                warnings.append(
                    f"Close call against {runner.scientific_name}. "
                    f"Check this character: {pairwise}"
                )
            if is_confusion_pair(top_ids[0], top_ids[1]) and margin < MARGIN_SPECIALIST:
                warnings.append(
                    f"Low-margin confusion pair: {top.common_en} vs "
                    f"{runner.common_en}. Confirm the checklist before recording a species."
                )
            if top.wpa != runner.wpa and (runner_p > 0.15 or margin < MARGIN_SPECIALIST):
                warnings.append(
                    f"LEGAL DIVERGENCE: {top.scientific_name} is {top.wpa} but "
                    f"{runner.scientific_name} is {runner.wpa}. The offence "
                    "category differs between the top two candidates. Do not "
                    "finalise paperwork on this determination alone."
                )
        if top_ids:
            top = self.db.get(top_ids[0])
            if state and top.occurs_in(state) == "absent":
                warnings.append(
                    f"{top.scientific_name} is not known from {state}. If the "
                    "determination holds, this is a trade, transport or release "
                    "record and should be reported as such, not logged as a wild "
                    "occurrence."
                )
            elif state and top.occurs_in(state) == "marginal":
                warnings.append(
                    f"{top.scientific_name} is marginal in {state}. Retain voucher "
                    "photographs before recording."
                )
            if top.is_threatened and top_posterior >= TIER_PROBABLE and tier != "REJECTED":
                warnings.append(
                    f"{top.scientific_name} is {top.iucn_label} ({top.iucn_status}) and "
                    f"{top.wpa}. Notify the Divisional Forest Officer and retain the "
                    "record regardless of outcome."
                )
        if specialist:
            warnings.append(
                "Head-crop specialist re-ranked a known confusion pair "
                f"(margin was under {MARGIN_SPECIALIST:.2f})."
            )

        questions = verification_questions(top_ids, limit=3) if len(top_ids) >= 2 else []
        matches = []
        for rank, position in enumerate(order[:3], start=1):
            sid = self.species_ids[int(position)]
            species = self.db.get(sid)
            confidence = float(posterior[position])
            matches.append(
                {
                    "species_id": sid,
                    "common_name": species.common_en,
                    "scientific_name": species.scientific_name,
                    "confidence": confidence,
                    "score": confidence,
                    "family": species.family,
                    "iucn_status": species.iucn_status,
                    "wpa_schedule": species.wpa,
                    "cites_appendix": species.cites,
                    "key_character": species.key_character,
                    "diagnostics": list(species.diagnostics),
                    "reference_image_url": self.reference_urls.get(sid),
                    "occurrence": species.occurs_in(state) if state else "unknown",
                    "geo_multiplier": float(multipliers[position]),
                    "rank": rank,
                }
            )

        weak = tier in {"TENTATIVE", "INDETERMINATE", "REJECTED"}
        abstain = bool(weak or margin < MARGIN_SPECIALIST or rejected)
        return StackPrediction(
            engine=engine,
            tier=tier,
            abstain=abstain,
            warnings=warnings,
            pairwise_discriminator=pairwise,
            verification_questions=questions,
            top_matches=matches,
            margin=margin,
            specialist=specialist,
            crop_method=crop_method,
            max_similarity=max_similarity,
            message=_TIER_ACTION.get(tier, ""),
            extra={"energy": energy, "entropy": entropy, "bundle_version": self.meta.get("bundle_version", BUNDLE_VERSION)},
        )


def _temperature() -> float:
    raw = os.environ.get("TURTLEID_TEMPERATURE", "").strip()
    if not raw:
        return float(DEFAULT_TEMPERATURE)
    value = float(raw)
    if value <= 0:
        raise ValueError("TURTLEID_TEMPERATURE must be positive")
    return value


def _similarity_floor() -> float | None:
    raw = os.environ.get("TURTLEID_SIM_FLOOR", "").strip()
    if raw:
        return float(raw)
    return float(GALLERY_SIMILARITY_FLOOR)


def package_ranked_matches(
    ranked,
    db: SpeciesDB,
    *,
    state: str | None = None,
    engine: str = "bioclip_v1",
) -> StackPrediction:
    """Wrap a BioCLIP v1 ranking in the same response shape as Stack v2."""
    if not ranked:
        return StackPrediction(
            engine=engine,
            tier="INDETERMINATE",
            abstain=True,
            warnings=["No reference match was produced."],
            pairwise_discriminator=None,
            verification_questions=[],
            top_matches=[],
            message=_TIER_ACTION["INDETERMINATE"],
        )
    species_ids = [species.id for species in db]
    index = {sid: i for i, sid in enumerate(species_ids)}
    scores = np.full(len(species_ids), -2.0, dtype=np.float64)
    for match in ranked:
        pos = index.get(match.species_id)
        if pos is not None:
            scores[pos] = float(match.score)
    # The v1 matcher already refused to abstain on cosine ties. Do not apply
    # the embedding-floor gate, which was calibrated for a different model.
    predictor_like = StackV2Predictor.__new__(StackV2Predictor)
    predictor_like.db = db
    predictor_like.species_ids = species_ids
    predictor_like.temperature = _temperature()
    predictor_like.calibrated = False
    predictor_like.reference_urls = reference_image_urls()
    predictor_like.meta = {}
    return predictor_like._package(
        scores,
        state=state,
        specialist=False,
        crop_method="bioclip_v1",
        max_similarity=None,
        engine=engine,
    )


# ---------------------------------------------------------------------- weights


def _torch_device(torch) -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(torch, device: str):
    raw = os.environ.get("TURTLEID_DTYPE", "").strip().lower()
    table = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if raw in table:
        requested = table[raw]
    elif device == "cpu":
        requested = torch.float32
    else:
        requested = torch.float16
    if device == "cpu" and requested is torch.float16:
        logger.info("TURTLEID_DTYPE=fp16 is not used on CPU; loading fp32 vision towers")
        return torch.float32
    return requested


def _empty_device_cache(torch, device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def detach_visual(model):
    """Keep ``model.visual`` and drop the text tower (~1.6 GB on BioCLIP 2.5)."""
    visual = model.visual
    modules = getattr(model, "_modules", {})
    for name in list(modules):
        if name != "visual":
            del modules[name]
    modules.pop("visual", None)
    for name in (
        "transformer",
        "token_embedding",
        "ln_final",
        "text_projection",
        "positional_embedding",
        "attn_mask",
    ):
        if hasattr(model, name):
            try:
                setattr(model, name, None)
            except Exception:
                pass
    del model
    gc.collect()
    return visual


def _visual_features(visual, tensor):
    out = visual(tensor)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


def _move_module(module, device: str, dtype):
    try:
        return module.to(device=device, dtype=dtype).eval()
    except Exception as exc:
        logger.warning("dtype cast failed (%s); retrying fp32", exc)
        import torch

        return module.to(device=device, dtype=torch.float32).eval()


class VisionTowers:
    """Flip-averaged vision embeddings. Text parameters are not retained."""

    def __init__(self, device: str, dtype):
        self.device = device
        self.dtype = dtype
        self.bioclip = None
        self.bioclip_preprocess = None
        self.dino = None
        self.dino_processor = None
        self.biocap = None
        self.biocap_preprocess = None
        self.torch = None

    @classmethod
    def load(cls, *, use_biocap_channel: bool = True) -> "VisionTowers":
        import torch

        device = _torch_device(torch)
        dtype = _resolve_dtype(torch, device)
        towers = cls(device, dtype)
        towers.torch = torch
        logger.info("Loading vision towers on %s dtype=%s", device, dtype)
        towers.bioclip, towers.bioclip_preprocess = _load_openclip_visual(BIOCLIP_ID, device, dtype)
        towers.dino, towers.dino_processor = _load_dino(device, dtype)
        if use_biocap_channel:
            try:
                towers.biocap, towers.biocap_preprocess = _load_openclip_visual(
                    BIOCAP_ID, device, dtype
                )
            except Exception as exc:
                logger.warning("BioCAP vision unavailable (%s); trait channel off", exc)
                towers.biocap = None
        gc.collect()
        _empty_device_cache(torch, device)
        return towers

    def embed_bioclip(self, images: list[Image.Image]) -> np.ndarray:
        return self._embed_visual(self.bioclip, self.bioclip_preprocess, images)

    def embed_dino(self, images: list[Image.Image]) -> np.ndarray:
        return _embed_dino_batch(self.torch, self.dino, self.dino_processor, images, self.device, self.dtype)

    def embed_biocap(self, images: list[Image.Image]) -> np.ndarray:
        if self.biocap is None or self.biocap_preprocess is None:
            raise RuntimeError("BioCAP vision tower is not loaded")
        return self._embed_visual(self.biocap, self.biocap_preprocess, images)

    def _embed_visual(self, visual, preprocess, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, 0), np.float32)
        rows = []
        torch = self.torch
        for image in images:
            tensor = preprocess(image.convert("RGB")).unsqueeze(0)
            tensor = tensor.to(device=self.device, dtype=self.dtype)
            flipped = torch.flip(tensor, dims=[3])
            with torch.no_grad():
                summed = _visual_features(visual, tensor) + _visual_features(visual, flipped)
            rows.append(summed.float().cpu().numpy())
        return l2_normalise(np.vstack(rows).astype(np.float32))


def _load_openclip_visual(model_id: str, device: str, dtype):
    from core.photo_match import _load_openclip_model_streaming

    visual, preprocess = _load_openclip_model_streaming(
        model_id, device=device, dtype=dtype, visual_only=True
    )
    logger.info("Loaded visual tower %s", model_id)
    return visual, preprocess


def _load_dino(device: str, dtype):
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(DINO_ID)
    try:
        model = AutoModel.from_pretrained(DINO_ID, dtype=dtype, low_cpu_mem_usage=True)
    except TypeError:
        model = AutoModel.from_pretrained(DINO_ID)
    model = _move_module(model, device, dtype)
    logger.info("Loaded %s", DINO_ID)
    return model, processor


def _embed_dino_batch(torch, model, processor, images, device: str, dtype) -> np.ndarray:
    if not images:
        return np.empty((0, DINO_DIM), np.float32)
    rows = []
    for image in images:
        pair = [image.convert("RGB"), image.convert("RGB").transpose(Image.FLIP_LEFT_RIGHT)]
        inputs = processor(images=pair, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)
        with torch.no_grad():
            cls = model(**inputs).last_hidden_state[:, 0]
        rows.append(cls.float().mean(dim=0).cpu().numpy())
    return l2_normalise(np.vstack(rows).astype(np.float32))


def preload_vision_weights() -> None:
    """Download vision weights into HF_HOME, then release them.

    Used by the image build and by ``scripts/preload_models.py``. The text
    towers are discarded inside ``VisionTowers.load``.
    """
    towers = VisionTowers.load(use_biocap_channel=use_biocap())
    if detector_mode() not in {"border", "border_crop", "none", "off"}:
        from core.stack_v2.crops import _load_grounding_dino

        _load_grounding_dino()
        release_detector()
    del towers
    gc.collect()


def text_prompts_for_species(species) -> list[str]:
    """Shared with the exporter so bundle text matches query-time names."""
    return prompts_for(species.id, species.scientific_name, species.common_en, species.family)
