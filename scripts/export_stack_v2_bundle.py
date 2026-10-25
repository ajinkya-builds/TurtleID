"""Export the Stack v2 query bundle (no photos, no text towers at runtime).

Full benchmark mode (Ajinkya's machine, when both are present)::

    python -m scripts.export_stack_v2_bundle --turtles-dir Turtles

uses ``Turtles/`` plus ``logs/turtles_ground_truth.json``, the iNaturalist
gallery, verified ``submissions/``, and the 28 TRAFFIC plates.

Repo-standalone mode (CI, cloud build, no seizure photos)::

    python -m scripts.export_stack_v2_bundle

uses ``reference_images/``, verified ``submissions/images/``, and
``data/field_gallery/`` (downloaded by ``ensure_inat_gallery`` when online).

The file written is ``data/stack_v2_bundle.npz``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image

from config import BASE_DIR, LOG_DIR, REFERENCE_IMAGE_DIR, REFERENCE_MANIFEST
from core.database import SpeciesDB
from core.photo_match import BioClipMatcher, _normalise
from core.stack_v2.bundle import assemble_bundle, save_bundle
from core.stack_v2.crops import crop_animal_and_head, release_detector
from core.stack_v2.gallery import ensure_inat_gallery
from core.stack_v2.runtime import (
    BIOCAP_ID,
    BIOCLIP_ID,
    text_prompts_for_species,
)

logger = logging.getLogger("export_stack_v2")

DEFAULT_OUT = BASE_DIR / "data" / "stack_v2_bundle.npz"
DEFAULT_TURTLES = BASE_DIR / "Turtles"
DEFAULT_TRUTH = LOG_DIR / "turtles_ground_truth.json"
DEFAULT_INAT = BASE_DIR / "data" / "field_gallery"
SUBMISSION_RECORDS = BASE_DIR / "submissions" / "records"
SUBMISSION_IMAGES = BASE_DIR / "submissions" / "images"
SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def iter_plates(db: SpeciesDB) -> list[tuple[str, Path]]:
    if not REFERENCE_MANIFEST.exists():
        return []
    manifest = json.loads(REFERENCE_MANIFEST.read_text(encoding="utf-8"))
    found: list[tuple[str, Path]] = []
    known = set(db.ids)
    for species_id, records in (manifest.get("images") or {}).items():
        if species_id not in known or not records:
            continue
        path = REFERENCE_IMAGE_DIR / records[0]["file"]
        if path.is_file():
            found.append((species_id, path))
    return found


def iter_verified_submissions(
    db: SpeciesDB,
    records_dir: Path = SUBMISSION_RECORDS,
    images_dir: Path = SUBMISSION_IMAGES,
) -> list[tuple[str, Path]]:
    if not records_dir.is_dir():
        return []
    known = set(db.ids)
    found: list[tuple[str, Path]] = []
    for path in sorted(records_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Skipping unreadable submission record %s", path.name)
            continue
        if record.get("certainty") != "verified":
            continue
        if record.get("kind") not in (None, "image"):
            continue
        species_id = record.get("species_id")
        if species_id not in known:
            continue
        image_name = record.get("image_file") or ""
        image_path = images_dir / image_name
        if image_path.is_file() and image_path.suffix.lower() in SUFFIXES:
            found.append((species_id, image_path))
    return found


def iter_turtles(
    db: SpeciesDB,
    turtles_dir: Path | None,
    truth_path: Path | None,
) -> list[tuple[str, Path]]:
    if turtles_dir is None or not turtles_dir.is_dir():
        return []
    if truth_path is None or not truth_path.is_file():
        logger.warning("No ground-truth file; seizure photos are not added to the bundle")
        return []
    payload = json.loads(truth_path.read_text(encoding="utf-8"))
    labels = payload.get("labels") or {}
    known = set(db.ids)
    found: list[tuple[str, Path]] = []
    for name, label in sorted(labels.items()):
        if not label or label == "unidentified" or label not in known:
            continue
        path = turtles_dir / name
        if path.is_file() and path.suffix.lower() in SUFFIXES:
            found.append((label, path))
    return found


def inventory(
    db: SpeciesDB,
    *,
    turtles_dir: Path | None = None,
    truth_path: Path | None = None,
    inat_root: Path = DEFAULT_INAT,
    skip_inat: bool = True,
) -> dict:
    """Count sources. Does not download the iNat gallery unless skip_inat is false."""
    plates = iter_plates(db)
    submissions = iter_verified_submissions(db)
    turtles = iter_turtles(db, turtles_dir, truth_path)
    inat: list[tuple[str, Path]] = []
    if not skip_inat:
        inat = [(sid, path) for sid, path in ensure_inat_gallery(inat_root) if sid in set(db.ids)]
    elif (inat_root / "manifest.json").exists():
        cached = ensure_inat_gallery(inat_root)
        inat = [(sid, path) for sid, path in cached if sid in set(db.ids)]
    mode = "full" if turtles else "standalone"
    return {
        "mode": mode,
        "plates": len(plates),
        "submissions": len(submissions),
        "turtles": len(turtles),
        "inat": len(inat),
        "species": len(db),
    }


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as img:
        img.load()
        return img.convert("RGB")


def _encode_prompt_means(matcher: BioClipMatcher, groups: list[list[str]]) -> np.ndarray:
    rows = []
    for prompts in groups:
        parts = []
        for start in range(0, len(prompts), 32):
            parts.append(matcher._encode_text(prompts[start:start + 32]))
        rows.append(np.vstack(parts).mean(axis=0))
    return _normalise(np.vstack(rows).astype(np.float32))


def _embed_paths_bioclip(matcher: BioClipMatcher, images: list[Image.Image], batch_size: int) -> np.ndarray:
    from core.stack_v2.bundle import BIOCLIP_DIM

    if not images:
        return np.empty((0, BIOCLIP_DIM), np.float32)
    prepared = [matcher.prepare(img, crop=False) for img in images]
    return matcher.embed(prepared, batch_size=batch_size)


def _embed_dino(images: list[Image.Image]) -> np.ndarray:
    if not images:
        from core.stack_v2.bundle import DINO_DIM

        return np.empty((0, DINO_DIM), np.float32)
    import torch
    from core.stack_v2.runtime import (
        _embed_dino_batch,
        _empty_device_cache,
        _load_dino,
        _resolve_dtype,
        _torch_device,
    )

    device = _torch_device(torch)
    dtype = _resolve_dtype(torch, device)
    model, processor = _load_dino(device, dtype)
    vectors = _embed_dino_batch(torch, model, processor, images, device, dtype)
    del model, processor
    _empty_device_cache(torch, device)
    return vectors


def export_bundle(
    out: Path,
    *,
    db: SpeciesDB | None = None,
    turtles_dir: Path | None = None,
    truth_path: Path | None = None,
    inat_root: Path = DEFAULT_INAT,
    skip_inat: bool = False,
    inat_per_species: int = 12,
    detector: str | None = None,
    batch_size: int = 2,
    skip_biocap: bool = False,
) -> Path:
    """Encode the available photos and write ``out``. Needs the vision models."""
    db = db if db is not None else SpeciesDB.load()
    species = list(db)
    species_ids = [s.id for s in species]

    plates = iter_plates(db)
    submissions = iter_verified_submissions(db)
    turtles = iter_turtles(db, turtles_dir, truth_path)
    if skip_inat:
        inat: list[tuple[str, Path]] = []
    else:
        inat = [
            (sid, path)
            for sid, path in ensure_inat_gallery(inat_root, per_species=inat_per_species)
            if sid in set(species_ids)
        ]

    field_items = submissions + turtles
    mode = "full" if turtles else "standalone"
    logger.info(
        "Bundle sources mode=%s plates=%d submissions=%d turtles=%d inat=%d",
        mode, len(plates), len(submissions), len(turtles), len(inat),
    )

    field_images = [
        crop_animal_and_head(_load_rgb(path), detector=detector).animal
        for _, path in field_items
    ]
    inat_images = [
        crop_animal_and_head(_load_rgb(path), detector=detector).animal
        for _, path in inat
    ]
    # Plates follow the benchmark: border crop inside BioClipMatcher.prepare.
    plate_images = [_load_rgb(path) for _, path in plates]
    release_detector()

    logger.info("Encoding BioCLIP 2.5 vision + text")
    matcher = BioClipMatcher(db, model_id=BIOCLIP_ID)
    bioclip_field = _embed_paths_bioclip(matcher, field_images, batch_size)
    bioclip_inat = _embed_paths_bioclip(matcher, inat_images, batch_size)
    if plate_images:
        prepared_plates = [matcher.prepare(img, crop=True) for img in plate_images]
        bioclip_plates = matcher.embed(prepared_plates, batch_size=batch_size)
    else:
        from core.stack_v2.bundle import BIOCLIP_DIM

        bioclip_plates = np.empty((0, BIOCLIP_DIM), np.float32)

    text_matrix = _encode_prompt_means(
        matcher, [text_prompts_for_species(s) for s in species]
    )
    families = sorted({s.family for s in species})
    genera = sorted({s.scientific_name.split()[0] for s in species})
    fam_vecs = _encode_prompt_means(
        matcher,
        [[f"Animalia Chordata Reptilia Testudines {name}"] for name in families],
    )
    gen_vecs = _encode_prompt_means(
        matcher,
        [[f"Animalia Chordata Reptilia Testudines {name}"] for name in genera],
    )
    del matcher
    _release_cuda()

    if skip_biocap:
        biocap_text = np.zeros((len(species), 512), np.float32)
    else:
        logger.info("Encoding BioCAP text")
        biocap = BioClipMatcher(db, model_id=BIOCAP_ID)
        biocap_text = _encode_prompt_means(
            biocap, [text_prompts_for_species(s) for s in species]
        )
        del biocap
        _release_cuda()

    logger.info("Encoding DINOv2 field support (%d)", len(field_images))
    dino_field = _embed_dino(field_images)

    arrays = assemble_bundle(
        species_ids=species_ids,
        bioclip_field=bioclip_field,
        field_labels=[sid for sid, _ in field_items],
        dino_field=dino_field,
        bioclip_inat=bioclip_inat,
        inat_labels=[sid for sid, _ in inat],
        bioclip_plates=bioclip_plates,
        plate_ids=[sid for sid, _ in plates],
        text_matrix=text_matrix,
        fam_vecs=fam_vecs,
        fam_names=families,
        gen_vecs=gen_vecs,
        gen_names=genera,
        species_families=[s.family for s in species],
        species_genera=[s.scientific_name.split()[0] for s in species],
        biocap_text_matrix=biocap_text,
        meta={
            "mode": mode,
            "bioclip": BIOCLIP_ID,
            "biocap": None if skip_biocap else BIOCAP_ID,
            "dino": "facebook/dinov2-with-registers-large",
            "n_submissions": len(submissions),
            "n_turtles": len(turtles),
            "n_inat": len(inat),
            "n_plates": len(plates),
            "detector": detector or os.environ.get("TURTLEID_DETECTOR") or "grounding_dino",
        },
    )
    save_bundle(out, arrays)
    logger.info("Wrote %s", out)
    return out


def _release_cuda() -> None:
    import gc

    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--turtles-dir",
        type=Path,
        default=None,
        help="Seizure photo directory. Omit to use ./Turtles when it exists.",
    )
    parser.add_argument("--truth", type=Path, default=None)
    parser.add_argument("--inat-root", type=Path, default=DEFAULT_INAT)
    parser.add_argument("--skip-inat", action="store_true")
    parser.add_argument("--inat-per-species", type=int, default=12)
    parser.add_argument("--detector", choices=["grounding_dino", "border"], default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--skip-biocap", action="store_true")
    parser.add_argument(
        "--inventory",
        action="store_true",
        help="Print source counts and exit. Does not load models or download iNat.",
    )
    args = parser.parse_args()

    if args.turtles_dir is not None:
        if not args.turtles_dir.exists():
            raise SystemExit(f"turtles dir not found: {args.turtles_dir}")
        turtles_dir = args.turtles_dir
        truth = args.truth or DEFAULT_TRUTH
    else:
        turtles_dir = DEFAULT_TURTLES if DEFAULT_TURTLES.is_dir() else None
        truth = args.truth or (DEFAULT_TRUTH if turtles_dir else None)

    db = SpeciesDB.load()
    if args.inventory:
        print(json.dumps(inventory(
            db,
            turtles_dir=turtles_dir,
            truth_path=truth if truth and truth.exists() else None,
            inat_root=args.inat_root,
            skip_inat=True,
        ), indent=2))
        return

    export_bundle(
        args.out,
        db=db,
        turtles_dir=turtles_dir,
        truth_path=truth if truth and Path(truth).exists() else None,
        inat_root=args.inat_root,
        skip_inat=args.skip_inat,
        inat_per_species=args.inat_per_species,
        detector=args.detector,
        batch_size=args.batch_size,
        skip_biocap=args.skip_biocap,
    )


if __name__ == "__main__":
    main()
