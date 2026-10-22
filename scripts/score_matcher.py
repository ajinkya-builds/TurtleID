"""Score the species matcher on the verified field photographs.

The queries are the contributed photographs under submissions/. The names on
those records are the answers. Two questions are scored separately, because
they fail for different reasons:

* Name. Does the photograph agree with the species name, ignoring the card?
* Card. Does the photograph land nearest the right identification-card plate?

The combined rank is what BioClipMatcher.rank returns. Chance, for a name
picked at random from the reference list, is printed beside each figure.

    python -m scripts.score_matcher
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from config import BASE_DIR, LOG_DIR
from core.database import SpeciesDB
from core.matcher import embed_images, load_backbone
from core.photo_match import BioClipMatcher

SUBMISSIONS = BASE_DIR / "submissions"
RECORDS = SUBMISSIONS / "records"
IMAGES = SUBMISSIONS / "images"


def queries() -> list[dict]:
    rows = []
    for path in sorted(RECORDS.glob("*.json")):
        record = json.loads(path.read_text())
        image_path = IMAGES / record["image_file"]
        if record.get("kind") != "image" or not image_path.exists():
            continue
        rows.append({
            "species_id": record["species_id"],
            "view": record.get("view") or "",
            "certainty": record.get("certainty") or "",
            "file": record["image_file"],
            "path": image_path,
        })
    return rows


def accuracy(hits: list[bool]) -> float:
    return float(np.mean(hits)) if hits else float("nan")


def main() -> None:
    db = SpeciesDB.load()
    pending = queries()
    if not pending:
        raise SystemExit(f"No labelled photographs under {SUBMISSIONS}")

    verified = [row for row in pending if row["certainty"] == "verified"]
    print(f"Field photographs: {len(pending)}  verified: {len(verified)}")
    print(f"Species in the reference list: {len(db)}")
    print("Loading BioCLIP 2 (weights download on first run)...")
    matcher = BioClipMatcher(db)

    images = [Image.open(row["path"]) for row in verified]
    cropped = [matcher.prepare(image, crop=True) for image in images]
    full = [matcher.prepare(image, crop=False) for image in images]

    print("Embedding field photographs and identification cards...")
    text = matcher.text_matrix
    crop_vectors = matcher.embed(cropped)
    full_vectors = matcher.embed(full)
    crop_plates = matcher.plate_matrix(crop=True)
    full_plates = matcher.plate_matrix(crop=False)
    plate_ids = [species_id for species_id, _ in matcher.plate_index]
    plate_position = {species_id: index for index, species_id in enumerate(plate_ids)}
    species_ids = [species.id for species in matcher.species]

    print("Embedding the same photographs with the current ResNet matcher...")
    backbone, transform = load_backbone("resnet50")
    resnet = embed_images(full, backbone, transform)
    resnet_plates = embed_images(
        [image for _, image in matcher.plate_index], backbone, transform
    )

    methods = {
        "name, photo as taken": _text_hits(full_vectors, text, species_ids, verified),
        "name, animal cropped": _text_hits(crop_vectors, text, species_ids, verified),
        "card, photo as taken": _plate_hits(full_vectors, full_plates, plate_position, verified),
        "card, animal cropped": _plate_hits(crop_vectors, crop_plates, plate_position, verified),
        "combined, photo as taken": _combined_hits(
            full_vectors, text, full_plates, species_ids, plate_position, verified
        ),
        "combined, animal cropped": _combined_hits(
            crop_vectors, text, crop_plates, species_ids, plate_position, verified
        ),
        "resnet card, photo as taken": _plate_hits(
            resnet, resnet_plates, plate_position, verified
        ),
    }

    chance_name = 1 / len(species_ids)
    chance_card = 1 / len(plate_ids)
    chance_top3_name = 3 / len(species_ids)
    print()
    print(f"{'method':<32} {'top-1':>8} {'top-3':>8}  {'chance top-1':>12}")
    for name, (top1, top3, _) in methods.items():
        chance = chance_card if name.startswith("card") or name.startswith("resnet") else chance_name
        print(f"{name:<32} {accuracy(top1):8.1%} {accuracy(top3):8.1%}  {chance:12.1%}")
    print(f"{'chance, top-3 of the name list':<32} {'':>8} {chance_top3_name:8.1%}")

    print()
    for method in ("card, photo as taken", "combined, photo as taken", "name, photo as taken"):
        print(f"Misses for {method}:")
        for row, detail in zip(verified, methods[method][2]):
            if detail["hit"]:
                continue
            print(
                f"  {row['species_id']:<24} {row['view']:<20} "
                f"-> {', '.join(detail['top'])}"
            )
        if all(detail["hit"] for detail in methods[method][2]):
            print("  none")
        print()

    print("Card match, photo as taken, one row per verified photograph:")
    _, _, details = methods["card, photo as taken"]
    by_species: dict[str, list[bool]] = defaultdict(list)
    for row, detail in zip(verified, details):
        mark = "yes" if detail["hit"] else "no"
        by_species[row["species_id"]].append(detail["hit"])
        print(
            f"  {mark:<3} {row['species_id']:<24} {row['view']:<20} "
            f"-> {detail['top'][0]}"
            + ("" if detail["hit"] else f"   (then {detail['top'][1]})")
        )

    print()
    print("Top-1 by species, card match on the photo as taken:")
    for species_id, hits in sorted(by_species.items()):
        print(f"  {species_id:<24} {sum(hits)}/{len(hits)}")

    report = {
        "model": "hf-hub:imageomics/bioclip-2",
        "verified": len(verified),
        "species_in_list": len(species_ids),
        "plates": len(plate_ids),
        "methods": {
            name: {"top1": accuracy(top1), "top3": accuracy(top3)}
            for name, (top1, top3, _) in methods.items()
        },
        "rows": [
            {
                "file": row["file"],
                "species_id": row["species_id"],
                "view": row["view"],
                "hit": detail["hit"],
                "top": detail["top"],
            }
            for row, detail in zip(verified, details)
        ],
    }
    destination = LOG_DIR / "matcher_scores.json"
    destination.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {destination.relative_to(BASE_DIR)}")


def _text_hits(vectors, text, species_ids, verified):
    scores = vectors @ text.T
    return _hits(scores, species_ids, verified)


def _plate_hits(vectors, plates, plate_position, verified):
    scores = vectors @ plates.T
    species_ids = [None] * len(plate_position)
    for species_id, index in plate_position.items():
        species_ids[index] = species_id
    return _hits(scores, species_ids, verified)


def _combined_hits(vectors, text, plates, species_ids, plate_position, verified):
    from core.photo_match import _zscore

    text_scores = vectors @ text.T
    plate_scores = vectors @ plates.T
    combined = np.zeros_like(text_scores)
    for row_index in range(len(vectors)):
        plate_row = np.full(len(species_ids), np.nan)
        for species_id, index in plate_position.items():
            plate_row[species_ids.index(species_id)] = plate_scores[row_index, index]
        combined[row_index] = _zscore(text_scores[row_index]) + _zscore(plate_row)
    return _hits(combined, species_ids, verified)


def _hits(scores: np.ndarray, species_ids: list, verified: list[dict]):
    top1, top3, details = [], [], []
    for row_index, row in enumerate(verified):
        order = np.argsort(-scores[row_index])
        names = [species_ids[int(index)] for index in order[:3]]
        top1.append(names[0] == row["species_id"])
        top3.append(row["species_id"] in names)
        details.append({"hit": names[0] == row["species_id"], "top": names})
    return top1, top3, details


if __name__ == "__main__":
    main()
