"""Run a local BioCLIP model on every photograph in Turtles/.

No Streamlit. No remote inference. Weights load through OpenCLIP from the
Hugging Face cache on this machine.

    python -m scripts.score_drive_batch
    python -m scripts.score_drive_batch --model hf-hub:imageomics/bioclip-2.5-vith14 --tag bioclip25

Writes:
  logs/turtles_<tag>.json
  logs/turtles_<tag>.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from config import BASE_DIR, LOG_DIR
from core.database import SpeciesDB
from core.photo_match import MODEL_ID, BioClipMatcher

BATCH = BASE_DIR / "Turtles"
SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def image_paths() -> list[Path]:
    return sorted(
        path for path in BATCH.iterdir()
        if path.is_file() and path.suffix.lower() in SUFFIXES
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=MODEL_ID,
        help=f"OpenCLIP / Hugging Face model id (default: {MODEL_ID})",
    )
    parser.add_argument(
        "--tag",
        default="bioclip",
        help="Output file tag under logs/turtles_<tag>.* (default: bioclip)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Image embed batch size (lower if the GPU runs out of memory)",
    )
    args = parser.parse_args()

    paths = image_paths()
    if not paths:
        raise SystemExit(f"No images under {BATCH}")

    print(f"Images: {len(paths)}")
    print(f"Loading {args.model} locally (OpenCLIP / Hugging Face cache)...")
    matcher = BioClipMatcher(SpeciesDB.load(), model_id=args.model)
    print(f"Device: {matcher.device_name}")
    print(f"Reference plates: {len(matcher.plate_index)}")

    images = [Image.open(path) for path in paths]
    prepared = [matcher.prepare(image, crop=False) for image in images]
    print("Embedding photographs...")
    vectors = matcher.embed(prepared, batch_size=args.batch_size)
    plates = matcher.plate_matrix(crop=False, batch_size=args.batch_size)
    plate_ids = [species_id for species_id, _ in matcher.plate_index]
    scores = vectors @ plates.T

    rows = []
    for index, path in enumerate(paths):
        order = scores[index].argsort()[::-1]
        top = []
        for rank in order[:3]:
            species_id = plate_ids[int(rank)]
            species = matcher.db.get(species_id)
            top.append({
                "species_id": species_id,
                "scientific_name": species.scientific_name,
                "common_en": species.common_en,
                "score": float(scores[index, rank]),
            })
        rows.append({
            "file": path.name,
            "top1": top[0]["species_id"],
            "top1_common": top[0]["common_en"],
            "top1_score": top[0]["score"],
            "top2": top[1]["species_id"],
            "top2_score": top[1]["score"],
            "top3": top[2]["species_id"],
            "top3_score": top[2]["score"],
            "top": top,
        })

    counts = Counter(row["top1"] for row in rows)
    print()
    print(f"{'file':<42} {'top-1':<28} {'score':>6}  top-2")
    print("-" * 100)
    for row in rows:
        print(
            f"{row['file']:<42} {row['top1']:<28} {row['top1_score']:6.3f}  "
            f"{row['top2']}"
        )

    print()
    print("Top-1 counts:")
    for species_id, count in counts.most_common():
        print(f"  {species_id:<32} {count}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "model": args.model,
        "device": matcher.device_name,
        "source": str(BATCH.relative_to(BASE_DIR)),
        "n_images": len(rows),
        "n_plates": len(plate_ids),
        "top1_counts": dict(counts.most_common()),
        "rows": rows,
    }
    json_path = LOG_DIR / f"turtles_{args.tag}.json"
    json_path.write_text(json.dumps(report, indent=2))

    csv_path = LOG_DIR / f"turtles_{args.tag}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "file", "top1", "top1_common", "top1_score",
                "top2", "top2_score", "top3", "top3_score",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    print(f"\nWrote {json_path.relative_to(BASE_DIR)}")
    print(f"Wrote {csv_path.relative_to(BASE_DIR)}")
    print("Review the CSV/JSON, then mark which top-1 calls look right.")


if __name__ == "__main__":
    main()
