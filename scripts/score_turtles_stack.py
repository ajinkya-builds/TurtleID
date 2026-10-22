"""Supplemented BioCLIP 2.5 stack on Turtles/ — blind predictions, then score.

Pipeline (labels are not used to pick the answer for a query):
  1. Foreground crop (border-based animal crop)
  2. Taxonomy gate (BioCLIP text prompts → coarse group)
  3. Leave-one-out field gallery + ID-card plates inside the gate
  4. Abstain flag when top-2 scores are nearly tied

    python -m scripts.score_turtles_stack
    python -m scripts.score_turtles_stack --compare

Writes:
  logs/turtles_stack.json
  logs/turtles_stack.csv
  logs/turtles_stack_vs_truth.csv   (with --compare)
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from config import BASE_DIR, LOG_DIR
from core.database import SpeciesDB
from core.photo_match import BioClipMatcher, _normalise, _zscore
from core.taxonomy_gate import GROUP_OF, GROUP_PROMPTS, species_in_groups

BATCH = BASE_DIR / "Turtles"
TRUTH_PATH = LOG_DIR / "turtles_ground_truth.json"
MODEL_ID = "hf-hub:imageomics/bioclip-2.5-vith14"
SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
GATE_TOP_K = 2
ABSTAIN_MARGIN = 0.05
GALLERY_WEIGHT = 1.25
PLATE_WEIGHT = 1.0
TEXT_WEIGHT = 0.35


def _image_paths() -> list[Path]:
    return sorted(
        path for path in BATCH.iterdir()
        if path.is_file() and path.suffix.lower() in SUFFIXES
    )


def _load_truth() -> dict[str, str]:
    if not TRUTH_PATH.exists():
        return {}
    payload = json.loads(TRUTH_PATH.read_text())
    return dict(payload.get("labels", {}))


def _group_text_matrix(matcher: BioClipMatcher) -> tuple[list[str], np.ndarray]:
    names = list(GROUP_PROMPTS)
    prompts: list[str] = []
    counts: list[int] = []
    for name in names:
        group = GROUP_PROMPTS[name]
        prompts.extend(group)
        counts.append(len(group))
    encoded = matcher._encode_text(prompts)
    rows: list[np.ndarray] = []
    offset = 0
    for count in counts:
        rows.append(encoded[offset:offset + count].mean(axis=0))
        offset += count
    return names, _normalise(np.vstack(rows).astype(np.float32))


def _species_scores(
    query: np.ndarray,
    *,
    species_ids: list[str],
    text: np.ndarray,
    plate_ids: list[str],
    plates: np.ndarray,
    gallery_ids: list[str],
    gallery_vectors: np.ndarray,
    exclude_gallery_index: int | None,
) -> np.ndarray:
    """One score per species_ids entry (higher is better)."""
    text_map = {sid: float(query @ text[i]) for i, sid in enumerate(species_ids)}
    plate_map = {
        sid: float(query @ plates[i]) for i, sid in enumerate(plate_ids)
    }

    gallery_best: dict[str, float] = defaultdict(lambda: float("-inf"))
    if len(gallery_vectors):
        sims = gallery_vectors @ query
        for index, sid in enumerate(gallery_ids):
            if exclude_gallery_index is not None and index == exclude_gallery_index:
                continue
            if sims[index] > gallery_best[sid]:
                gallery_best[sid] = float(sims[index])

    text_row = np.array([text_map[s] for s in species_ids], dtype=np.float64)
    plate_row = np.array(
        [plate_map.get(s, np.nan) for s in species_ids], dtype=np.float64
    )
    gallery_row = np.array(
        [
            gallery_best[s] if gallery_best[s] > float("-inf") else np.nan
            for s in species_ids
        ],
        dtype=np.float64,
    )

    combined = (
        TEXT_WEIGHT * _zscore(text_row)
        + PLATE_WEIGHT * _zscore(plate_row)
        + GALLERY_WEIGHT * _zscore(gallery_row)
    )
    # Species with neither plate nor gallery fall back to text only.
    for i, sid in enumerate(species_ids):
        has_plate = np.isfinite(plate_row[i])
        has_gallery = np.isfinite(gallery_row[i])
        if not has_plate and not has_gallery:
            combined[i] = text_row[i]
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--compare",
        action="store_true",
        help="After predicting, score against logs/turtles_ground_truth.json",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Disable foreground crop (ablation)",
    )
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="Disable taxonomy gate (ablation)",
    )
    args = parser.parse_args()
    crop = not args.no_crop
    use_gate = not args.no_gate

    paths = _image_paths()
    if not paths:
        raise SystemExit(f"No images under {BATCH}")
    truth = _load_truth()

    print(f"Images: {len(paths)}")
    print(f"Loading {MODEL_ID}...")
    db = SpeciesDB.load()
    matcher = BioClipMatcher(db, model_id=MODEL_ID)
    species_ids = [species.id for species in matcher.species]
    print(f"Device: {matcher.device_name}  crop={crop}  gate={use_gate}")

    print("Embedding queries (foreground crop)...")
    images = [Image.open(path) for path in paths]
    prepared = [matcher.prepare(image, crop=crop) for image in images]
    query_vectors = matcher.embed(prepared, batch_size=args.batch_size)

    print("Embedding ID-card plates...")
    plate_ids = [sid for sid, _ in matcher.plate_index]
    plates = matcher.plate_matrix(crop=crop, batch_size=args.batch_size)
    text = matcher.text_matrix

    group_names, group_text = _group_text_matrix(matcher)

    # Field gallery: identified Drive photos present locally (not unidentified).
    gallery_paths: list[Path] = []
    gallery_labels: list[str] = []
    path_by_name = {path.name: path for path in paths}
    for name, label in sorted(truth.items()):
        if label == "unidentified":
            continue
        if name not in path_by_name:
            continue
        if label not in matcher._position:
            continue
        gallery_paths.append(path_by_name[name])
        gallery_labels.append(label)

    print(f"Field gallery photos (leave-one-out): {len(gallery_paths)}")
    if gallery_paths:
        gallery_prepared = [
            matcher.prepare(Image.open(path), crop=crop) for path in gallery_paths
        ]
        gallery_vectors = matcher.embed(
            gallery_prepared, batch_size=args.batch_size
        )
    else:
        gallery_vectors = np.empty((0, query_vectors.shape[1]), dtype=np.float32)

    gallery_index_by_name = {
        path.name: index for index, path in enumerate(gallery_paths)
    }

    rows = []
    for q_index, path in enumerate(paths):
        query = query_vectors[q_index]

        # Taxonomy gate
        group_scores = query @ group_text.T
        group_order = np.argsort(-group_scores)
        top_groups = [group_names[int(i)] for i in group_order[:GATE_TOP_K]]
        if use_gate:
            allowed = species_in_groups(top_groups)
        else:
            allowed = set(species_ids)
            top_groups = ["all"]

        exclude = gallery_index_by_name.get(path.name)
        scores = _species_scores(
            query,
            species_ids=species_ids,
            text=text,
            plate_ids=plate_ids,
            plates=plates,
            gallery_ids=gallery_labels,
            gallery_vectors=gallery_vectors,
            exclude_gallery_index=exclude,
        )

        # Mask species outside the gate
        masked = scores.copy()
        for i, sid in enumerate(species_ids):
            if sid not in allowed:
                masked[i] = -1e9

        order = np.argsort(-masked)
        top = []
        for rank in order[:3]:
            sid = species_ids[int(rank)]
            species = db.get(sid)
            top.append({
                "species_id": sid,
                "scientific_name": species.scientific_name,
                "common_en": species.common_en,
                "score": float(masked[rank]),
                "group": GROUP_OF.get(sid, ""),
            })

        margin = float(top[0]["score"] - top[1]["score"]) if len(top) > 1 else 1.0
        abstain = margin < ABSTAIN_MARGIN and top[0]["group"] == top[1].get("group")

        rows.append({
            "file": path.name,
            "top1": top[0]["species_id"],
            "top1_common": top[0]["common_en"],
            "top1_score": top[0]["score"],
            "top2": top[1]["species_id"],
            "top2_score": top[1]["score"],
            "top3": top[2]["species_id"],
            "top3_score": top[2]["score"],
            "margin": margin,
            "abstain": abstain,
            "gate_groups": top_groups,
            "top": top,
        })

    counts = Counter(row["top1"] for row in rows)
    abstain_n = sum(1 for row in rows if row["abstain"])
    print()
    print(f"{'file':<42} {'top-1':<28} {'margin':>6} abstain  gate")
    print("-" * 110)
    for row in rows:
        print(
            f"{row['file']:<42} {row['top1']:<28} {row['margin']:6.3f} "
            f"{'yes' if row['abstain'] else 'no':<7} {','.join(row['gate_groups'])}"
        )

    print()
    print("Top-1 counts:")
    for sid, n in counts.most_common():
        print(f"  {sid:<32} {n}")
    print(f"Abstain flags (margin < {ABSTAIN_MARGIN}): {abstain_n}/{len(rows)}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "model": MODEL_ID,
        "stack": {
            "crop": crop,
            "taxonomy_gate": use_gate,
            "gate_top_k": GATE_TOP_K,
            "leave_one_out_gallery": True,
            "gallery_n": len(gallery_paths),
            "abstain_margin": ABSTAIN_MARGIN,
            "weights": {
                "text": TEXT_WEIGHT,
                "plate": PLATE_WEIGHT,
                "gallery": GALLERY_WEIGHT,
            },
        },
        "device": matcher.device_name,
        "n_images": len(rows),
        "top1_counts": dict(counts.most_common()),
        "abstain_n": abstain_n,
        "rows": rows,
    }
    json_path = LOG_DIR / "turtles_stack.json"
    json_path.write_text(json.dumps(report, indent=2))

    csv_path = LOG_DIR / "turtles_stack.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "file", "top1", "top1_common", "top1_score",
                "top2", "top2_score", "top3", "top3_score",
                "margin", "abstain", "gate_groups",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **{k: row[k] for k in writer.fieldnames if k != "gate_groups"},
                "gate_groups": "|".join(row["gate_groups"]),
            })

    print(f"\nWrote {json_path.relative_to(BASE_DIR)}")
    print(f"Wrote {csv_path.relative_to(BASE_DIR)}")

    if args.compare:
        _compare(rows, truth)


def _compare(rows: list[dict], truth: dict[str, str]) -> None:
    pred = {row["file"]: row for row in rows}
    scored = [
        name for name, label in truth.items()
        if label != "unidentified" and name in pred
    ]
    if not scored:
        print("No overlapping labelled photos to compare.")
        return

    def hit(name: str, k: int = 1) -> bool:
        label = truth[name]
        tops = [pred[name]["top1"], pred[name]["top2"], pred[name]["top3"]]
        return label in tops[:k]

    top1 = sum(1 for name in scored if hit(name, 1))
    top3 = sum(1 for name in scored if hit(name, 3))
    abstain_scored = [name for name in scored if pred[name]["abstain"]]
    abstain_correct = sum(1 for name in abstain_scored if hit(name, 1))

    # Baselines from earlier runs if present
    def load_baseline(path: Path, key: str) -> dict[str, str]:
        if not path.exists():
            return {}
        return {r["file"]: r[key] for r in csv.DictReader(path.open())}

    b2 = load_baseline(LOG_DIR / "turtles_bioclip.csv", "top1")
    b25 = load_baseline(LOG_DIR / "turtles_bioclip25.csv", "top1")
    gem = load_baseline(LOG_DIR / "turtles_gemini.csv", "top1_id")

    def acc(mapping: dict[str, str]) -> str:
        if not mapping:
            return "n/a"
        subset = [n for n in scored if n in mapping]
        if not subset:
            return "n/a"
        ok = sum(1 for n in subset if mapping[n] == truth[n])
        return f"{ok}/{len(subset)} ({ok/len(subset):.1%})"

    n = len(scored)
    print()
    print(f"=== Compare on {n} identified photos (same set as before where possible) ===")
    print(f"Stack top-1:     {top1}/{n} ({top1/n:.1%})")
    print(f"Stack top-3:     {top3}/{n} ({top3/n:.1%})")
    print(f"BioCLIP 2:       {acc(b2)}")
    print(f"BioCLIP 2.5:     {acc(b25)}")
    print(f"Gemini:          {acc(gem)}")
    print(
        f"Abstain among scored: {len(abstain_scored)} "
        f"(of which top-1 still correct: {abstain_correct})"
    )

    by = defaultdict(lambda: [0, 0])
    for name in scored:
        label = truth[name]
        by[label][1] += 1
        if hit(name, 1):
            by[label][0] += 1
    print("\nPer-species stack top-1:")
    for sid in sorted(by):
        c, t = by[sid]
        print(f"  {sid:<32} {c}/{t}")

    out = LOG_DIR / "turtles_stack_vs_truth.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "file", "truth", "stack_top1", "stack_top2", "stack_top3",
                "ok", "top3_ok", "abstain", "margin", "gate_groups",
                "bioclip2", "bioclip25", "gemini",
            ],
        )
        writer.writeheader()
        for name in sorted(scored):
            writer.writerow({
                "file": name,
                "truth": truth[name],
                "stack_top1": pred[name]["top1"],
                "stack_top2": pred[name]["top2"],
                "stack_top3": pred[name]["top3"],
                "ok": "yes" if hit(name, 1) else "no",
                "top3_ok": "yes" if hit(name, 3) else "no",
                "abstain": "yes" if pred[name]["abstain"] else "no",
                "margin": f"{pred[name]['margin']:.4f}",
                "gate_groups": "|".join(pred[name]["gate_groups"]),
                "bioclip2": b2.get(name, ""),
                "bioclip25": b25.get(name, ""),
                "gemini": gem.get(name, ""),
            })
    print(f"\nWrote {out.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
