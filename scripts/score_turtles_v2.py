"""Full high-accuracy TurtleID stack v2 — blind predict, then compare.

Stages:
  1. Soft Tip-Adapter + CL2N + soft family/genus prior (no hard gate)
  2. BioCAP trait channel + richer BioCLIP trait prompts
  3. GroundingDINO animal crop (+ head band) with border-crop fallback
  4. External iNat multi-prototype gallery + k-means view clusters
  5. DINOv2-registers fusion + LOO logistic probe
  6. Low-margin specialist re-score + GroupKFold comparison

    python -m scripts.score_turtles_v2
    python -m scripts.score_turtles_v2 --compare
    python -m scripts.score_turtles_v2 --skip-inat --skip-dino   # faster ablations
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import GroupKFold

from config import BASE_DIR, LOG_DIR, REFERENCE_IMAGE_DIR, REFERENCE_MANIFEST
from core.database import SpeciesDB
from core.photo_match import BioClipMatcher, _normalise
from core.stack_v2.crops import prepare_crops, release_detector
from core.stack_v2.gallery import ensure_inat_gallery
from core.stack_v2.groups import groups_for, summarize_groups
from core.stack_v2.scoring import (
    cl2n,
    combine_channels,
    is_confusion_pair,
    kmeans_prototypes,
    loo_logistic_probs,
    soft_hierarchy_bonus,
    tip_adapter_scores,
)
from core.stack_v2.traits import prompts_for

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("stack_v2")

BATCH = BASE_DIR / "Turtles"
TRUTH_PATH = LOG_DIR / "turtles_ground_truth.json"
INAT_ROOT = BASE_DIR / "data" / "field_gallery"
BIOCLIP_ID = "hf-hub:imageomics/bioclip-2.5-vith14"
BIOCAP_ID = "hf-hub:imageomics/biocap"
DINO_ID = "facebook/dinov2-with-registers-large"
SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
MARGIN_SPECIALIST = 0.15
PLATE_WEIGHT = 0.35


def _paths() -> list[Path]:
    return sorted(
        p for p in BATCH.iterdir()
        if p.is_file() and p.suffix.lower() in SUFFIXES
    )


def _truth() -> dict[str, str]:
    if not TRUTH_PATH.exists():
        return {}
    return dict(json.loads(TRUTH_PATH.read_text()).get("labels", {}))


def _encode_text_prompts(matcher: BioClipMatcher, prompts: list[str], batch: int = 32) -> np.ndarray:
    rows = []
    for start in range(0, len(prompts), batch):
        rows.append(matcher._encode_text(prompts[start:start + batch]))
    return _normalise(np.vstack(rows).astype(np.float32))


def _species_text_matrix(matcher: BioClipMatcher) -> np.ndarray:
    """Mean of taxonomic + trait prompts per species (BioCLIP space)."""
    vectors = []
    for species in matcher.species:
        prompts = prompts_for(
            species.id, species.scientific_name, species.common_en, species.family
        )
        encoded = _encode_text_prompts(matcher, prompts)
        vectors.append(encoded.mean(axis=0))
    return _normalise(np.vstack(vectors).astype(np.float32))


def _rank_text_matrices(matcher: BioClipMatcher):
    families = sorted({s.family for s in matcher.species})
    genera = sorted({s.scientific_name.split()[0] for s in matcher.species})
    fam_prompts = [f"Animalia Chordata Reptilia Testudines {f}" for f in families]
    gen_prompts = [f"Animalia Chordata Reptilia Testudines {g}" for g in genera]
    fam_vecs = _encode_text_prompts(matcher, fam_prompts)
    gen_vecs = _encode_text_prompts(matcher, gen_prompts)
    family_of = [s.family for s in matcher.species]
    genus_of = [s.scientific_name.split()[0] for s in matcher.species]
    return families, fam_vecs, genera, gen_vecs, family_of, genus_of


def _embed_dino(images: list[Image.Image], batch_size: int = 2) -> np.ndarray:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info("Loading %s on %s", DINO_ID, device)
    processor = AutoImageProcessor.from_pretrained(DINO_ID)
    model = AutoModel.from_pretrained(DINO_ID).to(device).eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            inputs = processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model(**inputs).last_hidden_state[:, 0]
            rows.append(out.float().cpu().numpy())
    del model
    if device == "mps":
        torch.mps.empty_cache()
    return _normalise(np.vstack(rows).astype(np.float32))


def _biocap_channel(
    animal_images: list[Image.Image],
    species_ids: list[str],
    db: SpeciesDB,
    batch_size: int = 2,
) -> np.ndarray:
    """Per-image × species BioCAP cosine to trait prompts (mean over prompts)."""
    try:
        matcher = BioClipMatcher(db, model_id=BIOCAP_ID)
    except Exception as exc:
        logger.warning("BioCAP unavailable (%s); skipping trait channel", exc)
        return np.zeros((len(animal_images), len(species_ids)), dtype=np.float32)

    logger.info("BioCAP embeddings...")
    prepared = [matcher.prepare(img, crop=False) for img in animal_images]
    img_vecs = matcher.embed(prepared, batch_size=batch_size)
    text_rows = []
    for sid in species_ids:
        species = db.get(sid)
        prompts = prompts_for(sid, species.scientific_name, species.common_en, species.family)
        text_rows.append(_encode_text_prompts(matcher, prompts).mean(axis=0))
    text = _normalise(np.vstack(text_rows).astype(np.float32))
    scores = img_vecs @ text.T
    del matcher
    return scores.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--skip-inat", action="store_true")
    parser.add_argument("--skip-dino", action="store_true")
    parser.add_argument("--skip-biocap", action="store_true")
    parser.add_argument("--inat-per-species", type=int, default=10)
    args = parser.parse_args()

    paths = _paths()
    if not paths:
        raise SystemExit(f"No images under {BATCH}")
    truth = _truth()
    db = SpeciesDB.load()

    logger.info("Images: %d", len(paths))
    logger.info("Preparing crops (GroundingDINO / border fallback)...")
    bundles = []
    crop_methods = Counter()
    for path in paths:
        bundle = prepare_crops(Image.open(path))
        bundles.append(bundle)
        crop_methods[bundle.method] += 1
    logger.info("Crop methods: %s", dict(crop_methods))
    release_detector()

    animal_images = [b.animal for b in bundles]
    head_images = [b.head if b.head is not None else b.animal for b in bundles]

    logger.info("Loading BioCLIP 2.5...")
    matcher = BioClipMatcher(db, model_id=BIOCLIP_ID)
    species_ids = [s.id for s in matcher.species]
    n_species = len(species_ids)
    species_pos = {sid: i for i, sid in enumerate(species_ids)}

    logger.info("Embedding queries (animal crops) with BioCLIP 2.5...")
    q_bioclip = matcher.embed(
        [matcher.prepare(img, crop=False) for img in animal_images],
        batch_size=args.batch_size,
    )
    logger.info("Embedding head crops...")
    q_head = matcher.embed(
        [matcher.prepare(img, crop=False) for img in head_images],
        batch_size=args.batch_size,
    )

    text_matrix = _species_text_matrix(matcher)
    families, fam_vecs, genera, gen_vecs, family_of, genus_of = _rank_text_matrices(matcher)

    plate_ids = [sid for sid, _ in matcher.plate_index]
    plate_imgs = [matcher.prepare(img, crop=True) for _, img in matcher.plate_index]
    plate_vecs = matcher.embed(plate_imgs, batch_size=args.batch_size) if plate_imgs else np.empty((0, q_bioclip.shape[1]), np.float32)

    # Local labelled gallery (paths present + identified)
    local_paths: list[Path] = []
    local_labels: list[str] = []
    path_by_name = {p.name: p for p in paths}
    for name, label in sorted(truth.items()):
        if label == "unidentified" or name not in path_by_name:
            continue
        if label not in species_pos:
            continue
        local_paths.append(path_by_name[name])
        local_labels.append(label)
    local_index = {p.name: i for i, p in enumerate(local_paths)}
    # Reuse query embeddings for local gallery members
    name_to_qindex = {p.name: i for i, p in enumerate(paths)}
    local_vecs = np.vstack([q_bioclip[name_to_qindex[p.name]] for p in local_paths]) if local_paths else np.empty((0, q_bioclip.shape[1]), np.float32)

    # External iNat gallery
    ext_vecs = np.empty((0, q_bioclip.shape[1]), np.float32)
    ext_labels: list[str] = []
    if not args.skip_inat:
        logger.info("Ensuring external iNat gallery...")
        ext_items = ensure_inat_gallery(INAT_ROOT, per_species=args.inat_per_species)
        if ext_items:
            ext_imgs = []
            for sid, path in ext_items:
                crop = prepare_crops(Image.open(path)).animal
                ext_imgs.append(matcher.prepare(crop, crop=False))
                ext_labels.append(sid)
            ext_vecs = matcher.embed(ext_imgs, batch_size=args.batch_size)
            logger.info("External gallery embedded: %d", len(ext_labels))

    # CL2N mean from local+external (not plates)
    support_for_mean = []
    if len(local_vecs):
        support_for_mean.append(local_vecs)
    if len(ext_vecs):
        support_for_mean.append(ext_vecs)
    if support_for_mean:
        mean_vec = np.vstack(support_for_mean).mean(axis=0)
    else:
        mean_vec = q_bioclip.mean(axis=0)
    q_bioclip_c, _ = cl2n(q_bioclip, mean_vec)
    q_head_c, _ = cl2n(q_head, mean_vec)
    local_c, _ = cl2n(local_vecs, mean_vec) if len(local_vecs) else (local_vecs, mean_vec)
    ext_c, _ = cl2n(ext_vecs, mean_vec) if len(ext_vecs) else (ext_vecs, mean_vec)
    plate_c, _ = cl2n(plate_vecs, mean_vec) if len(plate_vecs) else (plate_vecs, mean_vec)

    # K-means multi-prototypes from EXTERNAL gallery only (avoids LOO leakage).
    if len(ext_c):
        proto_v, proto_l = kmeans_prototypes(ext_c, ext_labels, k_max=5)
    else:
        proto_v, proto_l = np.empty((0, q_bioclip.shape[1]), np.float32), []

    # Free BioCLIP before loading BioCAP / DINOv2 (MPS / unified memory).
    import torch
    del matcher
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Released BioCLIP 2.5 from memory")

    # BioCAP channel
    if args.skip_biocap:
        biocap_scores = np.zeros((len(paths), n_species), dtype=np.float32)
    else:
        biocap_scores = _biocap_channel(animal_images, species_ids, db, batch_size=args.batch_size)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # DINOv2
    if args.skip_dino:
        q_dino = np.zeros((len(paths), 1), dtype=np.float32)
        local_dino = np.zeros((len(local_paths), 1), dtype=np.float32)
        dino_ready = False
    else:
        logger.info("Embedding with DINOv2 registers...")
        q_dino = _embed_dino(animal_images, batch_size=max(1, args.batch_size))
        local_dino = np.vstack([q_dino[name_to_qindex[p.name]] for p in local_paths]) if local_paths else np.empty((0, q_dino.shape[1]), np.float32)
        dino_mean = local_dino.mean(axis=0) if len(local_dino) else q_dino.mean(axis=0)
        q_dino, _ = cl2n(q_dino, dino_mean)
        local_dino, _ = cl2n(local_dino, dino_mean) if len(local_dino) else (local_dino, dino_mean)
        dino_ready = True
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Fused features for logistic probe
    if dino_ready and len(local_paths):
        fused_all = np.concatenate([q_bioclip_c, q_dino], axis=1)
        fused_local = np.concatenate([local_c, local_dino], axis=1)
    else:
        fused_all = q_bioclip_c
        fused_local = local_c

    rows = []
    for qi, path in enumerate(paths):
        query = q_bioclip_c[qi]
        head_q = q_head_c[qi]

        # Tip-Adapter cache: external + prototypes + local LOO (exclude this file)
        cache_v = []
        cache_l: list[str] = []
        if len(ext_c):
            cache_v.append(ext_c)
            cache_l.extend(ext_labels)
        if len(proto_v):
            cache_v.append(proto_v)
            cache_l.extend(proto_l)
        exclude = local_index.get(path.name)
        if len(local_c):
            for li, label in enumerate(local_labels):
                if exclude is not None and li == exclude:
                    continue
                cache_v.append(local_c[li:li + 1])
                cache_l.append(label)
        if cache_v:
            cache_mat = np.vstack(cache_v)
        else:
            cache_mat = np.empty((0, query.shape[0]), np.float32)

        tip = tip_adapter_scores(query, cache_mat, cache_l, species_ids, alpha=1.15, beta=5.0)
        text = query @ text_matrix.T
        hier = soft_hierarchy_bonus(
            query, fam_vecs, gen_vecs, family_of, genus_of, families, genera
        )
        plate = np.full(n_species, np.nan, dtype=np.float64)
        if len(plate_c):
            sims = query @ plate_c.T
            for sid, score in zip(plate_ids, sims):
                plate[species_pos[sid]] = float(score)

        # DINOv2 tip-adapter on local LOO
        if dino_ready and len(local_dino):
            d_cache_v = []
            d_cache_l = []
            for li, label in enumerate(local_labels):
                if exclude is not None and li == exclude:
                    continue
                d_cache_v.append(local_dino[li])
                d_cache_l.append(label)
            if d_cache_v:
                d_mat = _normalise(np.vstack(d_cache_v))
                tip_dino = tip_adapter_scores(q_dino[qi], d_mat, d_cache_l, species_ids, alpha=1.0, beta=5.0)
            else:
                tip_dino = np.zeros(n_species)
        else:
            tip_dino = np.zeros(n_species)

        # Logistic probe (only if this image is in labelled local set, else use all)
        log_probs = np.zeros(n_species)
        if len(local_paths) >= 8:
            if exclude is not None:
                log_probs = loo_logistic_probs(fused_local, local_labels, species_ids, exclude)
            else:
                from sklearn.linear_model import LogisticRegression
                from sklearn.preprocessing import LabelEncoder
                enc = LabelEncoder()
                y = enc.fit_transform(local_labels)
                clf = LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced")
                clf.fit(fused_local, y)
                pred = clf.predict_proba(fused_all[qi:qi + 1])[0]
                for ci, name in enumerate(enc.classes_):
                    log_probs[species_pos[name]] = float(pred[ci])

        biocap_row = biocap_scores[qi]

        channels = [
            (tip, 1.25),
            (tip_dino, 0.9 if dino_ready else 0.0),
            (text, 0.55),
            (hier, 0.35),
            (np.nan_to_num(plate, nan=np.nanmean(plate) if np.isfinite(plate).any() else 0.0), PLATE_WEIGHT),
            (biocap_row, 0.45 if not args.skip_biocap else 0.0),
            (log_probs, 1.05),
        ]
        scores = combine_channels(channels)

        order = np.argsort(-scores)
        top = [species_ids[int(i)] for i in order[:3]]
        top_scores = [float(scores[int(i)]) for i in order[:3]]
        margin = top_scores[0] - top_scores[1] if len(top_scores) > 1 else 1.0

        specialist = False
        if margin < MARGIN_SPECIALIST and is_confusion_pair(top[0], top[1]):
            specialist = True
            head_tip = tip_adapter_scores(head_q, cache_mat, cache_l, species_ids, alpha=1.4, beta=6.0)
            head_text = head_q @ text_matrix.T
            boost = combine_channels([(head_tip, 1.2), (head_text, 0.8), (scores, 0.5)])
            # Only reshuffle within the top-5 of original
            allowed = set(species_ids[int(i)] for i in order[:5])
            for i, sid in enumerate(species_ids):
                if sid not in allowed:
                    boost[i] = -1e9
            order = np.argsort(-boost)
            top = [species_ids[int(i)] for i in order[:3]]
            top_scores = [float(boost[int(i)]) for i in order[:3]]
            margin = top_scores[0] - top_scores[1]
            scores = boost

        rows.append({
            "file": path.name,
            "top1": top[0],
            "top1_common": db.get(top[0]).common_en,
            "top1_score": top_scores[0],
            "top2": top[1],
            "top2_score": top_scores[1],
            "top3": top[2],
            "top3_score": top_scores[2],
            "margin": margin,
            "specialist": specialist,
            "crop_method": bundles[qi].method,
            "abstain": margin < MARGIN_SPECIALIST,
        })

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    counts = Counter(r["top1"] for r in rows)
    report = {
        "stack": "v2",
        "bioclip": BIOCLIP_ID,
        "biocap": None if args.skip_biocap else BIOCAP_ID,
        "dino": None if args.skip_dino else DINO_ID,
        "inat_gallery": 0 if args.skip_inat else len(ext_labels),
        "local_gallery": len(local_labels),
        "crop_methods": dict(crop_methods),
        "n_images": len(rows),
        "top1_counts": dict(counts.most_common()),
        "rows": rows,
    }
    json_path = LOG_DIR / "turtles_v2.json"
    json_path.write_text(json.dumps(report, indent=2))
    csv_path = LOG_DIR / "turtles_v2.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        fields = [
            "file", "top1", "top1_common", "top1_score", "top2", "top2_score",
            "top3", "top3_score", "margin", "specialist", "crop_method", "abstain",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})

    print(f"\n{'file':<42} {'top-1':<28} {'margin':>6} crop")
    print("-" * 100)
    for row in rows:
        print(
            f"{row['file']:<42} {row['top1']:<28} {row['margin']:6.3f} "
            f"{row['crop_method']}{' *' if row['specialist'] else ''}"
        )
    print("\nTop-1 counts:")
    for sid, n in counts.most_common():
        print(f"  {sid:<32} {n}")
    print(f"\nWrote {json_path.relative_to(BASE_DIR)}")
    print(f"Wrote {csv_path.relative_to(BASE_DIR)}")

    if args.compare:
        _compare(rows, truth)


def _compare(rows: list[dict], truth: dict[str, str]) -> None:
    pred = {r["file"]: r for r in rows}
    scored = [
        name for name, label in truth.items()
        if label != "unidentified" and name in pred
    ]
    if not scored:
        print("No labelled overlap to compare.")
        return

    def hit(name: str, k: int = 1) -> bool:
        tops = [pred[name]["top1"], pred[name]["top2"], pred[name]["top3"]]
        return truth[name] in tops[:k]

    n = len(scored)
    top1 = sum(1 for name in scored if hit(name, 1))
    top3 = sum(1 for name in scored if hit(name, 3))

    def load_map(path: Path, key: str) -> dict[str, str]:
        if not path.exists():
            return {}
        return {r["file"]: r[key] for r in csv.DictReader(path.open())}

    b25 = load_map(LOG_DIR / "turtles_bioclip25.csv", "top1")
    stack = load_map(LOG_DIR / "turtles_stack.csv", "top1")
    gem = load_map(LOG_DIR / "turtles_gemini.csv", "top1_id")

    def acc(mapping: dict[str, str]) -> str:
        if not mapping:
            return "n/a"
        subset = [x for x in scored if x in mapping]
        if not subset:
            return "n/a"
        ok = sum(1 for x in subset if mapping[x] == truth[x])
        return f"{ok}/{len(subset)} ({ok / len(subset):.1%})"

    print(f"\n=== Compare on {n} identified photos ===")
    print(f"Stack v2 top-1:  {top1}/{n} ({top1 / n:.1%})")
    print(f"Stack v2 top-3:  {top3}/{n} ({top3 / n:.1%})")
    print(f"Prior stack:     {acc(stack)}")
    print(f"BioCLIP 2.5:     {acc(b25)}")
    print(f"Gemini:          {acc(gem)}")

    by = defaultdict(lambda: [0, 0])
    for name in scored:
        label = truth[name]
        by[label][1] += 1
        if hit(name, 1):
            by[label][0] += 1
    print("\nPer-species v2 top-1:")
    for sid in sorted(by):
        c, t = by[sid]
        print(f"  {sid:<32} {c}/{t}")

    # GroupKFold accuracy: for each fold, only score held-out group using
    # whether predictions already used LOO (image-level). This reports
    # accuracy if we treat whole seizure groups as the unit of evaluation.
    groups = groups_for(scored)
    print("\nSeizure groups (top):", list(summarize_groups(scored).items())[:12])
    gkf = GroupKFold(n_splits=min(5, len(set(groups))))
    fold_scores = []
    X = np.arange(len(scored)).reshape(-1, 1)
    y = np.array([truth[n] for n in scored])
    g = np.array(groups)
    for fold, (_, test_idx) in enumerate(gkf.split(X, y, g), start=1):
        names = [scored[i] for i in test_idx]
        ok = sum(1 for name in names if hit(name, 1))
        fold_scores.append(ok / len(names))
        print(f"  GroupKFold fold {fold}: {ok}/{len(names)} ({ok / len(names):.1%})")
    print(f"  GroupKFold mean top-1: {np.mean(fold_scores):.1%}")

    out = LOG_DIR / "turtles_v2_vs_truth.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "file", "truth", "v2_top1", "v2_top2", "v2_top3", "ok", "top3_ok",
                "margin", "specialist", "crop_method", "group",
                "stack_v1", "bioclip25", "gemini",
            ],
        )
        writer.writeheader()
        for name in sorted(scored):
            writer.writerow({
                "file": name,
                "truth": truth[name],
                "v2_top1": pred[name]["top1"],
                "v2_top2": pred[name]["top2"],
                "v2_top3": pred[name]["top3"],
                "ok": "yes" if hit(name, 1) else "no",
                "top3_ok": "yes" if hit(name, 3) else "no",
                "margin": f"{pred[name]['margin']:.4f}",
                "specialist": "yes" if pred[name]["specialist"] else "no",
                "crop_method": pred[name]["crop_method"],
                "group": seizure_group_safe(name),
                "stack_v1": stack.get(name, ""),
                "bioclip25": b25.get(name, ""),
                "gemini": gem.get(name, ""),
            })
    print(f"\nWrote {out.relative_to(BASE_DIR)}")


def seizure_group_safe(name: str) -> str:
    from core.stack_v2.groups import seizure_group
    return seizure_group(name)


if __name__ == "__main__":
    main()
