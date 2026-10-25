"""Teacher-to-student losses and the browser scoring bundle.

The losses are plain NumPy so a unit test can check them without downloading
DINOv2. The projection head itself (384 → 1024 → 2048) lives in
``scripts/distill_browser_student.py`` and is built with PyTorch when training
or exporting ONNX.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from core.morphkey import CHARACTERS, MATRIX
from core.stack_v2.scoring import CONFUSION_PAIRS

BROWSER_BUNDLE_VERSION = 1

# Channel weights copied from core/stack_v2/runtime.py (score_turtles_v2).
CHANNEL_WEIGHTS = {
    "tip": 1.25,
    "tip_alpha": 1.15,
    "tip_beta": 5.0,
    "tip_dino": 0.9,
    "tip_dino_alpha": 1.0,
    "tip_dino_beta": 5.0,
    "text": 0.55,
    "hier": 0.35,
    "hier_family": 0.15,
    "hier_genus": 0.20,
    "plate": 0.35,
    "biocap": 0.45,
    "log_probs": 1.05,
    "margin_specialist": 0.15,
    "head_tip": 1.2,
    "head_tip_alpha": 1.4,
    "head_tip_beta": 6.0,
    "head_text": 0.8,
    "head_scores": 0.5,
    "temperature": 1.0,
    "similarity_floor": 0.30,
    "max_normalised_entropy": 0.55,
    "tier_confirmed": 0.85,
    "tier_probable": 0.60,
    "tier_tentative": 0.45,
}


def l2_rows(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def cosine_alignment_loss(student: np.ndarray, teacher: np.ndarray) -> float:
    """Mean 1 − cosine over a batch. Both sides are L2-normalised first."""
    left = l2_rows(student)
    right = l2_rows(teacher)
    if left.shape != right.shape:
        raise ValueError(f"cosine shapes {left.shape} vs {right.shape}")
    cos = np.sum(left * right, axis=-1)
    return float(np.mean(1.0 - cos))


def _pairwise_distance(matrix: np.ndarray) -> np.ndarray:
    """Mean-normalised Euclidean distances, upper triangle excluded from the mean."""
    x = l2_rows(matrix)
    diff = x[:, None, :] - x[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    off = ~np.eye(dist.shape[0], dtype=bool)
    scale = float(dist[off].mean()) if off.any() else 1.0
    if scale < 1e-8:
        scale = 1.0
    return dist / scale


def relational_similarity_loss(student: np.ndarray, teacher: np.ndarray) -> float:
    """Distance-wise relational knowledge distillation (Park et al.)."""
    if len(student) < 2:
        return 0.0
    left = _pairwise_distance(student)
    right = _pairwise_distance(teacher)
    return float(np.mean((left - right) ** 2))


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z, axis=-1, keepdims=True)
    exp = np.exp(z)
    return exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-12)


def kl_divergence(student_logits: np.ndarray, teacher_logits: np.ndarray) -> float:
    """KL(teacher ‖ student) on Stack v2 channel logits. Batch or one row."""
    teacher = _softmax(teacher_logits)
    student = _softmax(student_logits)
    teacher = np.clip(teacher, 1e-8, 1.0)
    student = np.clip(student, 1e-8, 1.0)
    kl = np.sum(teacher * (np.log(teacher) - np.log(student)), axis=-1)
    return float(np.mean(kl))


def classification_cross_entropy(logits: np.ndarray, class_index: np.ndarray) -> float:
    """Mean cross-entropy. ``class_index`` indexes the last axis of ``logits``."""
    probs = _softmax(logits)
    index = np.asarray(class_index, dtype=np.int64).reshape(-1)
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
    picked = probs[np.arange(len(index)), index]
    return float(np.mean(-np.log(np.clip(picked, 1e-8, 1.0))))


def distillation_loss(
    student_bioclip: np.ndarray,
    teacher_bioclip: np.ndarray,
    student_dino: np.ndarray,
    teacher_dino: np.ndarray,
    student_logits: np.ndarray,
    teacher_logits: np.ndarray,
    class_index: np.ndarray | None = None,
    *,
    w_cosine: float = 1.0,
    w_rkd: float = 0.5,
    w_kl: float = 1.0,
    w_ce: float = 0.5,
) -> dict[str, float]:
    """Cosine alignment + relational similarity + KL + optional cross-entropy."""
    cosine = 0.5 * (
        cosine_alignment_loss(student_bioclip, teacher_bioclip)
        + cosine_alignment_loss(student_dino, teacher_dino)
    )
    fused_student = np.concatenate(
        [l2_rows(student_bioclip), l2_rows(student_dino)], axis=-1
    )
    fused_teacher = np.concatenate(
        [l2_rows(teacher_bioclip), l2_rows(teacher_dino)], axis=-1
    )
    rkd = relational_similarity_loss(fused_student, fused_teacher)
    kl = kl_divergence(student_logits, teacher_logits)
    ce = 0.0
    if class_index is not None:
        ce = classification_cross_entropy(student_logits, class_index)
    total = w_cosine * cosine + w_rkd * rkd + w_kl * kl + w_ce * ce
    return {
        "cosine": cosine,
        "rkd": rkd,
        "kl": kl,
        "ce": ce,
        "total": float(total),
    }


def pack_f16(array: np.ndarray) -> dict:
    """Little-endian IEEE float16, base64. Shape is the logical ndarray shape."""
    arr = np.asarray(array, dtype=np.float32)
    raw = np.ascontiguousarray(arr.astype("<f2")).tobytes()
    return {
        "dtype": "float16",
        "endian": "little",
        "shape": list(arr.shape),
        "b64": base64.b64encode(raw).decode("ascii"),
    }


def unpack_f16(spec: dict) -> np.ndarray:
    raw = base64.b64decode(spec["b64"])
    values = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    return values.reshape(spec["shape"]).copy()


def _strings(array) -> list[str]:
    return [str(item) for item in np.asarray(array).reshape(-1).tolist()]


def morphkey_payload() -> dict:
    characters = []
    for key, (label, states) in CHARACTERS.items():
        characters.append({
            "id": key,
            "label": label,
            "states": [{"value": code, "label": text} for code, text in states.items()],
        })
    matrix = {}
    for sid, row in MATRIX.items():
        encoded = {}
        for char, expected in row.items():
            if expected is None:
                encoded[char] = None
            elif isinstance(expected, tuple):
                encoded[char] = list(expected)
            else:
                encoded[char] = expected
        matrix[sid] = encoded
    return {"characters": characters, "matrix": matrix}


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def build_browser_bundle(arrays: dict, species_payload: dict, manifest: dict | None) -> dict:
    """JSON-ready Stack v2 bundle plus species metadata and the MorphKey."""
    species_ids = _strings(arrays["species_ids"])
    by_id = {rec["id"]: rec for rec in species_payload.get("species", [])}
    species = {}
    plates = (manifest or {}).get("images") or {}
    for sid in species_ids:
        rec = by_id.get(sid)
        if rec is None:
            raise KeyError(sid)
        record = _jsonable(rec)
        images = plates.get(sid) or []
        if images:
            rel = str(images[0].get("file") or "").replace("\\", "/")
            record["reference_image"] = "reference_images/" + rel if rel else None
            record["reference_credit"] = images[0].get("credit")
        else:
            record["reference_image"] = None
            record["reference_credit"] = None
        species[sid] = record

    bundle = {
        "bundle_version": BROWSER_BUNDLE_VERSION,
        "kind": "stack_v2_browser",
        "note": (
            "Precomputed Stack v2 matrices and the morphological key. "
            "Embeddings are little-endian float16. No pixels, filenames of "
            "field photos, or coordinates."
        ),
        "preprocess": {
            "size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "layout": "nchw",
            "input_name": "pixel_values",
            "outputs": ["bioclip_emb", "dino_emb"],
        },
        "student": {
            "backbone": "facebook/dinov2-with-registers-small",
            "cls_dim": 384,
            "head": "384-1024-2048",
            "onnx": "models/turtleid_student_int8.onnx",
            "hf_onnx": (
                "https://huggingface.co/ajinkyapatil60/turtleid-student-onnx/"
                "resolve/main/turtleid_student_int8.onnx"
            ),
        },
        "weights": dict(CHANNEL_WEIGHTS),
        "geo_prior": {
            "resident": 1.0,
            "introduced": 0.9,
            "marginal": 0.35,
            "absent": 0.06,
            "unknown": 0.6,
        },
        "confusion_pairs": [sorted(pair) for pair in CONFUSION_PAIRS],
        "legal_basis": _jsonable(species_payload.get("legal_basis") or {}),
        "species_ids": species_ids,
        "species": species,
        "morphkey": morphkey_payload(),
        "bioclip_mean": pack_f16(arrays["bioclip_mean"]),
        "dino_mean": pack_f16(arrays["dino_mean"]),
        "cache_mat": pack_f16(arrays["cache_mat"]),
        "cache_labels": _strings(arrays["cache_labels"]),
        "local_dino": pack_f16(arrays["local_dino"]),
        "local_labels": _strings(arrays["local_labels"]),
        "text_matrix": pack_f16(arrays["text_matrix"]),
        "fam_vecs": pack_f16(arrays["fam_vecs"]),
        "fam_names": _strings(arrays["fam_names"]),
        "gen_vecs": pack_f16(arrays["gen_vecs"]),
        "gen_names": _strings(arrays["gen_names"]),
        "species_families": _strings(arrays["species_families"]),
        "species_genera": _strings(arrays["species_genera"]),
        "plate_c": pack_f16(arrays["plate_c"]),
        "plate_ids": _strings(arrays["plate_ids"]),
        "biocap_text_matrix": pack_f16(arrays["biocap_text_matrix"]),
        "probe_coef": pack_f16(arrays["probe_coef"]),
        "probe_intercept": pack_f16(arrays["probe_intercept"]),
        "probe_classes": _strings(arrays["probe_classes"]),
    }
    return bundle


def write_browser_bundle(bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def load_browser_bundle(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_browser_bundle_from_disk(
    npz_path: Path,
    species_db_path: Path,
    manifest_path: Path | None,
) -> dict:
    from core.stack_v2.bundle import load_bundle

    arrays = load_bundle(npz_path)
    species_payload = json.loads(species_db_path.read_text(encoding="utf-8"))
    manifest = None
    if manifest_path is not None and Path(manifest_path).exists():
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return build_browser_bundle(arrays, species_payload, manifest)


def topk_hit(scores: np.ndarray, species_ids: list[str], truth: str, k: int) -> bool:
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    names = [species_ids[int(i)] for i in order[:k]]
    return truth in names


def oracle_one_question(ranked: list[str], truth: str) -> str:
    """Answer the single most even MorphKey split with the true species' state.

    Used only by the offline benchmark. A field officer is not assumed to be
    an oracle; this estimates the ceiling of one careful tap.
    """
    from core.morphkey import MATRIX, most_discriminating

    short = [sid for sid in ranked[:3] if sid]
    if truth not in short:
        return ranked[0] if ranked else truth
    character = most_discriminating({}, short)
    if not character:
        return ranked[0]
    expected = MATRIX.get(truth, {}).get(character)
    if expected is None:
        return ranked[0]
    allowed = expected if isinstance(expected, tuple) else (expected,)
    kept = []
    for sid in short:
        row = MATRIX.get(sid, {}).get(character)
        if row is None:
            kept.append(sid)
            continue
        states = row if isinstance(row, tuple) else (row,)
        if any(state in allowed for state in states):
            kept.append(sid)
    if truth in kept:
        return kept[0]
    return ranked[0]


def ranking_report(
    teacher_scores: np.ndarray,
    student_scores: np.ndarray,
    labels: list[str],
    species_ids: list[str],
    *,
    margins: np.ndarray | None = None,
) -> dict[str, float]:
    """Top-1 / Top-3 for teacher and student, plus one-question MorphKey ceiling."""
    n = len(labels)
    if n == 0:
        raise ValueError("no labelled queries")
    teacher_top1 = teacher_top3 = student_top1 = student_top3 = 0
    student_key = 0
    for i, truth in enumerate(labels):
        t_row = teacher_scores[i]
        s_row = student_scores[i]
        teacher_top1 += int(topk_hit(t_row, species_ids, truth, 1))
        teacher_top3 += int(topk_hit(t_row, species_ids, truth, 3))
        student_top1 += int(topk_hit(s_row, species_ids, truth, 1))
        student_top3 += int(topk_hit(s_row, species_ids, truth, 3))
        order = np.argsort(-np.asarray(s_row, dtype=np.float64))
        ranked = [species_ids[int(j)] for j in order]
        margin = None
        if margins is not None:
            margin = float(margins[i])
        elif len(order) >= 2:
            margin = float(s_row[order[0]] - s_row[order[1]])
        if margin is not None and margin < 0.15:
            picked = oracle_one_question(ranked, truth)
        else:
            picked = ranked[0]
        student_key += int(picked == truth)
    return {
        "n": float(n),
        "teacher_top1": teacher_top1 / n,
        "teacher_top3": teacher_top3 / n,
        "student_top1": student_top1 / n,
        "student_top3": student_top3 / n,
        "student_plus_one_question_top1": student_key / n,
    }


def group_kfold_splits(labels: list[str], n_splits: int = 5) -> list[np.ndarray]:
    """Hold out whole species groups. Returns boolean test masks, one per fold."""
    from sklearn.model_selection import GroupKFold

    groups = np.asarray(labels)
    unique = np.unique(groups)
    splits = min(n_splits, len(unique))
    if splits < 2:
        return [np.ones(len(labels), dtype=bool)]
    x = np.zeros((len(labels), 1))
    masks = []
    folder = GroupKFold(n_splits=splits)
    for _, test in folder.split(x, groups, groups):
        mask = np.zeros(len(labels), dtype=bool)
        mask[test] = True
        masks.append(mask)
    return masks
