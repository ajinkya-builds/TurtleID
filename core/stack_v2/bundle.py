"""Stack v2 query bundle: static matrices, no photos and no text towers.

The exporter fits these arrays once. The online runtime only multiplies.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.stack_v2.scoring import cl2n, kmeans_prototypes, l2_normalise

BIOCLIP_DIM = 1024
DINO_DIM = 1024
BIOCAP_DIM = 512
FUSED_DIM = BIOCLIP_DIM + DINO_DIM
BUNDLE_VERSION = 1

REQUIRED_KEYS = (
    "bioclip_mean",
    "dino_mean",
    "cache_mat",
    "cache_labels",
    "local_dino",
    "local_labels",
    "text_matrix",
    "fam_vecs",
    "fam_names",
    "gen_vecs",
    "gen_names",
    "species_families",
    "species_genera",
    "plate_c",
    "plate_ids",
    "biocap_text_matrix",
    "probe_coef",
    "probe_intercept",
    "probe_classes",
    "species_ids",
)


class BundleError(ValueError):
    """The npz is missing a matrix or the shapes disagree."""


def _as_float(matrix: np.ndarray, *, dim: int | None = None) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1) if array.size else np.empty((0, dim or 0), np.float32)
    if dim is not None and array.size and array.shape[1] != dim:
        raise BundleError(f"expected width {dim}, got {array.shape}")
    return array


def _strings(values, *, width: int = 128) -> np.ndarray:
    return np.asarray([str(v) for v in values], dtype=f"U{width}")


def _empty(rows: int, dim: int) -> np.ndarray:
    return np.empty((rows, dim), dtype=np.float32)


def _safe_l2(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise rows. A zero row stays zero (no 1e-12 blow-up)."""
    if matrix.size == 0:
        return np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    out = np.zeros(matrix.shape, dtype=np.float32)
    keep = norms.reshape(-1) > 1e-8
    if np.any(keep):
        out[keep] = (matrix[keep] / norms[keep]).astype(np.float32)
    return out


def fit_probe(features: np.ndarray, labels: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """L2 logistic regression. Coef is always (C, features), including binary.

    sklearn stores a binary problem as one row (the positive class). Query
    time applies a softmax over both rows, so the negative class is stored
    as the negation of that row.
    """
    names = list(labels)
    if features.shape[0] != len(names):
        raise BundleError("probe features and labels differ in length")
    if len(set(names)) < 2 or len(names) < 2:
        return (
            _empty(0, features.shape[1]),
            np.empty((0,), np.float32),
            _strings([]),
        )

    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(
        max_iter=1000,
        C=0.5,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(features, names)
    coef = np.asarray(clf.coef_, dtype=np.float32)
    intercept = np.asarray(clf.intercept_, dtype=np.float32).reshape(-1)
    classes = [str(c) for c in clf.classes_]
    if coef.shape[0] == 1 and len(classes) == 2:
        row = coef.reshape(-1)
        bias = float(intercept.reshape(-1)[0])
        coef = np.vstack([-row, row]).astype(np.float32)
        intercept = np.asarray([-bias, bias], dtype=np.float32)
    if coef.shape[0] != len(classes):
        raise BundleError(
            f"probe coef rows {coef.shape[0]} != classes {len(classes)}"
        )
    return coef, intercept, _strings(classes)


def assemble_bundle(
    *,
    species_ids: list[str],
    bioclip_field: np.ndarray,
    field_labels: list[str],
    dino_field: np.ndarray,
    bioclip_inat: np.ndarray,
    inat_labels: list[str],
    bioclip_plates: np.ndarray,
    plate_ids: list[str],
    text_matrix: np.ndarray,
    fam_vecs: np.ndarray,
    fam_names: list[str],
    gen_vecs: np.ndarray,
    gen_names: list[str],
    species_families: list[str],
    species_genera: list[str],
    biocap_text_matrix: np.ndarray,
    meta: dict | None = None,
    k_max: int = 5,
) -> dict[str, np.ndarray]:
    """Center, cache, and fit the probe. Inputs are L2 image/text embeddings.

    `bioclip_*` and `dino_field` are the pre-centering vectors (the same
    normalisation `BioClipMatcher.embed` writes). Family, genus, species
    text, and BioCAP text stay uncentered, matching the offline stack.
    """
    species_ids = [str(s) for s in species_ids]
    n_species = len(species_ids)
    if n_species < 2:
        raise BundleError("species_ids needs the reference list")

    field = _as_float(bioclip_field, dim=BIOCLIP_DIM)
    inat = _as_float(bioclip_inat, dim=BIOCLIP_DIM)
    plates = _as_float(bioclip_plates, dim=BIOCLIP_DIM)
    dino = _as_float(dino_field, dim=DINO_DIM)
    text = _as_float(text_matrix, dim=BIOCLIP_DIM)
    fam = _as_float(fam_vecs, dim=BIOCLIP_DIM)
    gen = _as_float(gen_vecs, dim=BIOCLIP_DIM)
    biocap = _as_float(biocap_text_matrix, dim=BIOCAP_DIM)

    if field.shape[0] != len(field_labels) or field.shape[0] != dino.shape[0]:
        raise BundleError("field BioCLIP, DINOv2, and labels must align")
    if inat.shape[0] != len(inat_labels):
        raise BundleError("iNat embeddings and labels must align")
    if plates.shape[0] != len(plate_ids):
        raise BundleError("plate embeddings and ids must align")
    if text.shape[0] != n_species or biocap.shape[0] != n_species:
        raise BundleError("text matrices must have one row per species")
    if len(species_families) != n_species or len(species_genera) != n_species:
        raise BundleError("family/genus lists must align with species_ids")
    if fam.shape[0] != len(fam_names) or gen.shape[0] != len(gen_names):
        raise BundleError("rank prompt embeddings and names must align")

    support = [part for part in (field, inat) if part.shape[0]]
    if support:
        bioclip_mean = np.vstack(support).mean(axis=0).astype(np.float32)
    elif plates.shape[0]:
        bioclip_mean = plates.mean(axis=0).astype(np.float32)
    else:
        bioclip_mean = np.zeros(BIOCLIP_DIM, np.float32)

    field_c = cl2n(field, bioclip_mean)[0] if field.shape[0] else _empty(0, BIOCLIP_DIM)
    inat_c = cl2n(inat, bioclip_mean)[0] if inat.shape[0] else _empty(0, BIOCLIP_DIM)
    plate_c = cl2n(plates, bioclip_mean)[0] if plates.shape[0] else _empty(0, BIOCLIP_DIM)

    if dino.shape[0]:
        dino_mean = dino.mean(axis=0).astype(np.float32)
        local_dino = cl2n(dino, dino_mean)[0]
    else:
        dino_mean = np.zeros(DINO_DIM, np.float32)
        local_dino = _empty(0, DINO_DIM)

    cache_parts: list[np.ndarray] = []
    cache_labels: list[str] = []
    if inat_c.shape[0]:
        cache_parts.append(inat_c)
        cache_labels.extend(str(s) for s in inat_labels)
        proto_v, proto_l = kmeans_prototypes(inat_c, [str(s) for s in inat_labels], k_max=k_max)
        if proto_v.shape[0]:
            cache_parts.append(l2_normalise(proto_v).astype(np.float32))
            cache_labels.extend(proto_l)
    if field_c.shape[0]:
        cache_parts.append(field_c)
        cache_labels.extend(str(s) for s in field_labels)

    if cache_parts:
        cache_mat = np.vstack(cache_parts).astype(np.float32)
    else:
        cache_mat = _empty(0, BIOCLIP_DIM)

    if field_c.shape[0]:
        fused = np.concatenate([field_c, local_dino], axis=1).astype(np.float32)
        probe_coef, probe_intercept, probe_classes = fit_probe(
            fused, [str(s) for s in field_labels]
        )
    else:
        probe_coef = _empty(0, FUSED_DIM)
        probe_intercept = np.empty((0,), np.float32)
        probe_classes = _strings([])

    payload = dict(meta or {})
    payload.setdefault("bundle_version", BUNDLE_VERSION)
    payload.setdefault("bioclip_dim", BIOCLIP_DIM)
    payload.setdefault("dino_dim", DINO_DIM)
    payload.setdefault("biocap_dim", BIOCAP_DIM)
    payload["n_species"] = n_species
    payload["n_cache"] = int(cache_mat.shape[0])
    payload["n_local"] = int(local_dino.shape[0])
    payload["n_plates"] = int(plate_c.shape[0])
    payload["n_probe_classes"] = int(probe_coef.shape[0])

    arrays = {
        "bioclip_mean": np.asarray(bioclip_mean, dtype=np.float32).reshape(BIOCLIP_DIM),
        "dino_mean": np.asarray(dino_mean, dtype=np.float32).reshape(DINO_DIM),
        "cache_mat": cache_mat,
        "cache_labels": _strings(cache_labels),
        "local_dino": local_dino.astype(np.float32),
        "local_labels": _strings(field_labels),
        "text_matrix": _safe_l2(text),
        "fam_vecs": _safe_l2(fam) if fam.shape[0] else fam.astype(np.float32),
        "fam_names": _strings(fam_names),
        "gen_vecs": _safe_l2(gen) if gen.shape[0] else gen.astype(np.float32),
        "gen_names": _strings(gen_names),
        "species_families": _strings(species_families),
        "species_genera": _strings(species_genera),
        "plate_c": plate_c.astype(np.float32),
        "plate_ids": _strings(plate_ids),
        "biocap_text_matrix": _safe_l2(biocap),
        "probe_coef": probe_coef.astype(np.float32),
        "probe_intercept": probe_intercept.astype(np.float32),
        "probe_classes": probe_classes,
        "species_ids": _strings(species_ids),
        "meta": np.asarray(json.dumps(payload, sort_keys=True)),
    }
    validate_bundle(arrays)
    return arrays


def validate_bundle(arrays: dict) -> None:
    missing = [key for key in REQUIRED_KEYS if key not in arrays]
    if missing:
        raise BundleError(f"bundle missing keys: {', '.join(missing)}")

    species = np.asarray(arrays["species_ids"]).reshape(-1)
    n_species = int(species.shape[0])
    bioclip_mean = np.asarray(arrays["bioclip_mean"], dtype=np.float32).reshape(-1)
    dino_mean = np.asarray(arrays["dino_mean"], dtype=np.float32).reshape(-1)
    if bioclip_mean.shape != (BIOCLIP_DIM,):
        raise BundleError(f"bioclip_mean shape {bioclip_mean.shape}")
    if dino_mean.shape != (DINO_DIM,):
        raise BundleError(f"dino_mean shape {dino_mean.shape}")

    def width(name: str, dim: int) -> np.ndarray:
        matrix = np.asarray(arrays[name], dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != dim:
            raise BundleError(f"{name} shape {matrix.shape}, expected (*, {dim})")
        return matrix

    cache = width("cache_mat", BIOCLIP_DIM)
    local = width("local_dino", DINO_DIM)
    text = width("text_matrix", BIOCLIP_DIM)
    fam = width("fam_vecs", BIOCLIP_DIM)
    gen = width("gen_vecs", BIOCLIP_DIM)
    plates = width("plate_c", BIOCLIP_DIM)
    biocap = width("biocap_text_matrix", BIOCAP_DIM)
    coef = width("probe_coef", FUSED_DIM) if np.asarray(arrays["probe_coef"]).size else np.zeros((0, FUSED_DIM), np.float32)
    if np.asarray(arrays["probe_coef"]).ndim != 2 or (
        np.asarray(arrays["probe_coef"]).size and np.asarray(arrays["probe_coef"]).shape[1] != FUSED_DIM
    ):
        raise BundleError(f"probe_coef shape {np.asarray(arrays['probe_coef']).shape}")

    if text.shape[0] != n_species or biocap.shape[0] != n_species:
        raise BundleError("text rows must match species_ids")
    if np.asarray(arrays["species_families"]).reshape(-1).shape[0] != n_species:
        raise BundleError("species_families length")
    if np.asarray(arrays["species_genera"]).reshape(-1).shape[0] != n_species:
        raise BundleError("species_genera length")
    if cache.shape[0] != np.asarray(arrays["cache_labels"]).reshape(-1).shape[0]:
        raise BundleError("cache_labels length")
    if local.shape[0] != np.asarray(arrays["local_labels"]).reshape(-1).shape[0]:
        raise BundleError("local_labels length")
    if plates.shape[0] != np.asarray(arrays["plate_ids"]).reshape(-1).shape[0]:
        raise BundleError("plate_ids length")
    if fam.shape[0] != np.asarray(arrays["fam_names"]).reshape(-1).shape[0]:
        raise BundleError("fam_names length")
    if gen.shape[0] != np.asarray(arrays["gen_names"]).reshape(-1).shape[0]:
        raise BundleError("gen_names length")

    intercept = np.asarray(arrays["probe_intercept"], dtype=np.float32).reshape(-1)
    classes = np.asarray(arrays["probe_classes"]).reshape(-1)
    if coef.shape[0] != intercept.shape[0] or coef.shape[0] != classes.shape[0]:
        raise BundleError("probe coef, intercept, and classes must align")


def save_bundle(path: Path | str, arrays: dict[str, np.ndarray]) -> Path:
    path = Path(path)
    validate_bundle(arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def load_bundle(path: Path | str) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise BundleError(f"bundle not found: {path}")
    with np.load(path, allow_pickle=False) as packed:
        arrays = {key: np.array(packed[key]) for key in packed.files}
    validate_bundle(arrays)
    return arrays


def bundle_meta(arrays: dict) -> dict:
    raw = arrays.get("meta")
    if raw is None:
        return {}
    text = str(np.asarray(raw).reshape(-1)[0]) if np.asarray(raw).shape != () else str(np.asarray(raw))
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}
