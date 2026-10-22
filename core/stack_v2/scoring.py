"""Tip-Adapter, CL2N, soft hierarchy, and LOO logistic fusion."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def cl2n(matrix: np.ndarray, mean: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """SimpleShot centering + L2 normalisation. Returns (normalised, mean used)."""
    if mean is None:
        mean = matrix.mean(axis=0)
    centered = matrix - mean
    return l2_normalise(centered.astype(np.float64)).astype(np.float32), mean.astype(np.float32)


def tip_adapter_scores(
    query: np.ndarray,
    cache_vectors: np.ndarray,
    cache_labels: list[str],
    class_names: list[str],
    *,
    alpha: float = 1.0,
    beta: float = 5.0,
) -> np.ndarray:
    """Training-free Tip-Adapter visual cache logits, one column per class_names."""
    out = np.zeros(len(class_names), dtype=np.float64)
    if len(cache_vectors) == 0:
        return out
    sims = cache_vectors @ query
    affinity = np.exp(-beta * (1.0 - sims))
    index = {name: i for i, name in enumerate(class_names)}
    for weight, label in zip(affinity, cache_labels):
        pos = index.get(label)
        if pos is not None:
            out[pos] += float(weight)
    return alpha * out


def soft_hierarchy_bonus(
    query: np.ndarray,
    family_vectors: np.ndarray,
    genus_vectors: np.ndarray,
    family_of: list[str],
    genus_of: list[str],
    family_names: list[str],
    genus_names: list[str],
    *,
    w_family: float = 0.15,
    w_genus: float = 0.20,
) -> np.ndarray:
    """Per-species bonus from family/genus text similarity (soft prior, not a mask)."""
    fam_score = {
        name: float(query @ family_vectors[i])
        for i, name in enumerate(family_names)
    }
    gen_score = {
        name: float(query @ genus_vectors[i])
        for i, name in enumerate(genus_names)
    }
    n = len(family_of)
    bonus = np.zeros(n, dtype=np.float64)
    for i in range(n):
        bonus[i] = (
            w_family * fam_score.get(family_of[i], 0.0)
            + w_genus * gen_score.get(genus_of[i], 0.0)
        )
    return bonus


def zscore_row(row: np.ndarray) -> np.ndarray:
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


def combine_channels(
    channels: list[tuple[np.ndarray, float]],
) -> np.ndarray:
    """Weighted sum of z-scored score channels."""
    total = None
    for row, weight in channels:
        part = weight * zscore_row(np.asarray(row, dtype=np.float64))
        total = part if total is None else total + part
    assert total is not None
    return total


def loo_logistic_probs(
    features: np.ndarray,
    labels: list[str],
    class_names: list[str],
    exclude_index: int,
) -> np.ndarray:
    """LOO logistic regression probabilities over class_names for one query index."""
    probs = np.zeros(len(class_names), dtype=np.float64)
    mask = np.ones(len(labels), dtype=bool)
    mask[exclude_index] = False
    train_y = [labels[i] for i in range(len(labels)) if mask[i]]
    if len(set(train_y)) < 2:
        return probs
    encoder = LabelEncoder()
    y = encoder.fit_transform(train_y)
    x = features[mask]
    query = features[exclude_index : exclude_index + 1]
    clf = LogisticRegression(
        max_iter=1000,
        C=0.5,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(x, y)
    pred = clf.predict_proba(query)[0]
    index = {name: i for i, name in enumerate(class_names)}
    for cls_i, name in enumerate(encoder.classes_):
        pos = index.get(name)
        if pos is not None:
            probs[pos] = float(pred[cls_i])
    return probs


def kmeans_prototypes(
    vectors: np.ndarray,
    labels: list[str],
    *,
    k_max: int = 5,
) -> tuple[np.ndarray, list[str]]:
    """Per-class k-means sub-prototypes (view/age clusters)."""
    from sklearn.cluster import KMeans

    proto_vecs: list[np.ndarray] = []
    proto_labels: list[str] = []
    by: dict[str, list[int]] = {}
    for i, label in enumerate(labels):
        by.setdefault(label, []).append(i)
    for label, indices in by.items():
        block = vectors[indices]
        k = min(k_max, len(indices))
        if k <= 1:
            proto_vecs.append(block.mean(axis=0))
            proto_labels.append(label)
            continue
        model = KMeans(n_clusters=k, n_init=5, random_state=0)
        model.fit(block)
        for center in model.cluster_centers_:
            proto_vecs.append(center.astype(np.float32))
            proto_labels.append(label)
    if not proto_vecs:
        return np.empty((0, vectors.shape[1]), dtype=np.float32), []
    return l2_normalise(np.vstack(proto_vecs)), proto_labels


# Known sister / lookalike pairs for low-margin specialist boost.
CONFUSION_PAIRS = {
    frozenset({"lissemys_punctata", "nilssonia_gangetica"}),
    frozenset({"lissemys_punctata", "nilssonia_hurum"}),
    frozenset({"lissemys_punctata", "nilssonia_leithii"}),
    frozenset({"pangshura_tecta", "pangshura_tentoria"}),
    frozenset({"pangshura_tentoria", "melanochelys_trijuga"}),
    frozenset({"hardella_thurjii", "pangshura_tentoria"}),
    frozenset({"geoclemys_hamiltonii", "geochelone_elegans"}),
    frozenset({"trachemys_scripta_elegans", "hardella_thurjii"}),
}


def is_confusion_pair(a: str, b: str) -> bool:
    return frozenset({a, b}) in CONFUSION_PAIRS
