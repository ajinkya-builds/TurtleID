"""Deterministic Stack v2 bundles for tests. No model weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.database import SpeciesDB
from core.stack_v2.bundle import (
    BIOCAP_DIM,
    BIOCLIP_DIM,
    DINO_DIM,
    assemble_bundle,
    save_bundle,
)


def one_hot(index: int, dim: int) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    vector[index % dim] = 1.0
    return vector


def make_bundle_arrays(
    db: SpeciesDB | None = None,
    *,
    copies: int = 2,
    inat_counts: dict[str, int] | None = None,
) -> dict[str, np.ndarray]:
    db = db if db is not None else SpeciesDB.load()
    species = list(db)
    ids = [s.id for s in species]
    field = np.vstack([one_hot(i, BIOCLIP_DIM) for i in range(len(ids)) for _ in range(copies)])
    labels = [ids[i] for i in range(len(ids)) for _ in range(copies)]
    dino = np.vstack([one_hot(i, DINO_DIM) for i in range(len(ids)) for _ in range(copies)])

    inat_rows = []
    inat_labels: list[str] = []
    for sid, count in (inat_counts or {}).items():
        index = ids.index(sid)
        for _ in range(count):
            inat_rows.append(one_hot(index, BIOCLIP_DIM))
            inat_labels.append(sid)
    if inat_rows:
        inat = np.vstack(inat_rows)
    else:
        inat = np.empty((0, BIOCLIP_DIM), np.float32)

    families = sorted({s.family for s in species})
    genera = sorted({s.scientific_name.split()[0] for s in species})
    return assemble_bundle(
        species_ids=ids,
        bioclip_field=field,
        field_labels=labels,
        dino_field=dino,
        bioclip_inat=inat,
        inat_labels=inat_labels,
        bioclip_plates=np.vstack([one_hot(i, BIOCLIP_DIM) for i in range(len(ids))]),
        plate_ids=list(ids),
        text_matrix=np.vstack([one_hot(i, BIOCLIP_DIM) for i in range(len(ids))]),
        fam_vecs=np.vstack([one_hot(i, BIOCLIP_DIM) for i in range(len(families))]),
        fam_names=families,
        gen_vecs=np.vstack([one_hot(i, BIOCLIP_DIM) for i in range(len(genera))]),
        gen_names=genera,
        species_families=[s.family for s in species],
        species_genera=[s.scientific_name.split()[0] for s in species],
        biocap_text_matrix=np.vstack([one_hot(i, BIOCAP_DIM) for i in range(len(ids))]),
        meta={"mode": "test", "fitted": False},
    )


def write_bundle(path: Path, db: SpeciesDB | None = None) -> Path:
    return save_bundle(path, make_bundle_arrays(db))
