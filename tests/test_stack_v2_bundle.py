"""Bundle schema, probe fit, and the exporter's two source modes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.database import SpeciesDB
from core.stack_v2.bundle import (
    BIOCAP_DIM,
    BIOCLIP_DIM,
    BUNDLE_VERSION,
    DINO_DIM,
    FUSED_DIM,
    REQUIRED_KEYS,
    BundleError,
    assemble_bundle,
    bundle_meta,
    fit_probe,
    load_bundle,
    save_bundle,
    validate_bundle,
)
from scripts.export_stack_v2_bundle import inventory, iter_turtles
from tests.synthetic_bundle import make_bundle_arrays, one_hot


@pytest.fixture(scope="module")
def db():
    return SpeciesDB.load()


def test_synthetic_bundle_has_the_query_matrices(db, tmp_path):
    arrays = make_bundle_arrays(db)
    path = save_bundle(tmp_path / "stack_v2_bundle.npz", arrays)
    loaded = load_bundle(path)

    for key in REQUIRED_KEYS:
        assert key in loaded
    n = len(db)
    assert loaded["bioclip_mean"].shape == (BIOCLIP_DIM,)
    assert loaded["dino_mean"].shape == (DINO_DIM,)
    assert loaded["text_matrix"].shape == (n, BIOCLIP_DIM)
    assert loaded["biocap_text_matrix"].shape == (n, BIOCAP_DIM)
    assert loaded["species_ids"].shape == (n,)
    assert loaded["plate_c"].shape[1] == BIOCLIP_DIM
    assert loaded["local_dino"].shape[1] == DINO_DIM
    assert loaded["local_dino"].shape[0] == loaded["local_labels"].shape[0]
    assert loaded["cache_mat"].shape[1] == BIOCLIP_DIM
    assert loaded["cache_mat"].shape[0] == loaded["cache_labels"].shape[0]
    assert loaded["probe_coef"].shape[1] == FUSED_DIM
    assert loaded["probe_coef"].shape[0] == loaded["probe_intercept"].shape[0]
    assert loaded["probe_coef"].shape[0] == loaded["probe_classes"].shape[0]
    assert loaded["probe_coef"].shape[0] >= 2
    meta = bundle_meta(loaded)
    assert meta["bundle_version"] == BUNDLE_VERSION
    assert meta["n_species"] == n


def test_kmeans_prototypes_join_the_cache(db):
    arrays = make_bundle_arrays(db, copies=1, inat_counts={"lissemys_punctata": 6})
    labels = [str(x) for x in arrays["cache_labels"].tolist()]
    # 6 gallery rows plus up to 5 prototypes, and one field row per species.
    assert labels.count("lissemys_punctata") >= 6 + 2


def test_binary_probe_is_stored_as_two_softmax_rows():
    features = np.vstack([one_hot(0, 8), one_hot(0, 8), one_hot(1, 8), one_hot(1, 8)])
    coef, intercept, classes = fit_probe(features, ["a", "a", "b", "b"])
    assert coef.shape == (2, 8)
    assert intercept.shape == (2,)
    assert list(classes.tolist()) == ["a", "b"]
    logits = features[0] @ coef.T + intercept
    assert logits[0] > logits[1]


def test_missing_key_is_rejected(db):
    arrays = make_bundle_arrays(db, copies=1)
    del arrays["plate_ids"]
    with pytest.raises(BundleError):
        validate_bundle(arrays)


def test_repo_standalone_inventory_has_plates_and_submissions(db):
    info = inventory(db, turtles_dir=None, truth_path=None, skip_inat=True)
    assert info["mode"] == "standalone"
    assert info["plates"] == 28
    assert info["submissions"] == 13
    assert info["turtles"] == 0
    assert info["species"] == len(db)


def test_full_mode_includes_labelled_seizure_photos(db, tmp_path: Path):
    turtles = tmp_path / "Turtles"
    turtles.mkdir()
    (turtles / "kept.jpg").write_bytes(b"jpeg-bytes")
    (turtles / "skip.txt").write_text("not an image")
    truth = tmp_path / "truth.json"
    truth.write_text(json.dumps({
        "labels": {
            "kept.jpg": "lissemys_punctata",
            "gone.jpg": "lissemys_punctata",
            "skip.txt": "lissemys_punctata",
            "mystery.jpg": "unidentified",
        }
    }))
    found = iter_turtles(db, turtles, truth)
    assert [(sid, path.name) for sid, path in found] == [("lissemys_punctata", "kept.jpg")]
    info = inventory(db, turtles_dir=turtles, truth_path=truth, skip_inat=True)
    assert info["mode"] == "full"
    assert info["turtles"] == 1


def test_committed_baseline_bundle_loads():
    """The npz shipped in git is a real export, not a schema stub."""
    from config import BASE_DIR

    path = BASE_DIR / "data" / "stack_v2_bundle.npz"
    loaded = load_bundle(path)
    meta = bundle_meta(loaded)
    assert loaded["plate_c"].shape == (28, 1024)
    assert loaded["text_matrix"].shape == (30, 1024)
    assert loaded["biocap_text_matrix"].shape == (30, 512)
    assert loaded["probe_coef"].shape[1] == 2048
    assert meta["n_species"] == 30
    assert meta["n_plates"] == 28
    assert meta["mode"] == "standalone"
    assert meta["detector"] == "grounding_dino"
    assert int(meta["n_turtles"]) == 0
    assert int(meta["n_probe_classes"]) == 3
    assert int(meta["n_cache"]) == loaded["cache_mat"].shape[0]
    assert np.isfinite(loaded["cache_mat"]).all()
    assert np.isfinite(loaded["text_matrix"]).all()


def test_inventory_cli_exits_without_models():
    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, "-m", "scripts.export_stack_v2_bundle", "--inventory"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["mode"] == "standalone"
    assert payload["plates"] == 28
