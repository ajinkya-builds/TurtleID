from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from config import SPECIES_DB_PATH
from core.database import SpeciesDB
from core.stack_v2.bundle import save_bundle
from core.stack_v2.distill import (
    BROWSER_BUNDLE_VERSION,
    build_browser_bundle,
    classification_cross_entropy,
    cosine_alignment_loss,
    distillation_loss,
    group_kfold_splits,
    kl_divergence,
    oracle_one_question,
    pack_f16,
    ranking_report,
    relational_similarity_loss,
    unpack_f16,
)
from core.stack_v2.runtime import fuse_channels
from core.stack_v2.runtime import center_query

ROOT = Path(__file__).resolve().parent.parent


def _unit(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    arr = rng.normal(size=shape).astype(np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def _synthetic_arrays(seed: int = 7, d_b: int = 16, d_d: int = 16) -> dict:
    rng = np.random.default_rng(seed)
    db = SpeciesDB.load()
    species_ids = [sp.id for sp in db]
    n_sp = len(species_ids)
    fam_names = sorted({sp.family for sp in db})
    gen_names = sorted({sp.scientific_name.split()[0] for sp in db})
    return {
        "bioclip_mean": np.zeros(d_b, dtype=np.float32),
        "dino_mean": np.zeros(d_d, dtype=np.float32),
        "cache_mat": _unit(rng, (n_sp, d_b)),
        "cache_labels": list(species_ids),
        "local_dino": _unit(rng, (n_sp, d_d)),
        "local_labels": list(species_ids),
        "species_ids": species_ids,
        "text_matrix": _unit(rng, (n_sp, d_b)),
        "fam_names": fam_names,
        "fam_vecs": _unit(rng, (len(fam_names), d_b)),
        "gen_names": gen_names,
        "gen_vecs": _unit(rng, (len(gen_names), d_b)),
        "species_families": [sp.family for sp in db],
        "species_genera": [sp.scientific_name.split()[0] for sp in db],
        "plate_c": _unit(rng, (n_sp, d_b)),
        "plate_ids": list(species_ids),
        "biocap_text_matrix": _unit(rng, (n_sp, d_b)),
        "probe_coef": rng.normal(scale=0.2, size=(n_sp, d_b + d_d)).astype(np.float32),
        "probe_intercept": np.zeros(n_sp, dtype=np.float32),
        "probe_classes": list(species_ids),
    }


def test_distillation_losses_vanish_when_student_matches_teacher():
    rng = np.random.default_rng(11)
    b = _unit(rng, (8, 32))
    d = _unit(rng, (8, 32))
    logits = rng.normal(size=(8, 10)).astype(np.float32)
    targets = np.arange(8) % 10

    assert cosine_alignment_loss(b, b) == pytest.approx(0.0, abs=1e-6)
    assert relational_similarity_loss(b, b) == pytest.approx(0.0, abs=1e-6)
    assert kl_divergence(logits, logits) == pytest.approx(0.0, abs=1e-6)
    ce = classification_cross_entropy(logits, targets)
    assert ce > 0.0

    parts = distillation_loss(b, b, d, d, logits, logits, targets)
    assert parts["cosine"] == pytest.approx(0.0, abs=1e-6)
    assert parts["rkd"] == pytest.approx(0.0, abs=1e-6)
    assert parts["kl"] == pytest.approx(0.0, abs=1e-6)
    assert parts["total"] == pytest.approx(0.5 * ce, rel=1e-5)


def test_pack_unpack_f16_roundtrip():
    rng = np.random.default_rng(19)
    arr = rng.normal(size=(6, 12)).astype(np.float32)
    packed = pack_f16(arr)
    assert packed["dtype"] == "float16"
    assert packed["shape"] == [6, 12]
    restored = unpack_f16(packed)
    assert restored.shape == (6, 12)
    assert np.allclose(arr, restored, atol=2e-3)


def test_build_browser_bundle_schema(tmp_path):
    arrays = _synthetic_arrays()
    species_payload = json.loads(Path(SPECIES_DB_PATH).read_text(encoding="utf-8"))
    bundle = build_browser_bundle(arrays, species_payload, manifest={"version": 2})
    assert bundle["bundle_version"] == BROWSER_BUNDLE_VERSION
    assert len(bundle["species_ids"]) == 30
    assert len(bundle["species"]) == 30
    assert "characters" in bundle["morphkey"]
    assert "matrix" in bundle["morphkey"]
    out = tmp_path / "browser_bundle.json"
    out.write_text(json.dumps(bundle), encoding="utf-8")
    assert out.stat().st_size < 500_000


def test_ranking_report_and_oracle_question():
    species_ids = ["lissemys_punctata", "nilssonia_gangetica", "chitra_indica"]
    teacher_scores = np.array([[2.0, 1.0, 0.0], [0.2, 1.8, 0.1]])
    student_scores = np.array([[1.00, 1.05, 0.0], [0.1, 1.9, 0.0]])
    labels = ["lissemys_punctata", "nilssonia_gangetica"]
    rep = ranking_report(teacher_scores, student_scores, labels, species_ids)
    assert rep["teacher_top1"] == 1.0
    assert rep["student_top1"] == 0.5
    assert rep["student_top3"] == 1.0
    assert rep["student_plus_one_question_top1"] == 1.0
    assert oracle_one_question(["nilssonia_gangetica", "lissemys_punctata"], "lissemys_punctata") == "lissemys_punctata"


def test_group_kfold_splits_non_overlapping():
    groups = ["sp_a", "sp_a", "sp_b", "sp_b", "sp_c", "sp_c", "sp_d", "sp_d", "sp_e", "sp_e"]
    masks = group_kfold_splits(groups, n_splits=5)
    assert len(masks) == 5
    covered = np.zeros(len(groups), dtype=int)
    for m in masks:
        covered += m.astype(int)
    assert np.all(covered == 1)


def test_js_client_matches_python_scoring(tmp_path):
    arrays = _synthetic_arrays(seed=42, d_b=1024, d_d=1024)
    species_payload = json.loads(Path(SPECIES_DB_PATH).read_text(encoding="utf-8"))
    browser_bundle = build_browser_bundle(arrays, species_payload, manifest={"version": 2})
    bundle_json = tmp_path / "browser_bundle.json"
    bundle_json.write_text(json.dumps(browser_bundle), encoding="utf-8")

    rng = np.random.default_rng(99)
    raw_b = _unit(rng, (1024,))
    raw_d = _unit(rng, (1024,))

    b_q = center_query(raw_b, arrays["bioclip_mean"])
    d_q = center_query(raw_d, arrays["dino_mean"])
    py_scores = fuse_channels(
        bioclip_q=b_q,
        dino_q=d_q,
        biocap_q=None,
        cache_mat=arrays["cache_mat"],
        cache_labels=list(arrays["cache_labels"]),
        local_dino=arrays["local_dino"],
        local_labels=list(arrays["local_labels"]),
        text_matrix=arrays["text_matrix"],
        fam_vecs=arrays["fam_vecs"],
        fam_names=list(arrays["fam_names"]),
        gen_vecs=arrays["gen_vecs"],
        gen_names=list(arrays["gen_names"]),
        species_families=list(arrays["species_families"]),
        species_genera=list(arrays["species_genera"]),
        plate_c=arrays["plate_c"],
        plate_ids=list(arrays["plate_ids"]),
        biocap_text=arrays["biocap_text_matrix"],
        probe_coef=arrays["probe_coef"],
        probe_intercept=arrays["probe_intercept"],
        probe_classes=list(arrays["probe_classes"]),
        species_ids=list(arrays["species_ids"]),
        use_biocap_channel=False,
    )
    py_order = [arrays["species_ids"][int(i)] for i in np.argsort(-py_scores)[:5]]

    js_code = f"""
    const fs = require('fs');
    const client = require({json.dumps(str(ROOT / 'docs/stack_v2_client.js'))});
    const bundle = JSON.parse(fs.readFileSync({json.dumps(str(bundle_json))}, 'utf8'));
    const b = new Float32Array({json.dumps(raw_b.tolist())});
    const d = new Float32Array({json.dumps(raw_d.tolist())});
    const unpacked = client.unpackBundle(bundle);
    const bQ = client.centerQuery(b, unpacked.matrices.bioclip_mean);
    const dQ = client.centerQuery(d, unpacked.matrices.dino_mean);
    const scores = Array.from(client.fuseChannels(unpacked, bQ, dQ));
    const res = client.scoreEmbeddings(unpacked, b, d, {{ state: 'Madhya Pradesh', enableSimGate: false }});
    console.log(JSON.stringify({{
        scores,
        top3: res.top_matches.map(m => m.species_id),
        questions: res.verification_questions.length
    }}));
    """
    proc = subprocess.run(["node", "-e", js_code], check=True, capture_output=True, text=True)
    js_out = json.loads(proc.stdout.strip())
    js_scores = np.asarray(js_out["scores"], dtype=np.float64)
    js_order = [arrays["species_ids"][int(i)] for i in np.argsort(-js_scores)[:5]]

    assert np.allclose(py_scores, js_scores, atol=2e-2)
    assert py_order[:3] == js_order[:3]
    assert len(js_out["top3"]) == 3
    assert js_out["questions"] >= 1
