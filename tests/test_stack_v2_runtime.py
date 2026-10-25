"""StackV2Predictor scoring with a fake vision tower (no weight download)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from core.database import SpeciesDB
from core.morphkey import verification_questions
from core.stack_v2.runtime import (
    MARGIN_SPECIALIST,
    StackV2Predictor,
    apply_geo_prior,
    apply_head_specialist,
)
from core.stack_v2.scoring import is_confusion_pair
from tests.synthetic_bundle import make_bundle_arrays, one_hot, write_bundle


class FakeVision:
    def __init__(self, index: int):
        self.index = index

    def embed_bioclip(self, images):
        return np.vstack([one_hot(self.index, 1024) for _ in images])

    def embed_dino(self, images):
        return np.vstack([one_hot(self.index, 1024) for _ in images])

    def embed_biocap(self, images):
        return np.vstack([one_hot(self.index, 512) for _ in images])


@pytest.fixture(scope="module")
def db():
    return SpeciesDB.load()


@pytest.fixture()
def predictor(db, tmp_path):
    path = write_bundle(tmp_path / "stack_v2_bundle.npz", db)
    return StackV2Predictor(
        path,
        db=db,
        vision=FakeVision(0),
        detector="border",
        use_biocap_channel=True,
    )


def test_predict_ranks_the_matching_species(predictor, db):
    image = Image.new("RGB", (80, 96), (40, 90, 30))
    prediction = predictor.predict(image, state=None)
    top = prediction.top_matches[0]
    assert top["species_id"] == db.ids[0]
    assert top["scientific_name"]
    assert top["common_name"]
    assert top["family"]
    assert top["iucn_status"]
    assert top["wpa_schedule"]
    assert top["cites_appendix"]
    assert top["key_character"]
    assert top["diagnostics"]
    assert top["reference_image_url"].startswith("/reference_images/")
    assert top["confidence"] == top["score"]
    assert top["confidence"] > 0.5
    assert prediction.engine == "stack_v2"
    assert prediction.tier in {"CONFIRMED", "PROBABLE"}
    assert prediction.abstain is False
    assert prediction.crop_method == "border_crop"
    assert prediction.max_similarity is not None and prediction.max_similarity > 0.5
    assert any("UNCALIBRATED" in warning for warning in prediction.warnings)
    assert 0 <= len(prediction.verification_questions) <= 3
    body = prediction.as_response()
    assert body["status"] == "success"
    assert body["top_match"]["species_id"] == top["species_id"]
    assert body["matches"][0]["species_id"] == top["species_id"]
    assert body["engine"] == "stack_v2"


def test_low_similarity_is_rejected(predictor):
    prediction = predictor._package(
        np.zeros(len(predictor.species_ids)),
        state=None,
        specialist=False,
        crop_method="border_crop",
        max_similarity=0.01,
        engine="stack_v2",
    )
    assert prediction.tier == "REJECTED"
    assert prediction.abstain is True


def test_state_prior_prefers_a_resident(db):
    ids = db.ids
    probs = np.full(len(ids), 1.0 / len(ids))
    posterior, multipliers = apply_geo_prior(probs, ids, db, "Kerala")
    assert posterior.shape == probs.shape
    assert not np.isclose(multipliers, 1.0).all()
    top = ids[int(np.argmax(posterior))]
    assert db.get(top).occurs_in("Kerala") != "absent"


def test_schedule_divergence_and_confusion_warnings(predictor, db):
    scores = np.full(len(predictor.species_ids), -4.0)
    tecta = predictor.species_ids.index("pangshura_tecta")
    smithii = predictor.species_ids.index("pangshura_smithii")
    scores[tecta] = 1.0
    scores[smithii] = 0.92
    assert db.get("pangshura_tecta").wpa != db.get("pangshura_smithii").wpa
    prediction = predictor._package(
        scores,
        state="Kerala",
        specialist=False,
        crop_method="border_crop",
        max_similarity=0.9,
        engine="stack_v2",
    )
    text = " ".join(prediction.warnings)
    assert "LEGAL DIVERGENCE" in text
    assert prediction.pairwise_discriminator
    assert prediction.top_matches[0]["wpa_schedule"]
    assert 1 <= len(prediction.verification_questions) <= 3
    assert prediction.verification_questions[0]["prompt"]
    assert prediction.verification_questions[0]["options"]


def test_head_specialist_flips_a_confusion_pair(db):
    ids = db.ids
    text = np.vstack([one_hot(i, 1024) for i in range(len(ids))])
    tecta = ids.index("pangshura_tecta")
    tentoria = ids.index("pangshura_tentoria")
    assert is_confusion_pair(ids[tecta], ids[tentoria])
    scores = np.full(len(ids), -1.0)
    scores[tecta] = 0.20
    scores[tentoria] = 0.20 - (MARGIN_SPECIALIST / 2)
    boosted, used = apply_head_specialist(
        scores,
        one_hot(tentoria, 1024),
        text,
        ids,
        text,
        ids,
    )
    assert used is True
    assert ids[int(np.argmax(boosted))] == "pangshura_tentoria"


def test_checklist_splits_tecta_and_tentoria():
    questions = verification_questions(
        ["pangshura_tecta", "pangshura_tentoria", "pangshura_smithii"],
        limit=3,
    )
    assert 2 <= len(questions) <= 3
    characters = {item["character"] for item in questions}
    assert "third_vertebral" in characters or "plastron_colour" in characters
    for question in questions:
        assert question["label"]
        assert question["prompt"]
        assert len(question["options"]) >= 2
        covered = set()
        for option in question["options"]:
            covered.update(option["species_ids"])
        assert "pangshura_tecta" in covered or "pangshura_tentoria" in covered
