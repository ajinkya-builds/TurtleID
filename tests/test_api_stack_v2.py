"""HTTP shape for Stack v2, the BioCLIP fallback, and EXIF stripping."""

from __future__ import annotations

import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import api
from core.database import SpeciesDB
from core.photo_match import RankedMatch
from core.stack_v2.runtime import StackV2Predictor
from tests.synthetic_bundle import write_bundle
from tests.test_stack_v2_runtime import FakeVision


@pytest.fixture
def database():
    return SpeciesDB.load()


@pytest.fixture(autouse=True)
def reset_api(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "model_ready", False)
    monkeypatch.setattr(api, "model_error", None)
    monkeypatch.setattr(api, "predictor", None)
    monkeypatch.setattr(api, "matcher", None)
    monkeypatch.setattr(api, "db", None)
    monkeypatch.setenv("TURTLEID_BUNDLE", str(tmp_path / "missing.npz"))
    monkeypatch.setenv("TURTLEID_DETECTOR", "border")
    yield


def _jpeg(note: str | None = None) -> bytes:
    image = Image.new("RGB", (64, 72), (25, 80, 40))
    buffer = io.BytesIO()
    if note:
        exif = Image.Exif()
        exif[270] = note
        image.save(buffer, format="JPEG", quality=90, exif=exif.tobytes())
    else:
        image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def _stack_predictor(database, path):
    return StackV2Predictor(
        path,
        db=database,
        vision=FakeVision(0),
        detector="border",
        use_biocap_channel=True,
    )


def test_health_names_the_engine_before_weights_load(monkeypatch, tmp_path):
    missing = tmp_path / "nope.npz"
    monkeypatch.setenv("TURTLEID_BUNDLE", str(missing))
    client = TestClient(api.app)
    cold = client.get("/health").json()
    assert cold["ready"] is False
    assert cold["engine"] == "bioclip_v1"

    present = tmp_path / "stack_v2_bundle.npz"
    present.write_bytes(b"x")
    monkeypatch.setenv("TURTLEID_BUNDLE", str(present))
    warm = client.get("/health").json()
    assert warm["engine"] == "stack_v2"
    assert warm["ready"] is False


def test_identify_stack_v2_json_and_strips_exif(database, monkeypatch, tmp_path, caplog):
    path = write_bundle(tmp_path / "stack_v2_bundle.npz", database)
    inner = _stack_predictor(database, path)
    seen = {}

    class Recording:
        def load(self):
            inner.load()

        def predict(self, image, state=None):
            seen["state"] = state
            seen["exif"] = dict(image.getexif())
            return inner.predict(image, state=state)

    monkeypatch.setenv("TURTLEID_BUNDLE", str(path))
    monkeypatch.setattr(api, "build_predictor", lambda db: Recording())
    note = "NEST-BANK-SECRET"
    payload = {
        "image_base64": base64.b64encode(_jpeg(note)).decode("ascii"),
        "state": "Madhya Pradesh",
    }
    with caplog.at_level("INFO"):
        response = TestClient(api.app).post("/identify", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["engine"] == "stack_v2"
    assert body["status"] == "success"
    assert body["tier"]
    assert "abstain" in body
    assert "warnings" in body
    assert "pairwise_discriminator" in body or body["pairwise_discriminator"] is None
    assert "verification_questions" in body
    assert body["top_match"]["species_id"] == database.ids[0]
    assert body["top_matches"][0]["score"] == body["top_match"]["confidence"]
    assert seen["state"] == "Madhya Pradesh"
    assert seen["exif"] == {}
    assert note not in caplog.text
    assert "exif" not in caplog.text.lower()


def test_identify_form_and_query_state(database, monkeypatch, tmp_path):
    path = write_bundle(tmp_path / "stack_v2_bundle.npz", database)
    inner = _stack_predictor(database, path)
    seen = {}

    class Recording:
        def load(self):
            inner.load()

        def predict(self, image, state=None):
            seen["state"] = state
            return inner.predict(image, state=state)

    monkeypatch.setenv("TURTLEID_BUNDLE", str(path))
    monkeypatch.setattr(api, "build_predictor", lambda db: Recording())
    client = TestClient(api.app)
    response = client.post(
        "/identify?state=Kerala",
        files={"image": ("turtle.jpg", _jpeg(), "image/jpeg")},
    )
    assert response.status_code == 200, response.text
    assert seen["state"] == "Kerala"
    assert response.json()["engine"] == "stack_v2"


def test_unknown_state_is_rejected(database, monkeypatch, tmp_path):
    path = write_bundle(tmp_path / "stack_v2_bundle.npz", database)
    monkeypatch.setenv("TURTLEID_BUNDLE", str(path))
    monkeypatch.setattr(api, "build_predictor", lambda db: _stack_predictor(db, path))
    response = TestClient(api.app).post(
        "/identify",
        json={"image_base64": base64.b64encode(_jpeg()).decode("ascii"), "state": "Atlantis"},
    )
    assert response.status_code == 400


def test_neutral_state_applies_no_prior(database, monkeypatch, tmp_path):
    path = write_bundle(tmp_path / "stack_v2_bundle.npz", database)
    inner = _stack_predictor(database, path)
    seen = {}

    class Recording:
        def load(self):
            return None

        def predict(self, image, state=None):
            seen["state"] = state
            return inner.predict(image, state=state)

    monkeypatch.setenv("TURTLEID_BUNDLE", str(path))
    monkeypatch.setattr(api, "build_predictor", lambda db: Recording())
    response = TestClient(api.app).post(
        "/identify",
        json={
            "image_base64": base64.b64encode(_jpeg()).decode("ascii"),
            "state": "All India / Unknown (Trade / Seizure)",
        },
    )
    assert response.status_code == 200
    assert seen["state"] is None


def test_bioclip_fallback_when_bundle_is_missing(database, monkeypatch):
    species = next(iter(database))

    class FakeMatcher:
        def rank(self, image, crop=False):
            return [RankedMatch(
                species.id,
                species.scientific_name,
                species.common_en,
                0.42,
                0.2,
                0.42,
            )]

    monkeypatch.setattr(api, "bundle_available", lambda: False)
    monkeypatch.setattr(api, "build_matcher", lambda db: FakeMatcher())
    response = TestClient(api.app).post(
        "/identify",
        json={"image_base64": base64.b64encode(_jpeg()).decode("ascii")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["engine"] == "bioclip_v1"
    assert body["top_match"]["species_id"] == species.id
    assert body["matches"][0]["scientific_name"] == species.scientific_name
    assert "tier" in body
    assert "verification_questions" in body
