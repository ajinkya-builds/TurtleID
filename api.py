"""Chelonid-ID mobile API.

``/identify`` serves Stack v2 when ``data/stack_v2_bundle.npz`` is present and
falls back to the BioCLIP 2 plate matcher only when that file is missing.
EXIF, including GPS, is stripped in memory and never logged.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

import config
from config import INDIAN_STATES, MAX_UPLOAD_BYTES
from core.contributions import ContributionError, strip_exif
from core.database import SpeciesDB
from core.stack_v2.runtime import BUNDLE_PATH, StackV2Predictor, package_ranked_matches

logger = logging.getLogger("turtleid.api")

app = FastAPI(title="Chelonid-ID Mobile API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ajinkya-builds.github.io",
        "https://ajinkya-patil60.github.io",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:7860",
        "http://127.0.0.1:7860",
    ],
    allow_origin_regex=r"https://([a-z0-9-]+\.)*(hf\.space|github\.io)",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_NEUTRAL_STATES = {
    "",
    "all",
    "all india",
    "unknown",
    "all india / unknown (trade / seizure)",
    "trade",
    "seizure",
}

db = None
predictor: StackV2Predictor | None = None
matcher = None
model_ready = False
model_error: str | None = None
engine_name = "stack_v2"
_load_lock = threading.Lock()


def bundle_available() -> bool:
    path = bundle_file()
    return path.is_file() and path.stat().st_size > 0


def bundle_file() -> Path:
    override = os.environ.get("TURTLEID_BUNDLE", "").strip()
    if not override:
        return BUNDLE_PATH
    path = Path(override)
    if not path.is_absolute():
        path = config.BASE_DIR / path
    return path


def active_engine() -> str:
    return "stack_v2" if bundle_available() else "bioclip_v1"


def parse_state(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.casefold() in _NEUTRAL_STATES:
        return None
    for name in INDIAN_STATES:
        if name.casefold() == text.casefold():
            return name
    raise HTTPException(status_code=400, detail=f"Unrecognised state: {text}")


def build_predictor(database: SpeciesDB) -> StackV2Predictor:
    return StackV2Predictor(bundle_path=bundle_file(), db=database)


def build_matcher(database: SpeciesDB):
    from core.photo_match import BioClipMatcher

    return BioClipMatcher(db=database)


def _load_models() -> None:
    global db, predictor, matcher, model_ready, model_error, engine_name
    if not os.path.exists("data/species_db.json"):
        raise RuntimeError("Missing data/species_db.json file.")

    database = SpeciesDB.load()
    engine_name = active_engine()
    if engine_name == "stack_v2":
        logger.info("Loading Stack v2 bundle %s", bundle_file())
        loaded = build_predictor(database)
        loaded.load()
        predictor = loaded
        matcher = None
    else:
        if not os.path.exists("reference_images"):
            raise RuntimeError("Missing reference_images/ directory.")
        logger.info("Stack v2 bundle not found; loading BioCLIP v1 matcher")
        predictor = None
        matcher = build_matcher(database)
    db = database
    model_ready = True
    model_error = None
    logger.info("Backend ready (engine=%s)", engine_name)


def ensure_model() -> None:
    global model_error, model_ready
    if model_ready:
        return
    with _load_lock:
        if model_ready:
            return
        try:
            _load_models()
        except Exception as exc:
            model_ready = False
            model_error = f"{type(exc).__name__}: {exc}"
            logger.error("Model load failed: %s", model_error)
            raise


@app.on_event("startup")
async def startup_event():
    logging.basicConfig(level=logging.INFO)
    logger.info(
        "API process started (engine=%s). Weights load on /warmup or /identify.",
        active_engine(),
    )


def _health_body() -> dict:
    engine = active_engine()
    if model_ready:
        return {
            "status": "ok",
            "ready": True,
            "engine": engine,
            "message": f"{engine} loaded and ready.",
        }
    if model_error:
        return {"status": "error", "ready": False, "engine": engine, "message": model_error}
    hint = (
        "Call GET/POST /warmup once to load Stack v2 vision towers."
        if engine == "stack_v2"
        else "Call GET/POST /warmup once to load the BioCLIP matcher."
    )
    return {"status": "cold", "ready": False, "engine": engine, "message": hint}


@app.get("/health")
def health_check():
    return _health_body()


@app.post("/warmup")
@app.get("/warmup")
def warmup():
    try:
        ensure_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model failed to load: {exc}") from exc
    return _health_body()


async def _image_bytes(request: Request) -> tuple[bytes, str | None]:
    """Return raw upload bytes and an optional state from the body or query."""
    query_state = request.query_params.get("state")
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Request body is not JSON") from exc
        if not isinstance(body, dict) or "image_base64" not in body:
            raise HTTPException(status_code=400, detail="JSON body needs image_base64")
        try:
            raw = base64.b64decode(body["image_base64"], validate=False)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="image_base64 is not valid base64") from exc
        state = body.get("state", query_state)
        return raw, None if state is None else str(state)

    form = await request.form()
    state = form.get("state")
    if state is None:
        state = query_state
    upload = form.get("image") or form.get("file")
    if upload is not None and hasattr(upload, "read"):
        raw = await upload.read()
        return raw, None if state is None else str(state)
    encoded = form.get("image_base64")
    if encoded:
        try:
            raw = base64.b64decode(str(encoded), validate=False)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="image_base64 is not valid base64") from exc
        return raw, None if state is None else str(state)
    raise HTTPException(status_code=400, detail="Provide an image file or image_base64")


def _open_stripped(raw: bytes) -> Image.Image:
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the upload limit")
    try:
        clean, _had_gps = strip_exif(raw)
    except ContributionError as exc:
        raise HTTPException(status_code=400, detail="Could not read that image") from exc
    # `_had_gps` is a boolean. It is intentionally not logged.
    try:
        image = Image.open(io.BytesIO(clean))
        image.load()
        return image.convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not read that image") from exc


@app.post("/identify")
async def identify(request: Request):
    try:
        ensure_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model failed to load: {exc}") from exc

    raw, state_text = await _image_bytes(request)
    state = parse_state(state_text)
    image = _open_stripped(raw)

    try:
        if predictor is not None and active_engine() == "stack_v2":
            prediction = predictor.predict(image, state=state)
        elif matcher is not None:
            ranked = matcher.rank(image, crop=False)
            prediction = package_ranked_matches(ranked, db, state=state, engine="bioclip_v1")
        else:
            raise HTTPException(status_code=503, detail="Model is not ready.")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Identification failed: %s", type(exc).__name__)
        raise HTTPException(status_code=500, detail="Identification failed") from exc

    top = prediction.top_matches[0]["species_id"] if prediction.top_matches else None
    logger.info(
        "identify engine=%s tier=%s top=%s abstain=%s",
        prediction.engine,
        prediction.tier,
        top,
        prediction.abstain,
    )
    return prediction.as_response()


if os.path.isdir("reference_images"):
    app.mount(
        "/reference_images",
        StaticFiles(directory="reference_images"),
        name="reference_images",
    )
