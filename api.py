import os
import io
import logging
import base64
import threading
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Setup basic configuration and models
import config
from core.database import SpeciesDB
from core.photo_match import BioClipMatcher

# Initialise FastAPI app
app = FastAPI(title="Chelonid-ID Mobile API")

# Allow CORS for GitHub Pages frontend and local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ajinkya-builds.github.io",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
db = None
matcher = None
model_ready = False
model_error = None
_load_lock = threading.Lock()


def _load_models() -> None:
    global db, matcher, model_ready, model_error
    try:
        if not os.path.exists("reference_images"):
            raise RuntimeError("Missing reference_images/ directory.")
        if not os.path.exists("data/species_db.json"):
            raise RuntimeError("Missing data/species_db.json file.")

        logging.info("Loading Species Database...")
        db = SpeciesDB.load()

        logging.info("Loading BioClipMatcher (foundation models)...")
        # Downloads weights into HF_HOME / models/hf_cache on first boot.
        matcher = BioClipMatcher(db=db)

        model_ready = True
        model_error = None
        logging.info("Backend is ready for identification.")
    except Exception as e:
        logging.error("Failed to load models: %s", e)
        model_ready = False
        model_error = str(e)
        raise


def ensure_model() -> None:
    """Load BioCLIP while Cloud Run is allocating CPU for this request.

    With default CPU throttling, a background startup thread is starved between
    short /health probes, so the model never finishes downloading.
    """
    if model_ready:
        return
    if model_error and not model_ready:
        # Allow one retry after a previous failure.
        pass
    with _load_lock:
        if model_ready:
            return
        _load_models()


@app.on_event("startup")
async def startup_event():
    logging.basicConfig(level=logging.INFO)
    logging.info("API process started; BioCLIP loads on first /identify or /warmup.")


class IdentifyRequest(BaseModel):
    image_base64: str


@app.post("/identify")
async def identify(request: IdentifyRequest):
    try:
        ensure_model()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model failed to load: {e}") from e

    if not model_ready or not matcher:
        raise HTTPException(status_code=503, detail="Model is not ready.")

    try:
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        ranked_matches = matcher.rank(image, crop=False)

        if not ranked_matches:
            return {
                "status": "success",
                "matches": [],
                "message": "No matches found.",
            }

        top_matches = []
        for match in ranked_matches[:3]:
            top_matches.append(
                {
                    "species_id": match.species_id,
                    "scientific_name": match.scientific_name,
                    "common_name": match.common_en,
                    "score": float(match.score),
                }
            )

        return {
            "status": "success",
            "matches": top_matches,
        }

    except Exception as e:
        logging.error("Error during identification: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/warmup")
@app.get("/warmup")
def warmup():
    """Trigger model download while CPU is allocated to this request."""
    try:
        ensure_model()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model failed to load: {e}") from e
    return {"status": "ok", "ready": True, "message": "BioCLIP matcher loaded and ready."}


@app.get("/health")
def health_check():
    if model_ready:
        return {"status": "ok", "ready": True, "message": "BioCLIP matcher loaded and ready."}
    if model_error:
        return {"status": "error", "ready": False, "message": model_error}
    return {
        "status": "cold",
        "ready": False,
        "message": "Call GET/POST /warmup once (may take several minutes) to load BioCLIP.",
    }
