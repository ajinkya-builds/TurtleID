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


def _load_models() -> None:
    """Load BioCLIP off the request path so Cloud Run can bind $PORT quickly."""
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


@app.on_event("startup")
async def startup_event():
    logging.basicConfig(level=logging.INFO)
    # Non-blocking: Cloud Run health checks need the port open before BioCLIP loads.
    threading.Thread(target=_load_models, name="bioclip-loader", daemon=True).start()


class IdentifyRequest(BaseModel):
    image_base64: str


@app.post("/identify")
async def identify(request: IdentifyRequest):
    if model_error and not model_ready:
        raise HTTPException(
            status_code=503,
            detail=f"Model failed to load: {model_error}",
        )
    if not model_ready or not matcher:
        raise HTTPException(
            status_code=503,
            detail="Model is still loading. Retry in a minute (first cold start downloads BioCLIP).",
        )

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


@app.get("/health")
def health_check():
    # Always 200 once the process is up so Cloud Run startup probes succeed.
    if model_ready:
        return {"status": "ok", "ready": True, "message": "BioCLIP matcher loaded and ready."}
    if model_error:
        return {"status": "error", "ready": False, "message": model_error}
    return {"status": "loading", "ready": False, "message": "BioCLIP matcher is still loading."}
