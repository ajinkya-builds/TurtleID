import os
import io
import json
import logging
import base64
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

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
        "http://127.0.0.1:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
db = None
matcher = None
model_ready = False

@app.on_event("startup")
async def startup_event():
    global db, matcher, model_ready
    logging.basicConfig(level=logging.INFO)
    try:
        if not os.path.exists("reference_images"):
            raise RuntimeError("Missing reference_images/ directory.")
        if not os.path.exists("data/species_db.json"):
            raise RuntimeError("Missing data/species_db.json file.")

        logging.info("Loading Species Database...")
        db = SpeciesDB.load()

        logging.info("Loading BioClipMatcher (foundation models)...")
        # This will trigger HF weight downloads to models/hf_cache on first boot
        matcher = BioClipMatcher(db=db)

        model_ready = True
        logging.info("Backend is ready for identification.")
    except Exception as e:
        logging.error(f"Failed to load models during startup: {e}")
        # We do not exit here so the /health endpoint can report the failure to the load balancer
        model_ready = False

class IdentifyRequest(BaseModel):
    image_base64: str

@app.post("/identify")
async def identify(request: IdentifyRequest):
    if not model_ready or not matcher:
         raise HTTPException(
             status_code=503,
             detail="Model is still loading or failed to load. Please check server logs."
         )

    try:
        # Decode base64 image and convert to PIL Image
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Rank the image using BioClipMatcher
        # This returns a list of core.photo_match.RankedMatch
        ranked_matches = matcher.rank(image, crop=False)

        if not ranked_matches:
             return {
                 "status": "success",
                 "matches": [],
                 "message": "No matches found."
             }

        # Format the top 3 matches
        top_matches = []
        for match in ranked_matches[:3]:
            top_matches.append({
                "species_id": match.species_id,
                "scientific_name": match.scientific_name,
                "common_name": match.common_en,
                "score": float(match.score)
            })

        return {
             "status": "success",
             "matches": top_matches
        }

    except Exception as e:
         logging.error(f"Error during identification: {e}")
         raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    if model_ready:
        return {"status": "ok", "message": "BioCLIP matcher loaded and ready."}
    else:
        raise HTTPException(status_code=503, detail="Model not ready.")
