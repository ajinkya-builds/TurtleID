import io
import json
import logging
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Setup basic configuration and models
import config
from core.database import SpeciesDB
from core.inference import (
    ChelonidIdentifier,
    DeterminationTier,
    TrainedClassifier,
    OODFilter,
)

# Initialise FastAPI app
app = FastAPI(title="Chelonid-ID Mobile API")

# Allow CORS for GitHub Pages frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, this should be restricted to the github pages URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load database and model at startup
db = None
identifier = None

@app.on_event("startup")
async def startup_event():
    global db, identifier
    logging.basicConfig(level=logging.INFO)
    try:
        logging.info("Loading database...")
        db = SpeciesDB(config.SPECIES_DB_PATH)
        logging.info("Loading model...")

        # Determine if we should load the trained classifier or just the matcher
        # For simplicity, we assume the trained classifier is available for the backend
        # In a real deployment, we might need logic from `app.py` to pick the best model.
        if config.CLASSIFIER_PATH.exists():
             ood = None
             if config.CALIBRATION_PATH.exists():
                 cal = json.loads(config.CALIBRATION_PATH.read_text())
                 ood = OODFilter(threshold=cal.get("energy_threshold", config.DEFAULT_ENERGY_THRESHOLD))
             model = TrainedClassifier.load(config.CLASSIFIER_PATH, config.CALIBRATION_PATH)
             identifier = ChelonidIdentifier(
                 model=model,
                 detector_path=config.DETECTOR_PATH if config.DETECTOR_PATH.exists() else None,
                 ood=ood
             )
             logging.info("Trained classifier loaded.")
        else:
            # Fallback to matcher if no trained classifier
            from core.matcher import Gallery, embed_images, load_backbone
            import numpy as np
            if config.GALLERY_PATH.exists():
                gallery = Gallery.load(config.GALLERY_PATH)
                # Need a fallback identifier wrapper here, or just raise error if model missing
                logging.warning("No trained classifier found. API requires trained models.")
            else:
                 logging.error("No models found!")
    except Exception as e:
        logging.error(f"Failed to load models: {e}")

class IdentifyRequest(BaseModel):
    image_base64: str

@app.post("/identify")
async def identify(request: IdentifyRequest):
    if not db or not identifier:
         raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)

        # Run inference
        result = identifier.identify(image_data)

        if not result.candidates:
             return {
                 "status": "success",
                 "tier": result.tier.name,
                 "species_id": None,
                 "confidence": 0.0,
                 "message": "No species identified or out of distribution."
             }

        top_match = result.candidates[0]
        species_info = db.get(top_match.species_id)

        return {
             "status": "success",
             "tier": result.tier.name,
             "species_id": top_match.species_id,
             "scientific_name": species_info.scientific_name if species_info else "Unknown",
             "common_name": species_info.common_name if species_info else "Unknown",
             "confidence": top_match.probability
        }

    except Exception as e:
         logging.error(f"Error during identification: {e}")
         raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}
