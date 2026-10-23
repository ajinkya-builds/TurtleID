# Cloud Run: TurtleID FastAPI + BioCLIP matcher (CPU).
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    HF_HOME=/app/models/hf_cache \
    HUGGINGFACE_HUB_CACHE=/app/models/hf_cache/hub \
    TRANSFORMERS_CACHE=/app/models/hf_cache/transformers \
    OPENCLIP_CACHE_DIR=/app/models/hf_cache/open_clip

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-cloudrun.txt .
RUN pip install --no-cache-dir -r requirements-cloudrun.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY config.py api.py ./
COPY core ./core
COPY data/species_db.json data/reference_images.json ./data/
COPY reference_images ./reference_images

RUN mkdir -p /app/models/hf_cache

EXPOSE 8080

# Cloud Run sets $PORT; bind 0.0.0.0.
CMD exec uvicorn api:app --host 0.0.0.0 --port ${PORT}
