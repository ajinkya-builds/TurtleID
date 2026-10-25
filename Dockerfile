# TurtleID Stack v2 — Hugging Face Spaces (cpu-basic, port 7860) and Cloud Run ($PORT).
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=7860 \
    HOME=/home/user \
    HF_HOME=/app/models/hf_cache \
    HUGGINGFACE_HUB_CACHE=/app/models/hf_cache/hub \
    TRANSFORMERS_CACHE=/app/models/hf_cache/transformers \
    OPENCLIP_CACHE_DIR=/app/models/hf_cache/open_clip \
    TURTLEID_DTYPE=fp32 \
    TURTLEID_USE_BIOCAP=1 \
    TURTLEID_DETECTOR=grounding_dino

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 user

COPY requirements-cloudrun.txt .
RUN pip install --no-cache-dir -r requirements-cloudrun.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY config.py api.py ./
COPY core ./core
COPY scripts/__init__.py scripts/preload_models.py ./scripts/
COPY data/species_db.json data/reference_images.json data/stack_v2_bundle.npz ./data/
COPY reference_images ./reference_images

# Optional weight bake into /app/models/hf_cache. Leave this at 0 for a light
# image. scripts/deploy_cloudrun.sh passes PRELOAD_MODELS=1 so a Cloud Run
# cold start loads vision towers from the image instead of Hugging Face.
ARG PRELOAD_MODELS=0
RUN mkdir -p /app/models/hf_cache \
    && if [ "$PRELOAD_MODELS" = "1" ]; then python -m scripts.preload_models; fi \
    && chown -R user:user /app /home/user

USER user

EXPOSE 7860

# Hugging Face Spaces uses 7860. Cloud Run injects $PORT (often 8080).
CMD ["sh", "-c", "exec uvicorn api:app --host 0.0.0.0 --port ${PORT:-7860}"]
