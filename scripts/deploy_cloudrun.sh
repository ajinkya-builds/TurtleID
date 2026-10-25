#!/usr/bin/env bash
# Build the Stack v2 image and deploy it to Cloud Run.
#
# The live service is https://turtleid-api-wgbusgxhnq-el.a.run.app
# (region asia-south1, service turtleid-api). Cloud Run injects $PORT and
# the container listens on it. Do not set PORT in the service env.
#
# Weights are baked into /app/models/hf_cache when PRELOAD_MODELS=1 (the
# default for this script). The Dockerfile ARG still defaults to 0 so a
# casual local build stays light.
#
# Usage:
#   export CLOUD_RUN_PROJECT=your-project-id   # or PROJECT_ID, or gcloud config
#   bash scripts/deploy_cloudrun.sh
#
# Overrides: REGION, SERVICE, ARTIFACT_REPO, PRELOAD_MODELS, IMAGE_TAG,
#            CLOUD_RUN_MEMORY (default 16Gi), CLOUD_RUN_CPU (default 4),
#            CLOUD_RUN_TIMEOUT (default 900).

set -euo pipefail

PROJECT_ID="${CLOUD_RUN_PROJECT:-${PROJECT_ID:-}}"
if [[ -z "${PROJECT_ID}" ]]; then
  PROJECT_ID="$(gcloud config get-value project 2>/dev/null || true)"
fi
if [[ -z "${PROJECT_ID}" || "${PROJECT_ID}" == "(unset)" ]]; then
  echo "Set CLOUD_RUN_PROJECT or PROJECT_ID, or run: gcloud config set project ID" >&2
  exit 1
fi

REGION="${REGION:-asia-south1}"
SERVICE="${SERVICE:-turtleid-api}"
REPO="${ARTIFACT_REPO:-turtleid}"
PRELOAD_MODELS="${PRELOAD_MODELS:-1}"
MEMORY="${CLOUD_RUN_MEMORY:-16Gi}"
CPU="${CLOUD_RUN_CPU:-4}"
TIMEOUT="${CLOUD_RUN_TIMEOUT:-900}"
TAG="${IMAGE_TAG:-$(date -u +%Y%m%d%H%M%S)}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/api:${TAG}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

if [[ ! -f data/stack_v2_bundle.npz ]]; then
  echo "Missing data/stack_v2_bundle.npz. Run: python -m scripts.export_stack_v2_bundle" >&2
  exit 1
fi

echo "Project:  ${PROJECT_ID}"
echo "Region:   ${REGION}"
echo "Service:  ${SERVICE}"
echo "Image:    ${IMAGE}"
echo "Preload:  PRELOAD_MODELS=${PRELOAD_MODELS}"

gcloud config set project "${PROJECT_ID}"

gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  || echo "API enable skipped or rate-limited; continuing..."

if ! gcloud artifacts repositories describe "${REPO}" --location "${REGION}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${REPO}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="TurtleID API images"
fi

BUILD_CONFIG="$(mktemp)"
cat > "${BUILD_CONFIG}" <<EOF
steps:
  - name: gcr.io/cloud-builders/docker
    args:
      - build
      - --build-arg
      - PRELOAD_MODELS=${PRELOAD_MODELS}
      - -t
      - ${IMAGE}
      - .
images:
  - ${IMAGE}
timeout: 3600s
EOF

# --source cannot pass Docker build-args, so the bake goes through Cloud Build.
gcloud builds submit --config "${BUILD_CONFIG}" --timeout=3600s .
rm -f "${BUILD_CONFIG}"

# PORT is intentionally absent. Cloud Run sets it; the image reads \${PORT:-7860}.
gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --execution-environment gen2 \
  --memory "${MEMORY}" \
  --cpu "${CPU}" \
  --timeout "${TIMEOUT}" \
  --concurrency 1 \
  --min-instances 0 \
  --max-instances 1 \
  --cpu-boost \
  --no-cpu-throttling \
  --set-env-vars "HF_HOME=/app/models/hf_cache,HUGGINGFACE_HUB_CACHE=/app/models/hf_cache/hub,TRANSFORMERS_CACHE=/app/models/hf_cache/transformers,OPENCLIP_CACHE_DIR=/app/models/hf_cache/open_clip,TURTLEID_DTYPE=fp32,TURTLEID_USE_BIOCAP=1,TURTLEID_DETECTOR=grounding_dino"

URL="$(gcloud run services describe "${SERVICE}" --region "${REGION}" --format='value(status.url)')"
echo ""
echo "Deployed: ${URL}"
echo "Health:   ${URL}/health"
echo "Identify: ${URL}/identify"
echo "Pages default is already https://turtleid-api-wgbusgxhnq-el.a.run.app"
