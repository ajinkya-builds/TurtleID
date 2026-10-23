#!/usr/bin/env bash
# Run this in Google Cloud Shell (not on your Mac).
# Usage:
#   export PROJECT_ID=your-project-id
#   bash scripts/deploy_cloudrun.sh

set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID to your Google Cloud project id}"
REGION="${REGION:-asia-south1}"
SERVICE="${SERVICE:-turtleid-api}"

echo "Project:  $PROJECT_ID"
echo "Region:   $REGION"
echo "Service:  $SERVICE"

gcloud config set project "$PROJECT_ID"
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com

# Deploy from source (Dockerfile in repo root). First build can take 15–30+ minutes.
gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --timeout 300 \
  --concurrency 1 \
  --min-instances 0 \
  --max-instances 1 \
  --cpu-boost \
  --set-env-vars "HF_HOME=/app/models/hf_cache,HUGGINGFACE_HUB_CACHE=/app/models/hf_cache/hub"

URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')"
echo ""
echo "Deployed: $URL"
echo "Health:   $URL/health"
echo "Identify: $URL/identify"
echo ""
echo "Next: put this in docs/index.html → PYTHON_API_URL = '${URL}/identify'"
