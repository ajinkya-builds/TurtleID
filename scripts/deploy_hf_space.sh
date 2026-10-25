#!/usr/bin/env bash
# Push TurtleID to a Hugging Face Docker Space (cpu-basic, port 7860).
#
#   export HF_TOKEN=hf_...
#   ./scripts/deploy_hf_space.sh
#   HF_SPACE_ID=ajinkya-builds/TurtleID PRELOAD_MODELS=1 ./scripts/deploy_hf_space.sh
#
# Requires the Hugging Face CLI (`hf` or `huggingface-cli`) and a bundle at
# data/stack_v2_bundle.npz (python -m scripts.export_stack_v2_bundle).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SPACE="${HF_SPACE_ID:-ajinkya-builds/TurtleID}"
PRELOAD="${PRELOAD_MODELS:-0}"
BUNDLE="${ROOT}/data/stack_v2_bundle.npz"

if [[ ! -f "${BUNDLE}" ]]; then
  echo "Missing ${BUNDLE}." >&2
  echo "Export it first: python -m scripts.export_stack_v2_bundle" >&2
  exit 1
fi

if command -v hf >/dev/null 2>&1; then
  HF=(hf)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF=(huggingface-cli)
else
  echo "Install huggingface_hub so the hf CLI is on PATH." >&2
  exit 1
fi

"${HF[@]}" repos create "${SPACE}" --type space --space-sdk docker --exist-ok \
  || "${HF[@]}" repos create "${SPACE}" --type space --space-sdk docker \
  || true

STAGE="$(mktemp -d)"
cleanup() { rm -rf "${STAGE}"; }
trap cleanup EXIT

mkdir -p "${STAGE}/core" "${STAGE}/data" "${STAGE}/scripts" "${STAGE}/reference_images"
cp "${ROOT}/Dockerfile" "${ROOT}/requirements-cloudrun.txt" "${ROOT}/api.py" "${ROOT}/config.py" "${STAGE}/"
cp -R "${ROOT}/core/." "${STAGE}/core/"
cp "${ROOT}/scripts/__init__.py" "${ROOT}/scripts/preload_models.py" "${STAGE}/scripts/"
cp "${ROOT}/data/species_db.json" "${ROOT}/data/reference_images.json" "${BUNDLE}" "${STAGE}/data/"
cp -R "${ROOT}/reference_images/." "${STAGE}/reference_images/"
cp "${ROOT}/spaces/README.md" "${STAGE}/README.md"

if [[ "${PRELOAD}" == "1" ]]; then
  # The Space build honours the Dockerfile ARG. Flip the default for this push.
  python3 - "${STAGE}/Dockerfile" <<'PY'
import pathlib, sys
path = pathlib.Path(sys.argv[1])
path.write_text(path.read_text().replace("ARG PRELOAD_MODELS=0", "ARG PRELOAD_MODELS=1", 1))
PY
fi

echo "Uploading ${SPACE} (preload=${PRELOAD})"
"${HF[@]}" upload "${SPACE}" "${STAGE}" . --repo-type space \
  --commit-message "Deploy TurtleID Stack v2 runtime"
echo "Space: https://huggingface.co/spaces/${SPACE}"
