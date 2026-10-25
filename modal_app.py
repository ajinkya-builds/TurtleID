"""Modal serverless GPU for the TurtleID Stack v2 API.

Uses Modal's free monthly credit. Deploy from the repo root:

    modal deploy modal_app.py

The function serves ``api.app`` (FastAPI) on a T4. Vision weights are
downloaded while the image builds, so a cold container loads them from the
image instead of Hugging Face. Point the mobile page at the printed URL with
``?api=https://<your-app>.modal.run``.

The Cloud Run image pins CPU torch. This image installs CUDA torch on its own.
"""

from __future__ import annotations

import modal

APP_NAME = "turtleid-api"
HF_ROOT = "/cache/hf"

# Repos the vision runtime actually opens. snapshot_download stores the files
# Hugging Face Hub would otherwise fetch on the first request. The query path
# still drops the BioCLIP and BioCAP text towers after load.
MODEL_REPOS = (
    "imageomics/bioclip-2.5-vith14",
    "imageomics/biocap",
    "facebook/dinov2-with-registers-large",
    "IDEA-Research/grounding-dino-tiny",
)


def download_vision_weights() -> None:
    """Image build step. Does not import the TurtleID package."""
    import os

    os.environ["HF_HOME"] = HF_ROOT
    os.environ["HUGGINGFACE_HUB_CACHE"] = HF_ROOT + "/hub"
    os.environ["TRANSFORMERS_CACHE"] = HF_ROOT + "/transformers"
    os.environ["OPENCLIP_CACHE_DIR"] = HF_ROOT + "/open_clip"
    from huggingface_hub import snapshot_download

    for repo in MODEL_REPOS:
        snapshot_download(repo)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "fastapi==0.115.6",
        "uvicorn[standard]==0.34.0",
        "pydantic==2.10.4",
        "python-multipart==0.0.20",
        "pillow==11.0.0",
        "numpy==2.1.3",
        "open-clip-torch==3.3.0",
        "transformers==4.46.3",
        "safetensors==0.4.5",
        "huggingface_hub==0.26.5",
    )
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .env(
        {
            "HF_HOME": HF_ROOT,
            "HUGGINGFACE_HUB_CACHE": HF_ROOT + "/hub",
            "TRANSFORMERS_CACHE": HF_ROOT + "/transformers",
            "OPENCLIP_CACHE_DIR": HF_ROOT + "/open_clip",
            "TURTLEID_DTYPE": "fp16",
            "TURTLEID_USE_BIOCAP": "1",
            "TURTLEID_DETECTOR": "grounding_dino",
        }
    )
    .run_function(download_vision_weights, timeout=60 * 60)
    .add_local_dir(
        ".",
        remote_path="/root/turtleid",
        ignore=[
            ".git",
            ".git/**",
            ".venv",
            ".venv/**",
            "**/__pycache__",
            "**/*.pyc",
            "models/hf_cache",
            "models/hf_cache/**",
            "models/*.pt",
            "models/*.npz",
            "data/field_gallery",
            "data/field_gallery/**",
            "Turtles",
            "Turtles/**",
            "logs",
            "logs/**",
            "contributions",
            "contributions/**",
            "records",
            "records/**",
            "pool",
            "pool/**",
            "incoming",
            "incoming/**",
            "negatives",
            "negatives/**",
            "dataset",
            "dataset/**",
            ".cursor",
            ".cursor/**",
        ],
    )
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="T4",
    scaledown_window=120,
    timeout=900,
    memory=16384,
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def fastapi_app():
    """ASGI entry. ``api.app`` is the FastAPI instance in api.py."""
    import os
    import sys

    root = "/root/turtleid"
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)
    from api import app as api_app

    return api_app
