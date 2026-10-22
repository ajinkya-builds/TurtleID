# Model weight cache (Wednesday volume)

## Why this exists

Ajinkya does **not** give Gemini (or similar outside tools) access to  
`/Volumes/Wednesday/Conservation Projects/TurtleID`.

In October 2026, cleanup outside that volume wiped `~/.cache/huggingface`, which deleted the downloaded weights for the offline stack (BioCLIP 2.5, BioCAP, DINOv2 registers, Grounding DINO). Project **code and the `.venv` packages were fine**; only the weight cache was gone.

## Policy

- Always download Hugging Face / OpenCLIP / transformers weights into **`models/hf_cache/`** on this volume.
- `config.py` sets `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`, and `OPENCLIP_CACHE_DIR` to that path before model loads.
- Cursor rule: `.cursor/rules/hf-cache-on-volume.mdc` (always apply).
- `models/hf_cache/` is gitignored — weights are large and machine-local.

## Models the offline stack expects

| Hub ID | Used for |
|--------|----------|
| `imageomics/bioclip-2.5-vith14` | Main embeddings |
| `imageomics/biocap` | Trait channel |
| `facebook/dinov2-with-registers-large` | Geometry / texture fusion |
| `IDEA-Research/grounding-dino-tiny` | Animal crop |

First run after a wipe re-downloads into `models/hf_cache/` (needs network). Do not reintroduce home-only caching for TurtleID.
