# Hosting Stack v2

Stack v2 (84.8% top-1 / 96.7% top-3 on the 92-photo seizure benchmark) embeds
the photograph with vision towers only. Text matrices, the gallery cache,
plate vectors, and the logistic probe are precomputed into
`data/stack_v2_bundle.npz`. The server never opens `Turtles/` and never
loads the BioCLIP or BioCAP text towers.

The mobile page is `docs/index.html` on GitHub Pages. Its default API is
the existing Cloud Run service. A `?api=` query parameter points the same
page at a Modal URL, or any other host, without editing the HTML.

Hugging Face `cpu-basic` Docker Spaces now return **402 Payment Required**
on accounts that are not PRO. The Space card and `scripts/deploy_hf_space.sh`
stay in the repo for a PRO account. They are not the default host.

## What is in the committed bundle

`data/stack_v2_bundle.npz` is a real standalone export, not a schema stub:

| Field | Value |
|-------|--------|
| Mode | `standalone` (no `Turtles/` seizure photos) |
| Crops | GroundingDINO (`detector=grounding_dino`) |
| Plates | 28 TRAFFIC reference images |
| Verified submissions | 13 |
| iNat gallery | 96 (8 taxa × 12), cached locally under `data/field_gallery/` which is gitignored |
| Cache rows | 149 (iNat + k-means prototypes + field) |
| Probe classes | 3 (`geochelone_elegans`, `geoclemys_hamiltonii`, `pangshura_tecta`) — the labels present in verified submissions |
| Seizure photos | 0 |

This file is **not** the weight set behind the 84.8% figure. That figure
needs the seizure photographs. On the machine that has them:

```bash
python -m scripts.export_stack_v2_bundle \
  --turtles-dir Turtles \
  --truth logs/turtles_ground_truth.json
```

Replace `data/stack_v2_bundle.npz` with that export and redeploy.

## 1. Export the bundle

On a machine that can see the weights cache (`models/hf_cache/`, see
[model-cache.md](model-cache.md)):

```bash
# Repo-standalone: 28 TRAFFIC plates + verified submissions + iNat gallery.
python -m scripts.export_stack_v2_bundle

# See what would be included, without loading models.
python -m scripts.export_stack_v2_bundle --inventory
```

`--turtles-dir` is optional. If `Turtles/` is absent the exporter stays in
standalone mode. Pass `--skip-inat` to skip the iNat download. Pass
`--detector border` only when GroundingDINO will not be loaded at query
time. The query default is `TURTLEID_DETECTOR=grounding_dino`, and the
committed npz was exported with that crop.

Commit the npz. It holds embeddings, not pixels, filenames, or coordinates.
Do not commit `data/field_gallery/` or `models/hf_cache/`.

## 2. Run the API

`api.py` uses Stack v2 when the npz is present (`"engine": "stack_v2"` on
`/health` and `/identify`). If the file is missing it falls back to the
BioCLIP 2 plate matcher (`"engine": "bioclip_v1"`).

```bash
export HF_HOME="$PWD/models/hf_cache"
export TURTLEID_DTYPE=fp32          # fp16 | bf16 | fp32 (fp16 falls back to fp32 on CPU)
export TURTLEID_USE_BIOCAP=1        # 0 skips the 512-d trait tower
export TURTLEID_DETECTOR=grounding_dino   # or border
uvicorn api:app --host 0.0.0.0 --port 7860
```

`POST /identify` accepts JSON `{"image_base64", "state"}` or a multipart
form (`image` or `image_base64`, plus `state`). The state may also be a
query parameter. EXIF, including GPS, is stripped in memory and is not
logged.

`GET /reference_images/...` serves the TRAFFIC plates the response links to.

Warm the cache without serving:

```bash
python -m scripts.preload_models
```

## 3. Google Cloud Run

`Dockerfile` serves both Cloud Run and a Hugging Face Space. It listens on
`$PORT` (Cloud Run injects this, often 8080; the image default is 7860) and
runs as uid 1000.

`PRELOAD_MODELS=1` runs `python -m scripts.preload_models` during
`docker build` and writes vision-only weights to `/app/models/hf_cache`.
The Dockerfile ARG defaults to `0`. `scripts/deploy_cloudrun.sh` defaults
it to `1`, so a cold start does not spend several minutes downloading
BioCLIP, BioCAP, DINOv2, and GroundingDINO.

```bash
export CLOUD_RUN_PROJECT=your-gcp-project
bash scripts/deploy_cloudrun.sh
```

The script builds with Cloud Build (`--build-arg PRELOAD_MODELS=1`), pushes
to Artifact Registry, and deploys service `turtleid-api` in `asia-south1`
(the region already behind
`https://turtleid-api-wgbusgxhnq-el.a.run.app`). It does not set `PORT`.
Memory defaults to 16Gi and CPU to 4 (`CLOUD_RUN_MEMORY`, `CLOUD_RUN_CPU`).
Request timeout defaults to 900s. Concurrency is 1.

`gcloud run deploy --source` cannot pass a Docker build-arg, which is why
the script uses `gcloud builds submit`.

## 4. Modal free GPU

`modal_app.py` is a separate image. It installs CUDA torch (cu124), not the
CPU pin in `requirements-cloudrun.txt`, downloads the four vision repos
during the image build, and wraps `api.app`:

```bash
modal deploy modal_app.py
```

The web function requests `gpu="T4"` and `scaledown_window=120` (idle
containers stay up for two minutes, inside Modal's free monthly credit).
One request at a time. Dtype on this image is `fp16`.

After deploy, Modal prints an HTTPS URL. Open the Pages site as:

`https://ajinkya-builds.github.io/TurtleID/?api=https://<your-app>.modal.run`

That choice is stored in `localStorage` until another `?api=` replaces it.

## 5. Hugging Face Spaces (PRO only)

`spaces/README.md` is still a Docker Space card (`app_port: 7860`,
`startup_duration_timeout: 30m`). As of October 2026, Hugging Face returns
**402 Payment Required** for `cpu-basic` Docker and Gradio Spaces on
accounts that are not PRO. Do not point the mobile page at

`https://ajinkya-builds-turtleid.hf.space`

unless that Space is on a PRO account and answers `/health`.

```bash
export HF_TOKEN=hf_...
PRELOAD_MODELS=1 ./scripts/deploy_hf_space.sh
```

## 6. GitHub Pages

`docs/index.html` resolves the API in this order:

1. `?api=https://...` (saved in `localStorage` as `turtleid_api_base`)
2. that saved value, on later visits, unless it is an `hf.space` host
3. `https://turtleid-api-wgbusgxhnq-el.a.run.app`

A saved `hf.space` URL from an earlier visit is ignored. If an explicit
`?api=` host fails with a network error, HTTP 402, 404, 408, or 5xx, the
page retries Cloud Run.

The page sends the state dropdown (default **All India / Unknown (Trade /
Seizure)**, which applies no geographic prior). The response badge is the
determination tier. Schedule I / II, the legal-divergence and out-of-range
warnings, the “How to tell #1 from #2” line, and two or three morphological
questions are rendered under the top three. Tapping a trait fades candidates
that cannot show it.

Pages deploys from `docs/` via `.github/workflows/deploy-pages.yml` on
`main`.

## Memory knobs

| Variable | Effect |
|----------|--------|
| `TURTLEID_DTYPE` | `fp32` (CPU default), `fp16`, `bf16`. fp16 on CPU is loaded as fp32. Modal sets `fp16`. |
| `TURTLEID_USE_BIOCAP` | `0` skips BioCAP and its channel. |
| `TURTLEID_DETECTOR` | `border` skips GroundingDINO and uses the border crop. |
| `TURTLEID_RELEASE_DETECTOR` | `1` unloads the detector after each crop. |
| `TURTLEID_BUNDLE` | Alternate path to the npz. |
| `TURTLEID_SIM_FLOOR` | Cosine below this, against the cache and plates, is `REJECTED`. |
| `TURTLEID_ENERGY_GATE` | `1` also rejects on free-energy. Off by default: Stack v2 scores are not the calibrated classifier logits that threshold was fit on. |
| `TURTLEID_TEMPERATURE` | Softmax temperature. Default `1`. Probabilities stay marked uncalibrated until `TURTLEID_CALIBRATED=1`. |
