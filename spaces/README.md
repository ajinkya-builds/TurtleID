---
title: TurtleID
emoji: "🐢"
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
startup_duration_timeout: 30m
pinned: false
---

# TurtleID Stack v2

**402 on non-PRO accounts.** Hugging Face returns `402 Payment Required` for
`cpu-basic` Docker and Gradio Spaces unless the account is PRO. The mobile
page defaults to Cloud Run
(`https://turtleid-api-wgbusgxhnq-el.a.run.app`), not this Space. Use this
card only after a PRO Space answers `GET /health`.

Species identification for Indian tortoises and freshwater turtles. The Space
runs the Stack v2 vision runtime (`cpu-basic`, 2 vCPU / 16 GB) and reads
precomputed text embeddings from `data/stack_v2_bundle.npz`.

The mobile page on GitHub Pages calls `POST /identify`. See
`docs/free-hosting-stack-v2.md` in the source repository for how the bundle
is exported and how this Space is updated.

`GET /health` reports `"engine": "stack_v2"` once the bundle is in the image.
The first `GET /warmup` loads the vision towers into `HF_HOME` if they were
not baked in at build time (`PRELOAD_MODELS=1`).
