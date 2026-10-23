# TurtleID notes

Working notes from building and evaluating the offline photo identification stack (October 2026). These sit beside the Streamlit app docs in the repo root (`README.md`, `QUICKSTART.md`, `PUBLISHING.md`).

| Note | What it covers |
|------|----------------|
| [literature.md](literature.md) | Seizure trade context, why ID matters, published papers |
| [offline-stack.md](offline-stack.md) | Architecture and models used for the high-accuracy batch ID path |
| [benchmark-oct2026.md](benchmark-oct2026.md) | Accuracy on the organised field set (Gemini vs BioCLIP vs stack v2) |
| [feature-rule-engine.md](feature-rule-engine.md) | Top-3 → species feature checklist → human confirm (Prashant’s next step) |
| [model-cache.md](model-cache.md) | Why Hugging Face weights must live on the Wednesday volume |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Jules mobile UI: GitHub Pages + Drive Apps Script + FastAPI backend |
| **Live mobile shell** | [GitHub Pages](https://ajinkya-builds.github.io/TurtleID/) (`docs/index.html`) — camera/upload UI; ID needs API URLs |

### Development process

Prefer **Google Jules** for app scaffolding / productionization PRs (Ajinkya has Jules credits). Cursor reviews, stands up Pages, and keeps Wednesday-volume model/data work local. See `.cursor/rules/offload-dev-to-jules.mdc`.

Field photographs, iNaturalist gallery downloads, raw score CSVs under `logs/`, and downloaded model weights are **not** in git (see `.gitignore`). Summaries of the scores are in the benchmark note.
