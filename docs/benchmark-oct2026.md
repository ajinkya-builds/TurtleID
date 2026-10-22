# Benchmark notes — field set, October 2026

Evaluation of organised seizure / field photographs labelled by species folder on Google Drive, scored locally. **Raw per-image CSVs stay in `logs/` (gitignored)** because they name field files. This note keeps aggregate metrics and error patterns only.

## Dataset

- Source folder (Drive): organised by ground-truth species names.
- Local copy: `Turtles/` (gitignored).
- Identified photos used for accuracy: **92** (plus unidentified photos scored but not counted in %).
- Species in the labelled set: *Geochelone elegans*, *Geoclemys hamiltonii*, *Hardella thurjii*, *Lissemys punctata*, *Nilssonia gangetica*, *Pangshura tecta*, *Pangshura tentoria*, *Trachemys scripta elegans*.

## Headline accuracy (top-1 on 92 identified)

| System | Top-1 | Top-3 |
|--------|------:|------:|
| **Stack v2** (full supplement) | **78/92 (84.8%)** | **89/92 (96.7%)** |
| Stack v1 (crop + hard taxonomy gate + LOO) | 54/92 (58.7%) | 66/92 (71.7%) |
| BioCLIP 2.5 alone (plates / zero-shot style) | 29/92 (31.5%) | — |
| BioCLIP 2 alone | 24/92 (26.1%) | — |
| Gemini Pro (Drive-side labels) | 8/92 (8.7%) | — |

GroupKFold by seizure / encounter heuristic (stack v2 predictions): mean top-1 ≈ **84.9%**.

## Stack v2 per-species (top-1 / top-3)

| Species | Top-1 | Top-3 |
|---------|------:|------:|
| *Geochelone elegans* | 13/13 | 13/13 |
| *Geoclemys hamiltonii* | 8/8 | 8/8 |
| *Lissemys punctata* | 19/19 | 19/19 |
| *Trachemys scripta elegans* | 2/2 | 2/2 |
| *Pangshura tentoria* | 21/23 | 23/23 |
| *Pangshura tecta* | 15/20 | 18/20 |
| *Hardella thurjii* | 0/5 | 4/5 |
| *Nilssonia gangetica* | 0/2 | 2/2 |

## What failed on stack v2 (14 top-1 misses)

Patterns, not filenames:

| Pattern | Count (approx.) | Notes |
|---------|----------------:|-------|
| *Hardella* → *Pangshura* / *Geoclemys* / slider | 5 | Weakest taxon; often still in top-3 |
| *P. tecta* ↔ *P. tentoria* sister swap | ~5 | Classic MP enforcement confusion |
| *Nilssonia* → *Lissemys* | 2 | Softshell collapse; truth usually top-2 |
| Hard misses (truth not in top-3) | 3 | Need feature checklist / better views |

## Why BioCLIP 2.5 alone struggled earlier

Before the supplement stack:

1. Sister *Pangshura* photos collapsing onto a single TRAFFIC plate (e.g. *P. sylhetensis*).
2. Softshells (*Lissemys*) pulled toward giant softshell plates.
3. *Trachemys* (red-eared slider) with no useful plate match.
4. Background clutter (hands, trays) dominating centre-crop embeddings.

## Why the hard-gated stack (v1) was not enough

Of 38 top-1 errors under v1:

- **22** — taxonomy gate never admitted the true group (so ranking never saw the right species).
- **16** — gate OK, wrong species inside the group.
- Hard gate also **regressed 12 photos** that ungated BioCLIP 2.5 had gotten right.

Lesson: use family/genus as a **soft prior**, not a hard mask. That is what v2 does.

## Local log files (not in git)

Useful if you have the Wednesday volume:

- `logs/turtles_v2_vs_truth.csv` — full v2 vs truth
- `logs/turtles_stack_vs_truth.csv` — v1 stack vs truth
- `logs/turtles_vs_ground_truth.csv` — BioCLIP 2 / 2.5 / Gemini vs truth
- `logs/turtles_ground_truth.json` — filename → species map

## Next accuracy lever

Top-3 is already ~97%. Closing the remaining sisters (*tecta*/*tentoria*, *Hardella*, *Nilssonia*/*Lissemys*) should be **feature rules + human confirm on the top-3**, not another end-to-end finetune. See [feature-rule-engine.md](feature-rule-engine.md).
