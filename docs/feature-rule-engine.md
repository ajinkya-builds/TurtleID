# Feature checklist + lightweight rule engine

Notes from discussion with Prashant Jijaji (Bhopal) and Ajinkya Patil, 14–15 October 2026, after the stack v2 benchmark.

## What Prashant asked for

1. Model predicts a **top 3** (already ~97% hit rate on the field set).
2. Then use **species-specific identifying features** as the ultimate decider among those three — e.g. orange vertebral streak / coral plastron for *Pangshura tecta* vs yellow plastron with dark blotches for *P. tentoria*; plastron blotch shape; etc.
3. Optionally show **multiple reference images for the top 3** and let the officer decide.

That is better than asking the vision model alone to settle Madhya Pradesh sister species.

## What already exists in the repo

| Asset | Role |
|-------|------|
| Stack v2 top-3 | Shortlist from photos |
| `core/morphkey.py` | Multi-access morphological characters (keel, third vertebral, plastron colour, head marks, femoral flaps, …) |
| `data/species_db.json` | `key_character`, `diagnostics`, pairwise `discriminator` text (tecta vs tentoria already written) |
| `core/stack_v2/traits.py` | Trait text prompts for BioCAP / BioCLIP |

So the next step is wiring, not inventing a new science pack from scratch: **top-3 → morphkey restricted to those three → UI with diagnostics + reference plates → officer confirm**.

## Lightweight “text → rule” engine (Ajinkya’s design)

Do **not** retrain BioCLIP when someone adds a feature. Update a small rule file only.

| Step | What happens |
|------|----------------|
| Expert adds a feature in plain language | e.g. “orange line on vertebral keel → favors tecta over tentoria” |
| System stores a rule | `feature_id`, allowed states, which species allow / forbid each state |
| At ID time | Vision → top 3 → only those species stay in play → rules score or eliminate → optional human confirm |

Conceptual rule:

```text
IF top3 includes tecta AND tentoria
AND observer (or soft auto-assist) sees: plastron = coral_red
THEN boost tecta, downweight tentoria
```

## Two modes

1. **Human-in-the-loop (build first for enforcement)**  
   Show top 3 + checklist of discriminating features + 2–3 example photos each → officer ticks what they see → rule engine confirms.

2. **Auto feature assist (later)**  
   Ask BioCAP / part crops soft questions (“coral plastron?”, “spined third vertebral?”, “red ear stripe?”). Treat as soft evidence only — orange lines and blotches are easy to miss on bad phone JPEGs.

**Ultimate decider for casework:** rules + human. Vision is the shortlister.

## Practical message for Prashant

> Yes. Pipeline stays: photo → AI shortlists top 3 → feature checklist (and reference images) decides among those three. New ID features can be added as plain-language rules without retraining the whole model. We already have the morphological key and tecta/tentoria discriminators in the database; next step is connecting them to the top-3 output and the UI.

## Smallest useful build

1. After stack v2 (or Streamlit photo tab), take top-3 species IDs.  
2. Restrict `morphkey` / diagnostics to that set.  
3. Show TRAFFIC / gallery reference images for those three.  
4. Officer confirms or ticks characters → logged determination.  
5. Later: JSON/YAML rule bank editable without a deploy of model weights.
