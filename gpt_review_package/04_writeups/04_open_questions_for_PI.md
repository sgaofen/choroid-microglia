# Prep for next Huixin meeting

## Status

I spent a session running the three sample images through Cellpose + a custom supplementary detector and tried to verify the output by eye. I can run code; I cannot judge biologically whether a given outlined object is a real cell, a process fragment, a process node, or a multi-cell merge. Without that judgment I cannot validate the pipeline.

## What I need from her

A 15-minute calibration session, ideally with one of the crop images open on screen, walking through her own answers to:

1. **What is a cell?** Specifically — does she require a visible bright soma (compact local peak), or does she also count "obvious process-bearing structure where the soma is buried under overlap" as a cell?
2. **Bright node where multiple processes meet** — is that one cell, or is it process convergence without a cell?
3. **Long elongated bright structure with no clear center** — one cell? multiple? not a cell?
4. **Touching cells** — when two somata are next to each other and processes interlock, how does she normally count them? One or two?
5. **What precision does her science question need?** ±5% per-cell count? Or distribution-level only (e.g., "fraction round vs. ramified" approximately right)?
6. **Does she want Kolmer cells and stromal microglia analyzed together, or separately?**

## What to show her

Open these in Preview from the project folder (`~/Documents/choroid-microglia/project_A_morphology/experiments/final_review/`):

- `WT_2_B_dense_clean.png` — unmarked. Ask her: "How many cells do you count in this view?"
- `WT_2_B_dense_dots.png` — same crop, my centroids as dots. Compare to her count.
- `WT_2_B_missed_zoom_clean.png` + `_v4.png` — the rescue case. Ask: "Is the green-outlined structure one cell or multiple?"
- `WT_2_D_right_clean.png` + `_dots.png` — the densest case.

## Pipeline state (one-line summary)

Three detection paths exist:

- **Cellpose `cyto3` default** — clean, conservative, 1099–1293 cells/image. Misses cells with faint soma + bright processes.
- **Cellpose `cyto3` at lower threshold ("extended")** — 1772–1848 cells/image. Catches more, broadly clean.
- **Extended + distance-based supplementary ("v4")** — 2330–2752 cells/image. Catches the previously-missed ramified cells but introduces ~40–60% false positives in dense regions.

CSV with all v4 cells and their metrics (area, circularity, eccentricity, source) is at `experiments/v4_distance_supplement/all_cells_v4.csv`.

## After her answers

Two likely paths:

- **If she's happy with bulk approximation** — pick one of the three detection paths based on her tolerance for missed cells vs false positives, compute morphology distributions, compare across genotypes. Project A done.
- **If she wants per-cell precision** — she or I hand-annotate ~50 cells in one image (using her judgment from the calibration session), fine-tune Cellpose on those annotations, re-run. This is a 1–2 week loop.

## Other things to mention while there

- Confirm summer start date and Bio Sci 199 enrollment status (her email said summer enrollment may not be open yet)
- Get added to the lab Slack channel
- Get the calendar for joint Lehtinen-lab meetings (Thu/Fri, biweekly starting June)
- Ask whether she has any image where someone has already manually counted/classified cells — that becomes ground truth for validation
