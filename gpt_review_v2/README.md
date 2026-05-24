# Choroid plexus microglia analysis — round 2 consultation

Update since first GPT consultation: abandoned Cellpose entirely. Built pure-algorithm pipeline with threshold + erosion + watershed + skeleton. Read `PROMPT.md` first.

## What's in here

- `PROMPT.md` — the focused question
- `01_pipeline/` — current pipeline visuals (3 images showing the steps that work)
- `02_current_best/` — output of final pipeline + cross-genotype JSON
- `03_issues/` — remaining problems (over-split, HET verification, normalization sensitivity)
- `04_writeups/` — detailed findings doc

## Cliffs notes

**Pipeline (no Cellpose, no deep learning)**:
1. Normalize percentile [1, 99.5]
2. Gaussian σ=1
3. `> Otsu × 0.7` (lowered from default Otsu, fixes dim-process gaps)
4. Closing r=2
5. Skeletonize + spur prune (max=5) + filter skeleton CCs < 8 px
6. Erode binary r=2 → distance transform → local maxima → watershed → somata
7. For each soma: 1-px boundary ring, count skeleton CCs crossing it → process count

**Current numbers**:

| Image | n_somata | mean process count | bin 5+ ramified |
|---|---|---|---|
| WT_2  | 6163 | 1.98 | 2.8 % |
| HET_1 | 5131 | 1.82 | 2.5 % |
| HET_3 | 4943 | 2.00 | 4.1 % |

**Two takeaways from current state**:
- Cell density: WT > HET by ~25 % (stable across methods)
- Per-cell morphology: no consistent genotype effect (means within 10 %, bin 5+ direction inconsistent)

## Imaging details

- 3 confocal max-projection 2D images, 3168 × 3168 px, 16-bit
- 0.207 µm/px (4.84 px/µm), so each image covers ~655 × 655 µm
- Single channel: microglia marker (likely Iba1 or CX3CR1-GFP)
- n=1 WT, n=2 HET — pilot only, can't draw biological conclusion
