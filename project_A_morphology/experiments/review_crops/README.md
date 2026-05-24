# Review crops — 2026-05-21

9 sample crops (3 per image), each at 500×500 native resolution upscaled 2× for display (1000×1000 PNG).

## What to check on each `_overlay.png`

- **Red boundaries** — extended-Cellpose mask outlines (cellprob_threshold = −2.0)
- **Yellow `#NN`** — global cell ID (corresponds to row in `all_cells_extended.csv`)
- **Colored number underneath** — process count (cyan = 0, light-green = 1, orange = 2, red = 3+)

## How to verify

Open the matching `*_clean.png` (no annotation) side-by-side. Count cells yourself, then compare to the overlay.

Specifically look for:

1. **False positives** — red outlines around things that aren't cells (noise, vessel, artifact). Mark cell_id.
2. **Missed cells** — bright cell-like structures in the clean image with no outline in the overlay. Approximate (y, x) location is enough.
3. **Merged cells** — one outline that should be two adjacent cells. Mark cell_id.
4. **Wrong process count** — the displayed number doesn't match how many processes you see emerging. Mark cell_id and what you'd say it should be.

You can mark corrections directly in `all_cells_extended.csv` by adding a `note` column, or just list cell_ids in a chat reply.

## Counts per crop

| File | Cells detected |
|---|---|
| WT_2_crop1  | 57 |
| WT_2_crop2  | 55 |
| WT_2_crop3  | 57 |
| HET_1_crop1 | 49 |
| HET_1_crop2 | 64 |
| HET_1_crop3 | 60 |
| HET_3_crop1 | 55 |
| HET_3_crop2 | 66 |
| HET_3_crop3 | 63 |

## Crop locations in source image (top-left y, x of 500×500 native window)

```
WT_2  : (1100, 600)  (1600, 1400)  (700, 1800)
HET_1 : (1100, 1200) (1700, 1700)  (500, 800)
HET_3 : (1300, 1100) (1900, 1400)  (600, 1700)
```

## Caveats already known

- Process counts can be deflated when the cell mask grew large enough to swallow proximal processes.
- The process-count number is a quick heuristic; circularity / eccentricity in the CSV may be more reliable per-cell metrics.
- Touching cells may be assigned to one watershed territory; the count for one of them is then inflated.
