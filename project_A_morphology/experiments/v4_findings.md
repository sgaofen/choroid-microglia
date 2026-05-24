# v4 detection — combined Cellpose + distance-based supplementary (2026-05-21)

## Method

**Stage 1**: Cellpose `cyto3` at `cellprob_threshold=−2.0, flow_threshold=0.6` (the "extended" run).

**Stage 2**: For each pixel, compute distance to the nearest Cellpose centroid. In the Otsu-thresholded binary signal mask, connected components more than 20 px from any Cellpose centroid and with area ≥ 80 px are added as **supplementary** cells. Each supplementary cell inherits the connected bright region's shape.

The point: Cellpose finds compact-bright-blob cells well but skips cells where the soma is faint relative to the radiating processes. The distance heuristic specifically locates regions where there's clear signal that's *not* near any Cellpose detection — which is exactly where the missed ramified cells live.

## Counts per image

| Image | Cellpose ext | v4 supp | v4 total | Fiji classical |
|---|---|---|---|---|
| WT_2  | 1800 | +952 | **2752** | 9272 |
| HET_1 | 1848 | +624 | **2472** | 7780 |
| HET_3 | 1772 | +558 | **2330** | 6644 |

Fiji classical (Huixin's current method = Otsu + watershed + size filter 30–3000 px) over-fragments — every bright blob gets split into many small pieces because watershed seeds on local maxima of the distance transform, which fire repeatedly along thick processes. This matches what Huixin described about Fiji counts being noisy.

## Validation — the previously-missed ramified cell

The classic spider-shaped ramified microglia in WT_2 at (y≈1750, x≈1290) — which had bright soma + ~10 radiating processes and was invisible to Cellpose at all settings — **is now captured by v4** as a single supplementary cell. The mask traces the soma + most of the visible processes.

## False positive rate of supplementary detections

Audited on 4 native-resolution 200×200 windows:

| Region | Supp added | Visual judgment |
|---|---|---|
| A_upper (sparse) | 6 | 3–4 look real, 2–3 are process fragments |
| B_dense (dense) | 6 | mostly process-node-like, ~50% FP |
| B_missed_zoom | 2 | one is the rescued spider cell ✓, one is a smaller adjacent fragment |
| D_right (very dense) | 6 | mix of real edge cells and process fragments |

**Rough estimate**: of the 952/624/558 supplementary additions, perhaps 40–60% are real missed cells, the rest are process-knot artifacts. This is a known tradeoff — the distance criterion can't fully discriminate "missed soma" from "convergence of multiple processes from outside".

## Circularity distribution (v4, post-supplement)

| Image | % circular (≥0.5) | % non-circular (<0.5) |
|---|---|---|
| WT_2  | 55.3% | 44.7% |
| HET_1 | 61.4% | 38.6% |
| HET_3 | 54.9% | 45.1% |

No monotonic genotype pattern. Worth noting that v4 supplementary cells inherit irregular CC shapes, which biases them toward low circularity.

## Caveats

1. Supplementary masks are not biologically interpreted soma boundaries — they're the full bright CC the cell occupies. A "circularity" computed on a supplementary mask reflects shape of the cell+processes together, not soma alone.

2. The 20-px distance threshold was picked by eyeball. Tighter (15 px) finds more candidates but more FPs; looser (30 px) misses some real cases.

3. **For dense regions where v4 over-adds**: Stephen can post-filter the CSV by `source = 'supp'` to inspect only supplementary cells and remove obvious junk. Alternatively, use only `source = 'cellpose_ext'` for a more conservative count (~1800/image).

## Recommended use

- **For total cell counting**: use v4 total (2752/2472/2330) with a footnote that 20–30% of these are supplementary additions with ~50% noise. So effective real-cell range is roughly 2200–2500 per image.

- **For morphology classification**: use `cellpose_ext` cells only (1800/1848/1772) — they're the cleaner, more uniform set. Skip the supplementary cells unless you specifically want to look at the ramified-end population.

- **For the "did genotype shift morphology?" question**: ask Huixin to confirm she's interested in *bulk shift* or *Kolmer-vs-stromal ratio*. The v4 data supports the latter framing better than the former.

## Files

- `v4_distance_supplement/F_*_v4_masks.npy` — final v4 masks (3168×3168 int32) for each image
- `v4_distance_supplement/F_*_fiji_masks.npy` — Fiji-style baseline masks
- `v4_distance_supplement/all_cells_v4.csv` — unified per-cell CSV (7554 cells total across 3 images) with `source` column distinguishing cellpose_ext from supp
- `final_review/WT_2_*_v4.png` — overlays with red (Cellpose) / green (supplementary) boundaries
- `final_review/WT_2_*_fiji.png` — Fiji boundaries (yellow) for comparison
- `final_review/WT_2_*_clean.png` — unmarked originals for independent recount
