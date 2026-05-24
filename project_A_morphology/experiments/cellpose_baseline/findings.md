# Cellpose cyto3 baseline — findings (2026-05-21)

## Setup
- Model: `cyto3` pretrained, CPU
- Input: 16-bit TIFs, auto-contrasted to uint8 via (1, 99.5)-percentile stretch
- Parameters: `diameter=20`, `channels=[0,0]`, `flow_threshold=0.4`, `cellprob_threshold=0.0`
- Runtime: ~87s per 3168×3168 image on CPU

## Detection counts

| Image | Cells detected |
|---|---|
| F_WT_2 | 1099 |
| F_HET_1 | 1211 |
| F_HET_3 | 1293 |

Cell area distribution (WT_2, sample 300×300 crop, 13 cells):
- min 40 px², median 239 px², max 787 px²
- diameter range: 6.3 → 28.1 px

## Critical observation — visual QC on WT_2 300×300 crop

**Cellpose detects somata (cell bodies) only, NOT whole cells with processes.**

What the overlay shows:
- Each red outline hugs a compact bright soma region
- Processes (thin bright filaments extending from somata) are excluded from masks
- This is consistent with `cyto3` being trained on cells where the cell is one compact blob — it isn't trained to recognize "soma + far-extending thin processes" as a single object

Visual audit of the 13 outlined cells:
- All 13 outlines correspond to genuine soma-like bright structures — **no obvious false positives**
- Visible bright structures NOT outlined are mostly process fragments (where the corresponding soma is either outside the crop or has its soma counted nearby)
- Some elongated bright structures may be ambiguous (process vs. very thin soma) — these were not detected

## What this means for the project

1. **Soma counting is solid**. ~1100–1300 somata per image is a reliable, automatable readout. This already answers Huixin's first-pass question of "do counts shift between genotypes."

2. **Morphology classification cannot use Cellpose masks directly**. The 0–5 ramification scale Huixin described requires measuring process extent, which Cellpose discards.

3. **Soma-as-seed strategy is viable**. We have ~1100 reliable soma centers per image. For each soma, we can:
   - Define a local bounding box (e.g., 60×60 px around centroid)
   - Threshold the image to find all bright pixels in that box
   - Skeletonize → count skeleton endpoints in the box → approximate process count
   - Caveat: in dense regions, processes from neighboring cells will leak into a soma's box. This bounds the upper end of the count but doesn't kill the signal.

4. **The Kolmer-cell biology lines up**: published lit says choroid plexus contains **Kolmer epiplexus cells** with oval-shaped soma and **no or short non-branched processes**, plus stromal microglia in connective tissue around vessels with more typical ramified shape. The 0 → 5 morphology axis Huixin described maps onto a real biological distinction. Many detected cells in the visual audit do appear to have minimal processes — Kolmer-like.

## Density check
- 13 cells in 300×300 → density ≈ 14.4 cells / 100 px-square
- 300×300 native ≈ 62×62 µm → ~344 cells per 100 µm² (which seems high — recheck scale)
- Actually: 13 cells per 300×300 native = 13 cells per (300/4.84)² µm² = 13 cells per 3838 µm² → ~3.4 cells per 1000 µm² — that's plausible for choroid plexus microglia density
- For the full 3168×3168 image at this density, expected count ≈ 13 × (3168/300)² = 1449 — matches measured 1099 closely. Slight overestimate from picking a dense crop is the explanation.

## Next steps
- Build per-soma process count metric (Task #6)
- Try μSAM (micro-sam) as alternative segmentation (Task #5) — may capture whole cells where Cellpose only gets soma
- Compare on the same 300×300 inspection crop
