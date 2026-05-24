# Choroid plexus microglia analysis — review package

This bundle is a snapshot of a stuck research task. Read `PROMPT.md` first — it's the actual ask. Everything else is supporting context.

## Folder map

- `PROMPT.md` — the question to answer
- `01_samples/` — three whole-image previews (auto-contrasted PNG, source TIFs are 16-bit 3168×3168). Two genotypes: WT (wildtype, 1 sample) and HET (heterozygous, 2 samples). All from the Lehtinen lab (Boston Children's), already published, safe to share.
- `02_crops/` — native-resolution crops from WT_2 with three detection outputs side-by-side
  - `*_clean.png` — unmarked original (use to count by eye)
  - `*_v4.png` — best current detection: red = Cellpose extended, green = distance-based supplementary
  - `*_dots.png` — same detection as centroid dots only (no boundary outlines)
  - `*_fiji.png` — Fiji classical threshold + watershed baseline (yellow boundaries)
  - `rubric_gallery.png` — labeled grid of canonical example cells
- `03_data/all_cells_v4.csv` — per-cell metrics for all 3 images (7554 cells total): area, circularity, eccentricity, solidity, source flag (cellpose_ext vs supp)
- `04_writeups/` — markdown writeups in chronological order
  - `01_initial_findings.md` — first Cellpose run
  - `02_personal_audit.md` — manual scan of native-resolution regions
  - `03_v4_method.md` — final pipeline + Fiji comparison
  - `04_open_questions_for_PI.md` — what we plan to ask the wet-lab PI

## TL;DR detection numbers per image

| Pipeline | WT_2 | HET_1 | HET_3 |
|---|---|---|---|
| Cellpose cyto3 (default threshold) | 1099 | 1211 | 1293 |
| Cellpose cyto3 (cellprob_threshold=−2) | 1800 | 1848 | 1772 |
| **v4** = above + distance-based supplementary | **2752** | **2472** | **2330** |
| Fiji classical (Otsu + watershed + size 30–3000) | 9272 | 7780 | 6644 |

Image resolution: 4.84 px/µm, so a 3168×3168 image is ~655 × 655 µm of tissue per organ.
