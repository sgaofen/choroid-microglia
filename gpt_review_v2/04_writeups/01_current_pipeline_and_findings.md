# Final pipeline + honest findings — 2026-05-22

## The pipeline (no Cellpose, pure algorithm)

```
1. Normalize: percentile [1, 99.5]
2. Smooth: Gaussian σ=1
3. Threshold: > Otsu × 0.7   ← key fix; previous Otsu cut dim distal processes
4. Closing: disk(r=2)
5. Skeletonize
6. Spur prune: remove branches < 5 px from endpoint to nearest branchpoint
7. Filter: remove skeleton CCs < 8 px (kills FP noise speckles)
8. Soma detection:
   a. Erode binary with disk(r=2) — only thick somata survive
   b. Distance transform inside eroded
   c. Local maxima as seeds (min_distance=8, threshold=1.0)
   d. Watershed to separate touching somata
   e. Filter somata < 10 px
9. Per-soma process count:
   a. Dilate soma boundary by 1 px → ring
   b. Skeleton CCs touching the ring = independent processes emerging
   c. Count CCs
```

## Final cross-genotype results

| Image | n_somata | mean processes | bin 0 (no processes) | bin 5+ (ramified) |
|---|---|---|---|---|
| WT_2  | 6163 | 1.98 | 6.8 % | 2.8 % |
| HET_1 | 5131 | 1.82 | 8.5 % | 2.5 % |
| HET_3 | 4943 | 2.00 | 7.3 % | 4.1 % |

## Honest interpretation

**Two findings:**

1. **Cell density WT > HET (~25 %).** WT_2 has 6163 somata vs HET_1 5131 and HET_3 4943. This direction was stable across every detection method we tried (Cellpose, erosion+watershed at r=2 and r=3, with and without size filter). This is the cleanest signal in the data.

2. **Per-cell morphology is approximately equal between genotypes.** Mean process counts: WT 1.98, HET 1.82–2.00. Bin distributions overlap heavily. HET_3 actually has MORE highly-ramified cells (bin 5+: 4.1 %) than WT (2.8 %), opposite direction. No consistent genotype effect on per-cell complexity.

## What changed from earlier reports

Earlier I reported "WT cells more ramified than HET" with WT/HET ratios up to 1.5×. That was an **artifact of using too strict a threshold** (Otsu alone). HET images are slightly dimmer (background autofluorescence + possibly different staining intensity); Otsu's global cut excluded more of HET's dim distal processes than WT's. Once threshold was relaxed to Otsu × 0.7, the supposed per-cell difference disappeared.

This is a textbook case of **methodology-driven false positive** in image analysis.

## Threshold choice — why Otsu × 0.7

Tried several thresholds on the visible "spider cell" with known many processes:

| Threshold | Binary fragments | Skeleton continuity |
|---|---|---|
| Otsu (1.0) | Many gaps in processes | Heavy fragmentation |
| Otsu × 0.7 | Bridge most gaps, processes continuous | **Best balance** |
| Otsu × 0.6 + close r=3 | Slight over-flooding | Acceptable but starts merging cells |
| Sauvola local | Massive over-flooding | All signal merged into one blob |
| Shipley pct[20, 90] + Otsu × 0.7 | Over-inflates HET (amplifies noise) | Tested, rejected |

**Otsu × 0.7 + standard pct[1, 99.5] normalization** is the operating point.

## Validation: tiny-FP problem fixed

Before adding size filter, scattered tiny green endpoints in pure-background regions (Stephen's observation). After **filter CC < 8 px**:

- Endpoints: 6689 → 5009 (33 % reduction)
- Skeleton fragments: 2241 → 1031 (54 % reduction)
- Real cells visually preserved

## Limitations honestly stated

1. **n = 1 WT vs n = 2 HET — pilot only.** All numbers are descriptive, no inference.
2. **Per-cell morphology is sensitive to threshold/normalization choice.** Three different choices gave WT > HET, WT ≈ HET, and HET > WT directions. Without ground truth (Huixin's annotations), we cannot pick the "right" answer.
3. **Cell density signal is most robust** — direction stable across all methods.
4. **Touching cells may be over-split or under-split by watershed.** Some "two cells" might be one cell with elongated soma; some "one cell" might be two cells whose somata fused in erosion step.
5. **Z-stack max-projection gaps remain a partial confound** — closing r=2 mitigates but does not eliminate.

## What to bring to Huixin

- **Report cell density** with appropriate caveats (n=1/2 etc.)
- **Do not report per-cell morphology genotype effect** — data does not support a direction with confidence.
- **Request her annotation** on the 14 stratified crops we prepared. Her judgment is the only way to break the threshold-choice tie.
- **Discuss** whether the biological question is better framed as "cell number" (which we can show) vs "cell shape" (which we currently cannot).

## Files

- `F_WT_2_somata_FINAL.npy`, `F_HET_1_somata_FINAL.npy`, `F_HET_3_somata_FINAL.npy` — final soma masks (pure algorithm)
- `F_*_skel_FINAL.npy` — final skeletons
- `FINAL_results.json` — per-cell process counts
- `FINAL_big_crop.png` — 800×800 visualization on WT_2 showing 525 somata in this field
- `filter_size_compare.png` — visual proof FP filter works
- This document
