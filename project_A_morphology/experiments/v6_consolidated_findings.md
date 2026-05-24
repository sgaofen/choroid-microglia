# Consolidated findings post-GPT-Pro review (2026-05-21)

After a full GPT Pro consultation flagged a critical issue with the v4 analysis, I ran four follow-up analyses. Big picture: the previous v4 morphology comparison was confounded; the new patch-level and collar-based analyses show a clear, robust WT > HET ramification signal.

## TL;DR

1. **v4 morphology comparison is invalid as previously reported.** Supplementary detections are 34.6 % of WT_2 cells but only 24–25 % of HET cells, and supplementary objects have radically different morphology (median area 200 vs 400 px, circularity 0.27 vs 0.65) — they are process fragments, not cells. Mixing them with Cellpose detections in a distribution comparison manufactures genotype shifts.

2. **Cell density is not the genotype signal.** Once normalized by tissue area, WT_2 / HET_1 / HET_3 have ~4250 / 4365 / 4165 cells per mm² — essentially flat.

3. **Patch-level signal complexity IS the genotype signal.** All four no-per-cell metrics (signal area fraction, skeleton length, endpoints, branchpoints per mm² of tissue) rank WT_2 > HET_1 > HET_3 monotonically. WT_2 has ~40 % more branchpoints per mm² tissue than HET_3.

4. **Collar-based process count (per-cell, stable definition) confirms the same pattern.** WT cells average 2.28 process branches crossing a 1.5 µm collar around the soma vs 1.75–1.79 in HET. WT has ~4× the fraction of highly-ramified cells (6+ branches: 5.1 % WT vs 1.2–1.5 % HET).

So the underlying biological signal — WT more ramified than HET — appears real once the methodology is fixed. With n = 1 WT and n = 2 HET, it's still a pilot, not a conclusion.

## Task 1 — Verifying the v4 contamination

```
WT_2:   952/2752 cells supplementary  =  34.6 %
HET_1:  624/2472 cells supplementary  =  25.2 %
HET_3:  558/2330 cells supplementary  =  23.9 %
```

Per-source median morphology, WT_2:

| Source | Median area | Median circularity | Median eccentricity |
|---|---|---|---|
| Cellpose_ext | 406 px | 0.65 | 0.83 |
| Supplementary | 210 px | 0.27 | 0.92 |

Supplementary objects are half the area, far less circular, and much more elongated than Cellpose cells. They are clearly a different object class — bright connected components anchored at process intersections rather than at compact somata.

**Decision: drop supplementary detections from any per-cell morphology distribution analysis.** They remain useful as evidence that ramified cells with faint somata exist in the tissue, but they cannot be statistically mixed with Cellpose detections.

## Task 2 — Tissue-area-normalized density

```
Image    Tissue area     Cells (ext only)    Cells per mm² tissue
WT_2     0.424 mm²        1800                4248
HET_1    0.423 mm²        1848                4365
HET_3    0.425 mm²        1772                4165
```

Tissue areas are nearly identical across the three images. Cell density per tissue area is flat. **Density is not the genotype signal.**

## Task 3 — Patch-level genotype metrics

These metrics are computed on the whole-image signal mask within tissue. They do not require per-cell assignment.

| Metric | WT_2 | HET_1 | HET_3 |
|---|---|---|---|
| Signal area fraction (%) of tissue | **8.42** | 7.15 | 6.41 |
| Skeleton length per mm² tissue | **373,196** | 300,905 | 289,127 |
| Endpoints per mm² tissue | **29,350** | 26,202 | 25,434 |
| Branchpoints per mm² tissue | **40,195** | 31,394 | 27,761 |

Monotonic WT > HET_1 > HET_3 across all four metrics. The branchpoint metric shows the strongest separation: WT has ~28 % more branchpoints per mm² than HET_1 and ~45 % more than HET_3. This is the "WT more ramified than HET" signal in a form that doesn't depend on per-cell judgment at all.

## Task 4 — Collar-based process count

GPT's principled fix for the unstable "process count" metric: for each Cellpose_ext cell, count distinct skeleton branches crossing a fixed-physical-size collar (1.5 µm = 7 px) around the soma. The metric is invariant to whether the Cellpose mask happens to include 1–2 px of proximal process.

```
Bin      WT_2          HET_1         HET_3
0        149 ( 8.3%)   243 (13.1%)   253 (14.3%)
1        562 (31.2%)   700 (37.9%)   609 (34.4%)
2        445 (24.7%)   472 (25.5%)   459 (25.9%)
3        297 (16.5%)   257 (13.9%)   264 (14.9%)
4        187 (10.4%)   105 ( 5.7%)   116 ( 6.5%)
5         68 ( 3.8%)    48 ( 2.6%)    44 ( 2.5%)
6+        92 ( 5.1%)    23 ( 1.2%)    27 ( 1.5%)

Mean     2.28          1.75          1.79
```

WT distribution is right-shifted (more processes per cell) on every bin from 3 upward. The 6+-process tail is ~4× more populated in WT than either HET. Mean process count: 2.28 vs 1.75/1.79 — about half a process per cell difference.

This is the same WT > HET pattern from Task 3, expressed at the per-cell level.

## Task 5 — Annotation set prepared for the PI

14 blinded 448×448-px crops stratified across:

- 3 × sparse (low cell density, isolated cells)
- 4 × dense (heavy tangle)
- 3 × edge (mixed tissue + background)
- 2 × faint_soma (high skeleton density, low bright-peak fraction — ramified-cell candidates)
- 2 × mixed (everything else)

All saved as both 16-bit TIF (for Cellpose-SAM fine-tuning later) and 8-bit PNG (for visual annotation). Folder: `experiments/huixin_annotation_set/`. Two manifests:

- `manifest_blinded.csv` — given to PI, no genotype info
- `manifest_unblinded.json` — kept locally, used after annotation to map back

Ready to feed into Cellpose-SAM fine-tuning the moment annotations come back.

## Task 6 — μSAM install

Blocked. micro-sam requires Python ≥ 3.10; the project venv is 3.9.6. Deferred until a new env is created.

## What this means for the project

The signal Huixin asked about — "is there a morphology shift between WT and HET?" — appears to be present in the data at the population level. WT has a more complex, more ramified microglia network than HET, by every metric I trust:

- Signal area, skeleton length, branchpoints, endpoints per mm² tissue
- Per-cell collar branch count distribution

With n = 1 / n = 2, this is a pilot result. But it's now expressed in a defensible analytic framework, not in the v4 confounded analysis.

## Next concrete actions

1. **Send the 14 annotation crops to Huixin** when she replies to the email. Even 50–100 hand-labeled cells from her gives us a Cellpose-SAM fine-tune target that should fix the faint-soma miss problem at its root.

2. **Pivot the project framing from per-cell morphology bins to a two-metric report:** (a) patch-level skeleton/branchpoint density per mm² tissue, (b) per-cell collar-branch distribution among Cellpose_ext-detected somata. The 0–5 ramification ordinal scale becomes a derived summary, not the primary measurement.

3. **Re-run all the above on real data when it arrives**, with the fine-tuned Cellpose-SAM in the soma-detection slot.

## Files

- `v6_tissue_normalized.json` — per-image tissue area and patch metrics
- `v6_collar_metric/collar_branch_counts.json` — per-cell collar branch counts (cellpose_ext only)
- `huixin_annotation_set/` — 14 crops + manifests
- `v4_distance_supplement/` — kept for archive but morphology comparison is **deprecated**; use cellpose_ext only going forward
