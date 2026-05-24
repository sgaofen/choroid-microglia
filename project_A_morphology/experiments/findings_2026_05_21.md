# Project A first-pass findings — 2026-05-21

End-to-end pipeline tried on the three Lehtinen-lab sample images. Goal was to test whether automated microglia detection + process counting is tractable, and to pull out anything worth bringing back to Huixin before scoping the project further.

## Pipeline

1. **Read 16-bit TIF**, normalize via (1, 99.5)-percentile stretch to uint8.
2. **Cellpose `cyto3`** (CPU, diameter=20, default thresholds) — instance segmentation of cell bodies.
3. **Otsu threshold** of Gaussian-smoothed (σ=1) image → global binary signal mask. Signal occupies 6–8% of pixels per image.
4. **Watershed** seeded by Cellpose soma centroids, masked by the binary signal → each soma owns a "territory" of connected signal.
5. **Process count per cell** = number of connected components in (territory − soma) that touch the soma boundary (dilated by 2 px). Components smaller than 3 px are discarded.

Runtime: ~90 s/image on CPU for Cellpose, ~10 s/image for the watershed + counting step. Total under 10 min for all three.

## Results

### Soma detection counts

| Image | Cells | Signal fraction (after Otsu) |
|---|---|---|
| F_WT_2  | 1099 | 8.4 % |
| F_HET_1 | 1211 | 7.1 % |
| F_HET_3 | 1293 | 6.4 % |

### Per-cell process-tree count distributions

| | 0 | 1 | 2 | 3 | 4 | 5+ | median | mean |
|---|---|---|---|---|---|---|---|---|
| F_WT_2  | **43.6 %** | 22.7 % | 19.1 % | 9.9 % | 3.4 % | 1.4 % | 1 | 1.11 |
| F_HET_1 | 35.1 % | 28.9 % | 20.8 % | 10.4 % | 3.9 % | 0.9 % | 1 | 1.22 |
| F_HET_3 | 29.2 % | 30.5 % | 23.7 % | 11.6 % | 4.5 % | 0.5 % | 1 | 1.34 |

### Suggestive pattern (not a result — n is too small)

WT_2 has the largest "0-process" fraction (44 %), HET_3 the smallest (29 %). HET samples have correspondingly more cells with 1–3 processes. Mean process count rises monotonically WT_2 → HET_1 → HET_3 (1.11 → 1.22 → 1.34).

Signal fraction goes the *opposite* direction (WT_2 is brightest), so the count shift is unlikely to be a pure brightness artifact — if anything, WT should be detection-biased upward.

## Visual QC — what I confirmed by eye

On a 300 × 300 native-resolution crop of WT_2:

- **All 13 Cellpose outlines correspond to genuine soma-like bright structures**. No false positives in the QC crop.
- **Cellpose detects somata only, not whole cells with processes**. Outlines hug the compact bright soma; thin filaments extending outward are excluded. This matches the `cyto3` training distribution (compact-blob cells in culture).
- **Process counts on each outlined cell broadly match what's visually countable** at the soma's local neighborhood. A cell labeled "3" has roughly 3 visible filaments emerging; a "0" is a featureless oval; an "8" is the most spider-like soma in the crop.

## Critical caveat — Cellpose under-detects the most ramified cells

In the same QC crop, **one of the most-ramified-looking cells in the field was not detected by Cellpose at all**. Its soma is faint and small relative to its processes, so the detector — trained to find compact bright objects — passes over it. Its bright processes then get assigned by watershed to neighboring cells, inflating their process counts.

**Implication**: the genuinely ramified-end cells (the "5" bin Huixin described) are systematically missed. The current pipeline shows a distribution that probably *over-represents* the Kolmer/oval-cell end and *under-represents* the ramified end. The HET-shift pattern above could reflect real biology, but it could also reflect HET having a brightness/contrast profile that happens to expose more cells to the detector.

## What this tells us about scope

1. **Soma counting is a reliable, automatable readout** (~1100–1300 per image, all visually verified as real). This alone answers "do counts differ between genotypes."

2. **The 0 → 5 morphology axis Huixin described lines up with real biology in this tissue**. Published lit (e.g. on choroid plexus immune cells) describes two coexisting populations:
   - **Kolmer epiplexus cells**: oval soma, no/short processes — these are the "0" bin
   - **Stromal microglia**: ramified, around vessels in connective tissue — these are the upper bins
   The bimodal-looking distribution (large 0-bin + tail) matches this dichotomy.

3. **A single pretrained Cellpose isn't enough**. The fix is one of:
   - **Fine-tune Cellpose on hand-annotated examples** that explicitly include ramified-soma cases (~50 corrected cells in one image should be enough; Cellpose human-in-the-loop GUI is the standard workflow)
   - **Add a complementary detector for faint-soma cells**: scan the binary signal mask for compact-but-dim regions not covered by Cellpose output, treat them as additional soma candidates
   - **Try μSAM (`micro-sam`)** as a second segmenter — Nature Methods 2025, fine-tuned SAM on light microscopy. May handle faint somata better than `cyto3`. GitHub: `computational-cell-analytics/micro-sam`. *Not tried yet — heavy install, deferred.*

4. **For "morphology classification on a 0–5 scale" specifically, MorphoCellSorter (eLife 2024) is the obvious next downstream tool**. It explicitly does *not* do segmentation — it takes per-cell binary masks as input and computes 20 dimensionless morphology indices, then ranks cells along a continuous morphology axis using Andrews plots. The binning into discrete classes is done after ranking, by the user. GitHub: `Pascuallab/MorphCellSorter`. The natural pipeline becomes: Cellpose-soma (or successor) + dilate to capture processes → MorphoCellSorter → ranked morphology score → bin into 0–5.

## Questions for Huixin

- **Confirm the biology**: are you measuring Kolmer cells + stromal microglia together, or only one population? If you want them separated, the "0 process" bin and the "1+" bins map onto that distinction roughly.
- **Confirm the comparison target**: is the comparison WT vs HET per-cell *morphology distribution* (the current framing), or *cell-count shift between Kolmer-like and stromal-like populations* (an alternative framing the data here might support better)?
- **Faint-soma cells**: should they be detected? If yes, we need to fine-tune Cellpose (hand-correct ~50 cells in one image). If they're a small minority that can be skipped, the current pipeline is closer to usable.
- **Ground-truth labels**: do you have any image where you (or someone in the lab) manually counted/classified cells? Even one image with a few dozen counted cells lets us validate the pipeline numerically.

## Files generated

```
experiments/cellpose_baseline/
  F_HET_1_*_masks.npy           # Cellpose soma masks (3168×3168 int32)
  F_HET_3_*_masks.npy
  F_WT_2_*_masks.npy
  findings.md                    # detailed cellpose-only findings

experiments/process_counting/
  WT_2_process_metrics.json      # initial endpoint-based counts (deprecated)
  WT_2_process_tree_counts.json  # final tree-count metric, WT_2 only
  all_images_process_counts.json # all 3 images, current best metric
```

## Tools surveyed

| Tool | Used? | Notes |
|---|---|---|
| Cellpose 3 (cyto3) | yes | soma-only, fast, no training needed |
| μSAM (`micro-sam`) | not yet | Nature Methods 2025, possibly better on faint objects |
| MorphoCellSorter | not yet | eLife 2024, downstream morphology ranking on masks |
| MicrogliaJ | no | Fiji plugin, ImageJ-native auto detection |
| 3DMorph | n/a | requires z-stacks (we have max-projected 2D) |
| PrestoCell | n/a | also z-stack only |
| Fiji + circularity (Huixin's current) | no | semi-manual baseline; could implement as reference for comparison |
