# Project A — Microglia morphology classification

## Goal

For each confocal image of choroid plexus tissue, detect every microglia and assign each one a morphology label on a 0–5 ordinal scale.

## Morphology scale

- **0** — pure round cell body, no processes
- **5** — fully ramified, many long thin processes
- **1–4** — intermediate states between the two extremes

Biological morphology is continuous; the discrete bins are a human-imposed quantization so populations can be compared. Choice of bin count and boundaries is part of the design space.

## Downstream use

Per-image output is a count vector across the six bins (plus total cell count). The biological question is whether genotype or condition shifts the distribution — for example, deramification (mass shift toward 0) in disease models, or specific intermediate-bin enrichment.

## Image properties

- Source format: Zeiss CZI (multi-channel volumetric), exported as max-projected 16-bit TIF
- 20× objective, tile-scanned to cover the whole organ; one image per organ
- Resolution: 4.84 px/µm (0.207 µm/px); field is ~655×655 µm per stitched tile
- Typically 3–4 channels per CZI (nuclei, vasculature, microglia, sometimes a functional marker); the three sample images here are **single-channel C2 only** (microglia)
- Whole-mount tissue, not cell culture — background varies with vessel density, folds, edges
- Cell density observed in samples: high (visually, hundreds to low thousands per image)

## Sample data

In `data/raw/` — three 16-bit TIFs, 3168×3168 px each, ~20 MB:

- `F_WT_2` — wildtype female, sample 2 (brightest of the three)
- `F_HET_1` — heterozygous female, sample 1 (dimmest)
- `F_HET_3` — heterozygous female, sample 3

`data/previews/` holds 1024-px auto-contrasted PNG renders for quick visual inspection.

## Prior art tried in lab

- **Fiji / ImageJ + circularity threshold** — semi-automated. Human picks a circularity cut (default ~0.5) and the script reports counts above/below. Sensitive to background, manual per-image tuning.
- **Cellpose** — works on uniform cell-culture images but fails on whole tissue with variable background, vasculature, and edge effects.
- **A pre-COVID commercial ML demo** — "crashed" on these images per Huixin.

## Acceptance criteria (first pass)

- Detects substantially all visible cells across the three sample images, in one fixed configuration (no per-image manual tuning)
- Per-cell morphology label assigned via some reproducible scheme (continuous metric → binned)
- Output is a CSV per image: one row per cell with centroid, area, morphology score, bin label
- Visual QC overlay: cells shown on top of the original, color-coded by bin

## Open design questions

- Detection method: Cellpose retrained on a small tissue-labeled subset, classical (DoG + watershed + threshold), or hybrid
- Morphology metric: circularity is one option, but solidity, skeleton branch count, or hull-area ratio may discriminate the 1–4 range better than circularity alone
- Bin boundaries: derive from sample distribution or fix a priori
- Handling tissue mask: separate organ-shape mask first, only detect inside it
