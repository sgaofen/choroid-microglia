# Sample data — visual inspection notes (2026-05-21)

Cached observations from looking at the three TIFs and their auto-contrasted PNG previews.

## File inventory

| File | Dimensions | dtype | min | max | p99 | p99.9 |
|---|---|---|---|---|---|---|
| `F_WT_2` | 3168 × 3159 | uint16 | 0 | 52085 | 40559 | 46839 |
| `F_HET_1` | 3168 × 3168 | uint16 | 0 | 52646 | 39402 | 45337 |
| `F_HET_3` | 3168 × 3168 | uint16 | 0 | 52995 | 39896 | 45660 |

All three are well below saturation (max ~52000 of 65535). Dynamic range is usable end-to-end. No clipping.

## Visual observations

- All three show the characteristic lobulated whole-mount choroid plexus shape — looks like a hand or a stylized leaf, with several rounded lobes branching from a stem.
- Microglia (C2 channel) appear as small bright structures distributed densely across the tissue.
- At preview resolution (1024 px) individual cells are 5–10 px wide; at native 3168 px they are 15–30 px — small but tractable for instance segmentation.
- Cells overwhelmingly appear elongated with visible processes; very few obviously-round cells visible at this resolution. The "0 = round" bin may be sparse in healthy tissue.
- Density: visually estimating, several hundred to ~2000 cells per image. Comparable to the pupa-counter dense-region scale.

## Cross-sample notes

- WT_2 is the brightest; HET_1 is the dimmest; HET_3 is intermediate. This could be (a) biological signal from genotype, (b) staining or acquisition variability, (c) sample preparation differences. With n=1 WT and n=2 HET, no biological inference is possible — three images are for method development, not biological readout.
- Tissue shapes differ — they are different physical samples, not different views of the same organ.
- The tissue boundary against background is clean and high-contrast; an organ mask is easy to derive from a coarse threshold + morphological close.

## Practical implications for Project A

- Single channel keeps things simple — start without worrying about multi-channel composites.
- Resolution is fine (0.207 µm/px); no need to downsample for detection.
- Background outside tissue is dark; intra-tissue background varies but is consistently lower than cell signal — a tissue-mask-then-detect-inside approach should work.
- The "0 (round)" extreme may be empty in healthy tissue, so the morphology distribution will likely be a unimodal shape centered in the ramified end. Disease shifts toward 0 would be tail enrichment, not bulk movement.
