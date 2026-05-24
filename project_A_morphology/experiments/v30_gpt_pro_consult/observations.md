# Stephen's 3 cell examples — over-split pattern analysis

Stephen identified 3 cell images where the v29 algorithm over-splits elongated structures into multiple "cells". Need expert input.

## Example 1 (example_1.png) — "vertical long process"
- Visual: 1 clear soma + 1 short Y in middle/lower + 1 very long thin process extending up to top + small fork at top
- Algorithm split: 4 cells
- My honest visual judgment: likely **1 cell** with 1 unusually long primary process (microglia processes can extend 30-50 µm)
- But 2-3% chance the bright spots along the long process are Kolmer cells in different z-planes

## Example 2 (example_2.png) — "horizontal long structure"
- Visual: long horizontal bright structure ~70-100 px = 15-21 µm + 1 small separate fragment above
- Algorithm split: probably 3-5 cells along the horizontal length
- My honest visual judgment: likely **1-2 cells** along a vessel
  - Choroid plexus stromal microglia frequently align along blood vessels
  - Could be 1 elongated cell OR 2 cells touching end-to-end
- The small upper fragment we don't care about

## Example 3 (example_3.png) — "two bright spots connected"
- Visual: 2 distinct compact bright "soma-like" regions, connected via skeleton, with short processes radiating
- Algorithm split: probably 2-4 cells (correct might be 2)
- My honest visual judgment: most likely **2 cells touching** at their processes
- But could be 1 elongated cell with two thick regions

## Common pattern across all 3

ALL three are ELONGATED structures where my algorithm over-splits because:
- h_maxima finds local peaks in distance transform
- Long structures have multiple thickness peaks along their length
- Each peak becomes an independent soma core
- Watershed splits the structure into many "cells"

## Algorithm details (current v29)

```python
binary = signal > Otsu * 0.7
binary_closed = closing(binary, disk(2))
dist = distance_transform_edt(binary_closed)
dist_smooth = gaussian(dist, sigma=1.0)
h_max = h_maxima(dist_smooth, h=1.5)
seed_mask = h_max & (dist_smooth >= 2.5)
markers = label(seed_mask)
labels = watershed(-dist_smooth, markers, mask=(dist_smooth >= 2.0))
# Then per-soma: collar dilation 3px, BFS endpoint attribution
```

## Stephen's preference

- He's NOT an expert (his words: "我对这个根本不是专家")
- Will go in-person to PI Huixin Xu to learn cell identification
- Wants algorithm improvement BEFORE that visit
- Acknowledges some structures are genuinely ambiguous

## Question to ask GPT Pro

How to distinguish in single-channel 2D max projection:
- (a) ONE elongated cell with a long primary process
- (b) MULTIPLE small cells aligned along a vessel
- (c) ONE cell with multiple thick regions (Type A)
- (d) TWO cells touching at processes (Type B)

Specifically: how to suppress h_maxima from generating multiple seeds inside a single elongated cell, while still detecting genuinely separate adjacent cells?

Possible approaches to discuss:
1. Higher thickness threshold (`dist >= 4` instead of `2.5`) — risks missing small Kolmer cells
2. Skeleton-graph-based seed merging: if two seeds connected via thin (≤3 px) skeleton path → same cell
3. Use Hessian eigenvalues to distinguish "tube-like" (process) vs "blob-like" (soma)
4. Sholl analysis at each seed location to test if it has "soma signature"
5. Accept this as fundamental limit of 2D max projection
