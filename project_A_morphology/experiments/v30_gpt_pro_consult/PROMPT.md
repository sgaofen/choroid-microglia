# Prompt for GPT Pro — Round 3

Round 3 consultation on the same choroid-plexus microglia analysis. Previous 2 rounds we discussed pipeline architecture and skeleton/topology methods. We've now reached a new failure mode that the user (not a biologist) caught visually that we need help with.

**Read `observations.md` for the specific 3 example cells and my current pipeline. Then look at `example_1.png`, `example_2.png`, `example_3.png`.**

## The problem in one sentence

Elongated bright structures in the 2D max projection — whether a single ramified microglia with a long primary process, or multiple small cells aligned along a blood vessel, or two cells touching at their processes — all get **over-split** into 2-4 "cells" by my h-maxima soma detector, because thickness varies along their length and each local maximum becomes an independent seed.

## What I've already tried in this codebase

```
Soma seeding methods tried:
  - peak_local_max(min_distance=8) → over-split everywhere (28% in close groups)
  - peak_local_max(min_distance=14) → less over-split but misses Kolmer cells
  - h_maxima(h=1.5) + dist≥2.5 + watershed (current v29) → still over-splits elongated cells
  - Cellpose pretrained → misses faint-soma ramified cells (different failure mode)

Skeleton cleanup tried:
  - filter CC < 30 px (removes isolated noise) — good
  - spur prune ≤ 3 px (removes tip+junction artifacts) — good
  - Larger spur prune (≤10 px) — removed real branches, abandoned

Counting tried:
  - 1-px collar boundary CC count → under-count
  - 5-px band CC count → over-count from skeleton gaps
  - Graph: edges crossing collar → still misses pass-through cases
  - Graph: BFS endpoint attribution (v28/29) → matches visual for most cells
```

## The 3 cells (in attached PNGs)

**example_1.png — "vertical long-process cell"**
My visual judgment: 1 cell with a 30-50 µm primary process extending upward
Algorithm gives: 4 cells
The long process has 2-3 local thickness peaks that become independent seeds

**example_2.png — "horizontal long structure"**
My visual judgment: 1-2 cells, likely aligned along a blood vessel (which is a known biological pattern in choroid plexus stromal macrophages)
Algorithm gives: 3-5 cells
Same failure mode

**example_3.png — "two bright soma-like spots connected"**
My visual judgment: probably 2 cells touching at processes, but could be 1 elongated cell
Algorithm gives: 3-4 cells
This case is genuinely ambiguous even to me

## What I'm asking

Give me your honest, specific advice on ONE or BOTH of these:

### Q1: Algorithmic fix for h-maxima over-splitting

Possible approaches I've considered (rank/reject and add yours):

A. **Higher thickness threshold** (`dist_s ≥ 4` instead of `2.5`) — simple, but risks missing real small Kolmer cells (which have soma thickness ~3 px)

B. **Skeleton-graph-based seed merging**: 
- For each pair of h-maxima seeds, find their geodesic distance along the skeleton
- If the skeleton path between them passes through only thin regions (dist_s consistently < some threshold), merge them
- If the path crosses through another thick region (= real second soma), keep separate
- This essentially asks "is there a continuous thick spine connecting these, or are there alternating thick/thin segments?"

C. **Hessian eigenvalue test at each seed**:
- Compute local Hessian eigenvalues at each seed location
- Tube-like structure: one large eigenvalue, one small → reject as process artifact
- Blob-like structure: both eigenvalues similar → keep as real soma
- This uses vesselness-style discrimination

D. **Sholl-signature test at each seed**:
- For each h-maxima seed, do a tiny Sholl analysis (count skeleton crossings at radii 5, 10, 15 px)
- A real soma has rotationally symmetric crossing pattern
- A process thickening has linear pattern (crossings only along the process axis)

E. **Accept the limit and report differently**:
- Detect "elongated structures" (length/width ratio > some threshold)
- Don't try to split them; report as "1 elongated unit, possibly multi-cell"
- Down-weight these in per-cell statistics

### Q2: Specifically for vessel-following microglia

The example_2 case is biologically real and common — choroid plexus stromal macrophages line blood vessels. Algorithmically:
- Without a vessel channel marker, can we distinguish "vessel-following cells" from "single cell along vessel" in single-channel images?
- Are there published methods that handle this?

## What I do NOT need

- Reminders to fine-tune Cellpose with annotations (already planned)
- Reminders that 2D max projection has fundamental limits (acknowledged)
- General advice; want specific algorithm recommendations
- Summary of what I wrote
- "Talk to your PI" — already doing that

## Format I'd like back

For Q1: rank A-E with reasoning, plus any approach I missed. For B specifically, want concrete pseudocode if you'd recommend it.
For Q2: yes/no + reference if exists.
