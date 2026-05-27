# Re-review (round 2): choroid-plexus microglia morphology — demo on 3 images

You are a senior quantitative-microscopy reviewer. You reviewed an earlier
version and gave a detailed, correct critique. This is the revised analysis.
**Context: this is an exploratory DEMO to fully characterize 3 images, not a
publication.** I am not trying to make a statistical WT-vs-HET claim with n=3 —
I know I'd need more animals for that. Please critique the EXTRACTION and METHOD
on these images: is what I now measure correct and trustworthy, and is the
exploratory conclusion (HET = de-ramified/fragmented) well-supported on this
data? Read `RESULTS.md` first.

## What I fixed from your last review
1. **Topology counting unified (your #1 issue).** Branch/endpoint counts now come
   from ONE global pass (`clean_topology.merged_branches`, exit≥3 + endpoints)
   assigned to tiles/components. Tile branch counts now SUM to the whole-image
   count (was 2.7–2.9× over-counted by the old dilated-degree≥3 count). See
   `pipeline.py` (single source of truth).
2. **One morphotype clustering (your #2 issue).** The canonical clustering uses
   SHAPE/topology features only (no coverage, no abundance). Coverage-inclusive
   clustering is kept ONLY as a labelled sensitivity. Maps/spot-check/JSON all use
   the canonical labels now.
3. **Denominators separated (your #3/#7).** Densities are reported per tissue-tile
   area AND per foreground AND per 100 µm skeleton. Per-foreground was indeed
   inconsistent across the 2 HET; per-tissue-area is clean and shows HET also has
   lower process abundance — reported as a SEPARATE finding from fragmentation.
4. **Pre-registered fragmentation score (your #6)**: z(ep/br)+z(ep/100µm)
   −z(branch/100µm)−z(mean seg len), summarized per image (mean, hotspot %).
5. **Segmentation sensitivity (your #4/#11)**: re-binarized + re-skeletonized
   under otsu×0.7, otsu, local-adaptive, hysteresis, frangi. Fragmentation
   direction holds in 4/5 (frangi over-fragments everything and saturates);
   abundance↓ holds in 5/5. Also: a hysteresis low-threshold set to capture faint
   processes FLOODS — they sit at background intensity (a real SNR limit).
6. **Grid robustness (your #5/#8)**: fragmentation-hotspot enrichment in HET holds
   across 5 grid-origin offsets + a sliding window (6/6), and tile size {31,41,62}
   × k {3,4,5}.
7. **Wording (your #12)**: reframed as exploratory/demo; "DAM-like" morphology not
   "DAM"; confound "controlled, not ruled out"; thickness = apparent width;
   no significance claims.

## Pipeline (brief; full code in `code/`)
normalize(1–99.5pct) → Gaussian σ=1 → segment (default otsu×0.7) → close → remove
<20px → skeletonize → clean_topology (drop <12px components, prune spurs ≤8px,
merge junctions exit≥3, break tree loops at dimmest px) → global topology assigned
to 200-px tiles / connected components → shape-only k-means morphotypes +
fragmentation score → replicate-consistency screen → segmentation & grid
sensitivity.

## Data & figures in this zip
- `RESULTS.md` (read first), `aggregate.json` (whole-image negative baseline)
- `data/region_features.csv`, `region_morphotype.json` — corrected tile analysis
- `data/cc_features.csv`, `cc_morphotype.json` — connected-component analysis
- `data/F_*_segments.csv` — per skeleton segment
- `images/<stem>_morphotype_map.png` (canonical), `composition_bar.png`,
  `fragmentation_distribution.png`
- `images/<stem>_cc_map.png`, `cc_size_distribution.png`
- `images/segmentation_compare_crop.png` — same HET window under each method
- `images/spot_check_WT_vs_HET.png`, `images/CURRENT_<stem>.png`
- `code/` — pipeline, clean_topology, region_morphotype, cc_morphotype,
  segmentation_compare, grid_robustness, spot_check, analyze

## Please critique (focused on these 3 images, not publishability)
1. Is the corrected topology counting now sound? Any remaining bias in the
   spur-prune / exit≥3 merge / loop-break rules (esp. loop-breaking in a 2D
   projection — could real overlaps be cut)?
2. With topology fixed, is the **fragmentation score** the right primary readout?
   Better composite, or should I report its parts separately?
3. Given the segmentation sensitivity (4/5 methods agree; faint recovery is
   SNR-limited), how much should I trust the fragmentation signal as biological
   vs residual segmentation effect? Is a pixel classifier worth building for a
   demo, or overkill?
4. Connected-component unit: is treating dense clumps as single units, and
   reading the component-size distribution, defensible? Does the region↔component
   convergence add real weight?
5. Anything else you'd measure on THESE images to understand them better
   (e.g., spatial statistics on the foci — Moran's I, hotspot clustering)?
