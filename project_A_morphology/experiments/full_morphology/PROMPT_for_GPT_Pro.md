# Review request: choroid-plexus microglia morphology extraction & analysis

You are a senior image-analysis / quantitative-microscopy reviewer. I'm a
student building a pipeline to compare **microglia morphology** between two
conditions in **choroid-plexus whole-mount** fluorescence images. Please
critically review whether my **data extraction is correct**, my **analysis
method is sound**, and whether it is **sufficient** — and suggest improvements.
I am NOT confident the topology, length, thickness, or regional division are
done right. Be rigorous and specific.

## Biological context
- Tissue: choroid plexus whole-mount, single channel (Iba1-type microglia marker).
- 3 images: F_WT_2 (control / WT) and F_HET_1, F_HET_3 (disease — an Alzheimer's
  model; the two HET are replicates). The morphological difference is expected
  to be SUBTLE, which is why we need quantification.
- 16-bit TIF, ~3168x3168 px, 0.207 um/px.
- Goal: quantify microglia morphology (process length, thickness, branching
  complexity, shape) and its spatial heterogeneity, to detect WT-vs-HET
  differences. Per-cell instance COUNTING is NOT reliable yet (dense, touching
  processes), so we deliberately focus on morphology metrics that do NOT require
  perfect single-cell segmentation.

## Pipeline (what produced the data)
1. **Binarization**: normalize (1–99.5 percentile) → Gaussian σ=1 → threshold =
   Otsu×0.7 → closing(disk2) → remove objects <20 px. (We tested raising the
   threshold to keep only bright signal: it dropped faint noise but BROKE
   processes into fragments → many spurious endpoints. Lowering it floods
   background. So Otsu×0.7 is a compromise. The "keep faint continuation vs drop
   isolated faint" decision seems unsolvable by a single threshold.)
2. **Skeletonize** (Lee/medial-axis) → 1-px skeleton (precomputed "v29").
3. **Topology cleanup** (clean_topology.py, included):
   - drop skeleton connected-components < 12 px (isolated fragments)
   - prune terminal spurs ≤ 8–10 px
   - branch points = clusters of degree≥3 pixels merged within ~8 px, kept only
     if ≥3 distinct skeleton arms leave the cluster (so a path's terminal is an
     ENDPOINT, never a branch)
   - break loops (microglia are trees): cut each small enclosed loop at its
     dimmest pixel
4. **Per-segment metrics** (via `skan`): each skeleton branch between
   junctions/endpoints → length_um (branch-distance) and thickness_um
   (mean distance-transform value along the branch × 2 × px size = diameter).
5. **Aggregate per image**, normalized by foreground area (mm²): branch density,
   endpoint density, skeleton-length density, mean/median/p90/CV segment length,
   mean/CV thickness, branches per 100 µm skeleton (complexity).
6. **Regional analysis**: tile the image into 400-px (~83 µm) tiles, compute
   per-tile skeleton-length density, endpoint density, mean thickness; report
   the CV across tiles as a spatial-heterogeneity measure (hypothesis: disease
   tissue is more heterogeneous).

## Data included in this zip
- `aggregate.json` — per-image aggregate metrics + `wt_vs_het_deltas.json`
- `<stem>_segments.csv` — every skeleton segment: length_um, thickness_um, type
- `<stem>_regional.csv` — per-tile densities + thickness
- `<stem>_regional_heatmap.png` — spatial skeleton-density map
- `seglen_dist.png`, `thickness_dist.png` — WT vs HET distributions
- `CURRENT_<stem>.png` — raw vs skeleton+branch(green)+endpoint(yellow) overlays
- `clean_topology.py`, `analyze.py` — the actual code

## Please critique specifically
1. **Topology extraction**: is skeleton→(branch/endpoint) reasonable? Are the
   spur-prune / branch-merge / loop-break / exit≥3 rules sound, or do they bias
   the counts? Better practice (e.g., AnalyzeSkeleton, skan conventions)?
2. **Length**: is per-segment branch-distance the right length metric? Should we
   use per-cell total process length, max process reach, or Sholl instead?
3. **Thickness**: is mean-distance-transform-along-skeleton ×2 a valid diameter?
   Pitfalls (skeleton not at true centerline, anisotropy, partial volume)?
4. **Regional division**: is a 400-px fixed grid appropriate? Should tiles be
   tissue-aware? Is CV-across-tiles a defensible heterogeneity measure?
5. **Normalization**: is normalizing by foreground (signal) area correct, or
   should it be tissue area / cell count? Implications for WT-vs-HET.
6. **Sufficiency & validity**: with n=1 WT vs n=2 HET, what can/can't be claimed?
   Which metrics are most likely real vs artifact? What additional
   measurements/controls would make this publishable?
7. **The threshold dilemma** (keep faint-continuation vs drop isolated-faint):
   is there a better classical approach (vesselness/Frangi, tubeness, local
   adaptive, ML) we should adopt before any of these metrics are trustworthy?

Give concrete, prioritized recommendations.
