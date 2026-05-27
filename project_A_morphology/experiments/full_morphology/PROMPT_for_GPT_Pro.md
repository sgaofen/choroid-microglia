# Review request: choroid-plexus microglia morphology — region-morphotype analysis

You are a senior quantitative-microscopy / image-analysis reviewer. I'm a
student comparing **microglia morphology** between two conditions in
**choroid-plexus whole-mount** fluorescence images. Please critically review
whether my **data extraction is correct**, my **analysis is sound**, and whether
it is **sufficient** — and give concrete, prioritized improvements. Be rigorous;
I am NOT confident any of this is done right.

## Biological context
- Choroid plexus whole-mount, single channel (Iba1-type microglia marker).
- 3 images: F_WT_2 (WT/control); F_HET_1, F_HET_3 (HET — Alzheimer's model, two
  biological replicates of one treatment). The expected difference is SUBTLE.
- 16-bit TIF, ~3168², 0.207 µm/px.
- Per-cell instance COUNTING is NOT reliable (dense, touching processes), so we
  deliberately use metrics that do not require single-cell segmentation.

## The key finding (please scrutinize this hardest)
Whole-image AVERAGE metrics do NOT separate WT from HET (mean process length
−1.8%, KS D=0.017 ≈ identical distributions). My interpretation: the disease
change is FOCAL, so whole-image averaging washes it out.

So I changed the unit of analysis, two independent ways, and BOTH recover the
same signal — HET microglia are **de-ramified / fragmented**:
- **(A) Connected-component units (sharpest):** treat each connected skeleton
  component as a unit (sparse region → one cell; dense region → one clump,
  treated as a single unit per the biology). HET has +45% more components, each
  shorter (median length −28%, p90 −42%), shifting from large branched arbors to
  small simple cells and bare fragments. Every metric is consistent across both
  HET replicates.
- **(B) Region (200-px tile) composition:** tile the image, compute a 6-feature
  fingerprint per tile, cluster tiles into 4 morphotypes, compare composition.
  HET converts ~14–21 pts of tissue to a de-ramified morphotype, forming
  contiguous spatial foci.

(A) and (B) use completely different unit definitions yet agree — convergent
evidence the conclusion is not an artifact of how the unit is drawn.

To decide which signals to trust I used a **replicate-consistency criterion**:
since the two HET are replicates, I only trust a metric if HET_1 ≈ HET_3 AND
both differ from WT in the same direction (replicate spread < WT gap). The clean,
replicated signals all lie on ONE axis — increased fragmentation / de-ramification
(endpoint-to-branch ratio +18%, de-ramified morphotype fraction up, endpoint
density up, branch density down). See `RESULTS.md` for all tables.

## Pipeline (what produced the data)
1. **Binarize**: normalize (1–99.5 pct) → Gaussian σ=1 → threshold = Otsu×0.7 →
   closing(disk2) → remove <20 px. (Raising the threshold to keep only bright
   signal BROKE processes into fragments → spurious endpoints; lowering it floods
   background. Otsu×0.7 is a compromise. This faint-continuation-vs-isolated-faint
   decision seems unsolvable by a single threshold.)
2. **Skeletonize** → 1-px skeleton (precomputed "v29").
3. **Topology cleanup** (`clean_topology.py`): drop components <12 px; prune
   terminal spurs ≤8 px; branch points = degree≥3 clusters merged within ~8 px,
   kept only if ≥3 arms leave; break loops (microglia are trees) at the dimmest
   pixel.
4. **Per-region features** (`region_morphotype.py`): per 200-px tile (fg ≥3%):
   skeleton-length density (/mm² fg), branch density, endpoint density,
   **endpoint/branch ratio** (fragmentation), mean thickness (distance-transform
   along skeleton ×2), ramification (branches/100 µm skeleton).
5. **Cluster** tiles (pooled across all 3 images) with StandardScaler + k-means
   (k=4) → morphotypes; per-image composition = % of tiles per morphotype.
6. **Replicate-consistency screen** (`replicate_consistency.py`) and
   **robustness** over tile∈{31,41,62 µm} × k∈{3,4,5} (`robustness.py`).
7. Whole-image aggregate (`analyze.py`) kept as the negative-control baseline.

## Confound I already checked
HET coverage is ~17% lower (dimmer/sparser). I (a) re-clustered with the coverage
feature removed (de-ramified enrichment survives, +13.7 pts), and (b) noted two
morphotypes with equal low coverage but opposite branching, only the de-ramified
one rising in HET. So the signal has a structural component independent of
brightness. Is this enough to rule out the confound?

## Data & figures in this zip
- `RESULTS.md` — all results/tables (read this first)
- `data/cc_features.csv` — every connected component: length, span, branches,
  endpoints, thickness, morphotype label  (analysis A)
- `data/cc_morphotype.json` — per-image component summary + composition
- `images/cc_size_distribution.png` — component-size distribution, WT vs HET
- `images/<stem>_cc_map.png` — components colored by morphotype
- `data/region_features.csv` — every tile: 6 features + morphotype label
- `data/region_morphotype.json` — cluster profiles + per-image composition
- `data/aggregate.json`, `data/wt_vs_het_deltas.json` — whole-image baseline
- `data/<stem>_segments.csv`, `<stem>_regional.csv` — per-segment / per-tile raw
- `images/<stem>_morphotype_map.png` — spatial morphotype maps (focality)
- `images/composition_bar.png`, `feature_distributions.png`
- `images/spot_check_WT_vs_HET.png` — raw vs skeleton, WT-healthy vs HET-deramified
- `images/CURRENT_<stem>.png` — full-image raw vs skeleton+branch+endpoint overlays
- `code/` — clean_topology, region_morphotype, replicate_consistency, robustness,
  analyze, spot_check

## Please critique specifically
1. **Is the connected-component unit (analysis A) sound?** "Connected" depends on
   the binarization (the connect-vs-separate problem reappears as "one component
   or two"), and DAM fragmentation vs threshold-broken dim processes both produce
   more small components. Is the component-size distribution + morphotype
   composition a defensible readout despite this? Is treating dense clumps as
   single units acceptable? Does the convergence of analyses A and B (different
   units, same conclusion) meaningfully strengthen the claim?
2. **Is the region-morphotype + composition approach (B) valid** for a focal,
   subtle difference, or are there standard methods I should use instead
   (e.g., per-cell morphometric clustering, spatial point-pattern statistics,
   texture/Haralick, fractal/lacunarity per region)?
2. **Replicate-consistency criterion**: is "both replicates same side of control,
   spread < gap" a defensible screen with n=2 vs 1, or is it ad hoc? Better way?
3. **Feature extraction**: is endpoint/branch ratio a sound fragmentation metric?
   Is distance-transform-along-skeleton ×2 a valid thickness/diameter? Is
   branch-distance the right length unit? Pitfalls?
4. **Topology**: are the spur-prune / branch-merge / loop-break / exit≥3 rules
   sound, or do they bias counts? (vs ImageJ AnalyzeSkeleton / skan conventions)
5. **Tiling**: is a fixed 200-px grid appropriate? Should tiles be tissue-aware
   or multi-scale? Edge effects?
6. **Confound**: is my brightness-confound rule-out sufficient? What else could
   masquerade as "fragmentation" (threshold breaking dim processes, focus,
   staining)? How would you control it?
7. **Sufficiency / statistics**: with n=1 WT vs 2 HET and non-independent tiles
   (pseudo-replication), what can/can't be claimed? What is the minimum design
   (images/animals, mixed-effects model?) to make the de-ramification claim
   publishable?
8. **Threshold dilemma**: better classical approach (Frangi/vesselness, tubeness,
   local-adaptive) before any metric is trustworthy?

Give concrete, prioritized recommendations.
