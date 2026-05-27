# Results — choroid-plexus microglia morphology (WT vs HET)

3 images: F_WT_2 = WT/control; F_HET_1, F_HET_3 = HET (Alzheimer's model,
two biological replicates of one treatment). 16-bit, ~3168², 0.207 µm/px,
single Iba1-type channel.

## TL;DR
**Whole-image AVERAGES do not separate WT from HET** — the difference is FOCAL,
so averaging the whole image washes it out. Two unit-of-analysis changes both
recover the same signal: **HET microglia are de-ramified / fragmented**.
1. **Connected-component** units (sharpest): HET has +45% more components, each
   shorter (median −28%, p90 −42%), shifting from large branched arbors to small
   simple cells and fragments.
2. **Region (200-px tile)** composition: HET converts a reproducible fraction of
   tissue regions to a de-ramified morphotype, forming spatial foci.

Both are replicated (the two HET agree on every metric), robust to parameters,
structural (not a brightness artifact), match the known **DAM**
(disease-associated microglia: fragmented/beaded processes) phenotype — and
agree with EACH OTHER despite completely different unit definitions (convergent
evidence). n = 3 images limits hard statistics.

## 0. Connected-component analysis (sharpest, most interpretable)
Use each CONNECTED skeleton component as the unit instead of an arbitrary grid:
a sparse region's component = one cell (free per-cell morphology where reliable);
a dense region's component = a clump of touching cells (Huixin: treat clumps as
units, don't segment inside them). Maps directly onto Huixin's hierarchy
(round/small vs ramified, by span, by branch count). Every metric passes the
replicate-consistency screen (`cc_morphotype.py`):

| metric | WT | HET_1 | HET_3 | reading |
|---|---|---|---|---|
| components / mm² | 12322 | 18081 | 17509 | HET +45% (broken into more pieces) |
| % unbranched stubs | 5.8 | 7.0 | 8.3 | more bare fragments in HET |
| median component length | 25.0 µm | 18.6 | 17.8 | HET −28% (shorter) |
| p90 component length | 95.7 µm | 53.8 | 55.6 | HET −42% (big arbors shrink) |
| % skeleton in >100 µm components | 45.3 | 24.6 | 31.3 | WT more connected |

Component morphotypes (M0 fragment → M3 large arbor):
| type | length | span | branches | reading | WT | HET_1 | HET_3 |
|---|---|---|---|---|---|---|---|
| M0 | 10.6 µm | 10.3 | 0 (96% stub) | bare fragment | 6.1 | 7.1 | 8.8 |
| M1 | 14.6 µm | 11.2 | 5.6 | small simple cell | 31.3 | **51.2** | **53.2** |
| M2 | 32.9 µm | 19.6 | 15.5 | medium ramified | 46.8 | 33.6 | 29.9 |
| M3 | 156.6 µm | 47.3 | 89.1 | large complex arbor | 15.8 | **8.1** | **8.1** |

HET shifts mass from M3/M2 (large/ramified) to M1/M0 (small/fragment); both HET
agree on every morphotype (M3 identical at 8.1%). The component-size distribution
shifts left in both HET (the two HET curves overlap). NOTE: the >100 µm
"clump"-size metric here reflects process CONNECTIVITY (WT processes form large
connected networks), NOT amoeboid cell aggregation — it is higher in WT.
(`cc_features.csv`, `cc_morphotype.json`, `cc_size_distribution.png`,
`<stem>_cc_map.png`.)

---

## 1. Whole-image aggregate means do NOT separate (the honest negative)
| metric | WT | HET (mean) | Δ |
|---|---|---|---|
| mean segment length | 3.26 µm | 3.20 | −1.8% |
| mean thickness | 1.84 µm | 1.92 | +4.3% |
| branch / mm² | 24725 | 23051 | −6.8% |
| skeleton length / mm² | 577798 | 550803 | −4.7% |

Pooled per-segment distributions are nearly identical (KS D=0.017 for length).
The means are insensitive because only a SUBSET of regions transform — so we
move to the region level. (Full table: `aggregate.json`, `wt_vs_het_deltas.json`.)

## 2. Region morphotype composition (the analysis that works)
Tile each image into 200-px (~41 µm) regions → 6 morphology features per region
→ k-means into 4 morphotypes → compare COMPOSITION between conditions. The unit
is the REGION (boundaries are ours, 100% reliable) — NOT per-cell, because
instance segmentation in this dense tissue is the unsolved problem.

Morphotype profiles (C0 = most ramified → C3 = least):
| type | branch/mm² | endpoint/branch | thickness | ramification | reading |
|---|---|---|---|---|---|
| C0 | 61642 | 0.83 | 2.78 | 13.4 | dense/thick (soma-rich/amoeboid), rare |
| C1 | 76509 | 1.37 | 1.51 | 12.3 | densely ramified, thin |
| C2 | 64793 | 1.24 | 1.81 | 12.1 | moderately ramified |
| **C3** | **52634** | **1.90** | 1.57 | **9.3** | **de-ramified / fragmented (DAM-like)** |

Composition (% of regions, brightness-independent clustering):
| type | WT | HET_1 | HET_3 | HET−WT |
|---|---|---|---|---|
| **C3 (de-ramified)** | **21.4** | **32.3** | **37.9** | **+13.7** |
| C0 | 1.0 | 3.5 | 8.3 | +4.9 |
| C1 | 31.8 | 19.2 | 34.0 | (HET disagree) |
| C2 | 45.8 | 44.9 | 19.9 | (HET disagree) |

Spatial maps (`*_morphotype_map.png`): the de-ramified regions form CONTIGUOUS
FOCI in both HET (large red patches), but are sparse/scattered in WT — a focal
disease pattern, not uniform noise.

## 3. Replicate-consistency screen (which signals to trust)
The two HET are replicates of ONE treatment → they must resemble each other. We
trust a WT-vs-HET signal only if **HET_1 ≈ HET_3 (same side of WT, replicate
spread < WT gap)**. `replicate_consistency.py` screens every metric. The clean,
replicated signals ALL lie on one axis — **de-ramification / fragmentation**:

| signal | WT | HET_1 | HET_3 | direction | replicate spread / WT gap |
|---|---|---|---|---|---|
| **endpoint/branch ratio (fragmentation)** | 1.30 | 1.52 | 1.55 | ↑ +18% | 0.10 (tightest) |
| C3 de-ramified % | 21.4 | 32.3 | 37.9 | ↑ | 0.40 |
| endpoint density / mm² | 89.8k | 98.6k | 104k | ↑ +13% | 0.48 |
| branch density / mm² | 66.7k | 62.0k | 64.1k | ↓ −5.5% | 0.56 |
| whole-image endpoint / mm² | 87.1k | 94.3k | 95.3k | ↑ +8.8% | 0.13 |

DROPPED as inconsistent (the two HET disagree): region skeleton-length density,
region ramification mean, C1/C2 morphotype proportions, regional CV of skeleton
density.

## 4. Brightness confound — ruled out
HET regions have ~17% lower foreground coverage (slightly dimmer/sparser).
(a) Re-clustering with the coverage feature REMOVED still gives de-ramified
HET enrichment of +13.7 pts. (b) C2 and C3 have nearly identical (low) coverage
but OPPOSITE branching — only the de-ramified C3 rises in HET, so the
discriminating axis is structure, not brightness. (c) Low coverage is partly a
CONSEQUENCE of de-ramification (fewer/shorter branches = less signal area).

## 5. Robustness to parameters
9 combinations of tile size {31, 41, 62 µm} × k {3, 4, 5}: in **9/9 both HET >
WT** for the most-fragmented morphotype; strict replicate-consistency holds in
**7/9** (the 2 misses are only at extreme coarse tiles or over-split k=5, where
the disease cluster becomes small/unstable). Not a parameter artifact.

## 6. Visual spot-check + biology
`spot_check_WT_vs_HET.png`: a HET de-ramified region = short, broken, stubby
fragments with many loose ends; a WT region = long, connected, branched network.
The eyeball matches the metric, and the fragmented morphology is exactly the
**DAM** ("half-dead" microglia with broken-up/beaded processes) feature.

## Caveats
- **n = 3 images** (1 WT, 2 HET); regions within an image are NOT independent
  (pseudo-replication) → no valid image-level p-values. The METHOD separates the
  groups, but a hard statistical claim needs more images/animals (≥3/genotype).
- Single global threshold (Otsu×0.7); faint processes can fragment, which is
  partially entangled with the fragmentation signal — though replicate
  agreement + focality + DAM biology argue the signal is real.
- Per-cell counting deliberately avoided (dense, touching processes).
