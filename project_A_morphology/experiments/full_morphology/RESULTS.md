# Results — choroid-plexus microglia morphology (WT vs HET), demo on 3 images

**Scope.** This is an exploratory DEMO on 3 images (F_WT_2 = WT/control;
F_HET_1, F_HET_3 = HET, an Alzheimer's model, two biological replicates of one
treatment). The aim is to fully characterize what these images show, not to make
a publication-grade statistical claim — n is 3 images and tiles/components are
not independent, so no image-level p-values are reported (see Limits). 16-bit,
~3168², 0.207 µm/px, single Iba1-type channel.

This version incorporates the GPT-Pro review (2026-05-27): topology counting is
now consistent between region and whole-image analyses; the morphotype
clustering is a single shape-only definition (coverage kept only as a labelled
sensitivity); densities are reported per tissue area, per foreground, and per
skeleton length; a pre-registered fragmentation score is added; and the result
is tested across 5 segmentation methods and 6 grid placements.

## TL;DR
Whole-image AVERAGES do not separate WT from HET (mean process length −1.8%,
KS D=0.017). The difference is FOCAL and shows up only when the unit of analysis
changes. Two independent unit definitions agree: **HET microglia are
de-ramified / fragmented** — broken into more, shorter, less-branched pieces,
with a higher endpoint-to-branch ratio, concentrated in spatial foci. HET also
has lower overall process abundance (a separate signal). Every signal below is
replicate-consistent (the two HET agree), survives the brightness/coverage
control, holds across 4 of 5 segmentation methods and all 6 grid placements, and
matches the DAM-like (fragmented/beaded process) morphology Huixin described.

## 1. Replicate-consistent signals (corrected topology counting, P1)
Branch/endpoint counts now come from ONE global topology pass
(`merged_branches`, exit≥3) assigned to tiles/components — tile branch counts now
sum to the whole-image count (was 2.7–2.9× over-counted). All metrics screened by
the replicate criterion (trust only if HET_1≈HET_3 and both differ from WT).

| region-level metric | WT | HET_1 | HET_3 | verdict |
|---|---|---|---|---|
| fragmentation score (pre-registered) | −0.85 | +0.67 | +0.15 | ✓ |
| de-ramified region % | 8.9 | 22.7 | 17.5 | ✓ |
| fragmentation-hotspot region % | 12.5 | 34.3 | 28.2 | ✓ |
| endpoint/branch ratio | 4.34 | 5.44 | 5.28 | ✓ |
| branch per 100 µm skeleton | 4.13 | 3.91 | 3.99 | ✓ |
| endpoint per 100 µm skeleton | 15.5 | 18.4 | 17.8 | ✓ |
| skeleton per **tissue-tile** area | 108000 | 83319 | 87668 | ✓ (abundance ↓) |
| skeleton per **foreground** area | 580939 | 538290 | 584170 | ✗ HET disagree |
| mean segment length | 3.41 | 3.26 | 3.42 | ✗ HET disagree |

The fragmentation axis is clean and replicated; the two ✗ metrics are correctly
dropped. **Abundance (per tissue area) and fragmentation are separate findings**
(P3): HET has both less process per tissue area AND a more fragmented topology.
Per-foreground density is inconsistent, confirming it is the wrong denominator.

## 2. Two units converge
**Connected components** (each connected skeleton piece = a unit; sparse → one
cell, dense → one clump): HET has **+45% more components**, each shorter
(median 25.0→18.6/17.8 µm; p90 95.7→53.8/55.6 µm), with more unbranched stubs
(35%→43/44%), shifting composition from large arbors (M3 19→11/11%) and medium
(M2 30→20/16%) to small/fragment types (M0 25→36/39%, M1 25→33/34%). WT has more
skeleton in large connected networks (45 vs 25/31% — connectivity, not amoeboid
clumping). All ✓ replicate-consistent.

**Region tiles** (200 px) give the same story via a completely different unit
(see §1). Convergence across unit definitions argues the conclusion is not an
artifact of how the unit is drawn.

## 3. Brightness / coverage confound — controlled, not "ruled out"
HET foreground coverage is ~17% lower. The fragmentation signal is NOT fully
explained by this: (a) the canonical morphotype clustering uses SHAPE-only
features (no coverage, no abundance) and still enriches the de-ramified type in
HET; (b) a coverage-inclusive clustering (sensitivity) still gives de-ramified
WT 7.8 → HET 22.2/18.4; (c) coverage-matched tiles (fg 0.10–0.22) still show the
enrichment. Threshold/brightness remains a real confound (faint processes sit
near background), addressed next.

## 4. Segmentation sensitivity (P4) — the signal is not a single-threshold artifact
Re-binarized + re-skeletonized under 5 methods. Fragmentation direction
(HET: endpoint/branch ↑, branch-per-100µm ↓, endpoint-per-100µm ↑):

| method | ep/br | branch/100µm | ep/100µm | abundance |
|---|---|---|---|---|
| otsu×0.7 (default) | ✓ HET↑ | ✓ HET↓ | ✓ HET↑ | ✓ HET↓ |
| otsu (stricter) | ✓ HET↑ | ✓ HET↓ | ✓ HET↑ | ✓ HET↓ |
| adaptive (local) | ✓ HET↑ | ✓ HET↓ | ✓ HET↑ | ✓ HET↓ |
| hysteresis (bright seed→grow) | ✓ HET↑ | ✓ HET↓ | ✓ HET↑ | ✓ HET↓ |
| frangi (tubeness) | saturated ~17 | — | ✗ | ✓ HET↓ |

The fragmentation signal holds across **4/5** methods (frangi over-fragments
EVERYTHING to ep/br≈17, masking the contrast); abundance↓ holds in **5/5**.
**Finding on the threshold dilemma:** a hysteresis low threshold set to capture
faint processes (≤0.6×otsu) FLOODS — the faint processes sit at background
intensity, so no single intensity cut both captures them and excludes background.
This is a genuine SNR limit, not a tuning failure; a pixel classifier (ilastik /
small U-Net) is the principled next step if faint continuation must be recovered.

## 5. Robustness to grid placement (P5) and parameters
Fragmentation-hotspot % across 6 grid placements (5 origin offsets + a sliding
overlapping window), using the corrected topology counting: WT 12–16%,
HET_1 33–36%, HET_3 24–30% — **✓ HET↑ in all 6**. The result does not depend on
where the grid is drawn. (Tile size and cluster count k were also varied during
development with the same direction.)

## 6. Visual spot-check + biology
`spot_check_WT_vs_HET.png`: a HET de-ramified region = short, broken, stubby
fragments with many loose ends; a WT ramified region = long, connected, branched
network. The fragmented morphology matches the **DAM-like** ("half-dead"
microglia with broken-up/beaded processes) phenotype Huixin described — note this
is morphology only; single-channel Iba1 cannot establish a molecular DAM state.

## Limits (this is a demo, kept honest)
- 3 images (1 WT, 2 HET); tiles/components are not independent → no valid
  image-level statistics. The METHOD cleanly separates these images; a formal
  WT-vs-HET claim would need more animals (animal = n) — out of scope here.
- "Thickness" = binary-mask local width (distance-transform ×2), not true 3D
  process diameter; large values include soma/overlap. Reported as apparent width.
- "% of regions" uses Iba1-positive tiles (fg ≥3%), not a true tissue mask.
- Faint-process recovery is SNR-limited (see §4); a pixel classifier is the
  natural upgrade. Per-cell instance counting deliberately avoided.
