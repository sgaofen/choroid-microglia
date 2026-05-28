# WT vs HET Choroid-Plexus Microglia Morphology — Master Multi-Angle Report

## 1. Headline

**HET microglia are de-ramified and fragmented relative to WT: their Iba1 skeleton loses internal branch points and long-range connectivity, redistributing arbor mass into many small, terminal-rich fragments.** This conclusion is reproduced across all six independent analytical angles, holds across spatial scales (20–160 µm), is replicate-consistent between both HET images, and is anchored by the single most defensible metric — the collapse of large connected networks (skeleton in components >100 µm falls from **26.8% in WT to ~6% in HET, −77%**, CLEAN).

---

## 2. Master discriminator table

Ranked by CLEAN verdict first, then absolute effect size. (Bins/ratios whose magnitude is partly definitional, e.g. specific size-bins, are included but noted; the top rows are the cleanest, largest, most interpretable separators.)

| # | metric | angle | WT | HET_1 | HET_3 | %diff | verdict |
|---|--------|-------|----|-------|-------|-------|---------|
| 1 | % of skeleton in components >100µm | Component-size spectrum | 26.84 | 6.70 | 5.83 | −76.7 | CLEAN |
| 2 | Frac of TOTAL skeleton in 256+µm bin | Component-size spectrum | 4.64 | 1.00 | 0.00 | −89.2 | CLEAN |
| 3 | Frac of TOTAL skeleton in 128–256µm bin | Component-size spectrum | 14.42 | 3.51 | 3.24 | −76.6 | CLEAN |
| 4 | Frac of TOTAL skeleton in 8–16µm bin | Component-size spectrum | 8.07 | 20.27 | 24.09 | +175 | CLEAN |
| 5 | Largest hotspot patch (contiguous tiles) | Spatial focality | 7 | 48 | 40 | +528.6 | CLEAN |
| 6 | Largest-patch dominance | Spatial focality | 0.318 | 0.686 | 0.727 | +122 | CLEAN |
| 7 | Frac of COUNT in 8–16µm bin | Component-size spectrum | 16.73 | 29.51 | 35.39 | +94 | CLEAN |
| 8 | Hot-tile fraction (top-25% frag) | Spatial focality | 0.115 | 0.357 | 0.278 | +175.6 | CLEAN |
| 9 | Mean hotspot patch size | Spatial focality | 2.44 | 6.36 | 5.00 | +132.6 | CLEAN |
| 10 | E/J ratio @ EDGE (0–50µm) | Edge–interior gradient | 8.566 | 12.11 | 12.697 | +44.8 | CLEAN |
| 11 | pct_type0 endpt–endpt segments | Per-segment dist. | 2.634 | 3.80 | 3.867 | +45.6 | CLEAN |
| 12 | Components per tissue mm² | Component-size spectrum | 29013 | 39258 | 44582 | +44.5 | CLEAN |
| 13 | ratio_t1/t2 (terminal/internal) | Per-segment dist. | 1.360 | 1.822 | 2.019 | +41.3 | CLEAN |
| 14 | p90 component length (µm) | Component-size spectrum | 55.48 | 34.98 | 34.16 | −37.7 | CLEAN |
| 15 | Endpoint:junction ratio (global) | Junction-order spectrum | 12.33 | 16.46 | 16.70 | +34.5 | CLEAN |
| 16 | Junctions per mm² tissue | Junction-order spectrum | 1366.5 | 927.8 | 879.0 | −33.9 | CLEAN |
| 17 | ep/branch @160µm | Multi-scale tiles | 10.26 | 12.96 | 14.09 | +31.9 | CLEAN |
| 18 | E/J ratio @ MID (50–150µm) | Edge–interior gradient | 11.393 | 13.738 | 14.726 | +24.9 | CLEAN |
| 19 | ep/branch @80µm | Multi-scale tiles | 12.24 | 14.27 | 16.17 | +24.4 | CLEAN |
| 20 | skel-len/area @40µm | Multi-scale tiles | 0.1191 | 0.0912 | 0.0862 | −25.6 | CLEAN |
| 21 | Skeleton density @ EDGE (per area) | Edge–interior gradient | 136.7 | 98.92 | 95.81 | −28.8 | CLEAN |
| 22 | Median component length (µm) | Component-size spectrum | 16.97 | 13.46 | 12.83 | −22.6 | CLEAN |
| 23 | pct_type2 junction–junction (internal) | Per-segment dist. | 41.26 | 34.08 | 31.84 | −20.1 | CLEAN |
| 24 | pct_terminal (t0+t1) segments | Per-segment dist. | 58.73 | 65.90 | 68.16 | +14.1 | CLEAN |
| 25 | Endpoint/100µm @160µm | Multi-scale tiles | 18.91 | 22.09 | 21.74 | +15.9 | CLEAN |
| 26 | Junctions per 100µm skeleton (global) | Junction-order spectrum | 1.775 | 1.537 | 1.473 | −15.2 | CLEAN |
| 27 | Junction density @ EDGE (per 100µm) | Edge–interior gradient | 2.412 | 2.093 | 1.964 | −15.9 | CLEAN |
| 28 | branch/100µm @20µm | Multi-scale tiles | 1.788 | 1.641 | 1.434 | −14.0 | CLEAN |
| 29 | % 4-way junctions | Junction-order spectrum | 31.57 | 29.32 | 26.72 | −11.2 | CLEAN |
| 30 | Gini of component lengths | Component-size spectrum | 0.5455 | 0.4679 | 0.4420 | −16.6 | CLEAN |

(All 30 rows are CLEAN. The strongest, most interpretable separators are the connectivity/size-spectrum metrics at the top and the spatial-focality patch metrics; ratio metrics whose magnitude is scale- or radius-dependent — ep/branch @160µm, % 4-way — are real in direction but should not be over-read on absolute magnitude.)

---

## 3. By angle

**Multi-scale tile analysis (20/40/80/160 µm).** The WT–HET separation is scale-robust on four independent families: skeleton-length/area (−20 to −26%), endpoint/100µm (+13 to +16%), branch/100µm (−11 to −14%), and ep/branch ratio. Strongest metric: skel-len/area @40µm (0.119 → 0.087, −25.6%). New sub-difference: density metrics are sharpest at the *finest* scales while the ep/branch ratio sharpens at *coarser* scales (+13% @40µm climbing to +32% @160µm), so the most simultaneously-clean band is 40–80 µm — a scale-dependence no other angle resolves.

**Connected-component size spectrum.** This angle delivers the single most defensible finding: long-range connectivity collapses — skeleton in components >100 µm drops 26.8% → ~6% (−77%), the 256+ µm class essentially vanishes (4.64% → 0–1%), and mass piles into the 8–16 µm bin (+175% of total skeleton, count +94%). Strongest metric: 256+µm bin / >100µm connectivity. New sub-difference: the Gini coefficient *falls* (0.55 → 0.44, −17%), proving WT's size inequality is created by a few dominant giant networks that HET simply lacks — fragmentation is pervasive, not a few snapped giants.

**Spatial focality / clustering (Moran's I + hotspots).** HET fragmentation is not just more abundant but more *focal*: top-25% fragmentation tiles coalesce into one dominant contiguous patch (largest patch 7 → 40–48 tiles; dominance 0.32 → ~0.71), versus WT's scattered small clumps. Strongest metric: largest-patch dominance (spread 0.041 ≪ gap 0.388). New sub-difference: patch *count* barely changes (9 → 11) — focality comes from patches getting *bigger*, not more numerous; Moran's I is positive even in WT, so generic patchiness is baseline and only patch-size statistics discriminate.

**Per-segment distributions (type, length, thickness).** The fragmentation signature lives entirely in branch-*type* composition: internal junction–junction segments −20%, terminal segments +14%, terminal/internal ratio +41% (1.36 → ~1.92, the sharpest single composite index). Strongest metric: ratio_t1/t2. New sub-difference: segment *length* (median 2.5 µm, p90 ~6.7 µm, CV 0.84) and *thickness* (~1.2–1.3 µm) are all "disagree"/flat — HET does not lengthen, shorten, or thin individual processes; it only changes the topological mix. This rules out a "thinner/shorter process" interpretation.

**Edge-to-interior gradient.** Fragmentation is present at every depth but concentrated at the tissue edge: the E/J ratio excess is +45% edge → +25% mid → +16% interior, a clean monotonic decay. Strongest metric: E/J @ edge (8.57 → ~12.4, +45%). New sub-difference: it decomposes the effect into two superimposed components — a roughly *uniform* ~21–29% skeleton-per-area loss and ~15% junction-density loss across all bands (global de-ramification), plus an *edge-weighted* excess of free endpoints (+22% edge vs +6% mid) — and notes WT's edge is normally the *most* intact band (lowest E/J), an advantage HET specifically erases.

**Junction-order / degree spectrum.** Beyond having fewer junctions (−34%/mm², −15%/100µm), HET selectively loses *high-order* crossings: 4-way junctions −11%, with a compensating rise in simple 3-way junctions and +34% endpoints per junction. Strongest metric: junctions/mm² (−34%). New sub-difference: the surviving arbor is not just sparser but *topologically simpler* — the 4-way → 3-way redistribution is direction-robust across disk radii (r=2/3/4 px) even though absolute percentages shift, something no density-only angle could detect.

---

## 4. Convergence (highest confidence)

Conclusions that recur across multiple independent angles — and are therefore the most trustworthy:

- **Fewer branch points / de-ramification** — multi-scale (branch/100µm −11 to −14%), junction-order (junctions −34%/mm², −15%/100µm), edge-gradient (junction density −15% edge/mid), per-segment (internal junction–junction −20%). **4 angles.**
- **Shift toward terminal/dead-end structure** — endpoint:junction up in junction-order (+34%), multi-scale ep/branch (+13 to +32%), edge-gradient E/J (+16 to +45%), per-segment terminal/internal ratio (+41%), per-segment type1 +13%. **4 angles.**
- **Less skeleton per unit tissue area** — multi-scale skel-len/area (−20 to −26%), edge-gradient skeleton density (−21 to −29% all bands). **2 angles, all scales/depths.**
- **Loss of large connected networks / fragmentation into small pieces** — component spectrum (>100µm −77%, 8–16µm +175%, components/mm² +44%), spatial focality (hotspots are real and coalescent), per-segment (isolated endpt–endpt fragments +46%). **3 angles.**

The endpoint-rich, junction-poor, low-connectivity arbor is thus corroborated by every angle that can measure it — this is the report's robust core.

---

## 5. What is NOT clean (honest caveats)

- **Per-segment geometry is a true null.** Segment median length (−1.0%), p90 (−0.2%), CV (+1.1%), and mean thickness (−1.8%) are all "disagree" — HET replicates straddle WT. Read as "no detectable effect," and thickness especially is a coarse `2·pixel_um·mean(dist)` proxy on a float16 array, so sub-pixel differences are unreliable.
- **ep/branch @20µm is a genuine artifact** (verdict "disagree", +1.1%): 90–96% of 20-µm tiles hold <3 junctions, so the per-tile ratio is dominated by single-junction tiles. The ratio is only meaningful at ≥40 µm.
- **Moran's I is a weak/marginal separator.** It is significantly positive in *all* images including WT (p=0.002), so spatial patchiness is baseline; the HET increment (0.160 → 0.174) barely exceeds replicate spread, and the ratio-variant proxy flipped sign (HET_3 = −0.009). Do not lead with autocorrelation — lead with patch-*size* statistics.
- **5+-way junctions are counting noise** (4–7 events/image; the +37% is driven entirely by HET_1). Mean junction order (−0.8%) and combined ≥4-way (−9.7%, same-side) are weak; the clean signal is specifically the 4-way fraction.
- **Mid-band endpoint density (+6%, same-side)** and **interior junction density (−2%, same-side)** are the two non-clean cells in the edge-gradient — both weak-magnitude and non-contradictory.
- **Magnitude (not direction) is method-dependent** for two ratios: ep/branch climbs with tile size (definitional, not biological), and junction-order percentages shift with disk radius — directions hold, absolute numbers should not be quoted as fixed.
- **Component-size metrics are sensitive to foreground-area differences** (WT 0.0445 mm² vs HET 0.033–0.037 mm²); this is why the normalized/distributional forms (Gini, fraction-of-skeleton bins, % >100µm) are the trustworthy ones, not raw counts.

---

## 6. Bottom line for Huixin

Across three images (1 WT control, 2 HET replicates) and six independent morphology analyses, **HET choroid-plexus microglia are consistently de-ramified and fragmented**: they have ~15–34% fewer branch points, ~20–26% less skeleton per tissue area, far more free endpoints per junction (+34%), and a near-total collapse of large connected networks (skeleton in >100 µm components 27% → 6%), with arbor mass redistributing into small 8–16 µm fragments (+175%). The effect is present throughout the tissue but most severe at the tissue edge (endpoint/junction +45% edge vs +16% interior) and is spatially focal, clustering into one dominant hotspot patch in each HET image. Both **image brightness and branch-detection were corrected upstream**, and the two HET replicates agree on every headline metric (replicate spread < gap to WT), so the direction of every claim here is reproducible. This is a **descriptive demo on 3 images, not a statistically powered comparison** — the replicate-consistency check substitutes for, but does not replace, a significance test across animals.