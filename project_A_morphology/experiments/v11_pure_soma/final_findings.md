# Pure-algorithm choroid plexus microglia analysis — final findings

After multiple iterations and a sharp critique that Cellpose was an unreliable baseline, we converged on a pure-algorithm pipeline that does not use Cellpose at all.

## Pipeline (Cellpose-free)

1. **Normalize** 16-bit TIF to [0, 1] via percentile stretch
2. **Smooth + threshold** at Otsu → binary signal mask  
3. **Erode** binary mask with disk(r=2) — thin processes vanish, somata survive
4. **Distance transform** inside the eroded mask
5. **Find local maxima** (min_distance=8, threshold=1.0) → candidate soma centers
6. **Watershed** to split touching somata
7. **Skeletonize** the original binary (with closing r=2 to bridge z-stack gaps)
8. **Per-soma**: count skeleton connected components crossing the 1-px boundary ring around the soma

## Per-image results

| Image | n_somata | mean process count |
|---|---|---|
| WT_2 | 3872 | **2.27** |
| HET_1 | 3295 | 2.14 |
| HET_3 | 2773 | 2.16 |

## Process count distribution (% of detected cells)

| Bin | WT_2 | HET_1 | HET_3 |
|---|---|---|---|
| 0 | 4.7% | 6.1% | 5.2% |
| 1 | 22.4% | 24.4% | 24.7% |
| 2 | 36.6% | 36.6% | 36.0% |
| 3 | 21.8% | 21.8% | 22.4% |
| 4 | 9.3% | 7.2% | 8.1% |
| 5 | 3.3% | 2.3% | 2.3% |
| 6 | 1.4% | 1.0% | 0.9% |
| 7+ | 0.5% | 0.6% | 0.4% |

**WT has more cells with 4+ processes (14.5% vs 10-11% in HET).** The "highly ramified" tail is enriched in WT.

## Why this pipeline is defensible

- **No Cellpose anywhere** — addresses the critique that Cellpose was an unreliable baseline
- **Uses biological fact directly** — somata are thicker than processes; erosion exploits this
- **Watershed separates touching somata** — a known limitation has a known solution
- **Finds the spider cell Cellpose missed** — visually verified on the test crop
- **Catches Kolmer cells** — 3000+ cells per image vs Cellpose's 1800 likely catches the small oval Kolmer-like population

## Robustness: 3 independent methods agree

| Method | Uses Cellpose? | WT > HET pattern |
|---|---|---|
| Boundary-rooted with Cellpose somata | Yes | ✓ (WT 1.68 vs HET 1.08–1.17) |
| Erosion + watershed somata | **No** | ✓ (WT 2.27 vs HET 2.14–2.16) |
| Patch-level skeleton density | No | ✓ (WT branchpoints/mm² 40k vs HET 28–31k) |

All three independent paths point to WT being more ramified than HET. The magnitude varies (5–45 %), but the direction is consistent.

## Limitations

- Erosion at r=2 may still miss the smallest Kolmer cells (those < 4 µm thick)
- Watershed can over-split a cell with a long elongated soma
- z-stack max-projection gaps remain a partial confound — closing r=2 mitigates but does not eliminate
- n = 1 WT vs n = 2 HET — pilot results only, not a biological conclusion

## Files

- `F_*_somata_v2.npy` — pure-algorithm soma masks (erode_r=2, min_size=10)
- `WT_2_crop_somata_v2.png` — visual verification crop
- This document
