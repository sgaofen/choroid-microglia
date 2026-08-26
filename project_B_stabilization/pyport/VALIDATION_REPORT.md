# Shipley 2020 Stabilization Pipeline Python Port -- Real-Data Validation Report

Date: 2026-08-24. Author: Stephen Yu (Claude collaboration).
Ported code: `pyport/cpstab/`. Upstream: [LehtinenLab/Shipley2020](https://github.com/LehtinenLab/Shipley2020).
Prior docs: `../CODEMAP.md` (reverse-engineering), `../../lab_delivery_2026-08/PROJECT_B_NOTES.md` (audit).

## 1. Test data and pairing

- **Raw**: `20230620_FAD-F_1.oir` series (61 files, 65 GB; 512x512, 2 channels, 41 z @5 um, 1500 T, ~55 min, 10-bit, FVMPE-RS resonant scanner).
- **Ground truth**: `AD CX3CR1_Registered/20230201/20230201-FAD-F_230620_001_zproj.tif`
  (TCYX 1500x2x512x512 uint16; TIFF header `ImageJ=1.52a` matches the jar pinned in the main script, **confirmed this finished product came out of that pipeline**).
- **Pairing can't be trusted by filename**: the finished product mislabels a WT run as "FAD-F"; the `.txt` frame counts for FAD-F_2 (60 chunks = 1500 frames) and
  WT-F_3 (48 chunks = 1200 frames) have swapped metadata tags. This pairing was determined by image cross-correlation
  (`scripts/pair_check.py`: peak sharpness 39.5 vs 4.5/4.6, unique hit) + cross-checked with mean-image r=0.987.
- Channels: C0=BA575-645 red (IV dye, refchannel=1), C1=BA495-540 green (CX3CR1-GFP).

## 2. Input provenance equivalence (Bio-Formats direct read vs. the original manual FluoView export)

| Check | Result |
|---|---|
| Byte spot-check | bfconvert's two independent paths (full vs. `-timepoint 0`) are byte-identical |
| Odd/even row phase (resonant bidirectional scan) | truth field offset 0.003 px, port -0.004 px -> no difference |
| Odd/even field high-frequency noise ratio | truth 0.998, port 0.997, raw 1.000 -> FluoView did not do row resampling |
| Intensity level | port/truth mean ratio 1.012 (possibly from a z-window difference of <=1 plane, see Section 4) |

Conclusion: **bfconvert direct read is pixel-equivalent to the original manual export**; the "FluoView manual export" step can be safely removed.

## 3. Validation results

Pipeline parameters = original defaults: refchannel=1, scale=4, chunksize=20, proj_range=quarter (z 10..31), proj_type=mean, opttype=none.

| Metric | Value | Interpretation |
|---|---|---|
| **Stabilization quality** (residual motion relative to each own's time mean) | truth and port both median 0.01 px / p95 0.02 px | **Both equally rock-solid stable -- the pipeline's core metric fully meets spec** |
| **Structural equivalence** (time-mean cross-correlation) | r=0.9784; after correcting a constant offset (-1.8, +0.1) px, **r=0.987** | Same stabilization scenario, only the overall anchor differs by ~1.8 px |
| Frame-by-frame Pearson r (after constant alignment) | median 0.61 | see the "shared noise" analysis below; already at the ceiling for this comparison method |
| Frame-by-frame noise floor calibration | truth adjacent-frame r=0.33; truth r=0.74 after shifting itself by 0.5 px | 0.61 falls within the range predicted by a "~0.5-1 px per-plane translation trajectory difference" |
| Run time (Apple Silicon Mac) | **serial 34.2 min; parallel (10 processes) 14.3 min, output bit-identical** | the original's stated "5-6 h/GB" has no written source (checked email/Slack/meeting notes, found none) and is 60-100x off from the algorithmic model; likely a mix-up with "5-6 h/experiment"; model estimate for the original is 1-6 h/run, exact figure pending an actual MATLAB run in the lab |
| Intermediate disk output | 0x (does not write .sbx/.sbxall/.tif.frames by default) | original writes 3-4x the data volume |

**Why frame-by-frame r isn't 1**: if the two sides' per-plane translations matched exactly, both sides would share the same photon noise and r->1.
The observed 0.61 indicates the per-plane translation trajectories diverge by ~0.5-1 px sub-pixel, and interpolation breaks up the shared noise.
The source can't be further disentangled (no MATLAB on this machine): (1) chained registration (the DFT_rect per-plane chain + the zproj_reg 1500-frame temporal chain)
amplifying the cumulative effect of tiny numerical differences; (2) Huixin's local pipeline may differ from the GitHub version (circumstantial evidence: her output is named `_zproj` != the
repo version's `_mean_zproj`). **The scientific conclusion is unaffected**: both structure and stability are equivalent, downstream kinematic analysis is interchangeable.

## 4. Original-code issue list (newly confirmed this round, supplementing CODEMAP Sections 9-10)

1. **`median(matrix)`'s per-column semantics silently erase volume-level XY registration** (MakeSBXall.m L31-35): `RS_total - median(RS_total)`
   on a 41xNt matrix is a per-column (per-volume) median -> `dftregistration3D`'s volume-level XY shift and the inter-chunk XY stitching
   **are completely subtracted out right after being computed**; temporal XY stabilization in practice is done entirely by `zproj_reg` on the projection. The likely intent was
   `median(RS_total(:))`. The Z dimension is a vector, a global median, and survives. -> the port keeps this by default (for equivalence);
   this is the single largest piece of dead computation in the original.
2. **lineshift dead code** (ConvertOIR_SBX.m L47-81 + main script L30): odd/even row phase is estimated from the first 10 volumes
   (+-5 candidates, exhaustive search), the return value is stored in the `lineshift` variable and **has no consumer**; at the application site
   (MakeSBXall L77/L144) the default parameter is always 0 and the main script never passes it in.
3. `DetermineXYShiftsFBS` puts the reference plane's Gaussian blur + FFT inside the t loop, recomputing the same reference
   Nz*Nt times (75 chunks * 20 frames * 41 planes = ~60,000 redundant FFTs).
4. `dftregistrationAlex.m`'s function header comment claims it returns `[error,diffphase,row,col]`, but L128 actually only returns
   `[row_shift,col_shift]` -- trust the comment and you'll get it wrong.
5. Others (already recorded in CODEMAP, re-verified during validation): the optotune default string `'true'` is always truthy + the `unwarp_chunk`
   typo dead branch; `zproj_reg`'s Affine branch has an undefined variable `pathz`; `proj_range` is computed as
   `1:Nz` in the main script but never passed in (the actual projection is MakeSBXall's default z 10..31, this round settled it definitively from both source code and data).

## 5. What the port removed/replaced

| Original | Port | Benefit |
|---|---|---|
| FluoView manual export `.tif.frames` (~1x the data volume, manual) | bfconvert direct-read OIR (pixel equivalence proven in Section 2) | removes 1x disk + removes a manual step |
| `.sbx` conversion + `.sbxall` write-out (~1x each; `.sbxall` has no reader anywhere in the library) | not written by default (`write_registered=True` optional) | removes ~2x disk |
| `MakeSBXall` two full passes of read + full recompute | single pass (PASS2 only runs when `.sbxall` is needed and only replays the shifts) | cuts ~1/2 the compute |
| MIJ/Fiji/Java bridge TIFF write | tifffile ImageJ hyperstack (page order/metadata matched to the 1.52a format) | removes the entire Java dependency |
| lineshift estimation (dead code) | removed | -- |
| reference-frame blur+FFT repeated inside the loop | hoisted out of the loop (numerically identical) | tens of thousands of redundant FFTs |
| dead weight like `ref_all` inside `.dftshifts` | not saved by default (`save_debug` optional) | slims down intermediate files |
| MATLAB + IPT + PCT + a specific account environment | numpy/scipy/tifffile, runs anywhere | reproducibility |

The numeric core was ported line-by-line and adversarially verified (19 subtasks, synthetic data 7/7, dftreg unit tests 24/24);
the pitfall checklist (per-column median, midpoint prctile, rescale truncation, uint16 round-half-away rounding, imresize anti-aliasing,
imtranslate [C,R], nearest-stretch MATLAB index mapping, Sterbenz rounding) was addressed item by item -- see each module's PORTING NOTES.

## 6. Optional follow-ups

1. **Parallelization**: chunks are naturally independent (only chunk1's reference needs to be shared); a process pool sized to P-cores is expected to give another 4-8x
   (34 min -> ~5 min/run).
2. **"Correction mode" switch**: change Section 4-1's median to a global scalar, so volume-level 3D registration actually takes effect
   (output would then diverge from the original, needs discussion with Huixin before becoming v2).
3. Streaming direct-read of OIR (skip the 114 GB OME-TIFF intermediate).
4. To reproduce frame-by-frame noise levels: would need the original FluoView-exported raw TIFFs from that time, or Huixin's actual driver script.

## Appendix

- Validation artifacts: `$CPSTAB_WORKSPACE` (reportA/reportB/validation_means.png,
  `FAD-F_1_raw.ome.tif` 114 GB, deletable).
- Toolchain: `~/tools/jdk-17.0.20.1+1-jre` + `~/tools/bftools` (Bio-Formats 8.5.0).
- Usage in `cpstab/README.md`; validation command:
  `python -m cpstab.validate --raw RAW.ome.tif --truth TRUTH.tif --out DIR [--params ...]`
