# Cell Skeleton Tuning Workbench — Usage Guide

## Before you run this on a NEW image set

Three things are specific to the 2026-08 12-animal 10x set and will silently give you
wrong or empty results otherwise. Check all three first.

1. **`config.py` — `PIX_UM` (0.164827 µm/px) and `IMG_PX` (14661).** Every µm, mm² and
   per-mm² number scales off `PIX_UM`; a wrong value produces wrong numbers with no error
   and no crash. Read the real pixel size from your acquisition metadata.
2. **Input filename contract.** `server.py` only sees TIFs named
   `10x_<sample>_CCR2-CD45_<C0|C1|C2>.tif`. Anything else is invisible: the sample dropdown
   simply comes up empty, with no message. Either rename your scans to that pattern, or set
   `CHP_FILE_RE` / `CHP_FILE_FMT` (and `CHP_SRC` for the folder). `preprocess_clean_images.py`
   honours the same three variables and discovers its sample list from disk.
3. **`rois/` is specific to the 2026-08 set (74 hand-drawn boxes).** For a new image set move
   it aside (`mv rois rois_2026-08 && mkdir rois`) and draw fresh boxes in the tuner before
   running any stats script. Every stats script enumerates `rois/*.json` and will otherwise
   re-analyse the old animals, or die on a missing TIF.

## Launch

From inside this folder, after
`python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt`:

```bash
mkdir -p analysis_output cache
./.venv/bin/python server.py
```

Then open http://localhost:8901 in a browser. Data is memmap-read straight from
the `tif_max/` directory named by `config.py` (`SRC_TIF_MAX`) — no memory cost, response in
seconds.
Point it elsewhere with `CHP_SRC=/path/to/tif_max ./.venv/bin/python server.py`.

## ROI selection workflow (added 2026-08-13, recommended)

The UI starts in **simple mode** (thresholds and other parameters hidden; click ⚙ Params to
expand). The top bar shows batch progress: *image N of <total images> · M ROI(s) on this image
· K/<total images> images done · T ROIs total*.

1. In the top bar turn on **◻ Select mode** (turns green = selecting), and pick the selection box
   size (338/507/676 µm)
2. Click on the left image to place the big cyan box (the big box takes about 5-10 s to compute);
   if the area looks usable, click **★ Save ROI**
3. Switch to the next image (dropdown at top left) and repeat; once every image is done, run the
   analysis chain:
   ```bash
   ./.venv/bin/python stats_baseline_parallel.py       # roi_metrics_all.csv, animal_summary.csv, baseline_stats_<today>.md
   ./.venv/bin/python stats_corrected.py   | tee analysis_output/stats_corrected_stdout.txt
   ./.venv/bin/python stats_round_cells.py | tee analysis_output/stats_round_cells_stdout.txt
   ```
   `stats_corrected.py` is the **authoritative** one — it adds ROI clipping, a tissue-area
   denominator and overlap dedup, and it is where the published numbers come from. The last two
   print their headline effect sizes and p-values to **stdout only**, so capture them.

   `analysis_group_comparison.py` is a standalone alternative that recomputes everything from
   `server.py` in a single pass (`analysis_report.md` + `roi_metrics_groupcmp.csv` +
   `animal_summary_groupcmp.csv` + an optional scatter png). Use it for a quick look; it does
   **not** apply the three corrections above.
   ```bash
   ./.venv/bin/python analysis_group_comparison.py
   ```
4. **Background cleanup** is on by default (blank areas are uniformly pushed to black, i.e. the
   repo's brightness normalization rule); it affects display only, not the numbers

## Controls

- **Left, full image**: scroll to zoom, drag to pan; **single click** anywhere = inspect that
  spot; **Shift+drag a box** = save directly as an ROI; arrow keys nudge the inspect box
- **Middle**: four views — Overlay / Original / Component colors / Original|Overlay; below it are the live
  metrics of the current box
- **Right**: all extraction parameters; drag a slider and it recomputes live (a few hundred ms)
- **★ Save ROI**: adds the current inspect box to this sample's ROI list (stored in
  `rois/<sample>.json`)
- **Recompute all ROIs**: recompute the metrics of every ROI of this sample with the "current"
  parameters (slow — it re-runs the full extraction per ROI)
- **Export CSV (slow)**: recompute all ROIs of **all samples** with the "current" parameters and
  download — this guarantees the final numbers all come from one single recipe. It runs
  synchronously inside the HTTP handler and takes minutes on a 74-ROI set; the browser tab looks
  dead while it works, so watch the server console, which logs `export: <sample> R<id>` per ROI.

## Base image (settled 2026-08-13)

Clean-image base = **Fiji flat-fielded version fjff** (multiplicative illumination correction, cell
brightness gradient 3.7× → 1.2×) + rolling-ball residual background cleanup.
All three channels C0/C1/C2 are fully preprocessed (`clean/`, 36 files). The original tif_max is
only a fallback.
Soma radius slider 3.0 → **2.4 µm** (the fjff base gives slightly thinner masks; at 3.0 the weak
samples all lose their soma anchor and get wrongly killed by the fragment filter).
Known unrecoverable regions: the dense fold band on the right of F-WT1, most of M-Het5, the right
edge of M-WT3 — pure haze, extracting zero there is the correct behaviour, don't select them.

## Recommended recipe (chosen from two rounds of experiments 2026-08-09; these are the defaults)

**Sato σ0.6 µm + brightness gate z2.0 + union with brightness mask + local z3.5 + spur pruning
×1.2**, everything else as in v10. (The UI calls this slider **Spur pruning × soma radius**.)

Evidence (clear window for continuity, thick-structure window for false branch points, pure-haze
window M-Het5 for false positives):
- Versus the v10 baseline: in the clear window skeleton length ≈ ×1.8, fragment rate
  (components/mm) roughly halved, dim processes reconnected to the main branch
- **Union**: single-scale sato responds only along the two edges of a thick soma (thick structures
  get cut into parallel thin strips, false branch points explode); the brightness mask puts the
  thick bright block back whole. Branch points in the problem window 23 → 12
- **Spur pruning × soma radius**: only deletes end-branches whose "length < the local radius at their
  branch point" (geometric artifacts of thick-block skeletons); at fine structures the threshold
  automatically drops and real branches are untouched. Branch points in the problem window drop
  further 12 → 8; at ×2.0 it starts deleting real branches and degrades the tree identity, don't go
  above ~1.5
- In the pure-haze window every recommended combination extracts **0** (sato without the brightness
  gate leaks 70 µm; hysteresis + sato without the gate leaks 600 µm+ — do not turn off the
  brightness gate)
- **If you want more continuity** (dim segments breaking apart a structure that is obviously one
  piece): switch to "hysteresis z", start the low threshold at 1.5 (measured: reconnects many dim
  links, haze areas still 0); below a low threshold of ≈1.0 it starts eating haze, so watch it
  yourself while tuning
- Large round bright blobs still produce a short skeleton ("a round dot turned into a strip"); just
  remove them later at the data level by circularity/solidity
- **Round-cell filter**: skeleton <20 µm and ≥80% of it huddled near the soma = a bright dot with no
  processes, delete
- **Speckle-web filter v3** (two calibration rounds): only deletes uncontroversial junk — no soma +
  <25 µm + has a branch point + median brightness along the path <10σ.
  v2 used to delete by "<50 µm with 2 branch points", which wrongly killed real 30-50 µm cells in
  the dense band (their somas are too dim to pass the 3 µm solidity slider) — when in doubt, keep
- **Local window default 40 µm** (was 20): in dense areas a 20 µm window is all signal, which raises
  the local median and suppresses real cells; at 40 µm the skeleton is +21%, all of it real cells
  recovered in the dense band, and haze areas are still 0
- If you touch a parameter by accident, click **↺ Reset defaults** in the top bar to restore
  everything at once

## Two scientific red lines

1. **The criteria for choosing ROIs must be defined in advance and be the same for all groups**
   (e.g. sharpness ≥ some value, no folds, no bubbles); you cannot "pick whichever area gives nice
   results" — the former is routine histology (ROI sampling + normalization by area is standard
   practice), the latter is selection bias.
2. Doing single-cell skeletons on 10x/NA≈0.3 images **systematically underestimates the branch
   point count** (adjacent fine processes fuse optically); between-group comparison is still valid,
   but do not compare the absolute values with high-magnification literature; metrics like total
   length/area and soma density are the most stable.

## Files

**Configuration**
- `config.py` — `PIX_UM` and `IMG_PX`. Set these first for a new image set.
- `requirements.txt` — pinned versions the published numbers were produced with.

**Pipeline (run in this order)**
- `preprocess_clean_images.py` — Step 0: flat-field + rolling-ball background → `clean/`
- `server.py` + `index.html` — the tuner / ROI drawing UI (port 8901). `index.html` is pure
  static, no Node needed. Every batch script imports `server.py`.
- `stats_baseline_parallel.py` — baseline stats; also defines `hedges_g` / `perm_p` / `bh_fdr`,
  which the next two import — do not delete it
- `stats_corrected.py` — **AUTHORITATIVE** stats (ROI clipping, tissue denominator, overlap dedup)
- `stats_round_cells.py` — round / amoeboid-cell metrics

**Optional outputs**
- `render_round_cell_gallery.py` → `analysis_output/round_cell_gallery/` (JPEGs + index.html)
- `make_pdfs.py` → two PDFs; run the gallery first
- `analysis_group_comparison.py` — standalone one-pass alternative analysis

**Exploratory, kept for the record — not part of the pipeline, do not re-run**
- `experiment_connectivity.py`, `experiment_thick_structures.py` — parameter comparison sweeps
  (write `cache/exp*.png`); their hand-picked windows are pixel coordinates in the 2026-08 images
- `diagnose_missed_round_cells.py`, `calibrate_round_brightness.py` — round-cell gate calibration
  and missed-blob triage

**Data**
- `rois/*.json` — the hand-drawn ROIs per sample (irreplaceable human input)
- `clean/` — the analysis-ready images (not shipped; symlink the lab drive's copy in)
- `cache/` — per-sample noise-floor cache + experiment images; created automatically
- `analysis_output/` — everything the scripts write
