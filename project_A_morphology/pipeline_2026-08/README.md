# Choroid plexus immune-cell morphology — 2026-08 pipeline

Segments CCR2/CD45-labelled immune cells in mouse choroid plexus on 10x tile-scan max projections,
reduces them to a skeleton, and reports density per unit tissue area. Twelve animals, 74 hand-drawn
ROIs. **The unit of statistics is the animal, not the image field.**

Images and measured results are not in this repository — the measurements are unpublished and stay
with the Xu lab, along with the write-up that has the ROI-selection criteria, the full parameter
table and the results.

## What the extraction looks like

Left: background-subtracted input. Right: the same field with the extraction drawn on top.
Blue = mask, red = skeleton, yellow = soma outline, cyan = endpoints, magenta circle = round cell.

![ROI overlay](../../docs/figures/06_extraction_overlay.png)

At native resolution — one pixel of skeleton per image pixel, no upsampling:

![extraction detail](../../docs/figures/07_extraction_zoom.png)

Every one of the 74 ROIs was reviewed this way; `render_round_cell_gallery.py` regenerates the set.

## Run

```bash
python3.9 -m venv .venv
./.venv/bin/pip install -r requirements.txt "tabulate==0.9.0"   # see Gotchas

export CHP_SRC=/path/to/tif_max                 # or edit config.py
export CHP_FJFF=/path/to/tif_fiji_flatfielded
ln -s /path/to/clean ./clean

./.venv/bin/python stats_baseline_parallel.py                   # must run first
./.venv/bin/python stats_corrected.py   | tee stats_corrected.txt
./.venv/bin/python stats_round_cells.py | tee stats_round_cells.txt

./.venv/bin/python server.py                    # workbench, http://localhost:8901
```

`stats_corrected.py` is authoritative — it adds ROI clipping, a tissue-area denominator, and overlap
de-duplication. It and `stats_round_cells.py` print effect sizes to **stdout only**, so capture them.
`stats_baseline_parallel.py` must run first: the other two import `hedges_g` / `perm_p` / `bh_fdr`
from it, and it creates `analysis_output/`. `TOOL_USAGE.md` covers the workbench controls.

## Data layout

Three directories, produced in this order from the `.czi` originals:

| | what it is | read by |
|---|---|---|
| `tif_max/` | tile-stitched max-intensity Z projections | `server.py` (sample list), `stats_corrected.py` (tissue mask) |
| `tif_fiji_flatfielded/` | the above, illumination-corrected in Fiji | `preprocess_clean_images.py`, `rebuild_clean.py` |
| `clean/` | flat-fielded minus an 8 µm rolling ball, plus a per-image noise floor σ | everything else |

Every threshold in the pipeline is in units of that image's own σ — the batch has no brightness
calibration and σ spans a 6× range across it.

## Files

| | |
|---|---|
| `server.py` | The engine: the frozen 39-key `DEFAULTS` recipe, the per-ROI `process()`, and the workbench backend. Every other script imports it. |
| `index.html` | Workbench front end. Read from disk by `server.py`; must stay beside it. |
| `config.py` | The only place data paths live, plus `PIX_UM` / `IMG_PX`. |
| `preprocess_clean_images.py` | Step 0: flat-fielded TIF − 8 µm rolling ball → `clean/`, plus a σ sidecar. |
| `rebuild_clean.py` | The same step 0, standalone — paths on the command line, no `config.py`. Reproduces the shipped `clean/` byte for byte; `--check SHA256.txt` proves it per file. |
| `stats_baseline_parallel.py` | Animal-median collapse, Hedges g, 10 000-draw permutation, BH-FDR, Spearman confound check. Home of the shared helpers. |
| `stats_corrected.py` | The authoritative rerun. |
| `stats_round_cells.py` | Round/amoeboid-cell metrics. |
| `rois/` | 74 hand-drawn boxes, one JSON per animal: `{id, x, y, w, h}` in full-scan pixels. **Hand-made and recorded nowhere else** — the numbers cannot be reproduced without them. |
| `render_round_cell_gallery.py`, `make_pdfs.py` | Regenerate the review images above, and bundle them into PDFs. |
| `calibrate_round_brightness.py`, `diagnose_missed_round_cells.py` | One-off QC: the sweep that set `round_bright_z = 50`, and blob-by-blob attribution of misses. |
| `analysis_group_comparison.py`, `experiment_*.py` | Development leftovers, **not results**. The first overwrites `analysis_output/*.csv` with an incompatible schema — point it elsewhere or leave it alone. |

## Three ways this silently gives wrong numbers

1. **Missing `clean/`.** `server.py` does not fail. It falls back to raw images with local background
   subtraction re-enabled and returns different numbers with no warning.
2. **A wrong `PIX_UM` or `IMG_PX`.** Nothing validates them, and every µm and per-mm² figure scales
   off `PIX_UM`.
3. **`REUSE=1` after a parameter change.** `stats_baseline_parallel.py` skips extraction and hands
   back the previous CSV — old numbers under a new recipe.

No output carries a version stamp, so a CSV made under an edited `DEFAULTS` is indistinguishable
from one made under the frozen recipe. If you re-tune, re-run everything.

## Gotchas

- **Python 3.9.** On 3.12/3.13 the pinned `numpy` and `scikit-image` have no wheels and pip falls
  into a source build that never finishes. Keep `scikit-image==0.24.0` — `skeletonize`, `sato`,
  `apply_hysteresis_threshold` and `remove_small_objects` all changed behaviour across releases.
- **`tabulate` is missing from `requirements.txt`**, and `stats_baseline_parallel.py` ends in a
  `to_markdown()` call — the analysis runs to completion and *then* dies writing its report.
  `matplotlib` and `img2pdf` are omitted too; only the QC scripts need them.
- **`Pool(6)` is hard-coded** in both stats scripts. Lower it under 32 GB of RAM.
- The scripts memory-map ~14 GB of TIFs. That is fine — they are never read in full.

## On a different image set

- `server.py` finds samples by the filename pattern `10x_<sample>_CCR2-CD45_C0.tif`; anything else is
  invisible to the workbench. Override with `CHP_FILE_RE` / `CHP_FILE_FMT`.
- Group assignment is parsed from the sample name: `sex = sample[0]`, and `genotype = "WT" if "WT" in
  sample else "Het"`, case-sensitive. An ID like `Ctrl1` is silently labelled `Het` — you get a
  100 %-Het dataset and a NaN effect size rather than an error.
- Check `PIX_UM` and `IMG_PX`. They are frozen to this cohort (0.164827 µm/px).
- New ROIs must be drawn; `rois/` covers these twelve animals only.
