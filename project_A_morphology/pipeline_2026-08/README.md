# Choroid plexus immune-cell morphology — code bundle

The 2026-08 pipeline for the twelve-animal WT/Het choroid plexus cohort — the version every reported
number came from. This file is about running the code.

The companion write-up (`PROTOCOL.md`: ROI selection criteria, the full parameter table, the
statistics, and the results) lives with the Xu lab, alongside the images. It is not in this
repository because the measurements it contains are unpublished. Ask Stephen for a copy if you
need the ROI-selection criteria or the re-tuning procedure.

The images are not here either — see **Data layout** below for what the code expects.

These scripts are an English-renamed copy of the author's working set. The rename was checked, not
assumed: `process()` was run on 12 ROIs across 6 animals under both versions and every returned
metric and every mask/skeleton pixel came out identical, with the 39-key `DEFAULTS` recipe unchanged.

---

## Run order

```bash
cd /path/to/code
python3.9 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/pip install "tabulate==0.9.0"        # requirements.txt omits it — see Gotchas

rm -r clean && ln -s /path/on/lab/drive/clean ./clean    # point at the images
# edit the absolute paths listed below

./.venv/bin/python stats_baseline_parallel.py
./.venv/bin/python stats_corrected.py   | tee analysis_output/stats_corrected_stdout.txt
./.venv/bin/python stats_round_cells.py | tee analysis_output/stats_round_cells_stdout.txt
```

`stats_corrected.py` is the authoritative run: it adds the three robustness corrections (ROI
clipping, tissue-area denominator, overlap de-duplication). It and
`stats_round_cells.py` print their effect sizes and p-values to **stdout only**, so capture them.

`stats_baseline_parallel.py` must run first and must not be deleted — the other two import
`hedges_g`, `perm_p` and `bh_fdr` from it, and it is what creates `analysis_output/`.

Interactive workbench, for drawing ROIs and inspecting the extraction:

```bash
./.venv/bin/python server.py     # then open http://localhost:8901
```

`TOOL_USAGE.md` explains the controls. Step 0 (`preprocess_clean_images.py`) is only needed if
`clean/` does not exist yet; for the 2026-08 cohort those images are already on the lab drive.

## What each file is

**Pipeline**

| File | Role |
|---|---|
| `server.py` | The engine: the frozen recipe `DEFAULTS` (39 keys), the per-ROI pipeline `process()`, and the workbench backend. Every other script imports it. |
| `index.html` | Workbench front end, read from disk by `server.py`. Must stay beside it. |
| `config.py` | `PIX_UM` and `IMG_PX`. **Check both before running on a new image set.** |
| `preprocess_clean_images.py` | Step 0: flat-fielded TIF minus an 8 µm rolling-ball background → `clean/<sample>_<ch>_clean.tif` plus a sidecar holding that image's noise floor σ. |
| `rebuild_clean.py` | The same step 0, standalone: no `config.py`, paths on the command line, writes `clean/` beside `tif_max/` and `tif_fiji_flatfielded/` on the delivery drive. Use this one if `clean/` is ever lost — `python rebuild_clean.py <drive>/ChP_morphometry_2026-08/images`. Verified byte-identical to the shipped `clean/` on all 12 C0 images; pass `--check SHA256_clean.txt` to re-prove it per file. |
| `stats_baseline_parallel.py` | Primary statistics over the 74 ROIs: animal-median collapse, Hedges g, 10 000-permutation p, BH-FDR, Spearman confound check. Home of the shared stats helpers. |
| `stats_corrected.py` | The authoritative rerun with the three corrections. |
| `stats_round_cells.py` | Round/amoeboid-cell metrics and the skeleton-gap fraction. |
| `rois/` | The 74 hand-drawn ROI boxes, one JSON per animal. **Irreplaceable human input** — nothing else records them and the numbers cannot be reproduced without them. Back this up. |

**QC and diagnostics** — useful, not required

| File | Role |
|---|---|
| `render_round_cell_gallery.py` | Renders all 74 ROIs at full resolution, original beside marked-up, as a browsable gallery. This is how the round-cell calls were audited by eye. |
| `make_pdfs.py` | Assembles that gallery into two PDFs (74-ROI comparison; 12 full scans). Needs the gallery script to have run first. |
| `calibrate_round_brightness.py` | The one-off sweep over 685 round-cell candidates that produced `round_bright_z = 50`. |
| `diagnose_missed_round_cells.py` | Attributes missed bright blobs to the pipeline stage that dropped them. Written to find a specific bug (see `../results/README.md`); keep it for the next time detection looks wrong. |

**Exploratory — not results**

| File | Role |
|---|---|
| `analysis_group_comparison.py` | An earlier independent group analysis. **It writes `analysis_output/roi_metrics_all.csv` and `animal_summary.csv` with a different column schema than `stats_baseline_parallel.py`**, so running it afterwards overwrites those files with incompatible content. Use a separate output directory, or leave it alone. |
| `experiment_thick_structures.py`, `experiment_connectivity.py` | Parameter probes from development. They write PNGs to `cache/`. |

## Paths you must edit

All still point at the author's machine. Find every one with `grep -rn "/path/to" .`

| File | What | If you miss it |
|---|---|---|
| `server.py`, `SRC` | the `tif_max` directory | **the server refuses to start** — `list_samples()` calls `os.listdir(SRC)` unguarded, before the port is bound |
| `stats_corrected.py`, in `tissue_mask8()` | the same `tif_max` directory | the tissue-area correction fails |
| `preprocess_clean_images.py`, `SRC` and `FJFF` | `tif_max` and the flat-fielded TIFs | Step 0 only |
| `config.py`, `PIX_UM` / `IMG_PX` | not paths, but check them | wrong numbers, silently |

`clean/`, `rois/`, `cache/` and `analysis_output/` resolve **relative to the script's own directory**
and need no editing — but `clean/` must be populated and `cache/` must exist. Both ship with a
placeholder `README.txt`.

## Three ways this silently produces wrong numbers

1. **Missing `clean/`.** `server.py` does not fail. `get_clean()` returns nothing, the pipeline falls
   back to the raw images with local background subtraction re-enabled, and you get different numbers
   with no warning. Confirm `clean/` is populated before trusting output.
2. **A wrong `PIX_UM` or `IMG_PX`.** Nothing validates them. Every micrometre and per-mm² figure
   scales off `PIX_UM`.
3. **`REUSE=1` after a parameter change.** `stats_baseline_parallel.py` then skips extraction and
   hands back the previous `roi_metrics_all.csv` — old numbers under a new recipe.

Related: there is no version stamp in any output, so a CSV made under an edited `DEFAULTS` is
indistinguishable from one made under the frozen recipe. If you re-tune, say so in the notebook entry
and re-run everything.

## Gotchas

- **Python 3.9 specifically.** On 3.12/3.13 the pinned `numpy` and `scikit-image` have no wheels and
  pip falls into a source build that never finishes. Pin `scikit-image==0.24.0`: `skeletonize`,
  `sato`, `apply_hysteresis_threshold` and `remove_small_objects` change behaviour across releases.
- **`tabulate` is missing from `requirements.txt`.** `stats_baseline_parallel.py` ends in a
  `to_markdown()` call, so without it the whole analysis runs and *then* dies writing its report.
  `matplotlib` and `img2pdf` are likewise omitted; they are needed only by the QC scripts.
- **`Pool(6)` is hard-coded** in both stats scripts. Lower it below 32 GB of RAM.
- Both `server.py` and the stats scripts memory-map ~14 GB of TIFs. That is fine; they are not read
  in full.

## Applying this to a new image set

Three things are wired to the 2026-08 cohort and will not announce themselves:

- `server.py` finds samples by the filename pattern `10x_<sample>_CCR2-CD45_C0.tif`. Differently
  named files are invisible to the workbench.
- `preprocess_clean_images.py` has the twelve sample IDs hard-coded and produces nothing for animals
  not on that list.
- Group assignment is parsed from the sample string: `sex = sample[0]` (`F`/`M`) and
  `genotype = "WT" if "WT" in sample else "Het"`, case-sensitive. An ID like `Ctrl1` is silently
  labelled `Het`, which yields a 100 %-Het dataset and a NaN effect size rather than an error.

New ROIs must also be drawn — `rois/` covers these twelve animals only. `rois/<sample>.json` is a
list of `{id, x, y, w, h}` boxes in pixel coordinates on the full scan; they are hand-drawn and
nothing else records them, so they are the one irreplaceable input in this directory.

## Data layout

No images are in this repository. The code expects three directories, produced in this order from
the `.czi` originals:

| | what it is | read by |
|---|---|---|
| `tif_max/` | tile-stitched maximum-intensity Z projections | `server.py` (sample list), `stats_corrected.py` (tissue mask) |
| `tif_fiji_flatfielded/` | the above, illumination-corrected in Fiji | `preprocess_clean_images.py`, `rebuild_clean.py` |
| `clean/` | flat-fielded minus an 8 µm rolling-ball background, plus a per-image noise floor σ | everything else |

Point `CHP_SRC` and `CHP_FJFF` at the first two (or edit `config.py`), and put `clean/` beside the
scripts. `rebuild_clean.py` regenerates `clean/` from `tif_fiji_flatfielded/`:

```bash
python rebuild_clean.py /path/to/_work        # writes /path/to/_work/clean
```

It is a standalone copy of step 0 with the paths on the command line and no `config.py` import.
On the 2026-08 cohort it reproduces the shipped `clean/` byte for byte on all twelve C0 images.

`PIX_UM` and `IMG_PX` in `config.py` are frozen to that cohort (0.164827 µm/px). **Check both before
running on any other image set** — nothing validates them, and every micrometre and per-mm² figure
scales off `PIX_UM`.
