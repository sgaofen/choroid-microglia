# cpstab — Python port of the Shipley 2020 z-stack stabilization pipeline

Port of `references/Shipley2020/registration/` (LehtinenLab/Shipley2020;
Shipley FB et al., *Neuron* 2020;108(4):623-639). The MATLAB original is
**ground truth**: every ported function mirrors its `.m` counterpart's name
(snake_case), argument order, and defaults, and its docstring's first line
names the source file and line range.

Target: Python 3.9, `numpy` / `scipy` / `skimage` / `tifffile` only.
No MATLAB, no JVM, no Fiji.

## Quick start

```python
from cpstab import RegistrationConfig, run_pipeline

cfg = RegistrationConfig(
    input_path="/data/0309-0721-009_200813_002.sbx",  # sidecar .mat beside it
    refchannel=1,        # 1-based, MATLAB habit: 1 = red vessels, 2 = green
    scale=4,
    chunksize=20,
    proj_type="mean",
    proj_range="quarter",  # what the published pipeline actually used
    opttype="none",        # piezo; the only supported mode
    write_registered=False,
    mode="replicate",         # "improved" = the four corrections, see below
    compute_dtype="float64",  # "float32" = fast mode, see below
)
zproj = run_pipeline(cfg)   # also writes <stem>_mean_zproj.tif
```

The two port-extension knobs — `mode` and `compute_dtype` — are independent
and orthogonal: `mode` decides **which algorithm** runs, `compute_dtype`
decides **what precision** it runs in. Both default to the MATLAB-faithful
setting, so the snippet above is bit-for-bit the validated port.

## Conventions

- **Arrays**: single frame `[Y, X]` (MATLAB `[row, col]`); volume `[Y, X, Z]`;
  time block `[Y, X, Z, T]` — same axis order as the MATLAB code, *not* the
  C-order `[T, Z, Y, X]` habit. The channel dimension lives in the IO layer;
  the registration core sees one channel.
- **Indexing at module boundaries**: `refchannel` and `proj_range` are
  **1-based** (MATLAB habit), converted to 0-based only inside the IO/apply
  layers. Beware: the original's `Nx = info.sz(1)` is the **row** count.
- **dtype**: raw data is uint16; everything is promoted to float64 at the
  read boundary. FFT work stays complex128. `compute_dtype="float32"` opts
  into the fast mode described below, which lowers only the first of those.
- **Divergence from MATLAB is opt-in and has exactly one switch.** Every
  numeric difference from the original lives behind `cfg.mode="improved"`
  (`cpstab/improved.py`); with the default `"replicate"` every guarded branch
  takes the literal code that was there before. Two regressions enforce this
  and must both pass before any change lands: `tests/test_synthetic.py` 7/7,
  and the 40-volume subset reproducing
  `reportA/port_run/FAD-F_1_T0-39_mean_zproj.tif` **bit for bit**.

## Stage correspondence (MATLAB → Python)

`RegistrationMasterPipeline.m` (44-line entry script) → `cpstab.pipeline.run_pipeline(cfg)`:

| # | MATLAB (file, lines) | What it does | Python |
|---|---|---|---|
| 0 | `RegistrationMasterPipeline.m` L1 | `tic` (no matching `toc`) | timing in `run_pipeline` |
| 1 | L3-5 `javaaddpath` mij/ij-1.52a/TurboRegHL | JVM bootstrap for MIJ/Fiji | **dropped** — no JVM anywhere |
| 2 | L8-15 hardcoded mouse/date/run/server/fbase, `opttype`, `refchannel` | run identification | `RegistrationConfig` (`config.py`) |
| 3 | L17-18 `pipe.lab.datedir`/`datapath` (→ `pathbase` hostname→drive table) | machine-specific path resolution | **dropped** — explicit `cfg.input_path` / `cfg.out_dir` (same move as `clean/run_registration.m`) |
| 4 | L20 `GetDimensions.m` | `[Nchan,Nx,Ny,Nz,Nt]` | `VolumeSource` attributes / `get_dimensions` (IO module) |
| 5 | L22-26 `scale=4`, `chunksize=20`, `Nchunks=round(Nt/chunksize)`, `proj_range`, `proj_type` | parameters | `RegistrationConfig` defaults + `cfg.nchunks(nt)` (MATLAB half-away-from-zero `round`) |
| 6 | L29-32 `ConvertOIR_SBX.m` (+ `SpoofSBXinfo3D`, `RegWriter`, `load_tiff_nobar`) | FluoView TIFF export → `.sbx` | **dropped** — see "What was removed" |
| 7 | L34 `CalculateOptotuneWarp.m` | per-z affine warp; `'none'` → identity ×Nz (its L20-23) | inlined identity (`pipeline._optotune_tforms`); `'affine'/'rigid'` → `NotImplementedError` |
| 8 | L37-38 `DFT_warp_3D_2.m` (`reftype='mean'` explicit) → `.dftshifts` | **core**: chunked 3D DFT shift estimation (`DFT_rect`, `defineReference`, `DetermineXYShiftsFBS`, `ApplyXYShiftsFBS`, `dftregistrationAlex`, `dftregistration3D`) | `orchestrator.dft_warp_3d_2(...)` → `<stem>.dftshifts.npz` |
| 9 | L41 `MakeSBXall.m` (+ `zproj_reg.m` refinement) | apply warp+XY+Z shifts to **all** channels, z-project, write `.sbxall` | `apply_project.make_sbxall(...)`; registered-stack write now **opt-in** (`write_registered=False`) |
| 10 | L43-44 `write2chanTiff.m` via `uint16(...)` + MIJ/ImageJ | write `<run>_<proj>_zproj.tif` | `matlab_uint16(...)` (round-half-away + saturate + NaN→0) + `write_zproj_tiff` (`tifffile`, ImageJ-compatible) |

`clean/run_registration.m` / `clean/config_example.m` are the MATLAB-side
predecessors of this config design; `config.py` mirrors their fields.

## What was removed (relative to the original)

**Behavioral changes, all opt-in or externalized:**

- **`.sbxall` registered stack: not written by default.** The original
  unconditionally wrote a ~1× full-size `.sbxall` that *nothing in the repo
  reads back* (no sidecar `.mat`, no reader — CODEMAP §5-G/§6). This was the
  single biggest disk-bloat item. `write_registered=True` restores the
  original output — a byte-identical `.sbxall` stream next to the input
  (RegWriter contract, verified in `tests/scratch_apply_project.py`).
- **FluoView manual export + `ConvertOIR_SBX` conversion: dropped.** The
  original assumed a by-hand FluoView export to `<fbase>.tif.frames/`
  followed by an in-pipeline TIFF→`.sbx` conversion (which also wrote a dead
  `.txt` copy and held 2× data on disk during conversion). The port ingests
  `.sbx` (or TIFF) via `VolumeSource`; `.oir` inputs are converted once,
  externally, with Bio-Formats `bfconvert`.
- **MIJ / Miji / ImageJ / TurboReg: dropped entirely.** The only live Fiji
  dependency on the demo path was the final TIFF write (`write2chanTiff.m` →
  MIJ hyperstack); replaced by `tifffile`. The optotune
  MultiStackReg/TurboReg path (`opttype='affine'/'rigid'`) was already dead
  on the piezo/`'none'` path and is not ported (`NotImplementedError`).
- **Dead weight files not reproduced**: `<fbase>.txt` copy,
  `tforms_optotune.mat`; `ref_all`/`intermediate_shifts` inside the shift
  file are optional (nothing downstream reads them).

**Dead code not ported** (verified unreachable in CODEMAP §10):
`zproj.m`, `writeTiff.m` + `arrtoij.m`, `MultiStackReg_Fiji_affine.m` (GUI
variant; its only caller has an undefined-variable bug), `aligned.m`,
`metadata.m`, the whole `pathbase/mousedir/datedir/rundir/datapath` chain,
`load_tiff_nobar.m` (wrong function name), `parfor_progressbar.m`,
`sort_nat.m` (conversion-path only), and the `DFT_warp_3D_2.m` L48
`unwarp_chunk` typo branch (unreachable). The `vessel segmentation/` folder
is out of scope (separate downstream analysis).

## Algorithm modes (`cfg.mode`)

Port extension, no MATLAB counterpart. `cpstab/improved.py` owns the
switchboard; `run_pipeline` installs the setting for the duration of a run and
restores it afterwards (including on an exception).

`"replicate"` (**default**) is the validated port, bit-for-bit.
`"improved"` turns on four corrections to the original algorithm:

| # | feature flag | where | what it changes |
|---|---|---|---|
| 1 | `global_median` | `apply_project` (MakeSBXall.m L29-35) | RS/CS centring uses a **global scalar** median instead of MATLAB's per-timepoint one |
| 2 | `fourier_shift` | `apply_project` (L93-118, `zproj_reg` L56-64) | apply-side XY translation is a **phase ramp in the sqrt domain** instead of bilinear |
| 3 | `chain_refine` | `orchestrator` (DFT_warp_3D_2.m L52-57) | one **global refinement pass** after the DFT_rect plane chain, gated (see below) |
| 4 | `subplane_z` | `apply_project` (L111-118) | Z shift **interpolates between planes** instead of rounding to a whole one |

Corrections 1 and 3 change the **shifts**, so an improved run's
`.dftshifts.npz` differs from a replicate run's; 2 and 4 change only how a
given shift is applied.

**Why correction 2 shifts `sqrt(x)` and not `x`.** A plain phase ramp is the
exact band-limited translation, and this data is not band-limited: photon
counts on a near-zero background ring around every nearly-unresolved feature.
Measured on one interior frame, a plain ramp leaves **25.6% of the pixels
negative** (worst -373 counts, |negative mass| = 20% of the frame). Most
cancels in the z-projection, but 0.145% of the projection's pixels stay
negative and the `uint16` cast turns each into a black speckle with real data
all round it — 33480 of them on the 40-volume subset, against 3 for replicate.
Nonnegativity cannot be clipped back on: every clipping guard inflates the
projection's intensity by 11-14%, and a linear taper strong enough to remove
the negatives is a worse low-pass than the bilinear being replaced (a kernel
that cannot make a negative from nonnegative input *is* a nonnegative kernel,
hence a low-pass). Shifting `sqrt(x)` and squaring back is nonlinear, is the
variance-stabilizing transform for Poisson data, compresses exactly the
contrast that drives the ringing, and cannot return a negative. It costs about
30% of correction 2's headline gain and removes the speckle: correction 2
measured alone goes from 18637 isolated internal zeros to **49**. Full detail
and the rejected alternatives: `cpstab/fourier_shift.py`.

**Why correction 1 is a bug fix and not a preference.** MATLAB centres the
`(Nz, Nt)` shift matrices with `median()` at its default dim, which is a
*per-column* (per-timepoint) median. That subtracts back out, exactly, every
shift term that is constant along the plane axis — and two of the three
registration stages produce exactly such terms:

```
RS_total - median(RS_total, dim=1)
  = (RS + RS_chunk) - median(RS + RS_chunk, dim=1)
  = RS - median(RS, dim=1)          # RS_chunk cancels identically
```

`RS_chunk` (the inter-chunk stitch, DFT_warp_3D_2.m L110-118) is a `nearest`
stretch of a `1 × Nchunks` vector, and `RS2` (the per-volume 3-D registration,
L71-84) is `repmat`'d over planes — so **the volume-level registration and the
inter-chunk anchoring contribute nothing to the applied shifts in replicate
mode.** Verified on the validation subset: dropping `RS_chunk` entirely
changes the centred matrix by 3.6e-15 px. Measured effect on the volume-level
RS trace there — the subject moves ~-13 px over t=15..24; per-timepoint
centring reports that as **+10.8 px of the opposite sign**, and flattens the
CS excursion from -12.7 px to -0.1 px.

**Why correction 3 needs a trust gate** (`cfg.chain_refine_cap`,
`cfg.chain_refine_min_ncc`). Registering a plane against the volume mean only
means something where the two share content. Deep planes do not: on `FAD-F_1`
the plane-vs-mean correlation drops from 0.30-0.76 (z ≤ 23) to 0.039-0.245
(z ≥ 24), where the correlation surface is flat to 1-4% and its argmax is a
uniform draw over the ±N/2 search domain. Folded in unconditionally, that draw
became a real shift: **at t=1300, a quiet timepoint, 17 of 41 planes moved
95-225 px**, taking that volume's RS to `[-10.4, +218.4]` px against
replicate's `[-9.2, +19.0]`, and the zero-fill bands of those planes are
visible in the projection as rectangular seams and ghosting. It compounded
downstream too — the volume is rebuilt at the bogus shift, so
`DetermineXYShiftsFBS` then measured a mostly-black frame and its own CS
spread went 4.1 px → 189 px. Over the full run: 35.3% of (plane, timepoint)
cells past 50 px, in 1492 of 1500 timepoints.

So a plane's correction is applied only if it is both small and backed by a
real peak, and is otherwise **dropped in favour of the DFT_rect chain value**
— which is what `replicate` does for that plane, so a rejection can never be
worse than the MATLAB-faithful result:

```
accept  iff  max(|dR|, |dC|) <= cfg.chain_refine_cap        # default 3.0
             and NCC_zm(plane, volume mean) >= cfg.chain_refine_min_ncc   # 0.30
```

`chain_refine_cap` is in **registration-grid px** (× `cfg.scale` for
full-resolution px). The default 3.0 comes from a 4100-sample survey where the
correction magnitudes are bimodal: honest ones (correlation ≥ 0.40) have p99
1.0 and max 2.0 grid px, the mislocks start at 12.75, and the valley between
holds 0.7% of samples — every one a deep low-correlation plane. The NCC gate
is the independent second line: it catches a mislock that lands *near* the
origin, which no magnitude cap can. `cap=float('inf'), min_ncc=0.0` restores
the ungated behaviour. `tests/test_improved.py::test_4b/test_4c` pin both
halves (rejection is bit-exact fallback; a real ~1 px residual still gets
through).

Measured on the 40-volume subset (see `tests/test_improved.py` for the full
table, the pre-gate numbers, and `cpstab/metrics.py` for what the columns
mean):

| run | ch | resid px (median) | resid px (p95) | sharpness | field ratio |
|---|---|---|---|---|---|
| replicate | 1 | 0.0636 | 0.1203 | 0.00632 | 0.743 |
| replicate | 2 | 0.1962 | **3.3905** | 0.00565 | 0.836 |
| improved | 1 | 0.0316 | 0.0672 | 0.01259 | 0.955 |
| improved | 2 | 0.0685 | **0.1506** | 0.01138 | 0.935 |

Improved mode is also **~1.4× faster** (52.7 s → 38.0 s on this subset): the
phase-ramp shift is cheaper than the `scipy` bilinear + uint16 requantize loop
it replaces.

(Those two `improved` rows read 0.0224 / 0.0500 / 0.01758 / 0.985 and
0.0671 / 0.1204 / 0.01406 / 1.137 before correction 2 moved into the sqrt
domain. The drop is the price of the speckle fix above — note that
`field_noise_ratio`, the control on `sharpness`, fell with it, i.e. part of
what the plain ramp scored as sharpness was the ringing.)

Two caveats the table cannot show, both spelled out in
`tests/test_improved.py`: correction 2 does **not** reduce motion (it changes
only how a shift is applied — its apparent residual gain is the metric getting
a sharper image to localize), and 40 volumes over 2 chunks is far too short to
exercise correction 1, whose subject is inter-chunk drift. Treat the numbers
as evidence about the *mechanisms*, not as a result about the dataset.

**Ablation.** Each correction has an independent override that beats the mode
in both directions, for attributing a change to one of them:

```python
from cpstab import improved
with improved.feature_scope(chain_refine=False):   # improved minus #3
    run_pipeline(cfg_improved)
with improved.feature_scope(subplane_z=True):      # replicate plus #4
    run_pipeline(cfg_replicate)
```

`fast_run.py` takes `--mode improved`, plus `--chain-refine-cap` /
`--chain-refine-min-ncc` for the gate. Like the compute dtype these are
per-process globals, shipped in each worker's job tuple and installed on the
worker's first line; forgetting the mode makes a worker run `replicate` — the
MATLAB-faithful path, never a silently half-improved one.

## Measuring a run (`cpstab/metrics.py`)

```
python -m cpstab.metrics replicate=a_zproj.tif improved=b_zproj.tif --stride 4
```

Three numbers per channel, chosen so they cannot all be gamed at once:
**`residual_px_median` / `_p95`** (sampled frames phase-correlated against the
temporal mean at usfac=100 — the primary metric, and the only one that
measures displacement rather than contrast), **`sharpness`** (gradient energy
of the time-averaged image, normalized by mean intensity²), and
**`field_noise_ratio`** (power at the vertical Nyquist frequency — the
odd/even scan-line alternation — over the median upper-band power). The third
is the *control* on the second: gradient energy is also produced by noise, so
a run that merely preserved more high-frequency noise would score higher on
`sharpness` alone. All three are measured on a central 80% crop, because
stabilization opens a moving black border and including it would reward the
run with the *larger* shifts.

## Precision modes (`cfg.compute_dtype`)

Port extension, no MATLAB counterpart (MATLAB is `double` throughout).
`cpstab/precision.py` owns the boundary; `run_pipeline` installs the setting
for the duration of a run and restores it afterwards.

| | `"float64"` (default) | `"float32"` (fast mode) |
|---|---|---|
| pixel storage, interpolation, projection | float64 | **float32** |
| DFT phase correlation (every shift estimate) | complex128 | complex128 |
| shift bookkeeping + `.dftshifts.npz` | float64 | float64 |
| final `uint16()` cast of the projection | float64 | float64 |

**The rule: fast mode lowers the class of PIXELS; it never lowers the class of
the correlation arithmetic that DECIDES A SHIFT.** That exception is forced by
measurement, not caution. Every shift here is an `argmax` over a phase
correlation surface, and on this data that surface is nearly flat — the DC
term is ~91% of the peak height and the winning sample beats the runner-up by
a median 4.5e-4 relative (worst 5.6e-6), while float32 summation noise over
the 256×256 padded inverse transform is ~3e-5. Letting the engine follow the
compute dtype moved per-plane shifts by up to **15.5 px** and dropped the
output projection to **Pearson r = 0.756**. `tests/test_f32.py::test_4` is the
negative control that keeps that boundary in place.

Measured on the 40-volume validation subset (`FAD-F_1_T0-39.tif`,
refchannel=1 scale=4 chunksize=20 proj_range=quarter, macOS / numpy 2.0.2,
single process):

| | float64 | float32 |
|---|---|---|
| wall clock | 55.9 s | 40.0 s (**1.33–1.40×**) |
| registration stage | 7.0 s/chunk | 6.2 s/chunk (1.12×) |
| apply + refine | remainder | ~1.5× |
| `.dftshifts.npz` | — | bit-identical |
| `zproj_mean` (float) | — | max\|Δ\| 6.9e-06 on a 0..226 range (5.4e-08 relative), r = 0.999999999999977, NRMSE 3.1e-09 |
| written uint16 TIFF | — | **bit-identical** (0 of 20,971,520 pixels) |

The bit-identity is a *result on this dataset*, not a promise — the
float-level difference is ~5e-8 relative, so a pixel sitting that close to a
`.5` rounding tie can still come out one count different. Use `float64` when
exact reproduction is the requirement, and re-validate fast mode on a new
dataset before trusting it there.

`fast_run.py` (the multiprocessing driver) takes `--compute-dtype float32`.
The setting is a per-process global, so each worker installs it itself from
the dtype shipped in its job tuple; forgetting that makes a worker run
float64 — slow but correct, never the other way round.

## Running it at scale (`scripts/relayout.py` + `fast_run.py`)

`run_pipeline` is the reference implementation and stays serial. For the
114 GB production stack the driver is `fast_run.py`, and the input should be
converted once:

```bash
# 1. transpose the Z-major bfconvert OME-TIFF into a T-major volume store
python scripts/relayout.py --in RAW.ome.tif --out RAW.tzcyx.npy

# 2. run it
python fast_run.py --raw RAW.tzcyx.npy --out-dir OUT --workers 10 \
                   [--dtype float32] [--mode improved] [--read-mb 2048] \
                   [--compare SERIAL_zproj.tif]
```

`--raw` accepts either container; `VolumeSource` serves both through the same
interface and returns the same values bit for bit (io_rw PORTING NOTES #16).
Two things only the `.npy` path can do:

- **one volume is one contiguous byte range**, instead of `Nz` page reads
  scattered through the file — the fix for the parallel-read collapse
  (measured 2.4× on 10 processes before, because the workers thrashed the
  drive's request queue);
- **the apply stage reads in bulk**: each worker pulls its slab with a single
  `VolumeSource.read_block()` and computes out of RAM, so its sequential read
  overlaps the other workers' compute (`--read-mb` caps one read; `0` = the
  whole slab). On a TIFF source `read_block` returns `None` and the ordinary
  per-volume path runs unchanged — there is deliberately no second TIFF
  implementation to keep in agreement.

`--dtype` is an alias of `--compute-dtype`; both are the same setting.
Neither it nor `--mode` is ever inferred, so a forgotten flag yields
`replicate`/`float64` — slow and MATLAB-faithful, never silently improved.

**Benchmark ladder.** `scripts/bench_full.sh` runs the whole comparison —
relayout, then `float64/replicate` (bitwise-checked against a serial
reference), `float32/replicate` (diffed against it), `float64/improved`, then
`cpstab.metrics` over truth + both — appending everything to
`bench_log.txt` and collecting `bench_report.md`. It is idempotent per step,
aborts on the first failure, and `--subset` runs the identical ladder on the
40-volume subset. `--dry-run` prints the commands only.

**MATLAB side.** `matlab_bench/` holds what the lab's MATLAB machine needs to
time the ORIGINAL pipeline on the same data: `make_sbx.py` converts the
`.npy`/OME-TIFF into the `.sbx` + `.mat` sidecar that `run_registration.m` can
read (through the ported `RegWriter` / `SpoofSBXinfo3D` contracts, verified by
reading back with the ported `sbxRead`), and `run_benchmark.m` prechecks the
environment and wraps `run_registration` in `tic/toc`. See
`matlab_bench/README.md` (Chinese).

## Batch driver (`scripts/process_run.sh`)

One-button driver for turning a single experiment run's `.oir` volume chunks
into a stabilized z-projection and a metrics report — the tool for working
through the remaining runs one at a time, not a benchmark harness:

```bash
scripts/process_run.sh <oir-chunk-dir> <out-dir> [--dry-run] [--workers N]
```

It is a fixed five-step ladder, each step gated on **product + `<product>.ok`
both present** before it is skipped — looking at the product alone is not
enough, because a step killed mid-write leaves a plausible-looking half-file
behind, and `<product>.ok` is only ever written after that step's command
returns 0:

1. `bfconvert -bigtiff -nogroup <first .oir in the dir> <out>/raw.ome.tif`
   (Bio-Formats follows the volume-chunk naming and stitches the rest in
   automatically from the first file).
2. `scripts/relayout.py --in raw.ome.tif --out raw.tzcyx.npy --verify 16`.
3. Delete `raw.ome.tif` — **the only deletion anywhere in this script**, and
   only once step 2's `.ok` (i.e. `--verify 16` already passed) exists.
4. `fast_run.py --raw raw.tzcyx.npy --out-dir <out>/replicate --workers N
   --dtype float64 --mode replicate` (the MATLAB-faithful, bitwise-checked
   setting — see "Algorithm modes" / "Precision modes" above).
5. `python -m cpstab.metrics --stride 8
   replicate=<out>/replicate/*_mean_zproj.tif > <out>/report.md`, with a
   step-timing table appended.

A step whose product exists **without** a `.ok` is not silently trusted or
silently overwritten — the script has no deletion budget left to spend on it
(rule 3 above) and no way to tell a genuine half-write from a complete run
that just didn't get its marker written, so it halts with a diagnostic and
leaves the file for manual triage. `set -euo pipefail` throughout: any
non-zero exit stops the ladder immediately, and a rerun of the identical
command resumes exactly where it left off. Before touching anything it also
runs a `df` precheck: available space at `<out-dir>` must be `>= 2.2x` the
`.oir` chunk directory's size, or it exits without starting.

`--workers N` (default 10) is `fast_run.py`'s `--workers`, nothing else in
the ladder is parallel. `--dry-run` prints every command it would run,
including the disk precheck's numbers, without executing bfconvert/
relayout/fast_run/metrics or deleting anything. All output — the driver's
own progress lines and every child process's stdout/stderr — is teed to
`<out-dir>/process.log`.

`--skip-convert F` is a **rehearsal-only** escape hatch: it skips the real
`bfconvert` call and symlinks `<out>/raw.ome.tif -> F` instead, so step 3's
"delete `raw.ome.tif`" removes only the symlink, never `F` itself. It exists
to exercise steps 2-5 end to end against a small stand-in file without
touching production `.oir` data; production batch runs should never pass it.

## Testing / validation

- **`tests/test_synthetic.py`** — end-to-end suite on a generated
  64×64×8z×2c×7t uint16 ImageJ TIFF with known per-volume translations
  (≤4 px) + per-plane jitter (±0.5 px) + noise. Runs the full
  `run_pipeline`, asserts output shapes/TIFF/shift-file contents, shift
  recovery (max estimate error observed 0.83 px, asserted < 1.2 px), and
  stabilized-projection residual motion (observed 0.72 px vs 3.34 px raw,
  asserted < 1.0 px and ≥50% reduction). Also covers Nchunks=2 and
  `write_registered=True`. Standalone: `python tests/test_synthetic.py`
  (7/7 passing with the analysis venv).
- **`tests/test_f32.py`** — the float32 fast mode: that the mode is actually
  engaged and its precision boundary is where `precision.py` says (test 1),
  iron law (a) via `test_synthetic.py` (test 2), iron law (b) plus the
  float64-vs-float32 measurement from one pair of full 40-volume runs
  (test 3), and the negative control showing a single-precision correlation
  engine wrecks the shift estimates (test 4). Standalone:
  `python tests/test_f32.py` (4/4; the real-data tests SKIP without the
  workspace files).
- **`tests/test_improved.py`** — the `improved` mode: the switchboard defaults
  to replicate and both scopes restore (test 1), `fshift2` is an exact
  translation with the wrap band cleared on the correct side (test 2),
  `fshift2_vst` — the kernel correction 2 actually ships — cannot return a
  negative on the Poisson-with-puncta field where the plain ramp goes 17%
  negative, while holding the intensity and the mid-band amplitude (test 2b),
  the median annihilation and the sub-plane Z shift as unit facts (test 3), the
  chain refinement removes chain drift **and** the negative control showing it
  stops working without its central crop (test 4), iron law (a) via
  `test_synthetic.py` (test 5), and iron law (b) plus the replicate-vs-improved
  measurement from one pair of full 40-volume runs (test 6). Standalone:
  `python tests/test_improved.py` (9/9; the real-data tests SKIP without the
  workspace files).
- **`tests/test_tmajor.py`** — the T-major `.npy` ingest path: relayout reads
  the axis order from the file rather than assuming one (test 1, every TZCYX
  permutation), the 40-volume real subset relayouts to the expected shape
  (test 2), **every** `(t, c)` volume off the `.npy` equals the one off the
  TIFF against *both* TIFF read strategies (test 3), a full `run_pipeline` on
  the `.npy` reproduces the reference projection bit for bit (test 4), iron
  law (a) in a clean interpreter (test 5), relayout never publishes a store it
  did not fully write (test 6 — the sparse output would read back as
  legitimate zeros), and `read_block` returns exactly what the per-volume
  `get_volume`/`imread_fn` path returns in both compute dtypes (test 7).
  Standalone: `python tests/test_tmajor.py` (7/7; the real-data tests SKIP
  without the workspace files).
- **`validate.py`** — harness for comparing against a real MATLAB
  ground-truth projection:
  `python -m cpstab.validate --raw RAW.ome.tif --truth TRUTH_zproj.tif
  --out REPORT_DIR [--t-range a:b] [--params refchannel=1 proj_range=quarter ...]`.
  Emits a markdown report (per-frame Pearson r / NRMSE, phase-correlation
  residual-offset probes, key-frame comparison PNGs) plus a CSV; the port's
  own outputs land in `REPORT_DIR/port_run/`. Truth axis order is
  auto-detected from the TIFF series (override with `--truth-axes`).

## What is NOT yet verified

- **Bit-exact equivalence to MATLAB.** Validated against the lab's own MATLAB
  product for the FAD-F_1 run (see `../VALIDATION_REPORT.md`): structurally
  equivalent (temporal-mean r = 0.987 after a constant 1.8 px offset,
  stabilization quality tied at 0.01-0.02 px), but not bit-exact — per-frame
  r = 0.61, which the noise-floor calibration puts at the ceiling for two
  chains whose sub-pixel trajectories differ by ~0.5-1 px. Whether the
  remainder is FFT-library rounding or a lab-local script variant is
  unresolved; it does not affect measurements. The MATLAB side has never been
  re-run on this hardware (`matlab_bench/` exists for that).
- **`imresize` parity** (orchestrator): MATLAB `imresize(...,1/scale)` is
  bicubic (a=-0.5) **with antialiasing** (kernel widened when shrinking);
  scipy/skimage have no drop-in equivalent. Whether the port's
  implementation matches to float tolerance is the top equivalence risk.
- **`proj_range` question** (CODEMAP §13-Q9): the master computed `1:Nz` but
  never passed it; `MakeSBXall`'s `round(0.25·Nz):round(0.75·Nz)` default is
  what ran. Default `"quarter"` assumes the published projections used the
  default — confirm against a ground-truth projection.
- ~~Integration glue~~ **DONE**: sibling-module resolution is pinned
  (`pipeline.py` PORTING NOTES #1) and exercised end-to-end by
  `tests/test_synthetic.py` (TIFF `VolumeSource` ingest → orchestrator →
  apply/project → writer, incl. multi-chunk and `.sbxall` paths).
- **Nz derivation contract** (IO module): `nz` must be `len(info.otwave)` —
  the convention of the live MATLAB path (`MakeSBXall.m` L12,
  `DFT_warp_3D_2.m` L21, `CalculateOptotuneWarp.m` L18) — **not**
  `info.otlevels` (`GetDimensions.m` L15). `sbxInfo.m` L105-108 only sets
  `otlevels = length(otwave)` when optotune was used, so the two diverge for
  a plane-scan (`volscan==0`) file with non-empty `otwave` (where the MATLAB
  pipeline itself misbehaves). Identical on the demo path. See the `_dims`
  docstring in `pipeline.py`.
- **`.sbx` byte-contract self-consistency** (CODEMAP §13-Q8) — IO module's
  round-trip test pending.
- **`uint16` cast and `round` semantics** are implemented to MATLAB spec
  (half-away-from-zero, saturation, NaN→0) but only unit-testable once a
  reference vector is extracted from MATLAB.
- MATLAB ran `parfor` in places; the port is serial for determinism first —
  performance parity unmeasured.
