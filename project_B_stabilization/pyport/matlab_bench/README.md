# MATLAB benchmark kit — run the original pipeline on the lab Dell and time it

This directory has one purpose: **run the original MATLAB pipeline and the Python port on the same data to get comparable timings.**

No numerical code is changed. The subject under test is `clean/run_registration.m` and the registration/projection functions beneath it, untouched.

Target machine: Xu lab's Dell (Ultra 9 285 / 24 cores / 32 GB / RTX 2000 Ada).
Note the desktop directory is not writable — put data and output under `%TEMP%` or another drive.

---

## 0. What the three files are for

| File | Runs on | Purpose |
|---|---|---|
| `make_sbx.py` | **Mac** (the one with Python and the raw data) | Converts T-major `.npy` or OME-TIFF into the `.sbx` + `.mat` sidecar that's the only format MATLAB can read, and verifies bit-for-bit |
| `run_benchmark.m` | **Dell** (the one with MATLAB) | Sets up paths, checks the environment, wraps `run_registration` in `tic/toc`, prints elapsed time and outputs |
| `README.md` | You | This file |

Workflow: convert data on the Mac -> copy the two files to the Dell -> run `run_benchmark` on the Dell.

---

## 1. What needs to be installed on the Dell

| Dependency | Required? | Notes |
|---|---|---|
| MATLAB **R2018b or newer** | Required | Code uses `mean(...,'omitnan')` (R2018b+) and `rescale` (R2017b+) |
| **Image Processing Toolbox** | Required | `imresize` / `imtranslate` / `imwarp` / `imref2d` / `imgaussfilt` / `affine2d` / `fitgeotrans` are all in it — missing any one blocks the run |
| **Parallel Computing Toolbox** | Strongly recommended | `DFT_warp_3D_2.m` uses `parfor`. It'll still run without it (falls back to serial), but the registration-stage timing won't be comparable to the parallel version — the benchmark would be skewed |
| **Fiji + MIJ** (`mij.jar`) | Required | The last step, `write2chanTiff`, relies on MIJ to write TIFF. **This is the most fragile link in the whole chain** — see section 4 |
| Java heap >= 4 GiB | Required for the full run | Also a `write2chanTiff` requirement — see section 4 |

With `opttype='none'` (piezo — this is what our data uses) TurboReg is **not needed**;
only `opttype='affine'/'rigid'` requires installing `TurboRegHL_.jar` as well.

How to check (in the MATLAB command line):

```matlab
ver                                     % show version and installed toolboxes
license('test','image_toolbox')         % 1 = installed
license('test','distrib_computing_toolbox')
java.lang.Runtime.getRuntime.maxMemory / 2^30   % Java heap, in GiB
```

`run_benchmark.m` checks all of this automatically before it starts and prints the results — you don't need to track it by hand.

---

## 2. Step one (on the Mac): generate `.sbx` + `.mat`

The MATLAB-side entry point only recognizes `.sbx`: `GetDimensions` goes through
`pipe.io.sbxInfo`, and `MakeSBXall`/`DFT_warp_3D_2` go through `pipe.imread` ->
`sbxRead`'s `fseek/fread`. It can't read OME-TIFF or `.npy`. So a conversion
step is required first.

```bash
PY=python                      # or your venv's interpreter
PYPORT=/path/to/pyport         # this repo's project_B_stabilization/pyport
WS=/path/to/workspace          # where the .npy volume stores live

# --- small sample (40 frames, 1.6 GiB): use this first to validate the whole chain ---
$PY $PYPORT/matlab_bench/make_sbx.py \
    --in  $WS/FAD-F_1_T0-39.tzcyx.npy \
    --out-base $WS/matlab_bench/FAD-F_1_T0-39 \
    --verify-all

# --- full run (1500 frames, 60 GiB): confirm the target disk has space before running ---
$PY $PYPORT/matlab_bench/make_sbx.py \
    --in  $WS/FAD-F_1_raw.tzcyx.npy \
    --out-base /Volumes/BIG/FAD-F_1_raw \
    --verify 16

# --- just want to see how big it'll be ---
$PY $PYPORT/matlab_bench/make_sbx.py --in X.npy --out-base Y --dry-run
```

Two output files, **must be copied as a pair, same name, same directory**:

```
<out-base>.sbx    Nt*Nz records, each rows*cols*2*Nchan bytes
<out-base>.mat    info sidecar (MAT v5), readable directly by MATLAB's load
```

Without the sidecar, MATLAB can't even read Nx/Ny/Nz/Nt — `sbxInfo` finds it
by looking for a same-name `.mat` in the same directory.

### Can this conversion be trusted

`make_sbx.py` doesn't implement its own byte format — it's a line-by-line
port of the two functions MATLAB itself uses for conversion:

- write: `cpstab.io_rw.RegWriter` <- `RegWriter.m` L1-111
- sidecar: `cpstab.io_rw.spoof_sbx_info_3d` <- `SpoofSBXinfo3D.m` L1-34,
  saved via `save_sbx_info` <- `ConvertOIR_SBX.m` L39

Right after writing it reads back and compares bit-for-bit. Already verified
(40-frame subset, `--verify-all`):

```
(Nchan, Nx=rows, Ny=cols, Nz, Nt) = (2, 512, 512, 41, 40)  uint16
1640 records, 1.6 GiB
verify: 80 (t, c) volumes bit-identical
```

Separately verified once more **reading via the port of `sbxRead.m`** (i.e.
the exact call MATLAB-side `MakeSBXall.m` L74/L76 actually issues,
`pipe.imread(path, Nz*t+1, Nz, pmt, [])`):

```
sidecar-derived: nchan=2 sz=[512.0, 512.0] otlevels=41 nframes=1640 max_idx=1639
imread(sbxRead port) vs VolumeSource, all 40 volumes: mismatches=0
GetDimensions equivalent -> Nchan=2 Nx=512 Ny=512 Nz=41 Nt=40
```

In other words, the pixels MATLAB sees and the pixels the Python pipeline
consumes are the same numbers.

Every numeric field in the sidecar is written as **double** (only `area_line`
is logical), matching MATLAB's own `SpoofSBXinfo3D.m`. This isn't fussiness:
`scipy.io.savemat` stores Python `int` as `int64` by default, and MATLAB's
integer types are contagious and **integer division rounds**. `sbxInfo.m`'s
`max_idx`, `GetDimensions.m`'s `Nt = floor(...)`, and `run_registration.m`'s
`Nchunks = round(Nt/chunksize)` are the same expression chain propagating
downstream — on our data every step divides evenly so both give the same
value by coincidence, but a dataset that doesn't divide evenly would silently
produce a different chunk split. See the `matlabize()` docstring in
`make_sbx.py` for details.

> `.sbx` stores `65535 - pixel value` (inverted) — that's the format's own
> convention. Inverted on write, inverted again by `sbxRead` on read — the
> two cancel out, so you don't need to worry about it. But **don't** open it
> in a hex editor and conclude the data is wrong.

---

## 3. Step two (on the Dell): run the benchmark

Copy these to the Dell (keep the directory structure):

```
project_B_stabilization/
  clean/                     <- whole directory, don't drop anything (+pipe / registration / data / fiji)
  pyport/matlab_bench/
    run_benchmark.m
```

`run_benchmark.m` finds the pipeline via the relative path `../../clean`, so
the relative position of these two can't change. The data (`.sbx` + `.mat`)
can go anywhere — point `cfg.sbx_path` at it.

```matlab
cd('...\project_B_stabilization\pyport\matlab_bench');

cfg = struct();
cfg.sbx_path   = 'D:\shipley\FAD-F_1_raw.sbx';   % same-name .mat must be next to it
cfg.refchannel = 1;          % 1 = red vessel channel
cfg.scale      = 4;
cfg.chunksize  = 20;
cfg.proj_type  = 'mean';
cfg.opttype    = 'none';     % piezo, this is what our data uses
cfg.label      = 'Dell T2 / Ultra 9 285 / 24 cores';   % written into the log

r = run_benchmark(cfg);
```

Parameters must match the Python side to be comparable — the set above is
the same one used by the Python side's two ironclad regression tests
(`refchannel=1 scale=4 chunksize=20 proj_range=quarter proj_type=mean opttype=none`).

> `proj_range` has no cfg switch on the MATLAB side: the default in
> `MakeSBXall.m` L13 is `round(0.25*Nz):round(0.75*Nz)`, which is the Python
> side's `'quarter'`. Both defaults already agree — nothing to do here.

You can also skip cfg — it'll look under `matlab_bench/data/` for the one
`.sbx` file there:

```matlab
r = run_benchmark();
```

### What it prints / what it produces

Before it starts:

```
Environment:
  MATLAB    : 9.14.0.2206163 (R2023a) on PCWIN64
  host/cores: XULAB-T2 / 24
  Java heap : 0.48 GiB (Preferences > General > Java Heap Memory)
  Image Processing Toolbox  : yes
  Parallel Computing Toolbox: yes
  ...
Dimensions: Nchan=2 Nx(rows)=512 Ny(cols)=512 Nz=41 Nt=1500
  .sbx size : 60.1 GiB, 61500 records (Nz*Nt = 61500)
  Nchunks   = round(Nt/chunksize) = 75
  projection: 2 x 512 x 512 x 1500 uint16 = 1.46 GiB (goes into the JVM heap)

Precheck:
  [ OK ] Image Processing Toolbox
  [ OK ] Parallel Computing Toolbox (parfor available)
  [ OK ] Fiji + mij.jar
  [FAIL] Java heap 0.48 GiB < recommended 3.66 GiB.
```

After it finishes:

| Output | Size (full run) | Notes |
|---|---|---|
| `<stem>_mean_zproj.tif` | ~1.5 GB | **The main output** — this is what gets compared against the Python side's zproj |
| `<stem>.dftshifts` | ~2 MB | Shift set in MAT format (Python side's counterpart is `.dftshifts.npz`) |
| `<stem>.sbxall` | **~60 GB** | The full registered stack that `MakeSBXall` writes unconditionally. **No code in the repo reads it** — it's pure disk overhead — but deleting it would change the subject under test, so it has to stay for a benchmark run. **Confirm the target disk has 120 GB+ free beforehand** (60 for the raw data + 60 for this) |
| `<stem>_matlab_bench_*.log` | small | Full console diary, with progress timestamps for each registration chunk |
| `<stem>_matlab_bench_result.mat` | small | The `result` struct: elapsed time, dimensions, environment |

`result.elapsed_s` / `result.per_volume_s` are the numbers to compare against
the table in the Python side's `bench_report.md`.

---

## 4. Known pitfalls (ordered by likelihood of hitting them)

### 4.1 Java heap too small — blows up **after** the computation finishes

`write2chanTiff` -> `MIJ.createImage` needs to fit the **whole projection**
into the JVM heap. Full run is `2*512*512*1500*2 = 1.46 GiB`; add MIJ's
internal copy and the recommendation is >= 3.7 GiB. MATLAB's default is
often just a few hundred MB.

And this step is on the pipeline's **last line** — `zproj_mean` only exists
in `run_registration`'s return value. If the function doesn't return, that
memory goes away with the stack. **Hours of computation, for an
`OutOfMemoryError`.**

`run_benchmark.m` therefore checks this before `tic` even starts, and
refuses to run if it fails.

Fix: increase it under `Preferences > General > Java Heap Memory` ->
**restart MATLAB**. If you can't raise it (some site licenses restrict
this), use the segmented approach in 4.2.

### 4.2 Memory/disk can't handle the full run -> segment with `--limit-t` and extrapolate

```bash
$PY make_sbx.py --in ...raw.tzcyx.npy --out-base D:/shipley/FAD300 --limit-t 300
```

Run 300 volumes, then extrapolate `result.per_volume_s` linearly to 1500.
This extrapolation holds: the apply stage is per-volume independent, the
registration stage is chunked, and both are linear. On a Dell with 32 GB
RAM + 60 GB raw + 60 GB `.sbxall`, this is probably the more realistic path.

### 4.3 Fiji / MIJ won't install

`clean/fiji/NOTES.md` says: MIJ is the most fragile part of the whole
package, and the ground-truth version hasn't been confirmed.
`clean/fiji/setup_fiji.sh` installs Fiji; then manually drop `mij.jar` into
`<Fiji.app>/jars/`.

`run_benchmark.m` checks whether `<fiji_home>/jars/mij.jar` exists. You can
also point `cfg.fiji_home` at a Fiji install already on the system.

### 4.4 `clean/` has never actually been run

`clean/README.md` itself says **"Status: not yet executed"** — that
"pipeline repair" was done by code review on a machine with no MATLAB and no
data. So **use the 40-frame small sample for the first run** (section 2's
`--limit-t` / subset `.sbx`), confirm it runs and the output matches the
Python side, before going to the full run. Discovering a mistyped `addpath`
with 60 GB of data is too expensive.

### 4.5 `.m` file encoding

`run_benchmark.m` is **pure ASCII** — not a single Chinese character in it.
This is deliberate: MATLAB before R2020a reads `.m` files using the system
default encoding (GBK/windows-1252 on Windows), so UTF-8 Chinese comments
turn into mojibake, and Chinese in strings would garble `fprintf` output
too. All Chinese explanation stays in this README (read by an editor, not
by MATLAB).

### 4.6 The Dell's desktop isn't writable

Known issue (see memory: Xu lab Windows workstation). `cfg.out_base`
defaults to following the `.sbx` location, so as long as the `.sbx` isn't
on the desktop, this is fine.

---

## 5. How to compare against the Python side

The Python side's reference data is produced by `pyport/scripts/bench_full.sh`
and lands in `shipley_workspace/bench_report.md`:

| What to compare | Python side | MATLAB side |
|---|---|---|
| Timing | Table in `bench_report.md` section 1 | `result.elapsed_s` / `per_volume_s` |
| Main output | `bench/f64_replicate/*_mean_zproj.tif` | `<stem>_mean_zproj.tif` |
| Shifts | `*.dftshifts.npz` (RS/CS/ZS/...) | `*.dftshifts` (same field names, MAT format) |

Bit-level/statistical comparison of the main outputs from both sides can be
run directly:

```bash
$PY -m cpstab.metrics \
    matlab=<copied back from Dell>_mean_zproj.tif \
    python_f64=$WS/bench/f64_replicate/FAD-F_1_raw.tzcyx_mean_zproj.tif \
    --stride 8
```

> **Don't expect the two sides to match bit-for-bit.** The Python side's
> "bit-exact" guarantee is against **its own serial version**, not against
> MATLAB. The systematic difference between the port and the original
> MATLAB pipeline was already measured separately
> (`shipley_workspace/reportB/cpstab_validation.md`: reference-channel
> median Pearson r ~= 0.665, with a systematic residual shift of about
> 2 px). What's being compared here is **timing** — the main-output
> comparison is just corroborating evidence that nothing ran off the rails.
