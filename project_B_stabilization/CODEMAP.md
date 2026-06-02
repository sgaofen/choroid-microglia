# Shipley 2020 registration pipeline — code map

> Reverse-engineered reference for `references/Shipley2020/`. Every claim here was
> read against the source; line numbers are as-found in the cloned repo.
> Scope: the **registration core** (the z-stack stabilizer). The `vessel
> segmentation/` folder is confirmed separate and out of scope (see §11).
> This document records *what the code is*. The plan for changing it lives in the
> session discussion, not here.

## 1. Orientation

The pipeline takes an in-vivo choroid-plexus z-stack — geometrically incoherent
because the tissue drifts between slices — and re-registers every frame to a
reference using vasculature landmarks, by subpixel **DFT (phase) cross-correlation**.
The whole run is driven by one script, `registration/RegistrationMasterPipeline.m`,
which is a plain script (no `function` line), 44 lines, with all inputs hardcoded.

The repository is **45 `.m` files / ~3,500 lines** in three folders: `registration/`
(the pipeline), `data/` (IO and path helpers), and `vessel segmentation/` (unrelated
downstream analysis). It also reaches out of process into **ImageJ/Fiji** through the
MIJ MATLAB↔Java bridge for two jobs: the final TIFF write, and — only for
optotune-lens acquisitions — the per-slice affine registration via the MultiStackReg
(TurboReg) plugin.

## 2. The one-line truth: why it "only runs on one account"

Every path the script resolves goes through a `pipe.lab.*` / `pipe.io.*` /
`pipe.reg.*` / `pipe.imread` namespace — and **there is no `+pipe` package anywhere
in the repo** (verified: a search for `+*` package folders returns nothing). The
implementations exist only as *flat* files in `data/` and `registration/`. As
shipped, the script aborts at line 17 with `Undefined ... pipe`. It ran on one
Harvard account because that account had the Andermann lab's private `+pipe` library
on its MATLAB path. That single missing dependency — not the algorithm — is the
reproducibility wall.

The path-resolution chain underneath it (`pathbase → mousedir → datedir → rundir →
datapath`) is also machine-specific: `pathbase.m` maps a **hostname to a hardcoded
Windows drive letter** via a ~20-entry lookup table (e.g. `Pythagoras → G:\`), and
asserts the hostname contains no dots — so any FQDN hard-errors, and any unknown host
silently falls through to `H:\`.

## 3. File inventory

Status key: **live** = on the default production path (`opttype='none'`,
`regtype='DFT'`); **dead** = unreachable on that path; **support** = utility reached
indirectly.

### `registration/`
| File | Lines | Role | Status |
|---|---|---|---|
| `RegistrationMasterPipeline.m` | 44 | Entry script; hardcodes one run, sequences all stages | live |
| `ConvertOIR_SBX.m` | 106 | FluoView TIFF-export folder → monolithic `.sbx` | live (skipped if `.sbx` exists) |
| `SpoofSBXinfo3D.m` | 34 | Fabricates a Scanbox `info` struct for the `.sbx` | live |
| `CalculateOptotuneWarp.m` | 49 | Per-z affine warp for optotune lens; **returns identity early when `'none'`** | live (early-return) |
| `DFT_warp_3D_2.m` | 166 | **Core**: estimate XYZ shifts per chunk → write `.dftshifts` | live |
| `dftregistrationAlex.m` | 202 | 2D Guizar subpixel DFT registration (the engine, usfac up to 100) | live |
| `dftregistration3D.m` | 69 | 3D coarse DFT registration (usfac=2, no matrix-mult refine) | live |
| `DFT_rect.m` | 28 | Plane-to-plane axial rectification (chains `dftregistrationAlex`) | live |
| `DFT_reg.m` | 16 | 2D shift helper used by `zproj_reg` | live |
| `DetermineXYShiftsFBS.m` | 48 | Per-plane XY shift (central crop + gaussian blur → 2D DFT) | live |
| `ApplyXYShiftsFBS.m` | 25 | Apply XY shifts via `imtranslate` | live |
| `ApplyOptotuneWarp.m` | 7 | Per-slice `imwarp` of the affine transforms | live (identity in demo) |
| `defineReference.m` | 38 | Build the mean/median reference volume | live |
| `MakeSBXall.m` | 203 | Apply shifts; write registered `.sbxall`; return z-projection | live |
| `zproj_reg.m` | 81 | Time-series projection + 3-pass DFT stabilization | live (DFT branch) |
| `zproj.m` | 113 | Standalone projection entry | **dead** (bypassed by `MakeSBXall`) |
| `MultiStackReg_Fiji_affine_2.m` | 72 | Headless Fiji MultiStackReg, affine | dead in demo (optotune only) |
| `MultiStackReg_Fiji_rigid.m` | 75 | Headless Fiji MultiStackReg, rigid; own redundant `javaaddpath` | dead in demo |
| `MultiStackReg_Fiji_affine.m` | 71 | **GUI** (`Miji;`) affine variant | **dead** (only caller is broken) |
| `aligned.m` | 79 | Read-time registration (`pipe.reg.aligned`) | **dead** (forward build unused) |

### `data/`
| File | Lines | Role | Status |
|---|---|---|---|
| `sbxRead.m` | 90 | Read frames from a `.sbx` (applies `65535 − value`) | live |
| `sbxInfo.m` | 121 | Load `.sbx` sidecar info; **uses globals, leaks the fid** | live |
| `RegWriter.m` | 111 | Stream volumes into a `.sbx` (applies `65535 − value`) | live |
| `imread.m` | 74 | **SBX-only `imread` that shadows MATLAB's builtin** | live (hazard) |
| `GetDimensions.m` | 38 | Derive `[Nchan,Nx,Ny,Nz,Nt]` from info or TIFF filenames | live |
| `load_tiff.m` | 25 | TIFF loader (with waitbar) | support |
| `load_tiff_nobar.m` | 16 | Same loader, no waitbar; **declares wrong function name** | support |
| `write2chanTiff.m` | 33 | Final TIFF write via MIJ hyperstack | live |
| `writeTiff.m` | 38 | ImageJ `FileSaver` TIFF write | **dead** (only `zproj.m` calls it) |
| `arrtoij.m` | 87 | MATLAB array → ImageJ image | **dead** (only `writeTiff`/`zproj`) |
| `LoadTransforms.m` | 95 | Parse MultiStackReg `TransformationMatrices*.txt` | dead in demo |
| `Miji.m` | 103 | Set up Fiji classpath + start MIJ; **pins R2017a** | live (via `write2chanTiff`) |
| `metadata.m` | 14 | `pipe.metadata` | **dead** |
| `datapath.m` / `datedir.m` / `rundir.m` / `mousedir.m` / `pathbase.m` | 126/60/62/43/97 | Machine-specific path resolution chain | live (and broken off-machine) |
| `sort_nat.m` | 95 | Natural-order filename sort | support |
| `parfor_progressbar.m` | 178 | Waitbar + tempdir-IPC progress for `parfor` | support (GUI hazard) |

## 4. End-to-end call graph (production path)

```
RegistrationMasterPipeline.m  (script; tic at L1, no matching toc)
├─ javaaddpath ×3 (L3–5): mij.jar, ij-1.52a.jar (R2018a), TurboRegHL_.jar (Fred's Dropbox)
├─ pipe.lab.datedir / datapath (L17–18)  →  fdir, path     [UNRESOLVED without +pipe]
│    └─ datapath → rundir → datedir → mousedir → pathbase   (hostname → Windows drive)
├─ GetDimensions(path, fdir, fbase) (L20)  →  [Nchan,Nx,Ny,Nz,Nt]
│    ├─ try: sbxInfo
│    └─ catch: read .tif.frames/*.tif + sort_nat + regexp C###Z###T###
├─ if isempty(path): ConvertOIR_SBX (L30)  — only when no .sbx yet
│    └─ SpoofSBXinfo3D → save .mat ; load_tiff_nobar (parfor) ; RegWriter → .sbx
│       ; copyfile .txt ; rmdir .tif.frames
├─ CalculateOptotuneWarp(..,'regtype','none') (L34)
│    └─ 'none' → returns identity affine2d ×Nz at L20–22, BEFORE Fiji and BEFORE save L47
│       (affine/rigid branches would call MultiStackReg_Fiji_* → Miji → MIJ → TurboReg)
├─ DFT_warp_3D_2(..,'reftype','mean') (L38)
│    ├─ sbxInfo ; pipe.imread per chunk ; imresize(1/scale)
│    ├─ ApplyOptotuneWarp  (identity warp in demo)
│    ├─ DFT_rect → dftregistrationAlex (usfac=4)        — per-volume axial rectify
│    ├─ defineReference (mean/median)
│    ├─ DetermineXYShiftsFBS → dftregistrationAlex (usfac=100)  — per-plane XY
│    ├─ ApplyXYShiftsFBS (imtranslate [C,R])
│    ├─ dftregistration3D (usfac=2)                     — whole-volume + inter-chunk XYZ
│    └─ save → '<rundir>.dftshifts'   (a MAT-file)
├─ MakeSBXall(..,'refchannel',1) (L41)  →  zproj_mean
│    ├─ load(.dftshifts)
│    ├─ PASS 1 (L72–123): read RAW per volume → warp+XYshift+Zcircshift → accumulate zproj_raw
│    ├─ zproj_reg(.., 'zproj_raw',zproj_raw) (L125)  → DFT_reg ×2 + DFT_rect (refines shifts)
│    ├─ SpoofSBXinfo3D → RegWriter
│    └─ PASS 2 (L139–202): read RAW per volume AGAIN → same warp+shift → write '<base>.sbxall'
└─ write2chanTiff(uint16(zproj_mean), savepath) (L44)
     └─ Miji(false) → MIJ.createImage → 'Stack to Hyperstack...' → 'Save' Tiff → '<rundir>_mean_zproj.tif'
```

## 5. True data flow — on-disk intermediates

For the demo config (`opttype='none'`; input already `.sbx`, so conversion is skipped):

| | File | Produced by | Consumed by | Verdict |
|---|---|---|---|---|
| A | `<fbase>.tif.frames/*.tif` | FluoView (external) | `GetDimensions`; `ConvertOIR_SBX` (if it runs) | input; deleted at end of conversion |
| B | `<savename>.mat` (spoofed info) | `ConvertOIR_SBX` / `SpoofSBXinfo3D` | `sbxInfo` on every read | **required sidecar** |
| C | `<savename>.sbx` (raw stack, ~1×) | `ConvertOIR_SBX` → `RegWriter` | `DFT_warp_3D_2`, `MakeSBXall` (read **twice**) | **working format** |
| D | `<fbase>.txt` | `ConvertOIR_SBX` L44 | nobody | **dead weight** |
| E | `tforms_optotune.mat` | `CalculateOptotuneWarp` L47 | nobody (transforms travel in-memory) | **dead weight** — *and not even written in the `'none'` demo* |
| F | `<rundir>.dftshifts` (MAT) | `DFT_warp_3D_2` L140 | `MakeSBXall` (reads only `RS/CS/ZS`, `*_chunk`, `tforms_optotune_full`) | needed; but `ref_all` + `intermediate_shifts` inside it are **unread** (`ref_all` is the only heavy payload) |
| G | `<base>.sbxall` (registered stack, ~1×) | `MakeSBXall` L137/L193 | **no reader in the repo**; has no sibling `.mat` so can't even be reopened standalone | **prime bloat target** (pending out-of-repo check) |
| H | `<rundir>_mean_zproj.tif` | `write2chanTiff` via MIJ | humans / downstream | **the deliverable / ground truth** |

The deliverable **H** is computed in memory (`zproj_mean` returned by `MakeSBXall`) and
depends only on **B**, **C**, **F**. **G** is written and never read here; **D** and
**E** are pure dead weight; `ref_all` inside **F** is unread.

## 6. The 3–4× bloat, explained

The disk bloat is multiple full-size copies of the stack held at once, not the small
shift files:

1. **Raw `.sbx`** (~1×) — required working input.
2. **Registered `.sbxall`** (~1×) — written by `MakeSBXall`, consumed by **nothing** in
   the repo, with no `.mat` sidecar to reopen it. The single biggest removable artifact.
3. **`.tif.frames/` export** (~1×) — retained for the entire duration of
   `ConvertOIR_SBX` and only `rmdir`'d at the very end (L106), so during conversion you
   hold the full TIFF export *and* the growing `.sbx` simultaneously. (Not in play if
   the input is already `.sbx`.)

Small provably-removable writes: the `.txt` copy (**D**), `tforms_optotune.mat` (**E**),
and `ref_all`/`intermediate_shifts` inside `.dftshifts` (**F**).

**Compute bloat** (the ~5–6 h/GB driver, separate from disk): `MakeSBXall` reads the
raw `.sbx` and runs the full warp + XY-translate + Z-circshift pipeline **twice** in two
near-identical loops — once (L72–123) to build the projection, once (L139–202) to write
`.sbxall`. Dropping `.sbxall` deletes the second loop outright; keeping it allows fusing
the two loops into one pass, but the `zproj_reg` refinement at L125 (which adjusts the
shifts used by the write) imposes an ordering dependency that must be preserved.

## 7. The DFT registration core, in plain language

All alignment on the default path is **rigid translation only** (X, Y, Z) — no rotation
or affine — estimated by phase/DFT cross-correlation and applied by subpixel translation.

- **The engine** (`dftregistrationAlex.m`): cross-correlation of reference and moving
  image is, by the convolution theorem, `ifft( F(ref) · conj(F(moving)) )`; its peak is
  the integer shift. For subpixel accuracy it uses the Guizar-Sicairos/Thurman/Fienup
  (2008) trick: find the integer peak on a 2×-zero-padded FFT, then refine *only a small
  neighborhood* with a direct matrix-multiply DFT (`dftups`) at the requested upsampling
  factor `usfac` — giving 1/`usfac`-pixel accuracy without ever upsampling the whole
  image. Regimes: `usfac` 0 = none, 1 = integer, 2 = zero-pad refine, >2 = matrix-mult refine.
- **3D variant** (`dftregistration3D.m`): a stripped version that only does the
  zero-pad-and-IFFT step in all three dimensions and takes the integer argmax — so it is
  **coarse: half-pixel XY, half-plane Z** (`usfac=2`, no `dftups`). Used for whole-volume
  and inter-chunk XYZ correction.
- **Orchestration** (`DFT_warp_3D_2.m`): the stack is processed in chunks (`chunksize ≤
  20`). Per chunk: downsample by `scale=4`, axial-rectify plane-to-plane (`DFT_rect`,
  usfac=4), build a mean/median reference (`defineReference`), estimate per-plane XY
  shifts on a gaussian-blurred central crop (`DetermineXYShiftsFBS`, usfac=100), apply
  them, then a whole-volume 3D shift. Chunks are tied together by registering each
  chunk's reference back to chunk 1's. Shifts accumulate as
  `RS = RS0·scale + RS1·scale + RS2'`, `ZS = ZS1'`, and are saved to `.dftshifts`.

No MEX, no Java, no Fiji touches this math — it is pure FFT + array ops. (It maps almost
1:1 to `skimage.registration.phase_cross_correlation`, which *is* this algorithm.)

## 8. The Fiji/ImageJ surface and what to bundle

ImageJ is reached only through MIJ, for two jobs:

1. **Final TIFF write** (`write2chanTiff.m`, live): `Miji(false)` → `MIJ.createImage` →
   `MIJ.run('Stack to Hyperstack...')` → `MIJ.run('Save', 'Tiff..., path=[...]')` →
   `MIJ.exit`. Needs `mij.jar` + the ImageJ core jar. **This is the only Fiji dependency
   on the demo path** — and it is replaceable by `tifffile` (Python) or a plain MATLAB
   TIFF writer.
2. **Optotune affine registration** (`MultiStackReg_Fiji_*`, dead on the demo path):
   `MIJ.run('MultiStackReg', '... action=Align file=<fdir>TransformationMatrices{Affine|Rigid}.txt
   ... transformation={Affine|Rigid Body} save')`. MultiStackReg internally calls
   TurboReg, writes a `.txt` of per-slice transforms, which `LoadTransforms.m` reads back.
   Needs the MultiStackReg plugin + `TurboRegHL_.jar`.

**Bundling list for a self-contained setup:**

| Component | Why | How |
|---|---|---|
| Fiji.app (ImageJ core) | Final TIFF write; optotune registration | Bundle a known-good `Fiji.app` (`jars/` + `plugins/`); point `Miji.m`'s `FIJI_HOME` at it, bundle-relative. Pin the exact build that made ground truth. |
| `mij.jar` | MIJ bridge for all `MIJ.*` calls | Bundle **one** copy in `Fiji.app/jars/`. **Reconcile the R2018a (master) vs R2017a (`Miji.m` L26) mismatch** — two `mij.jar`s on the static classpath can corrupt it. |
| MultiStackReg plugin | Optotune affine step (dead in demo) | Drop into `Fiji.app/plugins/`; verify it writes the `.txt` format `LoadTransforms.m` parses. |
| `TurboRegHL_.jar` | TurboReg backend for MultiStackReg (dead in demo) | Bundle into `Fiji.app/plugins/`. **Custom "HL" fork** — stock TurboReg may change output; obtain the exact jar. |
| Bio-Formats / OIR reader | `.oir` is never decoded in-repo (assumes a manual FluoView export) | Bundle `bfmatlab` (MATLAB) or `bioio-bioformats`/`aicsimageio` (Python) — only if you want to drop the manual export step. |
| MATLAB ≥ R2018b | `rescale` needs R2017b+, `mean/max('omitnan')` needs R2018b+; jar paths say R2018a | Pin one release; reconcile `Miji.m`. Licensed — can't bundle. |
| Image Processing Toolbox | `imresize/imtranslate/imwarp/imref2d/imgaussfilt/affine2d/fitgeotrans` | Hard requirement (or port the core to Python `scipy.ndimage`+`skimage`). Can't bundle. |
| Parallel Computing Toolbox | `parfor` in conversion + `MakeSBXall` | Optional for correctness (degrades to serial), expected for speed. Can't bundle. |

## 9. Reproducibility blockers

| # | Blocker | Severity | Fix |
|---|---|---|---|
| 1 | **`pipe.*` namespace unresolved** — no `+pipe` package in repo; entry script aborts at L17 | critical | Create `+pipe/{io,lab,reg}/` and move the flat files in, *or* de-namespace every call site to the bare function names |
| 2 | **Hardcoded Windows jar paths** pinning R2018a + Fred's Dropbox (`RegistrationMasterPipeline` L3–5; `MultiStackReg_Fiji_rigid` L3–4) | critical | Bundle-relative `FIJI_HOME`, centralized once |
| 3 | **`pathbase.m` hostname→drive table** (no UCI host matches; malformed `else strcmpi` fallthrough to `H:\`; FQDN assert hard-errors) | critical | Replace the whole `pathbase/mousedir/datedir/rundir/datapath` chain with one explicit base-dir config / direct `.sbx` path argument |
| 4 | **`Miji.m` pins R2017a** (L26) vs R2018a elsewhere; assumes a `Fiji.app` tree two levels up; does `cd('..')` | high | Reconcile to one version; bundle a real `Fiji.app`; make `FIJI_HOME` configurable; remove the `cd` |
| 5 | **`data/imread.m` shadows builtin `imread`** (SBX-only; errors on TIFF). `load_tiff` calls the builtin → breaks if `data/` is ahead on the path | high | Rename to `sbxImread`, or move into `+pipe/+io` so the bare name reaches the builtin |
| 6 | **Hardcoded backslashes** at `sbxInfo.m` L28, `GetDimensions.m` L20, `zproj.m` L109 — break on macOS/Linux | high | Use `fullfile`/`filesep` |
| 7 | **`.oir` never decoded** (assumes manual FluoView export to `.tif.frames/`) | high | Add a Bio-Formats reader, or document the export step |
| 8 | **GUI/headless hazards**: `RegWriter` `warndlg`; `waitbar`/`parfor_progressbar`; MIJ — all hang under `-nodisplay` | medium | Gate GUI behind an interactive flag; run ImageJ headless |
| 9 | **`zproj_reg.m` L44 undefined `pathz`** (should be `p.pathz`) in the Affine branch | low | Fix or delete the unused branch |
| 10 | **Unpinned MATLAB version** with feature floors (`omitnan` R2018b+, `rescale` R2017b+) | medium | Pin ≥ R2018b in README |
| 11 | **`sbxInfo.m` globals + leaked fid** (`global info_loaded info`; `fopen` with no `fclose`); duplicate `nsamples` assignment | medium | Encapsulate; close fid deterministically |

## 10. Dead / legacy code (verified)

- `MultiStackReg_Fiji_affine.m` — GUI variant; only caller is `zproj_reg`'s Affine
  branch, which is itself broken (undefined `pathz`) and never taken (default `'DFT'`).
  Near-duplicate of `affine_2` (one functional line differs: `Miji;` vs `Miji(false);`).
- `zproj.m` — only caller is the bypassed `pipe.zproj` branch of `zproj_reg`
  (`MakeSBXall` always supplies `zproj_raw`).
- `writeTiff.m` + `arrtoij.m` — only caller is `zproj.m` (itself dead).
- `aligned.m` (`pipe.reg.aligned`) + `metadata.m` (`pipe.metadata`) — read-time
  registration path, not part of the forward build.
- `datapath.m` L23/25/34 — `path.io.datapath(...)` where `path` is the output char
  variable, not a package; the vector-date/run iteration branches are broken even with a
  real `+pipe`.
- `DFT_warp_3D_2.m` else branch (L48) — writes `unwarp_chunk` (typo; L52/55 read
  `unwarped_chunk`); unreachable anyway because `p.optotune` defaults to the string
  `'true'` (always truthy). Latent dead/broken.
- `load_tiff_nobar.m` — declares `function img = load_tiff(...)` (name ≠ filename);
  copy-paste of `load_tiff.m` minus the waitbar.
- Entire `vessel segmentation/` folder (see §11).

## 11. Vessel segmentation — confirmed out of scope

Three independent driver scripts (`Vessel_Segmentation_Dice[_batch]`,
`PerivascularProfile`) sharing one shape: load a hand-curated 2D PECAM TIFF →
`vesselness2D_fbs` (Frangi) → binarize → boundary/periluminal rings → Dice vs a manual
mask, or a radial intensity profile. **Their inputs are separately prepared 2D figure
images and manual masks, not the stabilized z-stacks.** No call edge or file path
connects this module to the registration pipeline in either direction (only shared
symbol: `load_tiff`). It is standalone figure analysis and plays no part in the
stabilization port.

## 12. Python port boundaries (if Path 2)

Clean seams for a rewrite, each independently testable against the original:

1. **IO reader** — one `SBX` class for the byte contract (`uint16`, `value = 65535 −
   raw`; record layout `[nchan, width, height]`; `otwave`/`otlevels` interleave;
   channels↔nchan inverse coding) replacing `sbxInfo`+`sbxRead`+`SpoofSBXinfo3D`+
   `RegWriter`; plus an OIR reader (`bioio`/`aicsimageio`) replacing the FluoView export.
2. **Reference selection** — `defineReference` (temporal mean/median of n-volume blocks)
   + the "chunk 1's reference is the global anchor" rule. Pure `numpy`.
3. **DFT shift estimation** — `dftregistrationAlex` + `dftregistration3D` + `DFT_rect` +
   `DetermineXYShiftsFBS`. Maps to `numpy.fft` + `scipy.ndimage` or
   `skimage.registration.phase_cross_correlation`. (Match the original's `usfac=2`
   coarseness before considering 3D refinement.)
4. **Warp / shift apply** — `ApplyOptotuneWarp` (affine `imwarp`) + `imtranslate([C,R])`
   + Z circshift-with-zero-fill. Maps to `scipy.ndimage.shift`/`affine_transform`.
   **Preserve the `[Col,Row]` argument order** — a transposed call silently swaps X/Y.
5. **Chunk orchestration** — `DFT_warp_3D_2`'s hierarchical local + inter-chunk scheme
   and shift accumulation. Pure bookkeeping.
6. **Z-projection** — mean/max/median collapse + `zproj_reg`'s 3-pass time stabilization.
7. **Output writer** — `tifffile.imwrite` (ImageJ-compatible) replaces the MIJ TIFF
   write, eliminating the JVM for IO. The only genuine Fiji dependency left is the
   optotune MultiStackReg step — **dead on the piezo/`'none'` path, so droppable for a
   piezo demo.**

## 13. Open questions (need Huixin / the real `+pipe` / the demo stacks)

1. Where does the real `+pipe` library live? Are the flat `data/` files faithful mirrors,
   or do they diverge from the lab version? (`sbxInfo` hints at a `.sbxreg` channel
   correction not present here.) Needed before trusting the local fallbacks to reproduce
   ground truth.
2. Is `<base>.sbxall` consumed by anything *outside* this repo? If not, it can be deleted
   outright — removing ~1× of the bloat.
3. For the demo stacks: input already `.sbx` (conversion skipped) or raw OIR/`.tif.frames`
   (conversion runs)? Determines whether the OIR/Bio-Formats and frame-bloat concerns
   apply at all.
4. Were the figures registered with `opttype='none'` (piezo) throughout, or did some use
   `'affine'` (optotune)? If always `'none'`, the entire Fiji optotune path + TurboReg can
   be dropped.
5. What exactly is `TurboRegHL_.jar` (Fred's "HL" fork)? Stock TurboReg may change output.
6. Which MATLAB/ImageJ/mij combo produced ground truth? (R2018a + ij-1.52a vs `Miji.m`'s
   R2017a.)
7. Does `dftregistration3D`'s `usfac=2` coarseness matter — intentional (coarse physical Z
   steps) or a perf compromise?
8. Verify the spoofed-info byte contract is self-consistent so `.sbx` round-trips
   byte-exact (`sbxInfo` `nsamples = sz(2)·recordsPerBuffer·2·nchan` vs `SpoofSBXinfo3D`
   `width·height·2·nchan`).
9. Is `proj_range` honored? The script computes `1:Nz` but passes nothing; `MakeSBXall`
   uses its own `round(0.25·Nz):round(0.75·Nz)`. Which z-range defines the published
   projection?
