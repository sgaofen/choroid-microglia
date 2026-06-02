# Shipley 2020 registration — cleaned, self-contained build

A runnable repackaging of the choroid-plexus z-stack stabilization pipeline
(`references/Shipley2020/registration` + `data`). The **numerical core is
unchanged** — every DFT-registration, warp, and projection function is the original
byte-for-byte. What changed is the plumbing that stopped it from running anywhere but
one Harvard account. See `../CODEMAP.md` for the full reverse-engineering.

> **Status: not yet executed.** These are plumbing-only edits made by inspection on a
> machine with no MATLAB and no test data. Nothing here has been run. Before trusting
> any output, run it on a short test stack and confirm the result matches the original
> pipeline (equivalence is the bar). The bloat/runtime fixes that *would* touch the
> numerics were deliberately **not** made yet — they need a stack to verify against.

## Run it

```matlab
addpath('/path/to/project_B_stabilization/clean');   % once, so run_registration is found

cfg = struct();
cfg.sbx_path   = '/path/to/0309-0721-009_200813_002.sbx';  % sidecar .mat next to it
cfg.refchannel = 1;          % 1 = red vessels, 2 = green
cfg.opttype    = 'none';     % 'none' = piezo; 'affine'/'rigid' = optotune lens
% cfg.fiji_home = '/Applications/Fiji.app';   % omit to use the bundled fiji/Fiji.app

zproj = run_registration(cfg);
```

`run_registration.m` puts `+pipe`, `registration/`, and `data/` on the path, resolves
output names, and runs the four stages (optotune warp → DFT 3D registration → apply +
z-project → TIFF write). Outputs land next to the input by default:
`<name>.dftshifts` and `<name>_mean_zproj.tif`. See `config_example.m`, including the
first-time **TIFF→SBX conversion** variant.

## Requirements

- **MATLAB ≥ R2018b** (the code uses `mean(...,'omitnan')` R2018b+ and `rescale` R2017b+).
- **Image Processing Toolbox** (`imresize/imtranslate/imwarp/imref2d/imgaussfilt/affine2d/fitgeotrans`).
- **Parallel Computing Toolbox** — optional; `parfor` degrades to serial without it.
- **Fiji + MIJ** — for the final TIFF write (always) and optotune registration (only
  `opttype≠'none'`). Run `fiji/setup_fiji.sh`, then add `mij.jar` and (for optotune)
  `TurboRegHL_.jar`. Read `fiji/NOTES.md` — MIJ is the fragile part and the
  ground-truth version is unconfirmed.

## What changed from the original

| Change | Why |
|---|---|
| Reconstructed the **`+pipe/`** package (`+io`, `+lab`, `+reg`) and moved the flat `data/` files into it | The code calls `pipe.io.*`/`pipe.lab.*`/`pipe.imread`; the package never existed in the repo, so the script died at line 17. This is the "only runs on one account" fix. |
| New **`run_registration.m`** entry (config struct) replacing the hardcoded script | Bypasses the machine-specific `pathbase` hostname→drive table; pass paths explicitly. |
| Moving `imread` into the package | `data/imread.m` shadowed MATLAB's builtin `imread` (SBX-only); as `pipe.imread` it no longer does, so `load_tiff`'s `imread` reaches the builtin. |
| Collapsed `MultiStackReg_Fiji_{rigid,affine,affine_2}` → one **`MultiStackReg_Fiji(...,mode)`**; deleted the broken GUI variant | Three near-duplicates; one was dead and reached only from a broken branch. |
| Reworked **`Miji.m`**: configurable bundle-relative `FIJI_HOME`, removed the hardcoded R2017a `mij.jar` and the `cd` | Was pinned to one machine and an inconsistent MATLAB version. |
| Fixed Windows backslashes in `sbxInfo.m`, `GetDimensions.m`, `+pipe/zproj.m`; use `fullfile` for the MultiStackReg transform path | These broke on macOS/Linux. |
| Removed a duplicate `inputParser` block in `ConvertOIR_SBX.m` | Verified inert. |
| **Dropped** the out-of-scope `vessel segmentation/` folder | Confirmed isolated from the stabilization pipeline (`../CODEMAP.md` §11). |

## Deliberately NOT changed (needs a test stack first)

- The unread `<name>.sbxall` write and `MakeSBXall`'s duplicate second loop (the ~1×
  disk bloat + a large share of the runtime).
- The `tforms_optotune.mat` / `.txt` / `ref_all` dead-weight writes.
- The optotune-default and `unwarped_chunk` typo bugs in `DFT_warp_3D_2.m`.
- Anything inside the DFT registration math.

These are catalogued with rationale in `../CODEMAP.md` (§6, "needs-data changes").
```
clean/
  +pipe/              reconstructed package (+io, +lab, +reg, imread, zproj, metadata)
  registration/       pipeline functions (numerics unchanged) + MultiStackReg_Fiji.m
  data/               flat helpers (GetDimensions, load_tiff, Miji, ...)
  run_registration.m  entry point
  config_example.m    copy + edit this
  fiji/               Fiji bundle (setup_fiji.sh + NOTES.md; binary git-ignored)
```
