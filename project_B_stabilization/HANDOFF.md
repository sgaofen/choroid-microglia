# Project B — Shipley 2020 z-stack stabilization — SESSION HANDOFF

> Self-contained context for a fresh session to work on Project B (the MATLAB
> z-stack motion-stabilization pipeline). The imaging/microglia work (Project A)
> is being handled in a separate session — keep this one focused on Project B.
> Complements `spec.md` (goals/constraints) with a concrete code map + first steps.
>
> **Update (2026-05-28):** the full reverse-engineering is done — see `CODEMAP.md`
> for the verified call graph, true on-disk data flow, the 3–4× bloat breakdown, the
> Fiji/ImageJ bundling list, every reproducibility blocker with a fix, dead-code
> inventory, and a confident-vs-needs-data cleanup split. The data-flow sketch below
> is the short version; `CODEMAP.md` supersedes it where they differ.

## 30-second orientation
Choroid plexus imaged in vivo floats freely in CSF (anchored at its base, the
rest drifts like seaweed). A z-stack takes ~4 s; between slices the tissue has
translated/rotated, so the raw stack is geometrically incoherent. Standard
cortex motion-correction fails (displacements are ~10× larger). The Shipley 2020
MATLAB pipeline re-registers every frame to a reference using **vasculature
landmarks** (IV fluorescent dye) via DFT-based 3D registration. It works but:
3–4× intermediate-file bloat, ~5–6 h per 1 GB stack, and only runs reproducibly
on **one** Harvard account's MATLAB setup.

**My job:** either (1) make it run reproducibly on the UCI lab computer, or
(2) refactor/Python-port it to produce equivalent stabilized stacks with far
less disk bloat. Original pipeline = ground truth for correctness.

## Where everything is
- This dir: `~/choroid-microglia/project_B_stabilization/`
  - `spec.md` — goals, two paths, constraints, data situation (read it)
  - `HANDOFF.md` — this file (code map + first steps)
  - `references/Shipley2020/` — the cloned repo (45 `.m` files). If missing:
    `git clone https://github.com/LehtinenLab/Shipley2020` into `references/`.
- Paper: Shipley FB et al., *Tracking Calcium Dynamics and Immune Surveillance at
  the Choroid Plexus Blood–CSF Interface.* Neuron 2020;108(4):623–639. PMID 32961128.
- Repo: https://github.com/LehtinenLab/Shipley2020

## Pipeline data-flow map  (entry: `registration/RegistrationMasterPipeline.m`, 44 lines)
1. **Init** — `javaaddpath` to mij.jar / ij-1.52a.jar / TurboRegHL_.jar (ImageJ + TurboReg via MATLAB-Java).
2. **Identify run** — hardcoded mouse/date/run/server/fbase; `refchannel` (1=red vessels, 2=green).
3. `GetDimensions(path,fdir,fbase)` → [Nchan,Nx,Ny,Nz,Nt].
4. **OIR→SBX** — `ConvertOIR_SBX(...)` converts Olympus `.oir` to Scanbox `.sbx` binary. *Skip when re-running.*
5. **Optotune warp** — `CalculateOptotuneWarp(path, refchannel, scale, 'regtype', opttype)`; `opttype='none'` if piezo, `'affine'` if optotune lens. Returns `tforms_optotune`.
6. **DFT 3D registration (CORE)** — `DFT_warp_3D_2(path, shiftpath, refchannel, scale, Nchunks, tforms_optotune, 'reftype','mean')` → writes `.dftshifts`. This is the heart: subpixel DFT cross-correlation registration in 3D. Supporting: `dftregistration3D.m`, `DFT_reg.m`, `DFT_rect.m`, `DFT_warp_3D_2.m`, `DetermineXYShiftsFBS.m`, `ApplyXYShiftsFBS.m`, `defineReference.m`.
7. **Apply + z-project** — `MakeSBXall(path, shiftpath, 'refchannel', refchannel)` applies shifts, writes registered SBX + z-projection (`zproj.m`, `zproj_reg.m`).
8. **Output** — `write2chanTiff(uint16(zproj_mean), savepath)` → `<run>_mean_zproj.tif`.
- Params: `scale=4`, `chunksize=20` (≤20), `proj_type='mean'` (or max/median).
- `vessel segmentation/` (separate, downstream): `Vessel_Segmentation_Dice*.m`, `vesselness2D_fbs.m`, `PerivascularProfile.m`, `MeasurePeriProfile.m` — Frangi-vesselness on PECAM-stained images + periluminal dilation; NOT part of the stabilization core. Likely not needed for Project B's stabilization goal.
- `data/` = IO/support helpers (`load_tiff*`, `sbxRead`, `sbxInfo`, `RegWriter`, `writeTiff`, `Miji.m`, `datapath/datedir/rundir`).

## Reproducibility blockers already visible in the entry script
1. **Hardcoded Harvard Windows paths** — `C:\Program Files\MATLAB\R2018a\java\...` and `C:\Users\LehtinenLab\Dropbox\AndermannLab\users\Fred\TurboRegHL_.jar`. Must be repointed for any other machine. Pins **MATLAB R2018a**.
2. **`pipe.lab.*` namespace** — calls `pipe.lab.datedir / datapath / rundir`. The repo's `data/` has `datedir.m`/`datapath.m`/`rundir.m` but they're invoked as a `+pipe/+lab` package → there is an expected MATLAB package folder structure (`+pipe/+lab/`) that may not be in the repo. **Check whether the `pipe` package exists; if not, that's the first "other users hit errors" cause.**
3. **Java / ImageJ (Miji/MIJ) + TurboReg** — fragile cross-version dependency; `Miji.m` present. ImageJ 1.52a pinned.
4. **`.sbx` (Scanbox) binary format** — proprietary intermediate; `sbxRead.m`/`sbxInfo.m`/`SpoofSBXinfo3D.m` handle it. A Python port needs an sbx reader (or skip sbx and work from OIR/TIFF).
5. **`.oir` (Olympus)** raw input — needs a reader (Bio-Formats) on the port side.

## Two paths (from spec.md)
- **Path 1 — revive in place:** clone onto UCI computer, reproduce env (MATLAB R2018a + toolboxes + MEX), fix the 5 blockers above, run on a SHORT test stack first. UCI has campus-wide MATLAB (Harvard only for grad students) → UCI side is friendlier.
- **Path 2 — port:** trace `RegistrationMasterPipeline.m` data flow (done — see map above), identify which intermediates are actually consumed vs retained (the 3–4× bloat), then either trim the MATLAB or rewrite the DFT registration in Python (scikit-image `phase_cross_correlation`, or `suite2p`/`DFT` libs; sbx/oir readers via Bio-Formats/sbx tools).

## Data situation
Raw z-stacks live on Huixin's external drives (multi-GB each, 5× external SSDs in a box) — NOT on this machine. For initial debugging, **ask Huixin for one short (~5 min) test stack** to use as the error-reproduction case before touching real 1–2 h videos.

## Suggested first session
1. Confirm `references/Shipley2020/` is present (else clone).
2. Grep for the `pipe` package / `+pipe` folder; determine if `pipe.lab.*` resolves → this likely explains "only runs on one account."
3. Read `DFT_warp_3D_2.m` + `dftregistration3D.m` fully (the core algorithm) and write a plain-language description of the registration math.
4. Decide Path 1 vs Path 2 with Huixin; if Path 2, draft the Python module boundaries (IO reader → reference selection → DFT shift estimation → warp apply → z-project).
5. Request a short test stack from Huixin.

## Constraints / status
- Equivalence to the original output is the bar (stabilized stacks indistinguishable on the same input).
- Secondary priority to Project A; this is "learn the shape + set up," not a deadline.
- Shipley 2020 is a calcium-dynamics/cell-motility paper — it has **no static microglia-morphology methods**, so nothing here feeds Project A.

## Lab / logistics
- PI: Huixin Xu (UCI EEB, McGaugh Hall 1217), trained in Lehtinen lab (Boston Children's/Harvard), co-author on Shipley 2020.
- Email thread with the paper link + repo + sample images: Gmail "Quick chat before summer start — research direction" (2026-05-20→).
- Stephen back in China late June→late July; Huixin in Japan late July→early Aug.
