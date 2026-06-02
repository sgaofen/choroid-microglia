# Fiji bundle

The pipeline reaches ImageJ through the **MIJ** MATLAB↔Java bridge (`data/Miji.m`),
for two jobs:

1. **Final TIFF write** (`write2chanTiff.m`) — used on every run, including the
   default piezo (`opttype='none'`) path.
2. **Optotune affine registration** (`MultiStackReg_Fiji.m`) — used **only** when
   `opttype` is `'affine'` or `'rigid'`. Dead on the piezo demo path.

`Miji.m` puts every `.jar` under `Fiji.app/jars/` and `Fiji.app/plugins/` onto the
Java classpath, so anything you place there is picked up automatically.

## What `setup_fiji.sh` fetches
- **Fiji.app** for your OS (includes the ImageJ core jar).
- **MultiStackReg** plugin → `Fiji.app/plugins/` (only needed for optotune runs).

## What you must add by hand
- **`mij.jar`** → `Fiji.app/jars/`. The MATLAB↔ImageJ bridge. It is *not* shipped
  with modern Fiji and is effectively unmaintained — it pairs best with an older
  ImageJ (the repo pinned **ij-1.52a**) and older MATLAB. Source it from the version
  that produced ground truth.
- **`TurboRegHL_.jar`** → `Fiji.app/plugins/`. Fred's custom "HL" TurboReg fork that
  MultiStackReg calls for the optotune affine step. Stock TurboReg may change output.
  Obtain the exact jar from the Lehtinen/Andermann lab. **Only needed for optotune
  acquisitions** — skip it for piezo (`'none'`) data.

## Version pinning (open question)
The original references are inconsistent: `ij-1.52a` + MATLAB R2018a in the master
script, but R2017a in `Miji.m`. Which MATLAB / ImageJ / mij combination produced the
published, ground-truth output is unconfirmed (see `../../CODEMAP.md` §13). Pin the
matching versions before trusting any registration result for equivalence.

## Pointing the pipeline at a Fiji elsewhere
You don't have to bundle here. Set one of:
- `cfg.fiji_home = '/path/to/Fiji.app'` in your config, or
- `setenv('FIJI_HOME','/path/to/Fiji.app')`, or
- `setpref('shipley_clean','fiji_home','/path/to/Fiji.app')`.
