"""Batch preprocessing: every image, every channel -> clean image (flat-field + uniform whole-image background removal).

Base = Fiji flat-fielded version (tif_fiji_flatfielded; multiplicative correction of the
illumination gradient/vignetting, signal amplitude normalized across the whole image.
Caveat: FIJI_research.md concluded that flat-fielding must be done per-Z before projection,
but fjff was in fact produced by flat-fielding the MAX projections after projection).
Clean = clip(fjff - rolling-ball background, 0), uint16 uncompressed BigTIFF (memmap-able).
Also saved: 8x downsampled overview PNG (2-sigma display floor + adaptive stretch) + meta (sigma, source).
Samples missing fjff fall back to tif_max. Already-produced outputs are skipped
(to redo, delete the corresponding files in clean/ first).

Samples are DISCOVERED from SRC by filename; there is no hardcoded animal list. The names
found are printed on startup - check them before letting it run. Only files matching
FILE_RE (default `10x_<sample>_CCR2-CD45_C0.tif`) are seen; override with CHP_FILE_RE /
CHP_FILE_FMT, and the source folder with CHP_SRC.

Pixel size comes from config.py (PIX_UM) - set it for a new image set before running.

Usage: .venv/bin/python preprocess_clean_images.py [C0 C1 C2]
"""
import json
import os
import re
import sys
import time

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import disk

from config import PIX_UM, SRC_FLATFIELDED, SRC_TIF_MAX

SRC = SRC_TIF_MAX
FJFF = SRC_FLATFIELDED
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clean")
FILE_RE = os.environ.get("CHP_FILE_RE", r"10x_(.+)_CCR2-CD45_C0\.tif$")
FILE_FMT = os.environ.get("CHP_FILE_FMT", "10x_{sample}_CCR2-CD45_{ch}.tif")
BG_UM = 8.0
DS = 4

os.makedirs(OUT, exist_ok=True)


def discover_samples():
    """Sample names present in SRC, from the filenames. No hardcoded animal list."""
    if not os.path.isdir(SRC):
        raise SystemExit(f"source directory not found: {SRC}\n"
                         "Point CHP_SRC at your tif_max folder, or edit SRC above.")
    names = sorted({m.group(1) for m in
                    (re.match(FILE_RE, p) for p in os.listdir(SRC)) if m})
    if not names:
        raise SystemExit(
            f"no files in {SRC} match {FILE_RE!r}.\n"
            "Rename your TIFs to <prefix>_<sample>_<stain>_C0.tif, or set CHP_FILE_RE / "
            "CHP_FILE_FMT to a pattern that matches them.")
    return names


def run(sample, ch):
    t0 = time.time()
    dst = f"{OUT}/{sample}_{ch}_clean.tif"
    if os.path.exists(dst) and os.path.exists(f"{OUT}/{sample}_{ch}_clean.json"):
        print(f"skip {sample} {ch} (already exists)", flush=True)
        return
    base = FILE_FMT.format(sample=sample, ch=ch)
    fjff = f"{FJFF}/{base[:-4]}_fjff.tif"
    src = fjff if os.path.exists(fjff) else f"{SRC}/{base}"
    a = tifffile.imread(src).astype(np.float32)

    # Rolling-ball residual background (same as v10: grey opening on a 4x downsample)
    small = a[::DS, ::DS]
    r = max(3, int(round(BG_UM / (PIX_UM * DS))))
    bg = ndi.gaussian_filter(ndi.grey_opening(ndi.uniform_filter(small, 3), footprint=disk(r)), r / 2)
    up = np.empty(a.shape, np.float32)
    ndi.zoom(bg, (a.shape[0] / bg.shape[0], a.shape[1] / bg.shape[1]), output=up, order=1)
    flat = np.clip(a - up, 0, None)
    del a, up, bg, small

    # sigma: MAD of the lowest 70% inside the tissue mask (v10 rule, estimated on the clean image)
    fs = flat[::8, ::8]
    sm = ndi.gaussian_filter(fs, 8)
    tm = sm > max(np.percentile(sm, 20), threshold_otsu(sm) * 0.25)
    vt = fs[tm] if tm.any() else fs.ravel()
    lo = vt[vt < np.percentile(vt, 70)]
    sigma = max(1.4826 * float(np.median(np.abs(lo - np.median(lo)))), 1e-6)

    tifffile.imwrite(dst, np.clip(flat, 0, 65535).astype(np.uint16), bigtiff=True)  # uncompressed, memmap-able

    # Overview image: 2-sigma floor + 1/99.7 stretch (the good-looking version)
    ov = np.clip(fs - 2 * sigma, 0, None)
    lo_, hi_ = np.percentile(ov, [1, 99.7])
    g = (np.clip((ov - lo_) / max(hi_ - lo_, 1e-6), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(g).save(f"{OUT}/{sample}_{ch}_clean_overview.png")

    json.dump(dict(sample=sample, ch=ch, sigma=sigma, bg_um=BG_UM, ds=DS,
                   src=os.path.basename(src)),
              open(f"{OUT}/{sample}_{ch}_clean.json", "w"), indent=1)
    print(f"done {sample} {ch} sigma={sigma:.1f} base={'fjff' if 'fjff' in src else 'max'} "
          f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    chans = sys.argv[1:] or ["C0", "C1", "C2"]
    SAMPLES = discover_samples()
    print(f"{len(SAMPLES)} samples discovered in {SRC}:", ", ".join(SAMPLES), flush=True)
    for s in SAMPLES:
        for c in chans:
            run(s, c)
    print("ALL DONE", flush=True)
