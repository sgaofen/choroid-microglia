"""Rebuild the analysis-ready `clean/` image set from the Fiji flat-fielded TIFs.

Standalone: no config.py, no other file from the bundle. Point it at the `_work`
folder that already holds tif_fiji_flatfielded/ and tif_max/, and it writes
`_work/clean/` beside them - the layout the rest of the pipeline expects.

    python rebuild_clean.py D:\\ChP_morphometry_2026-08\\images

Options
    --channels C0 C2      only these channels (default C0 C1 C2)
    --out DIR             write somewhere other than <work>/clean
    --pix-um 0.164827     micrometres per pixel; MUST match the image set
    --check SHA256.txt    after each image, compare against a reference digest

For each <sample>_<channel> it writes three files, exactly as the original
preprocess_clean_images.py does:
    <sample>_<ch>_clean.tif           uint16 uncompressed BigTIFF, memmap-able
    <sample>_<ch>_clean.json          that image's noise floor (sigma) and provenance
    <sample>_<ch>_clean_overview.png  8x downsampled look-at-me image

The .json is written last, so an interrupted run leaves an incomplete pair that the
next run redoes. Safe to re-run; finished images are skipped.

Needs: numpy scipy scikit-image tifffile pillow     (Python 3.9-3.12)
Uses ~4 GB of RAM per image and runs one image at a time. Roughly 1-3 min each,
so ~1-2 h for all 36. Output is ~14 GB for three channels, ~4.8 GB for C0 alone.
"""
import argparse
import hashlib
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

# Frozen for the 2026-08 delivery. Changing either changes every reported number.
BG_UM = 8.0     # rolling-ball radius, micrometres
DS = 4          # the background is estimated on a 4x downsample

FILE_RE = r"10x_(.+)_CCR2-CD45_C0_fjff\.tif$"
FJFF_FMT = "10x_{sample}_CCR2-CD45_{ch}_fjff.tif"
MAX_FMT = "10x_{sample}_CCR2-CD45_{ch}.tif"


def discover(fjff_dir):
    if not os.path.isdir(fjff_dir):
        sys.exit(f"not a directory: {fjff_dir}")
    names = sorted({m.group(1) for m in
                    (re.match(FILE_RE, p) for p in os.listdir(fjff_dir)) if m})
    if not names:
        sys.exit(f"no files in {fjff_dir} match {FILE_RE!r}")
    return names


def sha256(path, buf=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def run(sample, ch, fjff_dir, max_dir, out, pix_um, ref):
    t0 = time.time()
    dst = os.path.join(out, f"{sample}_{ch}_clean.tif")
    meta = os.path.join(out, f"{sample}_{ch}_clean.json")
    if os.path.exists(dst) and os.path.exists(meta):
        print(f"skip {sample} {ch} (already there)", flush=True)
        return True

    fjff = os.path.join(fjff_dir, FJFF_FMT.format(sample=sample, ch=ch))
    src = fjff if os.path.exists(fjff) else os.path.join(
        max_dir, MAX_FMT.format(sample=sample, ch=ch))
    if not os.path.exists(src):
        print(f"MISSING {sample} {ch}: neither {os.path.basename(fjff)} nor a tif_max fallback",
              flush=True)
        return False
    a = tifffile.imread(src).astype(np.float32)

    # Rolling-ball residual background: grey opening on a 4x downsample, then smoothed
    # and resampled back up. Estimating it small is what makes this tractable at 267 MP.
    small = a[::DS, ::DS]
    r = max(3, int(round(BG_UM / (pix_um * DS))))
    bg = ndi.gaussian_filter(
        ndi.grey_opening(ndi.uniform_filter(small, 3), footprint=disk(r)), r / 2)
    up = np.empty(a.shape, np.float32)
    ndi.zoom(bg, (a.shape[0] / bg.shape[0], a.shape[1] / bg.shape[1]), output=up, order=1)
    flat = np.clip(a - up, 0, None)
    del a, up, bg, small

    # sigma = MAD of the lowest 70% inside the tissue mask, measured on the clean image.
    # Every threshold downstream is expressed in units of this number.
    fs = flat[::8, ::8]
    sm = ndi.gaussian_filter(fs, 8)
    tm = sm > max(np.percentile(sm, 20), threshold_otsu(sm) * 0.25)
    vt = fs[tm] if tm.any() else fs.ravel()
    lo = vt[vt < np.percentile(vt, 70)]
    sigma = max(1.4826 * float(np.median(np.abs(lo - np.median(lo)))), 1e-6)

    tifffile.imwrite(dst, np.clip(flat, 0, 65535).astype(np.uint16), bigtiff=True)

    ov = np.clip(fs - 2 * sigma, 0, None)
    lo_, hi_ = np.percentile(ov, [1, 99.7])
    g = (np.clip((ov - lo_) / max(hi_ - lo_, 1e-6), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(g).save(os.path.join(out, f"{sample}_{ch}_clean_overview.png"))

    # written last: its absence marks the pair as unfinished
    with open(meta, "w") as f:
        json.dump(dict(sample=sample, ch=ch, sigma=sigma, bg_um=BG_UM, ds=DS,
                       src=os.path.basename(src)), f, indent=1)

    note = ""
    if ref is not None:
        want = ref.get(os.path.basename(dst))
        if want:
            note = "  sha256 MATCH" if sha256(dst) == want else "  sha256 *** MISMATCH ***"
        else:
            note = "  (no reference digest)"
    print(f"done {sample} {ch}  sigma={sigma:.1f}  base={'fjff' if src.endswith('_fjff.tif') else 'max'}"
          f"  {time.time()-t0:.0f}s{note}", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work", help="the folder holding tif_fiji_flatfielded/ and tif_max/ "
                                 "(on the delivery drive: ChP_morphometry_2026-08/images)")
    ap.add_argument("--out", default=None, help="output dir (default <work>\\clean)")
    ap.add_argument("--channels", nargs="+", default=["C0", "C1", "C2"])
    ap.add_argument("--pix-um", type=float, default=0.164827)
    ap.add_argument("--check", default=None, help="SHA256 reference file to verify against")
    a = ap.parse_args()

    fjff_dir = os.path.join(a.work, "tif_fiji_flatfielded")
    max_dir = os.path.join(a.work, "tif_max")
    out = a.out or os.path.join(a.work, "clean")
    os.makedirs(out, exist_ok=True)

    ref = None
    if a.check:
        ref = {}
        for line in open(a.check, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#"):
                digest, name = line.split(None, 1)
                ref[os.path.basename(name.strip())] = digest

    samples = discover(fjff_dir)
    print(f"source : {fjff_dir}")
    print(f"output : {out}")
    print(f"pix_um : {a.pix_um}   bg={BG_UM} um   ds={DS}")
    print(f"{len(samples)} samples: {', '.join(samples)}")
    print(f"channels: {', '.join(a.channels)}   -> {len(samples)*len(a.channels)} images\n", flush=True)

    t0, ok = time.time(), 0
    for s in samples:
        for c in a.channels:
            ok += bool(run(s, c, fjff_dir, max_dir, out, a.pix_um, ref))
    print(f"\nALL DONE  {ok}/{len(samples)*len(a.channels)} images  "
          f"{(time.time()-t0)/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
