"""Two PDFs: (1) raw|processed comparison of every ROI at native resolution;
(2) every clean full scan at 4x downsampling. img2pdf embeds the JPEGs losslessly.

Prerequisites: run render_round_cell_gallery.py FIRST (this reads its JPEGs - with an empty
               gallery img2pdf.convert([]) fails); clean/ must be present; and `img2pdf`
               must be installed (it is in requirements.txt).
Writes: analysis_output/PDF1_74_ROIs_raw_vs_processed.pdf
        analysis_output/PDF2_12_full_scans.pdf
        analysis_output/../cache/_scan_<sample>.jpg   (intermediates)
        The page counts baked into the filenames are from the 2026-08 set - rename them for
        a new dataset.
Usage: ./.venv/bin/python make_pdfs.py
"""
import glob, json, os
import img2pdf
import numpy as np
import tifffile
from PIL import Image

TOOL = os.path.dirname(os.path.abspath(__file__))
OUT = f"{TOOL}/analysis_output"
os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{TOOL}/cache", exist_ok=True)   # nothing else creates this

# PDF 1: ROI comparison (uses the gallery JPEGs directly)
jpgs = sorted(glob.glob(f"{TOOL}/analysis_output/round_cell_gallery/*.jpg"))
if not jpgs:
    raise SystemExit("analysis_output/round_cell_gallery/ has no JPEGs - "
                     "run render_round_cell_gallery.py first.")
p1 = f"{OUT}/PDF1_74_ROIs_raw_vs_processed.pdf"
with open(p1, "wb") as fh:
    fh.write(img2pdf.convert(jpgs))
print(p1, f"{os.path.getsize(p1)/1e6:.0f} MB, {len(jpgs)} pages")

# PDF 2: full scans (clean base images, 4x downsampled ~3665^2, enough to zoom and inspect)
tmp = []
for jp in sorted(glob.glob(f"{TOOL}/clean/*_C0_clean.json")):
    m = json.load(open(jp))
    s, sg = m["sample"], m["sigma"]
    mm = tifffile.memmap(f"{TOOL}/clean/{s}_C0_clean.tif", mode="r")
    a = np.asarray(mm[::4, ::4], np.float32)
    a = np.clip(a - 2 * sg, 0, None)
    lo, hi = np.percentile(a, [1, 99.7])
    g = (np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(np.stack([g]*3, -1))
    from PIL import ImageDraw, ImageFont
    d = ImageDraw.Draw(im)
    try: f_ = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 72)
    except Exception: f_ = ImageFont.load_default()
    d.text((30, 20), f"{s}  (full scan, flat-fielded + background-cleaned, 1px = 0.66um)", font=f_, fill=(255, 220, 100))
    t = f"{TOOL}/cache/_scan_{s}.jpg"
    im.save(t, quality=87)
    tmp.append(t)
    print(" ", s, flush=True)
p2 = f"{OUT}/PDF2_12_full_scans.pdf"
with open(p2, "wb") as fh:
    fh.write(img2pdf.convert(tmp))
print(p2, f"{os.path.getsize(p2)/1e6:.0f} MB, {len(tmp)} pages")
