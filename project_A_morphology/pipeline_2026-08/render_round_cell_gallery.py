"""Render every ROI in rois/ at native resolution as an original|overlay strip
(magenta circle = detected round cell), plus a browsable index page.

Needs:  rois/*.json, clean/<sample>_C0_clean.tif
Writes: analysis_output/round_cell_gallery/<sample>_R<nn>.jpg
        analysis_output/round_cell_gallery/index.html
        NOTE: that directory ships pre-created and, on the author's machine, pre-populated
        with the 2026-08 run. Empty it before rendering a new dataset, or make_pdfs.py will
        mix old and new pages into one PDF.
Usage:  ./.venv/bin/python render_round_cell_gallery.py     (6 processes)
"""
import json, os, sys
from multiprocessing import Pool
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TOOL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOL)
OUT = f"{TOOL}/analysis_output/round_cell_gallery"
os.makedirs(OUT, exist_ok=True)
from config import IMG_PX as IMG  # noqa: E402

def font(sz):
    for p in ["/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/Helvetica.ttc"]:
        try: return ImageFont.truetype(p, sz)
        except Exception: pass
    return ImageFont.load_default()

def work(task):
    sample, ro = task
    import server as S
    x0, y0 = max(0, ro["x"]), max(0, ro["y"])
    x1, y1 = min(IMG, ro["x"] + ro["w"]), min(IMG, ro["y"] + ro["h"])
    r = S.process(sample, "C0", x0, y0, x1 - x0, y1 - y0, dict(S.DEFAULTS))
    m = r["metrics"]
    orig, ov, _ = S.render(r, dict(S.DEFAULTS))
    h_, w_ = orig.shape[:2]
    strip = Image.new("RGB", (w_ * 2 + 12, h_ + 44), (16, 16, 20))
    d = ImageDraw.Draw(strip)
    d.text((10, 8), f"{sample} R{ro['id']}   round(magenta)={m['n_round']}  frac={100*m['round_frac']:.1f}%   "
                    f"soma={m['n_soma']}  skel={m['skel_len_um']:.0f}um", font=font(26), fill=(255, 220, 120))
    strip.paste(Image.fromarray(orig), (0, 44))
    strip.paste(Image.fromarray(ov), (w_ + 12, 44))
    fn = f"{sample}_R{ro['id']:02d}.jpg"
    strip.save(f"{OUT}/{fn}", quality=90)
    print(" ", fn, flush=True)
    return dict(fn=fn, sample=sample, rid=ro["id"], n_round=m["n_round"], frac=m["round_frac"])

if __name__ == "__main__":
    tasks = []
    for f in sorted(os.listdir(f"{TOOL}/rois")):
        if f.endswith(".json"):
            s = f[:-5]
            for ro in sorted(json.load(open(f"{TOOL}/rois/{f}")), key=lambda r: r["id"]):
                tasks.append((s, ro))
    with Pool(6) as pool:
        rows = pool.map(work, tasks)
    items = "\n".join(
        f'<div class="it"><h3>{r["sample"]} R{r["rid"]} · {r["n_round"]} round cells ({100*r["frac"]:.1f}% of cells)</h3>'
        f'<a href="{r["fn"]}" target="_blank"><img loading="lazy" src="{r["fn"]}"></a></div>'
        for r in rows)
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>Round-cell detection gallery (74 ROIs)</title>
<style>body{{background:#111;color:#ddd;font:14px/1.6 -apple-system,"PingFang SC",sans-serif;margin:0;padding:16px}}
h1{{font-size:20px}} .it{{margin:26px 0}} h3{{color:#ffd479;margin:4px 0}}
img{{width:100%;border:1px solid #333}} p{{color:#999}}</style></head><body>
<h1>Round-cell detection · all 74 ROIs (left = original, right = extracted overlay)</h1>
<p>Magenta circle = detected as a "round / process-free cell". Click any image to open it at full size in a new tab
(can be zoomed further, down to pixel level).
Legend: blue = mask, red = skeleton, green = branch points, cyan = endpoints, yellow = soma outline, magenta circle = round cell</p>
{items}</body></html>"""
    open(f"{OUT}/index.html", "w").write(html)
    print("Gallery:", f"{OUT}/index.html")
