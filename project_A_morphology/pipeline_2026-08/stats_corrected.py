"""Corrected statistics: (1) out-of-bounds ROIs are cropped (not shifted)
(2) density denominator = tissue area (glass/black regions excluded)
(3) overlapping ROIs counted only once (in id order; the later ROI drops its
intersection with earlier ones). Everything else matches the baseline stats.

This is the AUTHORITATIVE analysis: the published numbers come from here.

Needs: rois/*.json, clean/<sample>_C0_clean.tif, stats_baseline_parallel.py (imported for
       hedges_g / perm_p), and the raw tif_max stack for tissue_mask8() (config.py
       SRC_TIF_MAX, or set CHP_SRC).
Writes: analysis_output/roi_metrics_corrected.csv, analysis_output/animal_summary_corrected.csv
Usage: ./.venv/bin/python stats_corrected.py | tee analysis_output/stats_corrected_stdout.txt
       (~3 min, 6 processes; the effect sizes and p-values go to stdout ONLY - capture them)
"""
import importlib
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

TOOL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOL)
base = importlib.import_module("stats_baseline_parallel")

from config import IMG_PX as IMG, PIX_UM as PIX, SRC_TIF_MAX  # noqa: E402

RAW_SRC = SRC_TIF_MAX
RAW_FMT = os.environ.get("CHP_FILE_FMT", "10x_{sample}_CCR2-CD45_{ch}.tif")

_tm_cache = {}


def tissue_mask8(sample):
    if sample in _tm_cache:
        return _tm_cache[sample]
    import tifffile
    from scipy import ndimage as ndi
    from skimage.filters import threshold_otsu
    a = np.asarray(tifffile.memmap(
        f"{RAW_SRC}/" + RAW_FMT.format(sample=sample, ch="C0"),
        mode="r")[::8, ::8], np.float32)
    sm = ndi.gaussian_filter(a, 8)
    tm = sm > max(np.percentile(sm, 20), threshold_otsu(sm) * 0.25)
    _tm_cache[sample] = tm
    return tm


def work(task):
    sample, ro, prev_rects = task
    import server as S
    from scipy import ndimage as ndi
    from skimage.measure import label as sklabel
    from skimage.morphology import disk

    # (1) Crop to inside the image (no shifting)
    x0, y0 = max(0, ro["x"]), max(0, ro["y"])
    x1, y1 = min(IMG, ro["x"] + ro["w"]), min(IMG, ro["y"] + ro["h"])
    w, h = x1 - x0, y1 - y0
    P = dict(S.DEFAULTS)
    r = S.process(sample, "C0", x0, y0, w, h, P)
    sk, mask, soma = r["sk"].copy(), r["mask"].copy(), r["soma"].copy()

    # (2) Tissue mask (8x upsampled to the view region)
    tm8 = tissue_mask8(sample)
    tm = np.zeros((h, w), bool)
    ys = np.clip((y0 + np.arange(h)) // 8, 0, tm8.shape[0] - 1)
    xs = np.clip((x0 + np.arange(w)) // 8, 0, tm8.shape[1] - 1)
    tm[:] = tm8[np.ix_(ys, xs)]

    # (3) Drop the intersection with earlier ROIs
    ex = np.zeros((h, w), bool)
    for (px, py, pw, ph) in prev_rects:
        ox0, oy0 = max(x0, px), max(y0, py)
        ox1, oy1 = min(x1, px + pw), min(y1, py + ph)
        if ox1 > ox0 and oy1 > oy0:
            ex[oy0 - y0:oy1 - y0, ox0 - x0:ox1 - x0] = True

    keep = tm & ~ex
    sk &= keep
    mask &= keep
    soma &= keep

    eff_mm2 = float(keep.sum()) * (PIX / 1e3) ** 2
    if eff_mm2 < 0.001:
        return None
    K8 = np.ones((3, 3), bool)
    nc = ndi.convolve(sk.astype(np.uint8), K8.astype(np.uint8), mode="constant") - sk
    jpix = sk & (nc >= 3)
    if jpix.any():
        jlab, n_j = ndi.label(ndi.binary_dilation(jpix, structure=disk(9)), structure=K8)
        n_j = int(n_j)
    else:
        n_j = 0
    n_t = int(sklabel(sk & (nc == 1), connectivity=2).max())
    n_soma = int(sklabel(soma, connectivity=2).max())
    n_comp = int(sklabel(sk, connectivity=2).max())
    skel_um = float(sk.sum()) * PIX
    m = dict(sample=sample, roi_id=ro["id"], eff_mm2=round(eff_mm2, 5),
             tissue_frac=round(float(tm.mean()), 4),
             skel_mm_per_mm2=round(skel_um / 1e3 / eff_mm2, 3),
             soma_per_mm2=round(n_soma / eff_mm2, 1),
             fg_pct=round(100 * float(mask.sum()) / max(float(keep.sum()), 1), 3),
             comp_per_mm2=round(n_comp / eff_mm2, 1),
             junc_per_mm2=round(n_j / eff_mm2, 1),
             tips_per_mm2=round(n_t / eff_mm2, 1),
             junc_per_100um=round(100 * n_j / skel_um, 3) if skel_um else 0.0,
             mean_width_um=0.0)
    print(f"  {sample} R{ro['id']} done (tissue fraction {m['tissue_frac']:.0%})", flush=True)
    return m


if __name__ == "__main__":
    tasks = []
    for f in sorted(os.listdir(f"{TOOL}/rois")):
        if f.endswith(".json"):
            s = f[:-5]
            rois = sorted(json.load(open(f"{TOOL}/rois/{f}")), key=lambda r: r["id"])
            for i, ro in enumerate(rois):
                prev = [(p["x"], p["y"], p["w"], p["h"]) for p in rois[:i]]
                tasks.append((s, ro, prev))
    print(f"{len(tasks)} ROIs total (corrected: crop + tissue area + overlap removal)", flush=True)
    with Pool(6) as pool:
        rows = [r for r in pool.map(work, tasks) if r]
    R = pd.DataFrame(rows)
    R.to_csv(f"{TOOL}/analysis_output/roi_metrics_corrected.csv", index=False)

    R["sex"] = R["sample"].str[0]
    R["geno"] = np.where(R["sample"].str.contains("WT"), "WT", "Het")
    mets = ["skel_mm_per_mm2", "soma_per_mm2", "fg_pct", "comp_per_mm2", "junc_per_mm2", "tips_per_mm2"]
    ag = R.groupby("sample").agg({**{m: "median" for m in mets}, "roi_id": "count"})
    ag["geno"] = np.where(ag.index.str.contains("WT"), "WT", "Het")
    ag.round(3).to_csv(f"{TOOL}/analysis_output/animal_summary_corrected.csv")

    WT, HET = ag[ag.geno == "WT"], ag[ag.geno == "Het"]
    # The "before" column is recomputed from the baseline run's own output instead of being
    # pasted in as a constant. A hard-coded number goes stale the moment anything upstream
    # changes and nothing here would notice - the same failure mode README.md warns about.
    SHOW = ("skel_mm_per_mm2", "soma_per_mm2", "fg_pct")
    prev, old = f"{TOOL}/analysis_output/animal_summary.csv", {}
    if os.path.exists(prev):
        B = pd.read_csv(prev, index_col=0)
        if "geno" not in B:
            B["geno"] = np.where(B.index.str.contains("WT"), "WT", "Het")
        for met in SHOW:
            if met in B:
                w = B[B.geno == "WT"][met].values
                h = B[B.geno == "Het"][met].values
                old[met] = (f"{np.median(w):.3f}/{np.median(h):.3f} "
                            f"g={base.hedges_g(w, h):.2f} p={base.perm_p(w, h):.4f}")

    print("\n=== Key numbers after correction (vs before) ===")
    for met in SHOW:
        g = base.hedges_g(WT[met].values, HET[met].values)
        p = base.perm_p(WT[met].values, HET[met].values)
        print(f"{met:18s} WT={WT[met].median():.3f} Het={HET[met].median():.3f} "
              f"change={100*(HET[met].median()/WT[met].median()-1):+.1f}%  g={g:.2f} p={p:.4f}"
              f"   [before: {old.get(met, 'run stats_baseline_parallel.py first')}]")
