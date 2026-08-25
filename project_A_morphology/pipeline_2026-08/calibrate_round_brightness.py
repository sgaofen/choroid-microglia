"""EXPLORATORY one-off: calibrate the round-cell brightness gate (`round_bright_z`).

For every skeleton component that is <20 um long and >60% contained near its soma
(i.e. a round-cell candidate), report the peak brightness of its mask in units of the
image noise floor sigma. Inspect the printed distribution to choose the `round_bright_z`
cutoff; server.DEFAULTS currently uses 50.

Needs:  rois/*.json and clean/<sample>_C0_clean.tif
Writes: analysis_output/round_cell_candidate_brightness.csv
Usage:  ./.venv/bin/python calibrate_round_brightness.py     (6 processes)
"""
import json, os, sys
from multiprocessing import Pool
import numpy as np

TOOL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOL)
from config import IMG_PX as IMG, PIX_UM as PIX  # noqa: E402

def work(task):
    sample, ro = task
    import server as S
    from scipy import ndimage as ndi
    x0, y0 = max(0, ro["x"]), max(0, ro["y"])
    x1, y1 = min(IMG, ro["x"] + ro["w"]), min(IMG, ro["y"] + ro["h"])
    r = S.process(sample, "C0", x0, y0, x1-x0, y1-y0, dict(S.DEFAULTS))
    mm, sg = S.get_clean(sample, "C0")
    a = np.asarray(mm[y0:y1, x0:x1], np.float32)
    K8 = np.ones((3,3),bool)
    lab_v, ncv = ndi.label(r["sk"], structure=K8)
    out = []
    if ncv:
        near_v = ndi.binary_dilation(r["soma"], structure=K8, iterations=3)
        sz = ndi.sum_labels(r["sk"], lab_v, index=np.arange(1,ncv+1)) * PIX
        iv = ndi.sum_labels(r["sk"] & near_v, lab_v, index=np.arange(1,ncv+1)) * PIX
        cand = np.where((sz < 20.0) & (iv/np.maximum(sz,1e-9) > 0.6))[0] + 1
        objs = ndi.find_objects(lab_v)
        for i in cand:
            o = objs[i-1]
            oy0, oy1 = max(0,o[0].start-8), min(a.shape[0], o[0].stop+8)
            ox0, ox1 = max(0,o[1].start-8), min(a.shape[1], o[1].stop+8)
            reg = r["mask"][oy0:oy1, ox0:ox1]
            pk = float(a[oy0:oy1, ox0:ox1][reg].max()) / sg if reg.any() else 0.0
            out.append(dict(sample=sample, rid=ro["id"], peak_z=round(pk,1)))
    print(f"  {sample} R{ro['id']}: {len(out)}", flush=True)
    return out

if __name__ == "__main__":
    tasks = []
    for f in sorted(os.listdir(f"{TOOL}/rois")):
        if f.endswith(".json"):
            s = f[:-5]
            for ro in sorted(json.load(open(f"{TOOL}/rois/{f}")), key=lambda r: r["id"]):
                tasks.append((s, ro))
    with Pool(6) as pool:
        res = pool.map(work, tasks)
    import pandas as pd
    R = pd.DataFrame([r for sub in res for r in sub])
    R.to_csv(f"{TOOL}/analysis_output/round_cell_candidate_brightness.csv", index=False)
    v = R.peak_z.values
    print(f"\nTotal candidates {len(v)}")
    for q in [5,10,25,50,75,90,95]:
        print(f"p{q}: {np.percentile(v,q):.0f}σ")
    hist, edges = np.histogram(np.log10(np.clip(v,1,None)), bins=24)
    for h_, e0, e1 in zip(hist, edges[:-1], edges[1:]):
        print(f"{10**e0:6.0f}-{10**e1:6.0f}σ  {'#'*max(1,int(60*h_/max(hist.max(),1))) if h_ else ''} {h_}")
