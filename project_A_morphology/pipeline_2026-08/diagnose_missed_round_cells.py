"""EXPLORATORY one-off, kept for the record - not part of the pipeline.

Finds bright round blobs (>8 sigma, 30-1000 um^2, aspect ratio <2.5) that the pipeline did
NOT count, and reports which stage dropped each one (threshold / cleanup / size filter /
round-and-speckle-web filter) so the responsible parameter can be identified.

Needs:  rois/*.json, clean/<sample>_C0_clean.tif
Writes: analysis_output/missed_round_blobs.csv
Usage:  ./.venv/bin/python diagnose_missed_round_cells.py    (6 processes)
"""
import json, os, sys
from multiprocessing import Pool
import numpy as np
import tifffile

TOOL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOL)
from config import IMG_PX as IMG, PIX_UM as PIX  # noqa: E402

def work(task):
    sample, ro = task
    import server as S
    from scipy import ndimage as ndi
    x0, y0 = max(0, ro["x"]), max(0, ro["y"])
    x1, y1 = min(IMG, ro["x"] + ro["w"]), min(IMG, ro["y"] + ro["h"])
    P = dict(S.DEFAULTS); P["dbg"] = 1
    r = S.process(sample, "C0", x0, y0, x1 - x0, y1 - y0, P)
    mm, sg = S.get_clean(sample, "C0")
    a = np.asarray(mm[y0:y1, x0:x1], np.float32)

    # Bright-blob candidates: >8σ after smoothing, area 30-1000µm², aspect ratio < 2.5
    sm = ndi.gaussian_filter(a, 2)
    bl = ndi.binary_fill_holes(sm > 8 * sg)
    lab, n = ndi.label(bl, structure=np.ones((3,3),bool))
    out = []
    if n:
        objs = ndi.find_objects(lab)
        for i in range(1, n + 1):
            o = objs[i-1]
            m_ = lab[o] == i
            area = m_.sum() * PIX**2
            if not (30 <= area <= 1000):
                continue
            hh, ww = o[0].stop-o[0].start, o[1].stop-o[1].start
            if max(hh,ww)/max(min(hh,ww),1) > 2.5:
                continue
            def cov(arr): return float(arr[o][m_].mean())
            fin = cov(r["mask"])
            if fin > 0.05:
                continue  # already detected, skip
            # These four dbg keys are server.py's internal contract (set in process(), see the
            # `_dbg[...]` assignments) and must match it exactly, so they stay in Chinese.
            #   1_raw_mask_after_threshold            = stage 1, raw mask straight after thresholding
            #   2_mask_after_cleanup          = stage 2, mask after closing / small-object removal / hole fill
            #   3_skeleton_after_length_filter        = stage 3, skeleton after the tiered min-length filter
            #   4_skeleton_after_round_and_web_filters    = stage 4, skeleton after the round-cell and speckle-web filters
            # Renaming them means editing server.py and this file in the same commit.
            st = dict(thr=cov(r["dbg"]["1_raw_mask_after_threshold"]), cln=cov(r["dbg"]["2_mask_after_cleanup"]),
                      sk3=float((r["dbg"]["3_skeleton_after_length_filter"][o] & m_).sum()),
                      sk4=float((r["dbg"]["4_skeleton_after_round_and_web_filters"][o] & m_).sum()))
            if st["thr"] < 0.1: stage = "A_never_passed_threshold"
            elif st["cln"] < 0.1: stage = "B_lost_during_cleanup"
            elif st["sk3"] == 0: stage = "C_killed_by_tiered_min_skeleton"
            elif st["sk4"] == 0: stage = "D_removed_by_round_filter_but_not_counted?!"
            else: stage = "E_drop_nosk_stage(skeleton removed by a later step)"
            cy = (o[0].start+o[0].stop)//2; cx = (o[1].start+o[1].stop)//2
            out.append(dict(sample=sample, rid=ro["id"], y=cy, x=cx,
                            area_um2=round(area,1), bright=round(float(sm[o][m_].max()/sg),1), stage=stage))
    print(f"  {sample} R{ro['id']}: missed {len(out)}", flush=True)
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
    R.to_csv(f"{TOOL}/analysis_output/missed_round_blobs.csv", index=False)
    print("\nTotal missed bright blobs:", len(R))
    if len(R):
        print(R.stage.value_counts().to_string())
        print("\n10 brightest misses:")
        print(R.sort_values("bright", ascending=False).head(10).to_string(index=False))
