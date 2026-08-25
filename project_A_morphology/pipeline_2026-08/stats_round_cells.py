"""Round / amoeboid cell statistics over every ROI in rois/.

Per ROI: n_round, round_per_mm2, round_frac, skel_gap_frac (from server.process with the
frozen DEFAULTS) -> animal-level medians -> WT vs Het and F vs M, printed to stdout.

Needs: rois/*.json, clean/<sample>_C0_clean.tif, and stats_baseline_parallel.py
       (imported for hedges_g / perm_p - do not delete that file).
Writes: analysis_output/roi_round_cell_metrics.csv
        analysis_output/animal_round_cell_metrics.csv  (read by any summary report)
Usage: ./.venv/bin/python stats_round_cells.py | tee analysis_output/stats_round_cells_stdout.txt
       (the headline effect sizes go to stdout ONLY - capture them)
"""
import json, os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib

def work(task):
    sample, ro = task
    import server as S
    m = S.process(sample, "C0", ro["x"], ro["y"], ro["w"], ro["h"], dict(S.DEFAULTS))["metrics"]
    print(f"  {sample} R{ro['id']}", flush=True)
    return dict(sample=sample, roi_id=ro["id"], n_round=m["n_round"],
                round_per_mm2=m["round_per_mm2"], round_frac=m["round_frac"],
                skel_gap_frac=m["skel_gap_frac"])

if __name__ == "__main__":
    base = importlib.import_module("stats_baseline_parallel")
    TOOL = os.path.dirname(os.path.abspath(__file__))
    tasks = []
    for f in sorted(os.listdir(f"{TOOL}/rois")):
        if f.endswith(".json"):
            s = f[:-5]
            for ro in json.load(open(f"{TOOL}/rois/{f}")):
                tasks.append((s, ro))
    with Pool(6) as pool:
        rows = pool.map(work, tasks)
    R = pd.DataFrame(rows)
    R.to_csv(f"{TOOL}/analysis_output/roi_round_cell_metrics.csv", index=False)
    R["sex"] = R["sample"].str[0]
    R["geno"] = np.where(R["sample"].str.contains("WT"), "WT", "Het")
    ag = R.groupby("sample").agg(round_per_mm2=("round_per_mm2","median"),
                                 round_frac=("round_frac","median"),
                                 skel_gap_frac=("skel_gap_frac","median")).round(4)
    ag["sex"] = ag.index.str[0]
    ag["geno"] = np.where(ag.index.str.contains("WT"), "WT", "Het")
    ag.to_csv(f"{TOOL}/analysis_output/animal_round_cell_metrics.csv")
    print(ag.to_string())
    for met in ["round_per_mm2", "round_frac", "skel_gap_frac"]:
        WT, HET = ag[ag.geno=="WT"][met].values, ag[ag.geno=="Het"][met].values
        F, M = ag[ag.sex=="F"][met].values, ag[ag.sex=="M"][met].values
        print(f"{met}: WT={np.median(WT):.3f} Het={np.median(HET):.3f} g={base.hedges_g(WT,HET):.2f} p={base.perm_p(WT,HET):.3f}"
              f" | F={np.median(F):.3f} M={np.median(M):.3f} g={base.hedges_g(F,M):.2f} p={base.perm_p(F,M):.3f}")
