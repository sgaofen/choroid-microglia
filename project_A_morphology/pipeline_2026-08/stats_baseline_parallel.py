"""Baseline group statistics (parallel version, saturates all cores).

74 ROIs x (full extraction + ROI-level clarity score) computed in parallel ->
animal-level medians -> four comparisons (WT/Het pooled, F/M, within males,
within females) x Hedges g + 10,000 permutations + BH-FDR ->
noise (sigma) and clarity confound checks + quantitative check of the
"females are more out of focus" observation.

Needs: rois/*.json, clean/<sample>_C0_clean.tif (+ .json), and server.py.
Writes: analysis_output/roi_metrics_all.csv, analysis_output/animal_summary.csv,
        analysis_output/baseline_stats_<today>.md   (the date is today's, not a fixed one)
Also defines hedges_g / perm_p / bh_fdr, imported by stats_corrected.py and
stats_round_cells.py - do not delete this file.
Usage: ./.venv/bin/python stats_baseline_parallel.py       (6 processes)
       REUSE=1 ./.venv/bin/python stats_baseline_parallel.py   # re-stat an existing roi_metrics_all.csv
Requires `tabulate` at the very last step (pandas .to_markdown()).
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

TOOL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOL)
from config import PIX_UM  # noqa: E402

PRIMARY = ["skel_mm_per_mm2", "soma_per_mm2", "fg_pct", "comp_per_mm2"]      # insensitive to resolution / branch-detection error
SECONDARY = ["junc_per_mm2", "tips_per_mm2", "junc_per_100um", "mean_width_um"]  # branching-related, known to be underestimated, reference only
METRICS = PRIMARY + SECONDARY


def roi_clarity(sample, x, y, w, h):
    """ROI-level clarity = fine-scale (0.5-2 µm) / haze-scale (4-16 µm) energy ratio (mask-independent, averaged over the whole box)."""
    import tifffile
    from scipy import ndimage as ndi
    mm = tifffile.memmap(f"{TOOL}/clean/{sample}_C0_clean.tif", mode="r")
    a = np.asarray(mm[y:y + h:2, x:x + w:2], np.float32)   # ds2
    px2 = PIX_UM * 2

    def gb(s_um):
        return ndi.gaussian_filter(a, s_um / px2 * 0.5)

    fine = float(np.mean(np.abs(gb(0.5) - gb(2.0))))
    coarse = float(np.mean(np.abs(gb(4.0) - gb(16.0))))
    return fine / max(coarse, 1e-9)


def work(task):
    sample, ro = task
    import server as S
    P = dict(S.DEFAULTS)
    m = S.process(sample, "C0", ro["x"], ro["y"], ro["w"], ro["h"], P)["metrics"]
    m["comp_per_mm2"] = round(m["n_comp"] / m["area_mm2"], 1)
    m["clarity"] = round(roi_clarity(sample, ro["x"], ro["y"], ro["w"], ro["h"]), 3)
    m.update(sample=sample, roi_id=ro["id"], x=ro["x"], y=ro["y"], w=ro["w"], h=ro["h"])
    print(f"  {sample} R{ro['id']} done", flush=True)
    return m


def hedges_g(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sp = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    if sp == 0:
        return np.nan
    J = 1 - 3 / (4 * (na + nb) - 9)
    return J * (a.mean() - b.mean()) / sp


def perm_p(a, b, n=10000, seed=7):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    rng = np.random.default_rng(seed)
    obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    cnt = 0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(pool[:len(a)].mean() - pool[len(a):].mean()) >= obs - 1e-12:
            cnt += 1
    return (cnt + 1) / (n + 1)


def bh_fdr(ps):
    ps = np.asarray(ps, float)
    q = np.full_like(ps, np.nan)
    ok = ~np.isnan(ps)
    p, idx = ps[ok], np.where(ok)[0]
    order = np.argsort(p)
    m = len(p)
    qq = np.empty(m)
    prev = 1.0
    for rank, oi in enumerate(reversed(order), 1):
        i = m - rank
        prev = min(prev, p[order[i]] * m / (i + 1))
        qq[order[i]] = prev
    q[idx] = qq
    return q


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def compare(A, dfA, B, dfB, label):
    # The Chinese keys below become the column headers of the "Group comparisons" table in
    # analysis_output/baseline_stats_<date>.md and of the stdout table:
    #     comparison = comparison, metric = metric, "<group>_median" = "<group> median".
    # They are PRESENTATIONAL only - nothing in this bundle reads them back (stats_corrected.py
    # and stats_round_cells.py import only hedges_g / perm_p / bh_fdr from this module).
    # Renaming them to comparison / metric / f"{A}_median" is safe for the code but changes the
    # text of a delivered report, so it is deliberately left alone here; do it together with a
    # regeneration of that report.
    rows = []
    for met in METRICS:
        a, b = dfA[met].values, dfB[met].values
        rows.append(dict(comparison=label, metric=met,
                         **{f"{A}_median": round(float(np.median(a)), 3) if len(a) else np.nan,
                            f"{B}_median": round(float(np.median(b)), 3) if len(b) else np.nan},
                         g=round(hedges_g(a, b), 2), p_perm=perm_p(a, b)))
    q = bh_fdr([r["p_perm"] for r in rows])
    for r, qi in zip(rows, q):
        r["q_FDR"] = round(qi, 3) if not np.isnan(qi) else np.nan
        r["p_perm"] = round(r["p_perm"], 4) if not np.isnan(r["p_perm"]) else np.nan
    return rows


if __name__ == "__main__":
    tasks = []
    for f in sorted(os.listdir(f"{TOOL}/rois")):
        if f.endswith(".json"):
            s = f[:-5]
            for ro in json.load(open(f"{TOOL}/rois/{f}")):
                tasks.append((s, ro))
    csvp = f"{TOOL}/analysis_output/roi_metrics_all.csv"
    if os.environ.get("REUSE") == "1" and os.path.exists(csvp):
        R = pd.read_csv(csvp)
        print(f"Reusing existing {len(R)} rows of ROI metrics", flush=True)
    else:
        print(f"{len(tasks)} ROIs total, Pool(6) in parallel", flush=True)
        with Pool(6) as pool:
            rows = pool.map(work, tasks)
        R = pd.DataFrame(rows)
    os.makedirs(f"{TOOL}/analysis_output", exist_ok=True)
    R.to_csv(f"{TOOL}/analysis_output/roi_metrics_all.csv", index=False)

    # Animal level
    R["sex"] = R["sample"].str[0]
    R["geno"] = np.where(R["sample"].str.contains("WT"), "WT", "Het")
    ag = R.groupby("sample").agg({**{m: "median" for m in METRICS + ["clarity"]},
                                  "roi_id": "count"}).rename(columns={"roi_id": "n_roi"})
    ag["sex"] = ag.index.str[0]
    ag["geno"] = np.where(ag.index.str.contains("WT"), "WT", "Het")
    ag["sigma"] = [json.load(open(f"{TOOL}/clean/{s}_C0_clean.json"))["sigma"] for s in ag.index]
    ag.round(3).to_csv(f"{TOOL}/analysis_output/animal_summary.csv")

    # Comparisons
    out = []
    out += compare("WT", ag[ag.geno == "WT"], "Het", ag[ag.geno == "Het"], "Genotype (pooled)")
    out += compare("F", ag[ag.sex == "F"], "M", ag[ag.sex == "M"], "Sex (pooled)")
    Msub = ag[ag.sex == "M"]
    out += compare("WT", Msub[Msub.geno == "WT"], "Het", Msub[Msub.geno == "Het"], "Genotype within males")
    Fsub = ag[ag.sex == "F"]
    out += compare("WT", Fsub[Fsub.geno == "WT"], "Het", Fsub[Fsub.geno == "Het"], "Genotype within females")
    C = pd.DataFrame(out)

    # Confound check (animal level)
    conf = []
    for met in METRICS:
        conf.append(dict(metric=met,
                         rho_sigma=round(spearman(ag[met], ag["sigma"]), 2),
                         rho_clarity=round(spearman(ag[met], ag["clarity"]), 2)))
    CF = pd.DataFrame(conf)

    # Quality x sex (user observation: females are more out of focus)
    qF, qM = ag[ag.sex == "F"], ag[ag.sex == "M"]
    qual = (f"clarity: F median {qF['clarity'].median():.3f} vs M {qM['clarity'].median():.3f}"
            f" (g={hedges_g(qF['clarity'], qM['clarity']):.2f}, p={perm_p(qF['clarity'].values, qM['clarity'].values):.3f});"
            f" noise sigma: F median {qF['sigma'].median():.1f} vs M {qM['sigma'].median():.1f}"
            f" (g={hedges_g(qF['sigma'], qM['sigma']):.2f}, p={perm_p(qF['sigma'].values, qM['sigma'].values):.3f})")

    today = time.strftime("%Y-%m-%d")
    report = f"{TOOL}/analysis_output/baseline_stats_{today}.md"
    with open(report, "w") as fh:
        fh.write(f"# Baseline group statistics ({today}, fjff base, {len(R)} ROIs, animal as the unit)\n\n")
        fh.write("**Methods**: for each animal take the median across all its ROI metrics; Hedges g (small-sample corrected) + two-sided permutation test (10,000 iterations)"
                 " + BH-FDR (across metrics within each comparison). n=12 (5WT/7Het; 5F/7M), inherently only enough for descriptive conclusions.\n\n"
                 "**Known measurement limitations** (down-weight when interpreting): the 10x/NA limit causes systematic underestimation of branch counts; "
                 "despurring / junction-merging rules affect the junc-type metrics; bright round cells (amoeboid-like) are inherently underestimated by the skeleton method — "
                 "primary metrics are skeleton density / soma density / foreground % / component density; junc-type metrics are reference only.\n\n")
        fh.write("## Per-animal summary\n\n" + ag.round(3).to_markdown() + "\n\n")
        fh.write("## Group comparisons\n\n" + C.to_markdown(index=False) + "\n\n")
        fh.write("## Confound check (animal-level Spearman; |rho|>0.5 means that metric's significance is untrustworthy)\n\n"
                 + CF.to_markdown(index=False) + "\n\n")
        fh.write("## Image quality x sex (quantifying the out-of-focus observation)\n\n" + qual + "\n")
    print(f"Wrote analysis_output/baseline_stats_{today}.md")
    print(qual)
    print(C.to_string(index=False))
