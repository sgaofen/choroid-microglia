# -*- coding: utf-8 -*-
"""Standalone analysis script: run the final recipe per ROI to extract metrics
-> animal-level summary -> between-group statistics.

Usage:
    ./.venv/bin/python analysis_group_comparison.py

This is a STANDALONE ALTERNATIVE that recomputes everything in one pass; it is useful
for a quick look, but stats_corrected.py is the authoritative analysis and produces the
published numbers.

Reads only the server.py interface and rois/*.json; changes no other file in the
tool directory.
Writes to analysis_output/: roi_metrics_groupcmp.csv, animal_summary_groupcmp.csv,
analysis_report.md, group_comparison_scatter.png (the last only if matplotlib is
installed). The `_groupcmp` suffixes exist so this script cannot clobber
stats_baseline_parallel.py's roi_metrics_all.csv / animal_summary.csv, which carry a
different schema.

Statistical discipline (from the project handoff):
  1. The animal is always the unit: an animal's value = median over all of its ROI metrics.
  2. Comparisons: genotype WT vs Het (all animals), sex F vs M (all animals);
     n is small, so descriptive conclusions only.
  3. Per metric report: both groups' animal-level medians, Hedges g (small-sample
     correction), permutation test p (animal-level label permutation, 10000
     iterations, two-sided), BH-FDR q.
  4. Noise-confound check: Spearman of metric vs global_sigma; |rho|>0.5 is
     flagged "possible noise confound".
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats as sps

TOOL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOL)
import server as S  # noqa: E402

OUT_DIR = os.path.join(TOOL, "analysis_output")
ROI_DIR = os.path.join(TOOL, "rois")

# All 12 animals (samples)
ALL_SAMPLES = [
    "F-Het1", "F-Het2", "F-Het3", "F-WT1", "F-WT2",
    "M-Het1", "M-Het2", "M-Het4", "M-Het5",
    "M-WT1", "M-WT2", "M-WT3",
]

# Metrics used for between-group statistics (density/normalized metrics, so the
# median can be taken directly even when ROI areas differ)
ANALYSIS_METRICS = [
    "skel_mm_per_mm2", "junc_per_mm2", "tips_per_mm2", "soma_per_mm2",
    "junc_per_100um", "mean_width_um", "fg_pct",
]

N_PERM = 10000
PERM_SEED = 20260813
RHO_FLAG = 0.5  # |Spearman rho| above this -> possible noise confound


def parse_group(sample):
    sex = "F" if sample.startswith("F") else "M"
    genotype = "Het" if "Het" in sample else "WT"
    return sex, genotype


def load_rois(sample):
    p = os.path.join(ROI_DIR, f"{sample}.json")
    if not os.path.exists(p):
        return []
    try:
        rois = json.load(open(p))
    except Exception as e:
        print(f"[Warning] Failed to read {p}: {e}")
        return []
    return rois if isinstance(rois, list) else []


def hedges_g(a, b):
    """Hedges g (small-sample correction). a, b are animal-level arrays;
    returns nan if either group has n<2."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if sp == 0:
        return np.nan
    d = (a.mean() - b.mean()) / sp
    J = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)  # small-sample correction factor
    return d * J


def perm_pvalue(a, b, n_perm=N_PERM, seed=PERM_SEED):
    """Animal-level label permutation test (two-sided, statistic = difference of
    group means). Returns nan if either group has n<2."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    pool = np.concatenate([a, b])
    obs = abs(a.mean() - b.mean())
    rng = np.random.default_rng(seed)
    hit = 0
    for _ in range(n_perm):
        idx = rng.permutation(pool.size)
        pa = pool[idx[:n1]]
        pb = pool[idx[n1:]]
        if abs(pa.mean() - pb.mean()) >= obs - 1e-12:
            hit += 1
    return (hit + 1) / (n_perm + 1)


def bh_fdr(pvals):
    """Benjamini-Hochberg correction. nan values are returned as nan."""
    p = np.asarray(pvals, float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    m = ok.sum()
    if m == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * m / (np.arange(m) + 1)
    # Enforce monotonicity (running minimum from largest to smallest)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    qv = np.empty(m)
    qv[order] = np.clip(ranked, 0, 1)
    q[ok] = qv
    return q


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.{nd}g}" if abs(v) < 1e4 else f"{v:.3e}"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    P = dict(S.DEFAULTS)  # final recipe
    ch = P.get("ch", "C0")

    # ---------- 1) Run the pipeline ROI by ROI ----------
    roi_rows = []
    missing_samples = []   # samples with no ROIs drawn
    failed_rois = []       # ROIs that failed to process
    for sample in ALL_SAMPLES:
        rois = load_rois(sample)
        if not rois:
            missing_samples.append(sample)
            continue
        for roi in rois:
            rid = roi.get("id")
            x, y, w, h = roi.get("x"), roi.get("y"), roi.get("w"), roi.get("h")
            t0 = time.time()
            try:
                m = S.process(sample, ch, x, y, w, h, P)["metrics"]
            except Exception as e:
                print(f"[Warning] {sample} ROI#{rid} processing failed: {e}")
                failed_rois.append((sample, rid, str(e)))
                continue
            row = dict(sample=sample, roi_id=rid, x=x, y=y, w=w, h=h,
                       note=roi.get("note", ""))
            row.update(m)
            roi_rows.append(row)
            print(f"  {sample} ROI#{rid} ({w}x{h}) done, {time.time()-t0:.1f}s")

    roi_df = pd.DataFrame(roi_rows)
    # _groupcmp suffix: stats_baseline_parallel.py writes roi_metrics_all.csv /
    # animal_summary.csv with a DIFFERENT schema. Do not collide with it.
    roi_csv = os.path.join(OUT_DIR, "roi_metrics_groupcmp.csv")
    roi_df.to_csv(roi_csv, index=False, encoding="utf-8-sig")
    print(f"Wrote {roi_csv} ({len(roi_df)} rows)")

    # ---------- 2) Animal-level summary (median over ROIs) ----------
    metric_cols = [c for c in roi_df.columns
                   if c not in ("sample", "roi_id", "x", "y", "w", "h", "note")]
    animal_rows = []
    if len(roi_df):
        for sample, g in roi_df.groupby("sample"):
            sex, genotype = parse_group(sample)
            row = dict(sample=sample, sex=sex, genotype=genotype, n_roi=len(g))
            for c in metric_cols:
                row[c] = float(g[c].median())
            try:
                row["sigma"] = float(S.global_sigma(sample, ch))
            except Exception as e:
                print(f"[Warning] {sample} global_sigma failed: {e}")
                row["sigma"] = np.nan
            animal_rows.append(row)
    animal_df = pd.DataFrame(animal_rows)
    if len(animal_df):
        animal_df = animal_df.sort_values("sample").reset_index(drop=True)
    animal_csv = os.path.join(OUT_DIR, "animal_summary_groupcmp.csv")
    animal_df.to_csv(animal_csv, index=False, encoding="utf-8-sig")
    print(f"Wrote {animal_csv} ({len(animal_df)} animals)")

    # ---------- 3) Between-group statistics ----------
    def compare(df, col, g1, g2):
        """Returns [dict per metric]; g/p are nan when data are insufficient."""
        rows = []
        a_df = df[df[col] == g1]
        b_df = df[df[col] == g2]
        for met in ANALYSIS_METRICS:
            if met not in df.columns:
                rows.append(dict(metric=met, med1=np.nan, med2=np.nan,
                                 g=np.nan, p=np.nan))
                continue
            a = a_df[met].dropna().values
            b = b_df[met].dropna().values
            rows.append(dict(
                metric=met,
                n1=len(a), n2=len(b),
                med1=float(np.median(a)) if len(a) else np.nan,
                med2=float(np.median(b)) if len(b) else np.nan,
                g=hedges_g(a, b),
                p=perm_pvalue(a, b),
            ))
        q = bh_fdr([r["p"] for r in rows])
        for r, qi in zip(rows, q):
            r["q"] = qi
        return rows

    enough = len(animal_df) >= 4  # need at least 2v2 for anything to compute
    comp_geno = comp_sex = None
    if enough:
        comp_geno = compare(animal_df, "genotype", "WT", "Het")
        comp_sex = compare(animal_df, "sex", "F", "M")

    # ---------- 4) Noise-confound check (Spearman of metric vs sigma) ----------
    noise_rows = []
    if len(animal_df) >= 3 and "sigma" in animal_df.columns:
        sig = animal_df["sigma"].values
        for met in ANALYSIS_METRICS:
            if met not in animal_df.columns:
                continue
            v = animal_df[met].values
            ok = np.isfinite(v) & np.isfinite(sig)
            if ok.sum() >= 3:
                rho, pr = sps.spearmanr(v[ok], sig[ok])
            else:
                rho, pr = np.nan, np.nan
            noise_rows.append(dict(metric=met, rho=rho, p=pr,
                                   flagged=(np.isfinite(rho) and abs(rho) > RHO_FLAG)))
    flagged = {r["metric"] for r in noise_rows if r["flagged"]}

    # ---------- 5) Optional plotting (when matplotlib exists and data suffice) ----------
    fig_path = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        matplotlib.rcParams["font.family"] = ["Heiti TC", "Arial Unicode MS", "sans-serif"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        if enough:
            fig, axes = plt.subplots(2, len(ANALYSIS_METRICS),
                                     figsize=(3 * len(ANALYSIS_METRICS), 6))
            for j, met in enumerate(ANALYSIS_METRICS):
                for i, (col, groups) in enumerate(
                        [("genotype", ["WT", "Het"]), ("sex", ["F", "M"])]):
                    ax = axes[i, j]
                    for k, grp in enumerate(groups):
                        vals = animal_df.loc[animal_df[col] == grp, met].dropna()
                        ax.scatter(np.full(len(vals), k) +
                                   np.random.default_rng(1).uniform(-0.08, 0.08, len(vals)),
                                   vals, s=28, alpha=0.8)
                        if len(vals):
                            ax.hlines(vals.median(), k - 0.2, k + 0.2, colors="k", lw=1.5)
                    ax.set_xticks(range(len(groups)), groups)
                    ax.set_xlim(-0.6, len(groups) - 0.4)
                    if i == 0:
                        ax.set_title(met, fontsize=9)
            fig.suptitle("Animal-level metrics by group (bar = group median)")
            fig.tight_layout()
            fig_path = os.path.join(OUT_DIR, "group_comparison_scatter.png")
            fig.savefig(fig_path, dpi=130)
            plt.close(fig)
            print(f"Wrote {fig_path}")
    except ImportError:
        print("matplotlib unavailable, skipping figures, tables only.")

    # ---------- 6) Report ----------
    L = []
    L.append("# Choroid plexus immune cell morphology - group comparison report\n")
    L.append(f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"- Recipe: server.DEFAULTS (final recipe, sato + brightness gate), channel {ch}")
    L.append(f"- Unit of analysis: **animal** (animal value = median over all of its ROI metrics)")
    L.append(f"- Permutation test: animal-level label permutation, {N_PERM} iterations, two-sided; multiple-comparison correction: BH-FDR")
    L.append("")

    # Data overview
    L.append("## 1. Data overview\n")
    L.append("| Sample | Sex | Genotype | ROI count | sigma (noise floor) |")
    L.append("|---|---|---|---|---|")
    for sample in ALL_SAMPLES:
        sex, geno = parse_group(sample)
        if len(animal_df) and sample in set(animal_df["sample"]):
            r = animal_df[animal_df["sample"] == sample].iloc[0]
            L.append(f"| {sample} | {sex} | {geno} | {int(r['n_roi'])} | {fmt(r.get('sigma'))} |")
        else:
            L.append(f"| {sample} | {sex} | {geno} | **0 (not drawn)** | — |")
    L.append("")
    if missing_samples:
        L.append(f"**Samples with no ROIs drawn ({len(missing_samples)}/{len(ALL_SAMPLES)}), please draw them:** "
                 + ", ".join(missing_samples) + "\n")
    if failed_rois:
        L.append("**ROIs that failed to process:** " +
                 "; ".join(f"{s}#{r} ({e})" for s, r, e in failed_rois) + "\n")

    def comp_table(rows, g1, g2, title):
        L.append(f"## {title}\n")
        n1 = rows[0].get("n1", 0) if rows else 0
        n2 = rows[0].get("n2", 0) if rows else 0
        L.append(f"Group sizes: {g1} n={n1}, {g2} n={n2} (animals). "
                 f"Sample size is small, so the results below are **descriptive reference only**, not inferential conclusions.\n")
        L.append(f"| Metric | {g1} median | {g2} median | Hedges g | Permutation p | BH-FDR q | Note |")
        L.append("|---|---|---|---|---|---|---|")
        for r in rows:
            note = "⚠ possible noise confound" if r["metric"] in flagged else ""
            L.append(f"| {r['metric']} | {fmt(r['med1'])} | {fmt(r['med2'])} | "
                     f"{fmt(r['g'])} | {fmt(r['p'])} | {fmt(r['q'])} | {note} |")
        L.append("")

    if enough:
        comp_table(comp_geno, "WT", "Het", "2. Genotype comparison: WT vs Het (all animals)")
        comp_table(comp_sex, "F", "M", "3. Sex comparison: F vs M (all animals)")
    else:
        L.append("## 2. Group comparison\n")
        L.append(f"Only {len(animal_df)} animals currently have ROI data (<4); "
                 "**insufficient data, rerun once the ROIs are drawn**. The ROI-level and animal-level CSVs above were still generated.\n")

    # Noise-confound check
    L.append("## 4. Noise-confound check (metric vs sample global_sigma, Spearman)\n")
    if noise_rows:
        L.append(f"|rho| > {RHO_FLAG} counts as a possible noise confound; that metric's between-group significance is not trustworthy.\n")
        L.append("| Metric | Spearman rho | p | Verdict |")
        L.append("|---|---|---|---|")
        for r in noise_rows:
            mark = "⚠ possible noise confound" if r["flagged"] else "pass"
            L.append(f"| {r['metric']} | {fmt(r['rho'])} | {fmt(r['p'])} | {mark} |")
        L.append("")
    else:
        L.append("Fewer than 3 animals have sigma, so Spearman cannot be computed - insufficient data, rerun once the ROIs are drawn.\n")

    if fig_path:
        L.append(f"Grouped scatter figure: `{os.path.basename(fig_path)}`\n")

    # Conclusions
    L.append("## 5. Conclusions (honest version)\n")
    if not enough:
        L.append(f"- Only {len(animal_df)} animals currently have usable ROIs"
                 f" ({', '.join(animal_df['sample']) if len(animal_df) else 'none'}), "
                 "so no group comparison is possible at all.")
        if missing_samples:
            L.append(f"- Please draw ROIs for: {', '.join(missing_samples)}. Rerun this script once they are drawn.")
    else:
        L.append("- Designed sample size WT n=5 vs Het n=7 (n=12 overall), a small sample; "
                 "only the **direction** is reported below, with no inferential conclusions.")
        for rows, g1, g2, lab in [(comp_geno, "WT", "Het", "Genotype"),
                                   (comp_sex, "F", "M", "Sex")]:
            strong = [r for r in rows
                      if np.isfinite(r.get("g", np.nan)) and abs(r["g"]) >= 0.8
                      and r["metric"] not in flagged]
            if strong:
                desc = "; ".join(
                    f"{r['metric']} ({g1} {'higher' if r['g'] > 0 else 'lower'} than {g2}, g={fmt(r['g'])}, q={fmt(r['q'])})"
                    for r in strong)
                L.append(f"- Metrics with a large effect size (|g|>=0.8 and not noise-flagged) in the {lab} comparison: {desc}.")
            else:
                L.append(f"- No clean metric with a large effect size (|g|>=0.8) in the {lab} comparison.")
        if flagged:
            L.append(f"- For the noise-confound-flagged metrics ({', '.join(sorted(flagged))}), "
                     "the between-group difference may come from differences in the samples' noise floors; significance is not trustworthy.")
        sig_q = [r for c in (comp_geno, comp_sex) for r in c
                 if np.isfinite(r.get("q", np.nan)) and r["q"] < 0.05]
        if not sig_q:
            L.append("- No metric is significant at the BH-FDR q<0.05 level; "
                     "given the small n, this does not mean there is no effect, only that the current data cannot resolve one.")
        if missing_samples:
            L.append(f"- {len(missing_samples)} samples still have no ROIs drawn"
                     f" ({', '.join(missing_samples)}); the conclusions may change once they are added.")
    L.append("")

    report_path = os.path.join(OUT_DIR, "analysis_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"Wrote {report_path}")
    print("Done.")


if __name__ == "__main__":
    main()
