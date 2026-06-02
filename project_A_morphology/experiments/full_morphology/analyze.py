"""
Full-image microglia morphology analysis (no per-cell counting — that's the
unreliable part. Focus on MORPHOLOGY, which is robust): per-segment length &
thickness, complexity, and REGIONAL spatial heterogeneity (Huixin's point:
different regions differ). v29 skeleton + clean_topology baseline.

Outputs (experiments/full_morphology/out/):
  <stem>_segments.csv        per skeleton segment: length_um, thickness_um, type
  <stem>_regional.csv        per tile: densities + mean thickness
  aggregate.json             per-image aggregate + WT-vs-HET deltas
  <stem>_regional_heatmap.png, seglen_dist.png, thickness_dist.png
"""
import sys, json, csv
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage import filters, morphology
from skan import Skeleton, summarize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data/raw'
V29 = ROOT / 'experiments/v29_short_spur_audit'
OUT = ROOT / 'experiments/full_morphology/out'
sys.path.insert(0, str(ROOT / 'experiments'))
import clean_topology as ct

PIXEL_UM = 0.207
STEMS = ['F_WT_2', 'F_HET_1', 'F_HET_3']
COND = {'F_WT_2': 'WT', 'F_HET_1': 'HET', 'F_HET_3': 'HET'}
TILE = 400  # px tile for regional analysis (~83 um)


def find_raw(stem):
    for p in RAW.glob('*.tif'):
        if stem in p.name:
            return p
    raise FileNotFoundError(stem)


def normalize(img, p_lo=1.0, p_hi=99.5):
    lo, hi = np.percentile(img, [p_lo, p_hi])
    return np.clip((img - lo) / (hi - lo + 1e-9), 0, 1)


def fg_and_dist(raw):
    norm = normalize(raw)
    sm = filters.gaussian(norm, 1.0)
    binary = sm > filters.threshold_otsu(sm) * 0.7
    binary = morphology.binary_closing(binary, morphology.disk(2))
    binary = morphology.remove_small_objects(binary, 20)
    dist = ndi.distance_transform_edt(binary)
    return norm, binary, dist


def cv(x):
    x = np.asarray(x, float)
    return float(x.std() / x.mean()) if len(x) and x.mean() else 0.0


def analyze(stem):
    raw = tifffile.imread(find_raw(stem)).astype(np.float32)
    norm, binary, dist = fg_and_dist(raw)
    fg_mm2 = float(binary.sum()) * PIXEL_UM ** 2 / 1e6

    skel = np.load(V29 / f'{stem}_skel_pruned.npy').astype(bool)
    skel = ct.clean(skel)
    skel = ct.break_loops(skel, norm)
    skel = ct.prune_spurs(skel, 8)

    # per-segment via skan: length = branch-distance (px -> um).
    # thickness = sample the distance transform (=radius) along each branch's
    # OWN pixels and take the mean -> diameter in um. (Computed directly, not
    # via skan's mean-pixel-value which returned a constant.)
    sko = Skeleton(skel)  # pixel units
    df = summarize(sko, separator='-')
    seg_len = df['branch-distance'].to_numpy() * PIXEL_UM
    seg_thick = np.array([
        dist[tuple(np.round(sko.path_coordinates(j)).astype(int).T)].mean()
        for j in range(sko.n_paths)
    ]) * 2 * PIXEL_UM
    seg_type = df['branch-type'].to_numpy()

    branches = ct.merged_branches(skel)
    endpoints = ct.endpoints(skel)
    skel_len_um = float(skel.sum()) * PIXEL_UM

    # regional tiles
    H, W = skel.shape
    regional = []
    for ty in range(0, H, TILE):
        for tx in range(0, W, TILE):
            tb = binary[ty:ty+TILE, tx:tx+TILE]
            if tb.sum() < 200:   # skip near-empty tiles
                continue
            tsk = skel[ty:ty+TILE, tx:tx+TILE]
            tarea_mm2 = float(tb.sum()) * PIXEL_UM ** 2 / 1e6
            tdeg = ct.degree(tsk)
            n_ep = int((tsk & (tdeg == 1)).sum())
            n_bp = int((tsk & (tdeg >= 3)).sum())
            tthick = float(dist[ty:ty+TILE, tx:tx+TILE][tsk].mean()) * 2 * PIXEL_UM if tsk.any() else 0.0
            regional.append(dict(
                ty=ty, tx=tx,
                skel_len_um_per_mm2=round(float(tsk.sum())*PIXEL_UM/tarea_mm2, 1),
                endpoint_per_mm2=round(n_ep/tarea_mm2, 0),
                mean_thick_um=round(tthick, 3),
            ))

    reg_skeldens = np.array([r['skel_len_um_per_mm2'] for r in regional])
    reg_thick = np.array([r['mean_thick_um'] for r in regional])

    agg = dict(
        stem=stem, condition=COND[stem],
        fg_area_mm2=round(fg_mm2, 4),
        n_branches=int(len(branches)), n_endpoints=int(len(endpoints)),
        branch_per_mm2=round(len(branches)/fg_mm2, 0),
        endpoint_per_mm2=round(len(endpoints)/fg_mm2, 0),
        skel_len_um_per_mm2=round(skel_len_um/fg_mm2, 0),
        # morphology (per-segment)
        mean_seg_len_um=round(float(np.mean(seg_len)), 2),
        median_seg_len_um=round(float(np.median(seg_len)), 2),
        p90_seg_len_um=round(float(np.percentile(seg_len, 90)), 2),
        cv_seg_len=round(cv(seg_len), 3),
        mean_thick_um=round(float(np.nanmean(seg_thick)), 3),
        cv_thick=round(cv(seg_thick[~np.isnan(seg_thick)]), 3),
        # complexity
        branch_per_100um_skel=round(100*len(branches)/skel_len_um, 2),
        # regional heterogeneity (Huixin: disease tissue more variable)
        n_tiles=len(regional),
        regional_cv_skeldens=round(cv(reg_skeldens), 3),
        regional_cv_thick=round(cv(reg_thick), 3),
    )

    # save per-segment + regional CSV
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f'{stem}_segments.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['length_um', 'thickness_um', 'branch_type'])
        for L, T, ty in zip(seg_len, seg_thick, seg_type):
            w.writerow([round(float(L), 3), round(float(T), 3) if not np.isnan(T) else '', int(ty)])
    with open(OUT / f'{stem}_regional.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(regional[0].keys())); w.writeheader(); w.writerows(regional)

    return agg, seg_len, seg_thick, regional, skel, norm


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    aggs = {}; seglens = {}; thicks = {}; regionals = {}
    for stem in STEMS:
        a, sl, th, reg, skel, norm = analyze(stem)
        aggs[stem] = a; seglens[stem] = sl; thicks[stem] = th; regionals[stem] = reg
        print(f"{stem}({a['condition']}): seg_len mean={a['mean_seg_len_um']}um "
              f"thick={a['mean_thick_um']}um branch/mm2={a['branch_per_mm2']} "
              f"regional_CV(skeldens)={a['regional_cv_skeldens']}")
        # regional heatmap
        H, W = skel.shape
        grid = np.full((H//TILE+1, W//TILE+1), np.nan)
        for r in reg:
            grid[r['ty']//TILE, r['tx']//TILE] = r['skel_len_um_per_mm2']
        plt.figure(figsize=(6, 6)); plt.imshow(grid, cmap='viridis')
        plt.colorbar(label='skel len um per mm2'); plt.title(f'{stem} regional skeleton density')
        plt.savefig(OUT / f'{stem}_regional_heatmap.png', dpi=110, bbox_inches='tight'); plt.close()

    (OUT / 'aggregate.json').write_text(json.dumps(aggs, indent=1))

    # WT vs HET table
    print('\n=== WT vs HET (aggregate) ===')
    cols = ['mean_seg_len_um', 'median_seg_len_um', 'p90_seg_len_um', 'cv_seg_len',
            'mean_thick_um', 'cv_thick', 'branch_per_mm2', 'endpoint_per_mm2',
            'skel_len_um_per_mm2', 'branch_per_100um_skel', 'regional_cv_skeldens',
            'regional_cv_thick']
    wt = aggs['F_WT_2']
    print(f'{"metric":24}{"WT_2":>10}{"HET_1":>10}{"HET_3":>10}{"HETvsWT":>10}')
    deltas = {}
    for c in cols:
        h1, h3 = aggs['F_HET_1'][c], aggs['F_HET_3'][c]
        hm = (h1 + h3) / 2
        d = (100*(hm-wt[c])/wt[c]) if wt[c] else 0
        deltas[c] = round(d, 1)
        print(f'{c:24}{wt[c]:>10}{h1:>10}{h3:>10}{d:>+9.1f}%')
    (OUT / 'wt_vs_het_deltas.json').write_text(json.dumps(deltas, indent=1))

    # distribution plots WT vs HET
    plt.figure(figsize=(9, 5))
    for stem, col in [('F_WT_2', 'C0'), ('F_HET_1', 'C3'), ('F_HET_3', 'C1')]:
        plt.hist(seglens[stem], bins=np.linspace(0, 15, 40), density=True,
                 histtype='step', lw=2, label=f'{stem}({COND[stem]})', color=col)
    plt.xlabel('segment length (um)'); plt.ylabel('density'); plt.legend()
    plt.title('process segment length distribution'); plt.savefig(OUT / 'seglen_dist.png', dpi=120, bbox_inches='tight'); plt.close()

    plt.figure(figsize=(9, 5))
    for stem, col in [('F_WT_2', 'C0'), ('F_HET_1', 'C3'), ('F_HET_3', 'C1')]:
        t = thicks[stem][~np.isnan(thicks[stem])]
        plt.hist(t, bins=np.linspace(0, 4, 40), density=True, histtype='step', lw=2,
                 label=f'{stem}({COND[stem]})', color=col)
    plt.xlabel('process thickness/diameter (um)'); plt.ylabel('density'); plt.legend()
    plt.title('process thickness distribution'); plt.savefig(OUT / 'thickness_dist.png', dpi=120, bbox_inches='tight'); plt.close()
    print('\nsaved to', OUT)


if __name__ == '__main__':
    main()
