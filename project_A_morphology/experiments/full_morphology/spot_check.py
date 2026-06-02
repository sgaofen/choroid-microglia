"""
Visual spot-check (Stephen's request): crop a representative HET de-ramified
(C3) focus and a WT healthy (C1) region straight from the RAW fluorescence
image, show raw + skeleton overlay side by side, so the morphotype labels can be
eyeballed against the actual microglia. Pairs overlay with unmarked original.
"""
import sys
from pathlib import Path
import csv
import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage import filters, morphology
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import pipeline as pl
from pipeline import COND
ROOT = pl.ROOT
RAW = pl.RAW
OUTR = ROOT / 'experiments/full_morphology/out_region'
import clean_topology as ct

TILE = 200
WIN = 600  # crop window (3x3 tiles)


def prep(stem):
    raw = tifffile.imread(next(RAW.glob(f'*{stem}*.tif'))).astype(np.float32)
    lo, hi = np.percentile(raw, [1.0, 99.5])
    norm = np.clip((raw - lo) / (hi - lo + 1e-9), 0, 1)
    skel = np.load(V29 / f'{stem}_skel_pruned.npy').astype(bool)
    skel = ct.clean(skel); skel = ct.break_loops(skel, norm); skel = ct.prune_spurs(skel, 8)
    return norm, skel


def focus(stem, cluster):
    """Tile center whose 3x3 neighborhood is richest in `cluster`."""
    rows = [r for r in csv.DictReader(open(OUTR / 'region_features.csv')) if r['image'] == stem]
    pos = {(int(r['ty']), int(r['tx'])) for r in rows if int(r['cluster']) == cluster}
    best, bestn = None, -1
    for (ty, tx) in pos:
        n = sum(((ty + dy, tx + dx) in pos) for dy in (-TILE, 0, TILE) for dx in (-TILE, 0, TILE))
        if n > bestn:
            bestn, best = n, (ty + TILE // 2, tx + TILE // 2)
    return best


def crop(arr, cy, cx):
    y0, x0 = cy - WIN // 2, cx - WIN // 2
    return arr[y0:y0 + WIN, x0:x0 + WIN]


# canonical clustering: C0 = most ramified, C3 = de-ramified (last)
panels = [(pl.WT[0], 0, 'WT ramified (C0)'),
          (pl.HET[0], 3, 'HET de-ramified (C3: fragmented)')]

fig, axes = plt.subplots(2, 2, figsize=(11, 11))
for row, (stem, clu, title) in enumerate(panels):
    norm, skel = prep(stem)
    cy, cx = focus(stem, clu)
    rc, sc = crop(norm, cy, cx), crop(skel, cy, cx)
    deg = ct.degree(sc)
    axes[row, 0].imshow(rc, cmap='gray', vmin=0, vmax=np.percentile(rc, 99))
    axes[row, 0].set_title(f'{title}\nraw signal', fontsize=11)
    axes[row, 1].imshow(rc, cmap='gray', vmin=0, vmax=np.percentile(rc, 99))
    ys, xs = np.where(sc)
    axes[row, 1].scatter(xs, ys, s=0.3, c='#ff3b3b', marker='.')
    ey, ex = np.where(sc & (deg == 1))
    axes[row, 1].scatter(ex, ey, s=14, c='yellow', edgecolors='k', linewidths=0.3, label='endpoint')
    bp = ndi.binary_dilation(sc & (deg >= 3), iterations=3)
    bl, bn = ndi.label(bp)
    for i in range(1, bn + 1):
        yy, xx = np.where(bl == i)
        axes[row, 1].scatter(xx.mean(), yy.mean(), s=22, c='lime', edgecolors='k', linewidths=0.3)
    axes[row, 1].set_title(f'{title}\nskeleton (red) + branch (green) + endpoint (yellow)', fontsize=11)
    for c in (0, 1):
        axes[row, c].axis('off')
fig.suptitle('Spot-check: raw fluorescence vs skeleton, ~124um window', fontsize=13)
fig.tight_layout()
out = OUTR / 'spot_check_WT_vs_HET.png'
fig.savefig(out, dpi=120, bbox_inches='tight'); plt.close()
print('saved', out)
