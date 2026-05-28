"""Local zoom comparison on the NEW (background-normalized) pipeline:
raw | skeleton(+branch/endpoint) | connected-component coloring, WT vs HET.
Baseline = rolling-ball bg subtraction + otsu*0.7 (Huixin's normalization)."""
import sys
from pathlib import Path
import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.ndimage import white_tophat
from skimage import filters, morphology
from skimage.color import label2rgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
import pipeline as pl
from pipeline import COND
import clean_topology as ct
OUT = pl.ROOT / 'experiments/full_morphology/out_bg'
TOPHAT = 31
WIN = 400


def bg_prep(stem):
    raw = tifffile.imread(pl.find_raw(stem)).astype(np.float32)
    norm = pl.normalize(raw)
    sm = filters.gaussian(raw, 1.0)
    th = white_tophat(sm, size=TOPHAT)
    b = th > filters.threshold_otsu(th) * 0.7
    b = morphology.remove_small_objects(morphology.binary_closing(b, morphology.disk(2)), 20)
    skel = ct.prune_spurs(ct.break_loops(ct.clean(morphology.skeletonize(b)), norm), 8)
    lab, _ = ndi.label(skel, structure=np.ones((3, 3)))
    return norm, skel, lab


def pick_window(binary):
    """Densest window (most foreground) — compare WT vs HET at matched HIGH
    density so the contrast is structural, not 'WT happened to be emptier'."""
    H, W = binary.shape; best, bn = (0, 0), -1
    for ty in range(0, H - WIN, 150):
        for tx in range(0, W - WIN, 150):
            c = binary[ty:ty+WIN, tx:tx+WIN].sum()
            if c > bn:
                bn, best = c, (ty, tx)
    return best


fig, ax = plt.subplots(2, 3, figsize=(16, 11))
for row, s in enumerate(['F_WT_2', 'F_HET_1']):
    norm, skel, lab = bg_prep(s)
    reglabel = 'densest region'
    raw = tifffile.imread(pl.find_raw(s)).astype(np.float32)
    binary = pl.segment(pl.normalize(raw), 'bg', raw=raw)
    ty, tx = pick_window(binary)
    rc = norm[ty:ty+WIN, tx:tx+WIN]
    sc = skel[ty:ty+WIN, tx:tx+WIN]
    lc = lab[ty:ty+WIN, tx:tx+WIN]
    deg = ct.degree(sc)
    # col0 raw
    ax[row, 0].imshow(rc, cmap='gray', vmax=np.percentile(rc, 99)); ax[row, 0].axis('off')
    ax[row, 0].set_title(f'{s} ({COND[s]}) — {reglabel}, raw', fontsize=11)
    print(f'{s}: {reglabel}, {len(np.unique(lc[lc>0]))} pieces in window')
    # col1 skeleton + branch(green) + endpoint(yellow)
    ax[row, 1].imshow(rc, cmap='gray', vmax=np.percentile(rc, 99))
    ys, xs = np.where(sc); ax[row, 1].scatter(xs, ys, s=0.5, c='#ff3b3b', marker='.')
    ey, ex = np.where(sc & (deg == 1))
    ax[row, 1].scatter(ex, ey, s=10, c='yellow', edgecolors='k', linewidths=0.3)
    J = ct.merged_branches(sc)                       # validated >=3-arm branches only
    if len(J):
        ax[row, 1].scatter(J[:, 1], J[:, 0], s=20, c='lime', edgecolors='k', linewidths=0.3)
    ax[row, 1].axis('off'); ax[row, 1].set_title('skeleton(red)+branch(green)+endpoint(yellow)', fontsize=11)
    # col2 connectivity coloring
    rgb = label2rgb(ndi.grey_dilation(lc, size=3), bg_label=0, bg_color=(0.05, 0.05, 0.05))
    ax[row, 2].imshow(rgb); ax[row, 2].axis('off')
    npieces = len(np.unique(lc[lc > 0]))
    ax[row, 2].set_title(f'each connected piece = one color ({npieces} pieces)', fontsize=11)
fig.suptitle('Local comparison on BACKGROUND-NORMALIZED pipeline (~135um window)', fontsize=13)
fig.tight_layout(); fig.savefig(OUT / 'bg_zoom_compare.png', dpi=115, bbox_inches='tight'); plt.close()
print('saved bg_zoom_compare.png')
