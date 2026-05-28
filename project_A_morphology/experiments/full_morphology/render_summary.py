"""Render (A) the curated multi-angle stats table as a figure, and (B) a visual-
verification crop (densest matched WT vs HET region: raw | annotated skeleton |
connectivity coloring) with the region's MEASURED stats printed on it so the
numbers can be eyeballed against the picture."""
import sys
from pathlib import Path
import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.color import label2rgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
import pipeline as pl
from pipeline import COND
import clean_topology as ct
OUT = pl.ROOT / 'experiments/full_morphology/out_region'
CACHE = pl.ROOT / 'experiments/full_morphology/cache_bg'


# (table moved to render_table_clean.py — this file now only makes verify_crop)

# ---------- (B) visual verification crop ----------
WIN = 380
fig, ax = plt.subplots(2, 3, figsize=(16, 11))
for row, s in enumerate(['F_WT_2', 'F_HET_1']):
    d = np.load(CACHE / f'{s}.npz', allow_pickle=True)
    skel = d['skel']; binary = d['binary']; cc = d['cc']; J = d['J']; E = d['E']
    raw = tifffile.imread(pl.find_raw(s)).astype(np.float32); norm = pl.normalize(raw)
    H, W = skel.shape; best, bn = (0, 0), -1
    for ty in range(0, H-WIN, 120):
        for tx in range(0, W-WIN, 120):
            c = binary[ty:ty+WIN, tx:tx+WIN].sum()
            if c > bn:
                bn, best = c, (ty, tx)
    ty, tx = best
    rc = norm[ty:ty+WIN, tx:tx+WIN]; sc = skel[ty:ty+WIN, tx:tx+WIN]; lc = cc[ty:ty+WIN, tx:tx+WIN]
    # window-local stats
    jin = (J[:, 0] >= ty) & (J[:, 0] < ty+WIN) & (J[:, 1] >= tx) & (J[:, 1] < tx+WIN)
    ein = (E[:, 0] >= ty) & (E[:, 0] < ty+WIN) & (E[:, 1] >= tx) & (E[:, 1] < tx+WIN)
    nJ, nE = int(jin.sum()), int(ein.sum())
    npieces = len(np.unique(lc[lc > 0]))
    skum = sc.sum() * pl.PIXEL_UM
    # col0 raw
    ax[row, 0].imshow(rc, cmap='gray', vmax=np.percentile(rc, 99)); ax[row, 0].axis('off')
    ax[row, 0].set_title(f'{s} ({COND[s]}) raw  (~{WIN*pl.PIXEL_UM:.0f}um)', fontsize=11)
    # col1 annotated skeleton
    ax[row, 1].imshow(rc, cmap='gray', vmax=np.percentile(rc, 99))
    ys, xs = np.where(sc); ax[row, 1].scatter(xs, ys, s=0.5, c='#ff4444', marker='.')
    ax[row, 1].scatter(E[ein, 1]-tx, E[ein, 0]-ty, s=16, c='yellow', edgecolors='k', linewidths=0.3, label=f'endpoint ({nE})')
    ax[row, 1].scatter(J[jin, 1]-tx, J[jin, 0]-ty, s=34, c='lime', edgecolors='k', linewidths=0.5, label=f'branch ({nJ})')
    ax[row, 1].legend(loc='upper right', fontsize=8, framealpha=0.9); ax[row, 1].axis('off')
    ax[row, 1].set_title('skeleton(red) + branch(green) + endpoint(yellow)', fontsize=11)
    # col2 connectivity
    rgb = label2rgb(ndi.grey_dilation(lc, size=3), bg_label=0, bg_color=(0.05, 0.05, 0.05))
    ax[row, 2].imshow(rgb); ax[row, 2].axis('off')
    ax[row, 2].set_title(f'connectivity: {npieces} pieces', fontsize=11)
    txt = (f'pieces={npieces}\nbranches={nJ}\nendpoints={nE}\nendpoint/branch={nE/max(nJ,1):.1f}\n'
           f'skel={skum:.0f}um')
    ax[row, 2].text(0.015, 0.985, txt, transform=ax[row, 2].transAxes, fontsize=9, va='top', ha='left',
                    color='white', bbox=dict(boxstyle='round', fc='black', alpha=0.6))
fig.suptitle('Visual verification — densest WT vs HET region (background-normalized, fixed branches). '
             'Count branches/endpoints by eye and compare to the printed stats.', fontsize=12)
fig.tight_layout(); fig.savefig(OUT / 'verify_crop.png', dpi=135, bbox_inches='tight'); plt.close()
print('saved verify_crop.png')
