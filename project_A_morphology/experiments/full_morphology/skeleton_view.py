"""Show the CURRENT cleaned skeleton: full-image coverage (all 3) + a zoom with
branch(green, merged exit>=3)/endpoint(yellow) markers, paired with raw."""
import sys
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_dilation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
import pipeline as pl
from pipeline import STEMS, COND
import clean_topology as ct

OUT = pl.ROOT / 'experiments/full_morphology/out_region'

data = {}
for s in STEMS:
    norm, _, _, skel = pl.prep(s)
    J, E = pl.global_topology(skel)
    data[s] = (norm, skel, J, E)
    print(f'{s}: skeleton px={int(skel.sum())}, junctions={len(J)}, endpoints={len(E)}')

# ---- full-image: raw | skeleton(red) for each ----
fig, ax = plt.subplots(3, 2, figsize=(11, 16))
for i, s in enumerate(STEMS):
    norm, skel, J, E = data[s]
    ax[i, 0].imshow(norm, cmap='gray', vmax=np.percentile(norm, 99.5))
    ax[i, 0].set_title(f'{s} ({COND[s]}) raw', fontsize=11); ax[i, 0].axis('off')
    rgb = np.stack([norm]*3, -1) * 0.6
    rgb[binary_dilation(skel, iterations=1)] = [1, 0.1, 0.1]
    ax[i, 1].imshow(np.clip(rgb, 0, 1))
    ax[i, 1].set_title(f'skeleton ({int(skel.sum())} px, {len(J)} junctions)', fontsize=11)
    ax[i, 1].axis('off')
fig.suptitle('Current cleaned skeleton — full image', fontsize=14)
fig.tight_layout(); fig.savefig(OUT / 'skeleton_fullimage.png', dpi=95, bbox_inches='tight'); plt.close()

# ---- zoom (700px) for WT and HET: raw | skeleton+branch+endpoint ----
WIN = 700
fig, ax = plt.subplots(2, 2, figsize=(12, 12))
for row, s in enumerate([pl.WT[0], pl.HET[0]]):
    norm, skel, J, E = data[s]
    # window with most skeleton
    H, W = skel.shape; best, bn = (0, 0), -1
    for ty in range(0, H-WIN, 300):
        for tx in range(0, W-WIN, 300):
            c = skel[ty:ty+WIN, tx:tx+WIN].sum()
            if c > bn:
                bn, best = c, (ty, tx)
    ty, tx = best
    rc = norm[ty:ty+WIN, tx:tx+WIN]
    sc = skel[ty:ty+WIN, tx:tx+WIN]
    ax[row, 0].imshow(rc, cmap='gray', vmax=np.percentile(rc, 99)); ax[row, 0].axis('off')
    ax[row, 0].set_title(f'{s} ({COND[s]}) raw', fontsize=11)
    ax[row, 1].imshow(rc, cmap='gray', vmax=np.percentile(rc, 99))
    ys, xs = np.where(sc)
    ax[row, 1].scatter(xs, ys, s=0.5, c='#ff3b3b', marker='.')
    jin = (J[:, 0] >= ty) & (J[:, 0] < ty+WIN) & (J[:, 1] >= tx) & (J[:, 1] < tx+WIN)
    ax[row, 1].scatter(J[jin, 1]-tx, J[jin, 0]-ty, s=26, c='lime', edgecolors='k', linewidths=0.4)
    ein = (E[:, 0] >= ty) & (E[:, 0] < ty+WIN) & (E[:, 1] >= tx) & (E[:, 1] < tx+WIN)
    ax[row, 1].scatter(E[ein, 1]-tx, E[ein, 0]-ty, s=12, c='yellow', edgecolors='k', linewidths=0.3)
    ax[row, 1].axis('off')
    ax[row, 1].set_title('skeleton(red) + branch(green) + endpoint(yellow)', fontsize=11)
fig.suptitle('Current skeleton — zoom (~145um)', fontsize=14)
fig.tight_layout(); fig.savefig(OUT / 'skeleton_zoom.png', dpi=110, bbox_inches='tight'); plt.close()
print('saved skeleton_fullimage.png + skeleton_zoom.png')
