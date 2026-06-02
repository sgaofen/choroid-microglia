"""Compare branch detection: Stephen's SIMPLE definition (a skeleton pixel with
degree>=3 is a junction; directly-adjacent junction pixels = one junction; just
count 8-connected components of the degree>=3 mask) vs the current OVER-ENGINEERED
merged_branches (dilate-merge + arm-count validation). Overlay both on a crop +
print counts so we can SEE which matches the eye."""
import sys
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
import pipeline as pl
from pipeline import COND
import clean_topology as ct
CACHE = pl.ROOT / 'experiments/full_morphology/cache_bg'
OUT = pl.ROOT / 'experiments/full_morphology/out_bg'
WIN = 300


def branches_simple(sk):
    """degree>=3 pixels, 8-connected components -> one point per junction."""
    jp = sk & (ct.degree(sk) >= 3)
    l, n = ndi.label(jp, structure=np.ones((3, 3)))
    if n == 0:
        return np.empty((0, 2))
    return np.array(ndi.center_of_mass(jp, l, range(1, n + 1)))


fig, ax = plt.subplots(1, 2, figsize=(16, 8.5))
for col, s in enumerate([pl.WT[0], pl.HET[0]]):
    d = np.load(CACHE / f'{s}.npz', allow_pickle=True)
    skel = d['skel']; binary = d['binary']
    H, W = skel.shape; best, bn = (0, 0), -1
    for ty in range(0, H-WIN, 120):
        for tx in range(0, W-WIN, 120):
            c = skel[ty:ty+WIN, tx:tx+WIN].sum()
            if c > bn:
                bn, best = c, (ty, tx)
    ty, tx = best
    sc = skel[ty:ty+WIN, tx:tx+WIN]
    simple = branches_simple(sc)
    current = ct.merged_branches(sc)
    E = ct.endpoints(sc)
    ax[col].imshow(sc, cmap='gray_r', vmin=0, vmax=1)
    if len(E):
        ax[col].scatter(E[:, 1], E[:, 0], s=14, c='gold', edgecolors='k', linewidths=0.3, label=f'endpoint ({len(E)})', zorder=3)
    if len(current):
        ax[col].scatter(current[:, 1], current[:, 0], s=140, marker='x', c='cyan', linewidths=2.2,
                        label=f'CURRENT over-pruned ({len(current)})', zorder=4)
    if len(simple):
        ax[col].scatter(simple[:, 1], simple[:, 0], s=42, c='lime', edgecolors='k', linewidths=0.5,
                        label=f'SIMPLE degree>=3 ({len(simple)})', zorder=5)
    ax[col].set_title(f'{s} ({COND[s]})  —  simple={len(simple)} vs current={len(current)} junctions', fontsize=12)
    ax[col].legend(loc='upper right', fontsize=9, framealpha=0.95); ax[col].axis('off')
    print(f'{s}: simple(degree>=3)={len(simple)}  current(merged_branches)={len(current)}  endpoints={len(E)}')
fig.suptitle('Branch detection: SIMPLE degree>=3 (green) vs current over-engineered (cyan x). Skeleton shown black.', fontsize=13)
fig.tight_layout(); fig.savefig(OUT / 'branch_simple_diag.png', dpi=140, bbox_inches='tight'); plt.close()
print('saved branch_simple_diag.png')
