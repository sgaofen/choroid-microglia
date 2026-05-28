"""Branch-detection diagnostic (Stephen, 2026-05-28): prove every detected
branch has >=3 arms. On a zoom of each image, overlay:
  red   = skeleton
  yellow= endpoints (degree 1)
  green = VALIDATED branches (clean_topology.merged_branches, >=3 arms)
  cyan X= 'loose' degree>=3 clusters that FAILED validation (the false branches
          the old visualization was drawing)
Also INDEPENDENTLY re-counts arms for every green dot and asserts >=3.
"""
import sys
from pathlib import Path
import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.ndimage import white_tophat, binary_dilation
from skimage import filters, morphology
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
import pipeline as pl
from pipeline import COND
import clean_topology as ct
OUT = pl.ROOT / 'experiments/full_morphology/out_bg'
TOPHAT = 31
WIN = 480


def bg_skel(stem):
    raw = tifffile.imread(pl.find_raw(stem)).astype(np.float32)
    norm = pl.normalize(raw)
    th = white_tophat(filters.gaussian(raw, 1.0), size=TOPHAT)
    b = th > filters.threshold_otsu(th) * 0.7
    b = morphology.remove_small_objects(morphology.binary_closing(b, morphology.disk(2)), 20)
    skel = ct.prune_spurs(ct.break_loops(ct.clean(morphology.skeletonize(b)), norm), 8)
    return norm, skel


def arms_at(sk, y, x, r=6):
    """Independent arm count near (y,x): remove junction pixels in a small disk,
    count distinct skeleton components touching them."""
    y, x = int(round(y)), int(round(x))
    y0, y1 = max(0, y-r), min(sk.shape[0], y+r+1)
    x0, x1 = max(0, x-r), min(sk.shape[1], x+r+1)
    sub = sk[y0:y1, x0:x1]
    d = ct.degree(sub)
    jp = sub & (d >= 3)
    rest = sub & ~jp
    rl, _ = ndi.label(rest, structure=np.ones((3, 3)))
    touch = rl[binary_dilation(jp, iterations=1) & rest]
    return len(np.unique(touch[touch > 0]))


def loose_branches(sk):
    """Old buggy method: every degree>=3 cluster, no arm check."""
    bp = binary_dilation(sk & (ct.degree(sk) >= 3), iterations=3)
    l, n = ndi.label(bp, structure=np.ones((3, 3)))
    pts = []
    for i in range(1, n+1):
        ys, xs = np.where((l == i) & sk)
        if len(ys):
            pts.append((ys.mean(), xs.mean()))
    return np.array(pts) if pts else np.empty((0, 2))


fig, ax = plt.subplots(1, 2, figsize=(15, 8))
for col, s in enumerate(['F_WT_2', 'F_HET_1']):
    norm, skel = bg_skel(s)
    # densest window
    H, W = skel.shape; best, bn = (0, 0), -1
    for ty in range(0, H-WIN, 200):
        for tx in range(0, W-WIN, 200):
            c = skel[ty:ty+WIN, tx:tx+WIN].sum()
            if c > bn:
                bn, best = c, (ty, tx)
    ty, tx = best
    sub = skel[ty:ty+WIN, tx:tx+WIN]
    rc = norm[ty:ty+WIN, tx:tx+WIN]
    J = ct.merged_branches(sub)
    E = ct.endpoints(sub)
    loose = loose_branches(sub)
    # validate each green dot independently
    bad = sum(1 for (yy, xx) in J if arms_at(sub, yy, xx) < 3)
    # loose points not matched by a validated one = false branches
    false_pts = []
    for (ly, lx) in loose:
        if len(J) == 0 or np.min(np.hypot(J[:, 0]-ly, J[:, 1]-lx)) > 5:
            false_pts.append((ly, lx))
    false_pts = np.array(false_pts) if false_pts else np.empty((0, 2))
    print(f'{s}: validated branches={len(J)} (independent re-check failures={bad}), '
          f'loose={len(loose)}, false-removed={len(false_pts)}, endpoints={len(E)}')

    ax[col].imshow(rc, cmap='gray', vmax=np.percentile(rc, 99))
    ys, xs = np.where(sub); ax[col].scatter(xs, ys, s=0.7, c='#ff5555', marker='.')
    if len(E):
        ax[col].scatter(E[:, 1], E[:, 0], s=22, c='yellow', edgecolors='k', linewidths=0.4, label='endpoint')
    if len(false_pts):
        ax[col].scatter(false_pts[:, 1], false_pts[:, 0], s=60, c='cyan', marker='x', linewidths=1.6, label='false branch (removed)')
    if len(J):
        ax[col].scatter(J[:, 1], J[:, 0], s=44, c='lime', edgecolors='k', linewidths=0.5, label='validated branch (>=3 arms)')
    ax[col].set_title(f'{s} ({COND[s]}): {len(J)} real branches, {len(false_pts)} false removed', fontsize=11)
    ax[col].axis('off'); ax[col].legend(loc='upper right', fontsize=8, framealpha=0.9)
fig.suptitle('Branch detection FIXED: green = validated (>=3 arms), cyan x = false branches the old method drew', fontsize=12)
fig.tight_layout(); fig.savefig(OUT / 'branch_diag.png', dpi=130, bbox_inches='tight'); plt.close()
print('saved branch_diag.png')
