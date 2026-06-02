"""Junction-merge tolerance sweep (Stephen, 2026-05-28): merge 'branch-after-
branch' junctions that sit in one small branching area into a single point.
Merge rule = path-based: two degree>=3 junctions are merged if connected by a
skeleton segment SHORTER than merge_um (i.e. on the SAME process, a short hop
apart) — NOT merely close in straight-line distance (that would wrongly fuse
different cells). Show several thresholds on a crop so the right one can be
picked by eye, and report whole-image junction counts at each threshold."""
import sys
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import binary_dilation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
import pipeline as pl
from pipeline import COND, PIXEL_UM
import clean_topology as ct
CACHE = pl.ROOT / 'experiments/full_morphology/cache_bg'
OUT = pl.ROOT / 'experiments/full_morphology/out_bg'


def junctions(sk, merge_um=0.0):
    """degree>=3 components; optionally merge components joined by a skeleton
    segment shorter than merge_um (path-based 'same branching area')."""
    jp = sk & (ct.degree(sk) >= 3)
    jl, nj = ndi.label(jp, structure=np.ones((3, 3)))
    if nj == 0:
        return np.empty((0, 2))
    cents = list(ndi.center_of_mass(jp, jl, range(1, nj + 1)))
    if merge_um <= 0:
        return np.array(cents)
    seg = sk & ~jp
    sl, ns = ndi.label(seg, structure=np.ones((3, 3)))
    if ns == 0:
        return np.array(cents)
    seg_px = np.bincount(sl.ravel())[1:]
    short = np.where(seg_px * PIXEL_UM < merge_um)[0] + 1     # only short segments can merge
    objs = ndi.find_objects(sl)
    parent = list(range(nj + 1))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    H, W = sk.shape
    for si in short:
        sb = objs[si - 1]
        if sb is None:
            continue
        y0 = max(0, sb[0].start-2); y1 = min(H, sb[0].stop+2)
        x0 = max(0, sb[1].start-2); x1 = min(W, sb[1].stop+2)
        mask = (sl[y0:y1, x0:x1] == si)
        adj = np.unique((jl[y0:y1, x0:x1])[binary_dilation(mask, iterations=1) & (jl[y0:y1, x0:x1] > 0)])
        adj = adj[adj > 0]
        for k in adj[1:]:
            parent[find(int(k))] = find(int(adj[0]))
    groups = {}
    for ji in range(1, nj + 1):
        groups.setdefault(find(ji), []).append(ji)
    out = [(np.mean([cents[m-1][0] for m in g]), np.mean([cents[m-1][1] for m in g])) for g in groups.values()]
    return np.array(out)


# whole-image counts at each threshold
THRS = [0.0, 1.5, 3.0, 5.0]
print('whole-image junction counts by merge_um:')
print(f'{"merge_um":>9}' + ''.join(f'{s.replace("F_",""):>10}' for s in pl.STEMS) + f'{"HET/WT":>9}')
for t in THRS:
    cnts = {}
    for s in pl.STEMS:
        d = np.load(CACHE / f'{s}.npz', allow_pickle=True)
        cnts[s] = len(junctions(d['skel'], t))
    hetwt = np.mean([cnts[s] for s in pl.HET]) / np.mean([cnts[s] for s in pl.WT])
    print(f'{t:>9}' + ''.join(f'{cnts[s]:>10}' for s in pl.STEMS) + f'{hetwt:>9.2f}')

# crop sweep (dense WT region with branch-after-branch doublets)
d = np.load(CACHE / 'F_WT_2.npz', allow_pickle=True); skel = d['skel']
WIN = 200
H, W = skel.shape; best, bn = (0, 0), -1
for ty in range(0, H-WIN, 80):
    for tx in range(0, W-WIN, 80):
        sub = skel[ty:ty+WIN, tx:tx+WIN]
        nj = ndi.label(sub & (ct.degree(sub) >= 3), structure=np.ones((3, 3)))[1]
        if nj > bn:
            bn, best = nj, (ty, tx)
ty, tx = best
sub = skel[ty:ty+WIN, tx:tx+WIN]
fig, ax = plt.subplots(1, 4, figsize=(20, 5.4))
for a, t in zip(ax, THRS):
    J = junctions(sub, t)
    a.imshow(sub, cmap='gray_r', vmin=0, vmax=1)
    if len(J):
        a.scatter(J[:, 1], J[:, 0], s=60, c='lime', edgecolors='k', linewidths=0.6, zorder=3)
    a.set_title(f'merge ≤ {t} µm  →  {len(J)} junctions', fontsize=12); a.axis('off')
fig.suptitle('Junction-merge tolerance sweep (path-based) — pick the threshold that groups one branching area into one point', fontsize=13)
fig.tight_layout(); fig.savefig(OUT / 'merge_sweep.png', dpi=140, bbox_inches='tight'); plt.close()
print('saved merge_sweep.png')
