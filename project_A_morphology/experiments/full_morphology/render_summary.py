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

# ---------- (A) curated stats table ----------
# (section, metric, WT, HET_1, HET_3, pctdiff, verdict, plain meaning)
ROWS = [
 ('AMOUNT', 'skeleton length / tissue area', 102000, 77100, 75800, -25, 'CLEAN', 'less process material'),
 ('CONNECTIVITY', '% skeleton in >100um networks', 26.8, 6.7, 5.8, -77, 'CLEAN', 'big connected webs collapse'),
 ('CONNECTIVITY', '% skeleton in 256um+ giant nets', 4.64, 1.0, 0.0, -89, 'CLEAN', 'giant networks vanish'),
 ('CONNECTIVITY', 'Gini of piece sizes', 0.546, 0.468, 0.442, -17, 'CLEAN', "WT's size inequality = giants HET lacks"),
 ('SIZE', 'pieces / mm2 tissue', 29013, 39258, 44582, +44, 'CLEAN', 'broken into more pieces'),
 ('SIZE', '% skeleton in 8-16um fragments', 8.1, 20.3, 24.1, +175, 'CLEAN', 'mass piles into small fragments'),
 ('SIZE', 'median piece length (um)', 17.0, 13.5, 12.8, -23, 'CLEAN', 'pieces shorter'),
 ('SIZE', 'p90 piece length (um)', 55.5, 35.0, 34.2, -38, 'CLEAN', 'largest pieces shrink most'),
 ('BRANCHING', 'junctions / mm2 tissue', 1367, 928, 879, -34, 'CLEAN', 'fewer branch points'),
 ('BRANCHING', '% internal (jct-jct) segments', 41.3, 34.1, 31.8, -20, 'CLEAN', 'fewer internal connectors'),
 ('BRANCHING', '% 4-way junctions', 31.6, 29.3, 26.7, -11, 'CLEAN', 'loses high-order crossings -> simpler'),
 ('FRAGMENTATION', 'endpoint : junction ratio', 12.3, 16.5, 16.7, +34, 'CLEAN', 'more free dead-ends per junction'),
 ('FRAGMENTATION', 'terminal : internal segment ratio', 1.36, 1.82, 2.02, +41, 'CLEAN', 'arbor becomes dead-end-heavy'),
 ('SPATIAL', 'endpoint/junction @ tissue EDGE', 8.57, 12.1, 12.7, +45, 'CLEAN', 'worst at the edge (WT edge is intact)'),
 ('SPATIAL', 'largest fragmentation hotspot (tiles)', 7, 48, 40, +529, 'CLEAN', 'damage clusters into one big patch'),
 ('SPATIAL', 'hotspot patch dominance', 0.32, 0.69, 0.73, +122, 'CLEAN', 'focal, not uniform'),
 ('NOT CLEAN', 'median segment length (um)', 2.57, 2.55, 2.50, -1, 'null', 'processes NOT shorter (topology-only)'),
 ('NOT CLEAN', 'mean apparent thickness (um)', 1.29, 1.32, 1.20, -2, 'null', 'processes NOT thinner (HET replicates disagree)'),
]
SEC_COL = {'AMOUNT': '#1f78b4', 'CONNECTIVITY': '#33a02c', 'SIZE': '#ff7f00',
           'BRANCHING': '#6a3d9a', 'FRAGMENTATION': '#e31a1c', 'SPATIAL': '#b15928', 'NOT CLEAN': '#7f7f7f'}

fig, ax = plt.subplots(figsize=(15, 11)); ax.axis('off')
xcols = [0.005, 0.40, 0.515, 0.60, 0.685, 0.775, 0.86]   # metric, WT, H1, H3, diff, verdict, meaning
heads = ['metric', 'WT', 'HET_1', 'HET_3', 'HETvWT', 'verdict', 'plain meaning']
n = len(ROWS); top = 0.95; dy = top / (n + 3)
for xc, h in zip(xcols, heads):
    ax.text(xc, top, h, fontsize=11, fontweight='bold', va='top')
ax.plot([0, 1], [top-0.012, top-0.012], color='k', lw=1.2)
y = top - dy
last_sec = None
for sec, name, wt, h1, h3, d, verd, mean in ROWS:
    if sec != last_sec:
        ax.add_patch(plt.Rectangle((0, y-dy*0.18), 1, dy*0.92, color=SEC_COL[sec], alpha=0.13, zorder=0))
        ax.text(0.005, y, f'{sec}', fontsize=9.5, fontweight='bold', color=SEC_COL[sec], va='center')
        last_sec = sec
    yy = y - dy*0.42 if sec else y
    fmt = lambda v: (f'{v:,.0f}' if abs(v) >= 100 else (f'{v:.2f}' if abs(v) < 10 else f'{v:.1f}'))
    ax.text(xcols[0]+0.10, yy, name, fontsize=9.3, va='center')
    for xc, v in zip(xcols[1:4], (wt, h1, h3)):
        ax.text(xc, yy, fmt(v), fontsize=9.3, va='center')
    dcol = '#7f7f7f' if verd == 'null' else ('#c0392b' if d > 0 else '#16609a')
    ax.text(xcols[4], yy, f'{d:+d}%', fontsize=9.3, va='center', color=dcol, fontweight='bold')
    vcol = '#2ca02c' if verd == 'CLEAN' else '#999999'
    ax.text(xcols[5], yy, '✓ clean' if verd == 'CLEAN' else 'null', fontsize=9, va='center', color=vcol, fontweight='bold')
    ax.text(xcols[6], yy, mean, fontsize=8.4, va='center', color='#333333')
    y -= dy
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title('WT vs HET microglia morphology — multi-angle statistics (background-normalized, fixed branch detection)\n'
             '3 images (1 WT, 2 HET replicates); every CLEAN row: both HET replicates on the same side of WT. Demo, descriptive.',
             fontsize=12, loc='left')
fig.savefig(OUT / 'stats_table.png', dpi=130, bbox_inches='tight'); plt.close()
print('saved stats_table.png')

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
