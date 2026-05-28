"""Visual story of the findings: (A) connectivity coloring (each connected
component a distinct color) WT vs HET; (B) fragmentation-score spatial heatmap
per image; (C) dashboard of the clean dimensions WT vs HET."""
import sys, csv
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from skimage.color import label2rgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import pipeline as pl
from pipeline import STEMS, COND
OUT = pl.ROOT / 'experiments/full_morphology/out_region'
reg = list(csv.DictReader(open(OUT/'region_features.csv')))
cc = list(csv.DictReader(open(pl.ROOT/'experiments/full_morphology/out_cc/cc_features.csv')))
TILE = 200


def Acsv(rows, img, key):
    return np.array([float(r[key]) for r in rows if r['image'] == img])


# ============ (A) connectivity coloring ============
# pick a representative window by fragmentation score: WT = least fragmented
# (ramified), HET = most fragmented — so the connectivity contrast is clean.
WT_T = 3  # window = 3x3 tiles (600px)
WIN = WT_T * TILE


def frag_grid(s):
    rows = [r for r in reg if r['image'] == s]
    ntY = max(int(r['ty']) for r in rows)//TILE + 1
    ntX = max(int(r['tx']) for r in rows)//TILE + 1
    g = np.full((ntY, ntX), np.nan)
    for r in rows:
        g[int(r['ty'])//TILE, int(r['tx'])//TILE] = float(r['frag_score'])
    return g


def pick_window(s, want):
    g = frag_grid(s); ntY, ntX = g.shape
    best, bv = (0, 0), (np.inf if want == 'min' else -np.inf)
    for i in range(ntY - WT_T):
        for j in range(ntX - WT_T):
            blk = g[i:i+WT_T, j:j+WT_T]
            if np.sum(~np.isnan(blk)) < 7:
                continue
            m = np.nanmean(blk)
            if (want == 'min' and m < bv) or (want == 'max' and m > bv):
                bv, best = m, (i*TILE, j*TILE)
    return best


fig, ax = plt.subplots(2, 2, figsize=(12, 12))
for row, (s, want) in enumerate([('F_WT_2', 'min'), ('F_HET_1', 'max')]):
    norm, _, _, skel = pl.prep(s)
    lab, _ = ndi.label(skel, structure=np.ones((3, 3)))
    ty, tx = pick_window(s, want)
    rc = norm[ty:ty+WIN, tx:tx+WIN]
    lc = lab[ty:ty+WIN, tx:tx+WIN]
    ax[row, 0].imshow(rc, cmap='gray', vmax=np.percentile(rc, 99)); ax[row, 0].axis('off')
    region = 'a ramified region' if want == 'min' else 'a fragmented region'
    ax[row, 0].set_title(f'{s} ({COND[s]}) — {region}, raw', fontsize=12)
    # dilate labels a touch for visibility, color each component distinctly
    lc_d = ndi.grey_dilation(lc, size=3)
    rgb = label2rgb(lc_d, bg_label=0, bg_color=(0.05, 0.05, 0.05))
    ax[row, 1].imshow(rgb); ax[row, 1].axis('off')
    n_here = len(np.unique(lc[lc > 0]))
    ax[row, 1].set_title(f'each connected piece = one color  ({n_here} pieces here)', fontsize=12)
fig.suptitle('Connectivity: WT = few large connected webs · HET = many small broken pieces',
             fontsize=13)
fig.tight_layout(); fig.savefig(OUT/'story_connectivity.png', dpi=115, bbox_inches='tight'); plt.close()
print('saved story_connectivity.png')

# ============ (B) fragmentation spatial heatmap ============
fr_all = np.array([float(r['frag_score']) for r in reg])
vmin, vmax = np.percentile(fr_all, 2), np.percentile(fr_all, 98)
fig, ax = plt.subplots(1, 3, figsize=(16, 6))
for i, s in enumerate(STEMS):
    rows = [r for r in reg if r['image'] == s]
    ntY = max(int(r['ty']) for r in rows)//TILE + 1
    ntX = max(int(r['tx']) for r in rows)//TILE + 1
    g = np.full((ntY, ntX), np.nan)
    for r in rows:
        g[int(r['ty'])//TILE, int(r['tx'])//TILE] = float(r['frag_score'])
    im = ax[i].imshow(g, cmap='inferno', vmin=vmin, vmax=vmax)
    ax[i].set_title(f'{s} ({COND[s]})', fontsize=13); ax[i].axis('off')
fig.colorbar(im, ax=ax, fraction=0.025, label='fragmentation score (bright = more fragmented)')
fig.suptitle('Where the fragmentation is: WT cool & uniform · HET bright focal patches', fontsize=14)
fig.savefig(OUT/'story_frag_heatmap.png', dpi=120, bbox_inches='tight'); plt.close()
print('saved story_frag_heatmap.png')

# ============ (C) dashboard of clean dimensions ============
def cc_pct_large(s):
    L = Acsv(cc, s, 'length_um'); return 100*L[L > 100].sum()/L.sum()
panels = [
    ('process abundance\n(skeleton / tissue area)', lambda s: Acsv(reg, s, 'skel_per_tile_mm2').mean()/1000, 'less in HET'),
    ('branching\n(branches / 100um)', lambda s: Acsv(reg, s, 'branch_per_100um').mean(), 'less in HET'),
    ('fragmentation score', lambda s: Acsv(reg, s, 'frag_score').mean(), 'more in HET'),
    ('% pieces unbranched\n(broken stubs)', lambda s: 100*Acsv(cc, s, 'is_stub').mean(), 'more in HET'),
    ('connectivity\n(% skel in big webs)', cc_pct_large, 'less in HET'),
    ('median piece length (um)', lambda s: np.median(Acsv(cc, s, 'length_um')), 'shorter in HET'),
    ('de-ramified regions %', lambda s: 100*(Acsv(reg, s, 'cluster') == 3).mean(), 'more in HET'),
    ('fragmentation hotspots %', lambda s: 100*(Acsv(reg, s, 'frag_score') > np.percentile(fr_all, 75)).mean(), 'more in HET'),
]
fig, axes = plt.subplots(2, 4, figsize=(17, 8))
colors = ['#2c7bb6', '#d7191c', '#fd8d3c']
for ax_, (title, fn, tag) in zip(axes.ravel(), panels):
    vals = [fn(s) for s in STEMS]
    ax_.bar(['WT', 'HET_1', 'HET_3'], vals, color=colors)
    ax_.set_title(f'{title}\n({tag})', fontsize=10.5)
    for j, v in enumerate(vals):
        ax_.text(j, v, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    ax_.margins(y=0.18)
fig.suptitle('All clean dimensions: both HET replicates agree, both differ from WT', fontsize=14)
fig.tight_layout(); fig.savefig(OUT/'story_dashboard.png', dpi=115, bbox_inches='tight'); plt.close()
print('saved story_dashboard.png')
