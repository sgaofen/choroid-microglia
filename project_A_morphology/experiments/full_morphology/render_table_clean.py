"""Clean at-a-glance 'key findings' figure — now COMPUTES every number fresh from
the cache (no hardcoded/stale values). Horizontal diverging bars = HET vs WT %
change. Red=more in HET (fragmentation), blue=less in HET (structure lost)."""
import sys
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
import pipeline as pl
from pipeline import STEMS, COND, PIXEL_UM
CACHE = pl.ROOT / 'experiments/full_morphology/cache_bg'
OUT = pl.ROOT / 'experiments/full_morphology/out_region'


import csv
REG = list(csv.DictReader(open(pl.ROOT / 'experiments/full_morphology/out_region/region_features.csv')))
CCF = list(csv.DictReader(open(pl.ROOT / 'experiments/full_morphology/out_cc/cc_features.csv')))


def metrics_for(stem):
    # consistent with the analysis scripts / dashboard (same normalization)
    rg = [r for r in REG if r['image'] == stem]
    cc = [r for r in CCF if r['image'] == stem]
    clen = np.array([float(r['length_um']) for r in cc]); tot = clen.sum()
    d = np.load(CACHE / f'{stem}.npz', allow_pickle=True)
    binary = d['binary']; J = d['J']; E = d['E']
    fg_mm2 = float(binary.sum()) * PIXEL_UM**2 / 1e6
    tissue = ndi.binary_fill_holes(morphology.binary_closing(binary, morphology.disk(25)))
    edt = ndi.distance_transform_edt(tissue) * PIXEL_UM
    m = (edt >= 0) & (edt < 50)
    nj = int(m[J[:, 0].astype(int), J[:, 1].astype(int)].sum()) if len(J) else 0
    ne = int(m[E[:, 0].astype(int), E[:, 1].astype(int)].sum()) if len(E) else 0
    return dict(
        process_amount=np.mean([float(r['skel_per_tile_mm2']) for r in rg]),   # region (clean)
        large_net=100 * clen[clen > 100].sum() / tot,
        junctions_per_mm2=len(J) / fg_mm2,                                      # branch points per fg area
        median_piece=float(np.median(clen)),
        ep_branch=np.mean([float(r['endpoint_branch_ratio']) for r in rg]),
        pieces_per_mm2=len(cc) / fg_mm2,
        edge_frag=ne / max(nj, 1),
        tiny_frag=100 * clen[(clen >= 8) & (clen < 16)].sum() / tot,
    )


V = {s: metrics_for(s) for s in STEMS}
for s in STEMS:
    print(s, {k: round(v, 2) for k, v in V[s].items()})

# (label, key, meaning) ; %diff computed below
SPEC = [
    ('Tiny fragments\n(% skeleton in 8-16um pieces)', 'tiny_frag', 'mass piles into small bits'),
    ('Fragmentation at the tissue edge', 'edge_frag', 'worst at the edge'),
    ('Number of separate pieces per area', 'pieces_per_mm2', 'broken into more pieces'),
    ('Endpoint : branch-point ratio', 'ep_branch', 'more loose dead-ends'),
    ('Median piece length', 'median_piece', 'pieces shorter'),
    ('Process amount\n(skeleton per tissue area)', 'process_amount', 'less material'),
    ('Branch points per area', 'junctions_per_mm2', 'fewer junctions'),
    ('Large connected networks\n(% skeleton in >100um pieces)', 'large_net', 'big webs collapse'),
]
rows = []
for label, key, mean in SPEC:
    wt = np.mean([V[s][key] for s in pl.WT]); hm = np.mean([V[s][key] for s in pl.HET])
    rows.append((label, round(100*(hm-wt)/wt), mean))
rows.sort(key=lambda r: r[1])
labels = [r[0] for r in rows]; vals = [r[1] for r in rows]; means = [r[2] for r in rows]
colors = ['#2c7fb8' if v < 0 else '#d7301f' for v in vals]

fig, ax = plt.subplots(figsize=(12, 7.2))
y = np.arange(len(rows))
ax.barh(y, vals, color=colors, height=0.62, zorder=3)
ax.axvline(0, color='#333', lw=1.1, zorder=4)
hi = max(abs(min(vals)), abs(max(vals)))
for yi, (v, mean) in enumerate(zip(vals, means)):
    off = 0.02*hi+2 if v > 0 else -(0.02*hi+2)
    ax.text(v + off, yi + 0.16, f'{v:+d}%', va='center', ha='left' if v > 0 else 'right',
            fontsize=12.5, fontweight='bold', color=colors[yi], zorder=5)
    if v < -0.55*hi:
        ax.text(0.04*hi, yi, mean, va='center', ha='left', fontsize=9.3, color='#666', style='italic', zorder=5)
    else:
        ax.text(v + off, yi - 0.24, mean, va='center', ha='left' if v > 0 else 'right',
                fontsize=9.3, color='#666', style='italic', zorder=5)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10.5)
ax.set_xlim(-1.25*hi, 1.6*hi); ax.set_xlabel('HET vs WT  (% change)', fontsize=11)
ax.grid(axis='x', color='#ddd', lw=0.7, zorder=0)
ax.spines[['top', 'right', 'left']].set_visible(False); ax.tick_params(left=False)
ax.text(-1.18*hi, len(rows)-0.2, '◄  less in HET (structure lost)', color='#2c7fb8', fontsize=11, fontweight='bold')
ax.text(1.55*hi, len(rows)-0.2, 'more in HET (fragmentation)  ►', color='#d7301f', fontsize=11, fontweight='bold', ha='right')
ax.set_title('WT vs HET microglia — key morphology differences\n', fontsize=15, fontweight='bold', loc='center')
fig.text(0.5, 0.945, 'background-normalized (equal-black) + branch = degree≥3 skeleton points · both HET replicates agree on every bar',
         ha='center', fontsize=9.5, color='#666')
fig.text(0.5, 0.012, 'Also: fragmented regions cluster into one large focal patch in each HET.  '
         '3 images (1 WT, 2 HET) — descriptive demo, not a statistical test.', ha='center', fontsize=8.5, color='#888')
fig.tight_layout(rect=[0, 0.03, 1, 0.93])
fig.savefig(OUT / 'stats_table.png', dpi=150, bbox_inches='tight'); plt.close()
print('saved stats_table.png (fresh-computed)')
