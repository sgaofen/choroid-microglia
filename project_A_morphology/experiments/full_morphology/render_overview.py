"""Render FINAL_METRICS.csv into a grouped data-overview table figure (all 25
metrics, by section, with WT/HET values, %diff, verdict color-coded)."""
import sys, csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
import pipeline as pl
OUT = pl.ROOT / 'experiments/full_morphology/out_region'
rows = list(csv.DictReader(open(OUT / 'FINAL_METRICS.csv')))

SEC_COL = {'ABUNDANCE': '#1f78b4', 'SIZE / LENGTH': '#ff7f00', 'THICKNESS (apparent width)': '#999999',
           'BRANCHING / COMPLEXITY': '#6a3d9a', 'FRAGMENTATION (DAM-like)': '#e31a1c',
           'CONNECTIVITY': '#33a02c', 'HETEROGENEITY (within-image spread; Huixin: disease = more diverse)': '#b15928',
           'MORPHOTYPE COMPOSITION': '#1b7837', 'SPATIAL FOCALITY': '#d95f02'}

n = len(rows)
fig, ax = plt.subplots(figsize=(14, 0.42 * n + 2)); ax.axis('off')
xc = [0.01, 0.50, 0.64, 0.78, 0.90]
for x, h in zip(xc, ['metric', 'WT mean', 'HET mean', 'HET vs WT', 'verdict']):
    ax.text(x, 1.0, h, fontsize=11, fontweight='bold', va='top', transform=ax.transAxes)
ax.plot([0, 1], [0.985, 0.985], color='k', lw=1.1, transform=ax.transAxes)
dy = 0.97 / (n + len(set(r['section'] for r in rows)))
y = 0.97 - dy
last = None
def fmt(v):
    v = float(v)
    return f'{v:,.0f}' if abs(v) >= 1000 else (f'{v:.2f}' if abs(v) < 10 else f'{v:.1f}')
for r in rows:
    sc = r['section']
    if sc != last:
        col = SEC_COL.get(sc, '#444')
        short = sc.split(';')[0].split('(')[0].strip()
        ax.add_patch(plt.Rectangle((0, y-0.1*dy), 1, dy*0.95, color=col, alpha=0.12, transform=ax.transAxes, zorder=0))
        ax.text(0.01, y, short, fontsize=9.5, fontweight='bold', color=col, va='center', transform=ax.transAxes)
        last = sc; y -= dy
    clean = r['verdict'].startswith('✓')
    ax.text(xc[0]+0.02, y, r['metric'], fontsize=9.2, va='center', transform=ax.transAxes)
    for x, k in zip(xc[1:3], ['WT_mean', 'HET_mean']):
        ax.text(x, y, fmt(r[k]), fontsize=9.2, va='center', transform=ax.transAxes)
    d = float(r['HET_vs_WT_pct'])
    dcol = '#c0392b' if (clean and d > 0) else ('#16609a' if clean else '#888')
    ax.text(xc[3], y, f'{d:+.0f}%', fontsize=9.2, va='center', color=dcol, fontweight='bold', transform=ax.transAxes)
    vcol = '#2ca02c' if clean else '#999'
    vtxt = '✓ clean' if clean else ('~ weak' if r['verdict'].startswith('~') else '✗ noisy')
    ax.text(xc[4], y, vtxt, fontsize=8.8, va='center', color=vcol, fontweight='bold', transform=ax.transAxes)
    y -= dy
ax.set_title('WT vs HET microglia — full data overview (25 metrics)\n'
             'background-normalized · branch = degree≥3 + 3µm path-merge · ✓clean = both HET on same side of WT (spread<gap)',
             fontsize=12.5, loc='left')
fig.savefig(OUT / 'data_overview.png', dpi=140, bbox_inches='tight'); plt.close()
print('saved data_overview.png')
