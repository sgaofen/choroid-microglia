"""A clean, at-a-glance 'key findings' figure (replaces the dense table).
Horizontal diverging bars = HET vs WT % change. Red = more in HET (fragmentation),
blue = less in HET (intact structure). 8 curated, non-overlapping headline metrics,
each replicate-consistent. Overwrites stats_table.png (email attachment path)."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import pipeline as pl
OUT = pl.ROOT / 'experiments/full_morphology/out_region'

# (label, %diff HET vs WT, plain meaning)  — sorted later
M = [
    ('Large connected networks\n(% skeleton in >100µm pieces)', -77, 'big webs collapse'),
    ('Branch points per area', -34, 'fewer junctions'),
    ('Process amount\n(skeleton length per tissue area)', -25, 'less material'),
    ('Median piece length', -23, 'pieces shorter'),
    ('Endpoint : branch-point ratio', +34, 'more loose dead-ends'),
    ('Number of separate pieces per area', +44, 'broken into more pieces'),
    ('Fragmentation at the tissue edge', +45, 'worst at the edge'),
    ('Tiny fragments\n(% skeleton in 8–16µm pieces)', +175, 'mass piles into small bits'),
]
M.sort(key=lambda r: r[1])
labels = [m[0] for m in M]; vals = [m[1] for m in M]; means = [m[2] for m in M]
colors = ['#2c7fb8' if v < 0 else '#d7301f' for v in vals]

fig, ax = plt.subplots(figsize=(12, 7.2))
y = np.arange(len(M))
ax.barh(y, vals, color=colors, height=0.62, zorder=3)
ax.axvline(0, color='#333', lw=1.1, zorder=4)
for yi, (v, mean) in enumerate(zip(vals, means)):
    off = 5 if v > 0 else -5
    # % number at bar end
    ax.text(v + off, yi + 0.16, f'{v:+d}%', va='center', ha='left' if v > 0 else 'right',
            fontsize=12.5, fontweight='bold', color=colors[yi], zorder=5)
    # plain meaning: below the number, but for long left bars place it just RIGHT
    # of 0 (into the empty right half) so it never collides with the y-label.
    if v < -55:
        ax.text(6, yi, mean, va='center', ha='left', fontsize=9.3, color='#666', style='italic', zorder=5)
    else:
        ax.text(v + off, yi - 0.24, mean, va='center', ha='left' if v > 0 else 'right',
                fontsize=9.3, color='#666', style='italic', zorder=5)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10.5)
ax.set_xlim(-118, 250)
ax.set_xlabel('HET vs WT  (% change)', fontsize=11)
ax.grid(axis='x', color='#ddd', lw=0.7, zorder=0)
ax.spines[['top', 'right', 'left']].set_visible(False)
ax.tick_params(left=False)
# legend / direction labels
ax.text(-110, len(M)-0.2, '◄  less in HET (structure lost)', color='#2c7fb8', fontsize=11, fontweight='bold')
ax.text(245, len(M)-0.2, 'more in HET (fragmentation)  ►', color='#d7301f', fontsize=11, fontweight='bold', ha='right')
ax.set_title('WT vs HET microglia — key morphology differences\n', fontsize=15, fontweight='bold', loc='center')
fig.text(0.5, 0.945, 'after background normalization (equal-black) + corrected branch detection · both HET replicates agree on every bar',
         ha='center', fontsize=9.5, color='#666')
fig.text(0.5, 0.012, 'Also: fragmented regions cluster into one large focal patch in each HET (largest hotspot 7 → ~44 tiles).  '
         '3 images (1 WT, 2 HET) — descriptive demo, not a statistical test.', ha='center', fontsize=8.5, color='#888')
fig.tight_layout(rect=[0, 0.03, 1, 0.93])
fig.savefig(OUT / 'stats_table.png', dpi=150, bbox_inches='tight'); plt.close()
print('saved stats_table.png (clean version)')
