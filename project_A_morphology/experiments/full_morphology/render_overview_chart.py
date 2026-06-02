"""Visual data overview: every metric in FINAL_METRICS.csv as a horizontal
%-change bar (HET vs WT), grouped by section. Clean = solid (red up / blue down),
weak/noisy = faded gray. A visual chart, not a table."""
import sys, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
import pipeline as pl
OUT = pl.ROOT / 'experiments/full_morphology/out_region'
rows = list(csv.DictReader(open(OUT / 'FINAL_METRICS.csv')))
# drop z-score metrics (WT<=0): % change is meaningless across zero
rows = [r for r in rows if float(r['WT_mean']) > 0]

# build display list with section separators
SECS = []
for r in rows:
    s = r['section'].split(';')[0].split('(')[0].strip()
    if s not in [x[0] for x in SECS]:
        SECS.append((s, []))
    [x for x in SECS if x[0] == s][0][1].append(r)

labels, vals, cols, faded = [], [], [], []
yticklab = []
CAP = 120
for sname, items in SECS:
    # section header slot (blank bar)
    labels.append(f'§ {sname}'); vals.append(0); cols.append('none'); faded.append(True); yticklab.append(f'$\\bf{{{sname}}}$')
    for r in items:
        d = float(r['HET_vs_WT_pct']); clean = r['verdict'].startswith('✓')
        labels.append(r['metric']); vals.append(d)
        if not clean:
            cols.append('#bbbbbb'); faded.append(True)
        else:
            cols.append('#d7301f' if d > 0 else '#2c7fb8'); faded.append(False)
        yticklab.append(r['metric'])

y = np.arange(len(vals))[::-1]   # top-to-bottom
fig, ax = plt.subplots(figsize=(12, 0.34*len(vals)+1.5))
for yi, (v, c) in zip(y, zip(vals, cols)):
    if c == 'none':
        continue
    ax.barh(yi, max(min(v, CAP), -CAP), color=c, height=0.66, zorder=3)
ax.axvline(0, color='#333', lw=1.1, zorder=4)
for yi, (v, c, lab) in zip(y, zip(vals, cols, labels)):
    if c == 'none':
        ax.text(-CAP*1.02, yi, lab.replace('§ ', ''), fontsize=10, fontweight='bold', va='center', ha='left', color='#222')
        continue
    vc = max(min(v, CAP), -CAP)
    txt = f'{v:+.0f}%' + ('  (capped)' if abs(v) > CAP else '')
    ax.text(vc + (4 if v >= 0 else -4), yi, txt, va='center', ha='left' if v >= 0 else 'right',
            fontsize=8.8, color=c if c != '#bbbbbb' else '#888', fontweight='bold' if c != '#bbbbbb' else 'normal')
ax.set_yticks(y); ax.set_yticklabels([l if not l.startswith('§') else '' for l in labels], fontsize=8.8)
ax.set_xlim(-CAP*1.25, CAP*1.7); ax.set_xlabel('HET vs WT  (% change)', fontsize=11)
ax.grid(axis='x', color='#eee', lw=0.7, zorder=0)
ax.spines[['top', 'right', 'left']].set_visible(False); ax.tick_params(left=False)
ax.text(-CAP*1.2, len(vals)-0.3, '◄ less in HET', color='#2c7fb8', fontsize=11, fontweight='bold')
ax.text(CAP*1.65, len(vals)-0.3, 'more in HET ►', color='#d7301f', fontsize=11, fontweight='bold', ha='right')
# legend
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#d7301f', label='more in HET (clean)'),
                   Patch(color='#2c7fb8', label='less in HET (clean)'),
                   Patch(color='#bbbbbb', label='weak / noisy (not reliable)')],
          loc='upper right', fontsize=9, framealpha=0.95)
ax.set_title('WT vs HET microglia — full visual data overview (25 metrics)\n'
             'background-normalized · branch = degree≥3 + 3µm merge · ±120% cap',
             fontsize=12.5, loc='left')
fig.tight_layout(); fig.savefig(OUT / 'data_overview_chart.png', dpi=145, bbox_inches='tight'); plt.close()
print('saved data_overview_chart.png')
