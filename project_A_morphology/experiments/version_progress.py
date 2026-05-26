"""
One-page visualization summarizing every v30 iteration's cell counts and
mean endpoints per image. Designed for Stephen to scan first thing when
he wakes up.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/Users/stephenyu/Documents/choroid-microglia/project_A_morphology')
EXP = ROOT / 'experiments'

# Hard-coded from log table; no need to re-run pipelines
versions = ['v29', 'v30a', 'v30b', 'v30c', 'v30d', 'v30e', 'v30f']
counts = {
    'WT_2':  [2395, 2013, 1556, 1396, 1355, 1928, 1594],
    'HET_1': [2086, 1634, 1380, 1259, 1225, 1666, 1356],
    'HET_3': [1955, 1539, 1324, 1117, 1104, 1472, 1263],
}
mean_endpts = {
    'WT_2':  [2.22, 2.45, 3.33, 3.57, 3.67, 2.58, 3.06],
    'HET_1': [2.39, 2.67, 3.37, 3.42, 3.52, 2.59, 3.07],
    'HET_3': [2.62, 2.82, 3.62, 3.82, 3.87, 2.90, 3.29],
}
labels = {
    'v29':  'h-max seeds + soft prune',
    'v30a': '+ tube>blob hard filter',
    'v30b': '+ somaness graph merge',
    'v30c': '+ snap + density gate',
    'v30d': '+ compact-comp gate',
    'v30e': 'merge-bug fixes',
    'v30f': '+ weak0.26 + trunk-gate',
}

x = np.arange(len(versions))
colors = {'WT_2': '#1f77b4', 'HET_1': '#d62728', 'HET_3': '#ff7f0e'}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

for img, vals in counts.items():
    ax1.plot(x, vals, '-o', label=img, color=colors[img], lw=2, ms=8)
ax1.set_xticks(x); ax1.set_xticklabels(versions, fontsize=11)
ax1.set_ylabel('Accepted cells per image', fontsize=12)
ax1.set_title('Cell counts across v30 iterations', fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
for i, v in enumerate(versions):
    ax1.annotate(labels[v], (i, max(c[i] for c in counts.values())),
                 xytext=(0, 25 + (i%2)*12), textcoords='offset points',
                 ha='center', fontsize=8, rotation=15,
                 color='#555')
# highlight v30f as current
ax1.axvspan(len(versions)-1.4, len(versions)-0.6, alpha=0.15, color='lime',
            label='v30f (working)')

for img, vals in mean_endpts.items():
    ax2.plot(x, vals, '-o', label=img, color=colors[img], lw=2, ms=8)
ax2.set_xticks(x); ax2.set_xticklabels(versions, fontsize=11)
ax2.set_ylabel('Mean endpoints per cell', fontsize=12)
ax2.set_title('Endpoint richness across v30 iterations '
              '(higher = more ramified accepted cells)', fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.axvspan(len(versions)-1.4, len(versions)-0.6, alpha=0.15, color='lime')

fig.suptitle('Project A morphology pipeline — v30 progression  (2026-05-24)',
             fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = EXP / 'version_progress.png'
fig.savefig(out, dpi=140, bbox_inches='tight')
plt.close(fig)
print(f'wrote {out.name}')
