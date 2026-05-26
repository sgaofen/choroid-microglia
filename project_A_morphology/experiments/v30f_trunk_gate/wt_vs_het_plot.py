"""
Visualize WT vs HET per-cell distributions for v30f trunk metrics.
Output a single PNG with 4 panels: n_trunks, n_local_branches,
skel_len_local, score — KDE/histogram for WT (1 image) and HET (2 images
overlaid).
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path('/Users/stephenyu/Documents/choroid-microglia/project_A_morphology')
V30F = ROOT / 'experiments/v30f_trunk_gate'

WT = ['F_WT_2']
HET = ['F_HET_1', 'F_HET_3']
METRICS = [
    ('n_trunks', 'Primary trunks (annulus)'),
    ('n_local_branches', 'Local branch points (r=30 px)'),
    ('skel_len_local', 'Local skeleton length (r=30 px, px)'),
    ('score', 'Somaness score'),
]


def load(stem):
    return json.loads((V30F / f'{stem}_trunk_metrics_v30f.json').read_text())


def main():
    per_img = {stem: load(stem) for stem in WT + HET}

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.ravel()

    for i, (m, title) in enumerate(METRICS):
        ax = axes[i]
        wt_vals = np.concatenate([[c[m] for c in per_img[s]] for s in WT])
        het_vals = np.concatenate([[c[m] for c in per_img[s]] for s in HET])

        # discrete-vs-continuous: histogram bins
        if m in ('n_trunks',):
            bins = np.arange(0, max(wt_vals.max(), het_vals.max()) + 2) - 0.5
        elif m == 'score':
            bins = np.linspace(0.18, 0.7, 35)
        elif m == 'n_local_branches':
            bins = np.linspace(0, 35, 36)
        else:  # skel_len_local
            bins = np.linspace(0, 250, 30)

        ax.hist(wt_vals, bins=bins, density=True, alpha=0.55,
                label=f'WT  n={len(wt_vals)} (mean={wt_vals.mean():.2f})',
                color='#1f77b4', edgecolor='white')
        ax.hist(het_vals, bins=bins, density=True, alpha=0.55,
                label=f'HET n={len(het_vals)} (mean={het_vals.mean():.2f})',
                color='#d62728', edgecolor='white')

        u, mwu_p = stats.mannwhitneyu(wt_vals, het_vals, alternative='two-sided')
        ax.set_title(f'{title}\n'
                     f'MWU p={mwu_p:.2g}  '
                     f'delta={100*(het_vals.mean()-wt_vals.mean())/wt_vals.mean():+.1f}%',
                     fontsize=11)
        ax.set_xlabel(m)
        ax.set_ylabel('density')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle('v30f: WT vs HET per-cell distributions  '
                 '(WT=F_WT_2, HET=F_HET_1+F_HET_3, pooled per cell — exploratory)',
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = V30F / 'wt_vs_het_distributions.png'
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out.name}')


if __name__ == '__main__':
    main()
