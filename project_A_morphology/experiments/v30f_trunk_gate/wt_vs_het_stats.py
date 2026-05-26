"""
WT vs HET statistical comparison scaffold using v30f trunk metrics.

Project A's terminal goal: report cell-level statistics that distinguish WT
from HET microglia morphology. With just 3 images (1 WT, 2 HET), we cannot
do image-level inference, but we CAN compare per-cell distributions across
the conditions and tee up the test that will run once more images arrive.

Metrics compared:
  - n_trunks (primary processes leaving the soma)
  - n_local_branches (skeleton degree >= 3 within 30 px)
  - skel_len_local (skeleton px within 30 px)
  - score (somaness)

Tests:
  - per-image distribution summaries
  - WT vs HET pooled Mann-Whitney U (per-cell)
  - Kolmogorov-Smirnov 2-sample (shape difference)
  - Effect size: Cohen's d on pooled cells

NOTE: per-cell pooling treats each cell as independent — this is fine for an
EXPLORATORY readout but not for inference. Real per-genotype N needs N images
per genotype, not N cells. We report the analysis as such.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path('/Users/stephenyu/Documents/choroid-microglia/project_A_morphology')
V30F = ROOT / 'experiments/v30f_trunk_gate'

WT = ['F_WT_2']
HET = ['F_HET_1', 'F_HET_3']
METRICS = ['n_trunks', 'n_local_branches', 'skel_len_local', 'score']


def load(stem):
    return json.loads((V30F / f'{stem}_trunk_metrics_v30f.json').read_text())


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    sx = float(np.var(x, ddof=1)) if nx > 1 else 0.0
    sy = float(np.var(y, ddof=1)) if ny > 1 else 0.0
    s_pool = np.sqrt(((nx - 1) * sx + (ny - 1) * sy) / max(1, nx + ny - 2))
    if s_pool == 0:
        return float('nan')
    return (np.mean(x) - np.mean(y)) / s_pool


def main():
    per_img = {stem: load(stem) for stem in WT + HET}

    print('=== Per-image distribution summary ===')
    print(f'{"image":<10} {"n":>5}  ' + '  '.join(f'{m:>18}' for m in METRICS))
    print(' ' * 17 + '  '.join(f'{"mean[med]":>18}' for _ in METRICS))
    for stem in WT + HET:
        cells = per_img[stem]
        n = len(cells)
        parts = []
        for m in METRICS:
            vals = np.array([c[m] for c in cells])
            parts.append(f'{vals.mean():>9.2f}[{np.median(vals):>5.1f}]')
        print(f'{stem:<10} {n:>5}  ' + '  '.join(f'{p:>18}' for p in parts))

    print('\n=== Pooled WT vs HET (per-cell distributions — exploratory) ===')
    print(f'{"metric":<22} {"WT_mean":>9} {"HET_mean":>9}  '
          f'{"delta%":>7}  {"MWU_U":>10} {"MWU_p":>10}  '
          f'{"KS_p":>10}  {"Cohen_d":>9}')
    for m in METRICS:
        wt_vals = np.concatenate([[c[m] for c in per_img[s]] for s in WT])
        het_vals = np.concatenate([[c[m] for c in per_img[s]] for s in HET])
        u, mwu_p = stats.mannwhitneyu(wt_vals, het_vals, alternative='two-sided')
        ks, ks_p = stats.ks_2samp(wt_vals, het_vals)
        d = cohens_d(wt_vals, het_vals)
        delta = 100.0 * (het_vals.mean() - wt_vals.mean()) / wt_vals.mean()
        print(f'{m:<22} {wt_vals.mean():>9.2f} {het_vals.mean():>9.2f}  '
              f'{delta:>+6.1f}%  {u:>10.0f} {mwu_p:>10.2e}  '
              f'{ks_p:>10.2e}  {d:>9.3f}')

    print('\n=== Cell counts by image ===')
    for stem in WT + HET:
        print(f'  {stem}: {len(per_img[stem])} cells')

    print('\n=== Caveat ===')
    print('  With N=1 WT and N=2 HET images, image-level inference is')
    print('  underpowered. Numbers above are per-cell pooled, which inflates')
    print('  statistical power and ignores within-image correlation. Treat')
    print('  them as exploratory directional signals only. Real WT vs HET')
    print('  test needs N>=3 images per condition with image-level summary')
    print('  statistics (mean per image) compared by Mann-Whitney or t-test.')


if __name__ == '__main__':
    main()
