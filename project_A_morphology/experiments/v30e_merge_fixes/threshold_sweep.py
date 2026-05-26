"""
Threshold sweep over BLOB_HARD_MIN, FG_DENSITY_MIN, SCORE_WEAK, and the
strong-gate parameters using already-scored seeds. No re-scoring needed — the
v30e seeds JSON contains every per-seed metric.

We only re-derive the primary classification (not the merge step). This is a
"pre-merge" cell-count sweep that lets us see how hard gates affect the pool
of survivors before merging logic kicks in.

Output: a CSV per image and a multi-image summary table.
"""
import json
from pathlib import Path
from itertools import product

import numpy as np

ROOT = Path('/Users/stephenyu/Documents/choroid-microglia/project_A_morphology')
V30E = ROOT / 'experiments/v30e_merge_fixes'
STEMS = ['F_WT_2', 'F_HET_1', 'F_HET_3']

# v30e local-component gate held fixed (those were tuned in v30d audit)
LC_AREA_MIN = 6
LC_AREA_MAX = 200
LC_ECC_MAX = 0.92
LC_SOL_MIN = 0.50

SCORE_STRONG = 0.40


def classify(rec, blob_min, fg_min, score_weak, strong_rprom_min,
             strong_n_dirs_min):
    blob = rec['blob']
    fg = rec['fg_dens']
    score = rec['score']
    n_dirs = rec['n_dirs']
    rprom = rec['rprom']
    lc_area = rec['lc_area']
    lc_ecc = rec['lc_ecc']
    lc_sol = rec['lc_sol']

    hard = (blob < blob_min) or (fg < fg_min)
    lc_ok = (LC_AREA_MIN <= lc_area <= LC_AREA_MAX and
             lc_ecc < LC_ECC_MAX and lc_sol > LC_SOL_MIN)
    if not lc_ok:
        hard = True

    strong_ok = (n_dirs >= strong_n_dirs_min) and (rprom >= strong_rprom_min)

    if hard:
        return 'rej'
    if score >= SCORE_STRONG and strong_ok:
        return 'strong'
    if score >= score_weak:
        return 'weak'
    return 'rej'


def load_seeds(stem):
    return json.loads((V30E / f'{stem}_seeds_v30e.json').read_text())


def sweep_one_image(seeds, blob_grid, fg_grid, weak_grid,
                    rprom_grid, ndirs_grid):
    rows = []
    for bm, fm, sw, sr, sn in product(blob_grid, fg_grid, weak_grid,
                                       rprom_grid, ndirs_grid):
        n_strong = n_weak = n_rej = 0
        for r in seeds:
            t = classify(r, bm, fm, sw, sr, sn)
            if t == 'strong':
                n_strong += 1
            elif t == 'weak':
                n_weak += 1
            else:
                n_rej += 1
        rows.append(dict(
            blob_min=bm, fg_min=fm, score_weak=sw,
            strong_rprom_min=sr, strong_n_dirs_min=sn,
            n_strong=n_strong, n_weak=n_weak, n_accept=n_strong + n_weak,
        ))
    return rows


def main():
    blob_grid = [0.10, 0.15, 0.20, 0.25, 0.30]
    fg_grid = [0.35, 0.45, 0.55, 0.65]
    weak_grid = [0.18, 0.22, 0.26, 0.30]
    rprom_grid = [1.0]
    ndirs_grid = [3]

    print(f'sweep: {len(blob_grid)*len(fg_grid)*len(weak_grid)*len(rprom_grid)*len(ndirs_grid)} combos per image')

    image_rows = {}
    for stem in STEMS:
        seeds = load_seeds(stem)
        rows = sweep_one_image(seeds, blob_grid, fg_grid, weak_grid,
                               rprom_grid, ndirs_grid)
        image_rows[stem] = rows
        with open(V30E / f'sweep_{stem}.csv', 'w') as f:
            f.write('blob_min,fg_min,score_weak,strong_rprom_min,strong_n_dirs_min,'
                    'n_strong,n_weak,n_accept\n')
            for r in rows:
                f.write(f'{r["blob_min"]},{r["fg_min"]},{r["score_weak"]},'
                        f'{r["strong_rprom_min"]},{r["strong_n_dirs_min"]},'
                        f'{r["n_strong"]},{r["n_weak"]},{r["n_accept"]}\n')

    # Build a unified summary keyed on threshold tuple
    keys = [(r['blob_min'], r['fg_min'], r['score_weak'],
             r['strong_rprom_min'], r['strong_n_dirs_min'])
            for r in image_rows[STEMS[0]]]

    print('\n=== Threshold sweep — accepted counts (strong+weak), pre-merge ===')
    print(f'{"blob":>6} {"fg":>6} {"sw":>6}  '
          f'{"WT_2":>6} {"HET_1":>6} {"HET_3":>6}  '
          f'{"strong":>10}')
    # current v30e baseline highlight
    baseline = (0.20, 0.45, 0.22, 1.0, 3)
    for key in keys:
        bm, fm, sw, sr, sn = key
        cnts = []
        strong_cnts = []
        for stem in STEMS:
            row = next(r for r in image_rows[stem]
                       if (r['blob_min'], r['fg_min'], r['score_weak'],
                           r['strong_rprom_min'], r['strong_n_dirs_min']) == key)
            cnts.append(row['n_accept'])
            strong_cnts.append(row['n_strong'])
        marker = ' <-- v30e' if key == baseline else ''
        print(f'{bm:6.2f} {fm:6.2f} {sw:6.2f}  '
              f'{cnts[0]:6d} {cnts[1]:6d} {cnts[2]:6d}  '
              f'{"/".join(str(s) for s in strong_cnts):>10}{marker}')


if __name__ == '__main__':
    main()
