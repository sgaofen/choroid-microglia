"""
For each candidate stricter threshold, check whether the cells it KILLS have
low n_trunks (likely process pieces) vs high n_trunks (real microglia).

Hypothesis: a principled tightening should kill mostly n_trunks <= 1 cells
(which are vessel pieces / process beads) and preserve n_trunks >= 2 cells
(which look like real microglia).

Strategy:
1. Take v30e currently accepted cells (have trunk metrics).
2. For each candidate (blob_min, score_weak) setting, mark which currently
   accepted cells would now be REJECTED by hard gate / score floor.
3. Compare the n_trunks distribution of "would-be-killed" vs "would-be-kept".
4. Report ratio: killed_low_trunk / killed_total. Higher is better.
"""
import json
from pathlib import Path
from collections import Counter

import numpy as np

ROOT = Path('/Users/stephenyu/Documents/choroid-microglia/project_A_morphology')
V30E = ROOT / 'experiments/v30e_merge_fixes'
STEMS = ['F_WT_2', 'F_HET_1', 'F_HET_3']


def load_recs(stem):
    seeds = {r['label']: r for r in json.loads(
        (V30E / f'{stem}_seeds_v30e.json').read_text())}
    trunks = json.loads((V30E / f'{stem}_trunk_metrics_v30e.json').read_text())
    rows = []
    for t in trunks:
        s = seeds.get(t['label'])
        if not s:
            continue
        rows.append(dict(label=t['label'], type_v30e=t['type'],
                         n_trunks=t['n_trunks'],
                         n_branches=t['n_local_branches'],
                         skel_len=t['skel_len_local'],
                         score=s['score'], blob=s['blob'],
                         fg_dens=s['fg_dens'], n_dirs=s['n_dirs'],
                         rprom=s['rprom']))
    return rows


def kills_at(rows, blob_min, score_weak):
    """Return list of cells that this threshold would now kill."""
    killed = []
    for r in rows:
        # would the cell still pass under the new thresholds?
        if r['blob'] < blob_min:
            killed.append(r)
            continue
        # if score is below new weak floor AND it wouldn't be strong, kill
        is_strong = (r['score'] >= 0.40 and r['n_dirs'] >= 3
                     and r['rprom'] >= 1.0)
        if not is_strong and r['score'] < score_weak:
            killed.append(r)
    return killed


def main():
    settings = [
        ('v30e baseline', 0.20, 0.22),
        ('weak0.26',      0.20, 0.26),
        ('weak0.28',      0.20, 0.28),
        ('weak0.30',      0.20, 0.30),
        ('weak0.32',      0.20, 0.32),
        ('weak0.35',      0.20, 0.35),
    ]
    print(f'{"image":>8} {"setting":>15}  '
          f'{"kept":>5} {"killed":>6}  '
          f'{"k.n_tr=0":>8} {"k.n_tr=1":>8} {"k.n_tr>=2":>10}  '
          f'{"%lowtrunk":>10}')
    for stem in STEMS:
        rows = load_recs(stem)
        for name, bm, sw in settings:
            killed = kills_at(rows, bm, sw)
            kept = len(rows) - len(killed)
            cnt = Counter(min(k['n_trunks'], 2) for k in killed)
            n0 = cnt[0]; n1 = cnt[1]; n2 = cnt[2]
            total = max(1, len(killed))
            pct_low = 100.0 * (n0 + n1) / total
            print(f'{stem:>8} {name:>15}  '
                  f'{kept:>5} {len(killed):>6}  '
                  f'{n0:>8} {n1:>8} {n2:>10}  '
                  f'{pct_low:>9.1f}%')

    # also: what are kept-cell n_trunks distributions look like under each
    print('\n=== Kept-cell n_trunks distribution ===')
    print(f'{"image":>8} {"setting":>15}  n_trunks counts (0,1,2,3,4,5+)')
    for stem in STEMS:
        rows = load_recs(stem)
        for name, bm, sw in settings:
            killed = set(id(r) for r in kills_at(rows, bm, sw))
            kept = [r for r in rows if id(r) not in killed]
            dist = [0, 0, 0, 0, 0, 0]
            for k in kept:
                idx = min(k['n_trunks'], 5)
                dist[idx] += 1
            print(f'{stem:>8} {name:>15}  '
                  f'{dist[0]:>3}/{dist[1]:>3}/{dist[2]:>3}/'
                  f'{dist[3]:>3}/{dist[4]:>3}/{dist[5]:>3}  '
                  f'mean={np.mean([k["n_trunks"] for k in kept]):.2f}')


if __name__ == '__main__':
    main()
