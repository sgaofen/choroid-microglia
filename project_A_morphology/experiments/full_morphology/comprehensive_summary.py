"""All-dimensions summary from the corrected data (region + cc + aggregate).
Central tendency AND within-image spread (heterogeneity), with the replicate
verdict for each. CSV-only, no image processing."""
import csv, json
from pathlib import Path
import numpy as np

FM = Path('/Users/stephenyu/choroid-microglia/project_A_morphology/experiments/full_morphology')
reg = list(csv.DictReader(open(FM/'out_region/region_features.csv')))
cc = list(csv.DictReader(open(FM/'out_cc/cc_features.csv')))
STEMS = ['F_WT_2', 'F_HET_1', 'F_HET_3']
COND = {'F_WT_2': 'WT', 'F_HET_1': 'HET', 'F_HET_3': 'HET'}


def A(rows, img, key):
    return np.array([float(r[key]) for r in rows if r['image'] == img])


def verdict(wt, h1, h3):
    d1, d3 = h1-wt, h3-wt
    same = (d1 > 0) == (d3 > 0)
    gap = (abs(d1)+abs(d3))/2; spread = abs(d1-d3)
    return '✓ CLEAN' if (same and spread < gap and gap > 0) else ('~ same-side' if same else '✗ disagree')


_csv_rows = []
_section = ['']


def row(label, fn):
    v = [fn(s) for s in STEMS]
    vd = verdict(*v)
    print(f'{label:<34}{v[0]:>10.3g}{v[1]:>10.3g}{v[2]:>10.3g}   {vd}')
    hm = (v[1] + v[2]) / 2
    pct = round(100 * (hm - v[0]) / v[0], 1) if v[0] else float('nan')
    _csv_rows.append([_section[0], label, round(v[0], 3), round(v[1], 3), round(v[2], 3), pct, vd])


def sec(name):
    _section[0] = name.strip('— ').strip()
    print(name)


print(f'{"":34}{"WT":>10}{"HET_1":>10}{"HET_3":>10}')
sec('— ABUNDANCE —')
row('skeleton / tissue-tile mm2', lambda s: A(reg, s, 'skel_per_tile_mm2').mean())
row('foreground fraction (mean tile)', lambda s: A(reg, s, 'fg_fraction').mean())
row('components / image (count)', lambda s: len(A(cc, s, 'length_um')))

sec('— SIZE / LENGTH —')
row('region mean segment len (um)', lambda s: A(reg, s, 'mean_seg_len_um').mean())
row('component median len (um)', lambda s: np.median(A(cc, s, 'length_um')))
row('component p90 len (um)', lambda s: np.percentile(A(cc, s, 'length_um'), 90))
row('component median span (um)', lambda s: np.median(A(cc, s, 'span_um')))

sec('— THICKNESS (apparent width) —')
row('region mean thickness (um)', lambda s: A(reg, s, 'mean_thickness_um').mean())
row('component mean thickness (um)', lambda s: A(cc, s, 'thickness_um').mean())

sec('— BRANCHING / COMPLEXITY —')
row('branch per 100um skeleton', lambda s: A(reg, s, 'branch_per_100um').mean())
row('mean branches / component', lambda s: A(cc, s, 'n_branches').mean())

sec('— FRAGMENTATION (DAM-like) —')
row('region endpoint/branch ratio', lambda s: A(reg, s, 'endpoint_branch_ratio').mean())
row('region endpoint per 100um', lambda s: A(reg, s, 'endpoint_per_100um').mean())
row('fragmentation score (mean)', lambda s: A(reg, s, 'frag_score').mean())
row('% components unbranched (stub)', lambda s: 100*A(cc, s, 'is_stub').mean())

sec('— CONNECTIVITY —')
row('% skeleton in >100um components', lambda s: 100*A(cc, s, 'length_um')[A(cc, s, 'length_um') > 100].sum()/A(cc, s, 'length_um').sum())

sec('— HETEROGENEITY (within-image spread; Huixin: disease = more diverse) —')
row('CV thickness across tiles', lambda s: A(reg, s, 'mean_thickness_um').std()/A(reg, s, 'mean_thickness_um').mean())
row('CV seg-len across tiles', lambda s: A(reg, s, 'mean_seg_len_um').std()/A(reg, s, 'mean_seg_len_um').mean())
row('CV branch/100um across tiles', lambda s: A(reg, s, 'branch_per_100um').std()/A(reg, s, 'branch_per_100um').mean())
row('STD fragmentation score (tiles)', lambda s: A(reg, s, 'frag_score').std())
row('CV component length', lambda s: A(cc, s, 'length_um').std()/A(cc, s, 'length_um').mean())

sec('— MORPHOTYPE COMPOSITION —')
row('region de-ramified %', lambda s: 100*(A(reg, s, 'cluster') == 3).mean())
row('CC small/fragment M0 %', lambda s: 100*(A(cc, s, 'cluster') == 0).mean())
row('CC large-arbor M3 %', lambda s: 100*(A(cc, s, 'cluster') == 3).mean())

sec('— SPATIAL FOCALITY —')
fr = np.array([float(r['frag_score']) for r in reg]); thr = np.percentile(fr, 75)
row('fragmentation-hotspot tile %', lambda s: 100*(A(reg, s, 'frag_score') > thr).mean())


import csv as _csv
with open(FM/'out_region/FINAL_METRICS.csv','w',newline='') as f:
    w=_csv.writer(f); w.writerow(['section','metric','WT','HET_1','HET_3','HET_vs_WT_pct','verdict'])
    w.writerows(_csv_rows)
print('\nsaved FINAL_METRICS.csv  (', len(_csv_rows), 'metrics )')
