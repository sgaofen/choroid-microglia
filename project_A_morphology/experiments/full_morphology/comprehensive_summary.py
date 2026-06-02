"""All-dimensions summary from the corrected data (region + cc + aggregate).
Central tendency AND within-image spread (heterogeneity), with the replicate
verdict for each. CSV-only, no image processing."""
import sys, csv
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import pipeline as pl
from pipeline import STEMS, COND, WT, HET   # auto-discovered from data/raw

FM = pl.ROOT / 'experiments/full_morphology'
reg = list(csv.DictReader(open(FM/'out_region/region_features.csv')))
cc = list(csv.DictReader(open(FM/'out_cc/cc_features.csv')))


def A(rows, img, key):
    return np.array([float(r[key]) for r in rows if r['image'] == img])


_csv_rows = []
_section = ['']


def row(label, fn):
    vals = {s: fn(s) for s in STEMS}
    vd, pct = pl.group_verdict(vals)
    wt_mean = float(np.mean([vals[s] for s in WT])) if WT else float('nan')
    het_mean = float(np.mean([vals[s] for s in HET])) if HET else float('nan')
    cells = '  '.join(f'{vals[s]:.3g}' for s in STEMS)
    print(f'{label:<34}{cells:>34}   {vd}')
    _csv_rows.append([_section[0], label, round(wt_mean, 4), round(het_mean, 4), pct, vd,
                      ';'.join(f'{s}={round(vals[s], 4)}' for s in STEMS)])


def sec(name):
    _section[0] = name.strip('— ').strip()
    print(name)


print(f'{"metric":<34}{"  ".join(STEMS):>34}   verdict')
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


with open(FM/'out_region/FINAL_METRICS.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['section','metric','WT_mean','HET_mean','HET_vs_WT_pct','verdict','per_image'])
    w.writerows(_csv_rows)
print('\nsaved FINAL_METRICS.csv  (', len(_csv_rows), 'metrics )')
