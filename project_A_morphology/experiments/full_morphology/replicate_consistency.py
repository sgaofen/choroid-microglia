"""
Replicate-consistency screen (Stephen's criterion, 2026-05-27): the two HET are
biological replicates of ONE treatment, so they SHOULD resemble each other. A
trustworthy WT-vs-HET signal therefore requires BOTH:
  (1) HET_1 and HET_3 on the SAME side of WT (consistent direction), AND
  (2) the two HET closer to EACH OTHER than to WT (replicate spread < WT gap).
Metrics where the two HET disagree are noise — drop them.

Screens every metric we have: whole-image aggregate (out/aggregate.json),
per-region feature means, and morphotype-composition proportions (clustered on
brightness-independent structural features). Prints a clean->messy ranking.
"""
import sys, json, csv
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

FULL = Path('/Users/stephenyu/choroid-microglia/project_A_morphology/experiments/full_morphology')
OUTR = FULL / 'out_region'


def verdict(wt, h1, h3):
    """Return dict scoring replicate-consistency + WT-separation."""
    d1, d3 = h1 - wt, h3 - wt
    same_side = (d1 > 0) == (d3 > 0) and d1 != 0 and d3 != 0
    gap = (abs(d1) + abs(d3)) / 2.0          # mean HET->WT distance
    spread = abs(d1 - d3)                    # HET<->HET distance
    gap_pct = 100 * gap / abs(wt) if wt else float('nan')
    clean = same_side and spread < gap       # replicates tighter than the gap
    score = (gap / (spread + 1e-9)) if same_side else 0.0
    return dict(wt=wt, h1=h1, h3=h3, dir=('+' if d1 > 0 else '-') if same_side else 'X',
                gap_pct=round(gap_pct, 1), spread_vs_gap=round(spread / (gap + 1e-9), 2),
                clean=clean, score=score)


def screen(name_vals):
    """name_vals: dict name -> (wt, h1, h3). Print ranked table."""
    res = {n: verdict(*v) for n, v in name_vals.items()}
    order = sorted(res, key=lambda n: (-res[n]['clean'], -res[n]['score']))
    print(f'{"metric":<26}{"WT":>10}{"HET_1":>10}{"HET_3":>10}{"dir":>4}'
          f'{"gap%":>7}{"sprd/gap":>9}  verdict')
    for n in order:
        r = res[n]
        v = '✓ CLEAN' if r['clean'] else ('~ same-side' if r['dir'] != 'X' else '✗ HET disagree')
        print(f'{n:<26}{r["wt"]:>10.3g}{r["h1"]:>10.3g}{r["h3"]:>10.3g}{r["dir"]:>4}'
              f'{r["gap_pct"]:>7}{r["spread_vs_gap"]:>9}  {v}')
    return res


print('================ (1) WHOLE-IMAGE AGGREGATE ================')
agg = json.loads((FULL / 'out/aggregate.json').read_text())
metrics = [k for k, v in agg['F_WT_2'].items() if isinstance(v, (int, float))
           and k not in ('fg_area_mm2', 'n_branches', 'n_endpoints', 'n_tiles')]
screen({m: (agg['F_WT_2'][m], agg['F_HET_1'][m], agg['F_HET_3'][m]) for m in metrics})

print('\n================ (2) PER-REGION FEATURE MEANS ================')
rows = list(csv.DictReader(open(OUTR / 'region_features.csv')))
feats = ['fg_fraction', 'skel_len_per_mm2', 'branch_per_mm2', 'endpoint_per_mm2',
         'endpoint_branch_ratio', 'mean_thickness_um', 'ramification']
def img_mean(stem, f):
    return float(np.mean([float(r[f]) for r in rows if r['image'] == stem]))
screen({f: (img_mean('F_WT_2', f), img_mean('F_HET_1', f), img_mean('F_HET_3', f)) for f in feats})

print('\n================ (3) MORPHOTYPE COMPOSITION (brightness-independent) ================')
STRUCT = ['skel_len_per_mm2', 'branch_per_mm2', 'endpoint_per_mm2',
          'endpoint_branch_ratio', 'mean_thickness_um', 'ramification']
X = np.array([[float(r[f]) for f in STRUCT] for r in rows])
lab = KMeans(4, random_state=0, n_init=10).fit_predict(StandardScaler().fit_transform(X))
rami = np.array([float(r['ramification']) for r in rows])
order = np.argsort([-rami[lab == k].mean() for k in range(4)])
remap = {o: n for n, o in enumerate(order)}
lab = np.array([remap[l] for l in lab])
def comp(stem, k):
    idx = [i for i, r in enumerate(rows) if r['image'] == stem]
    return 100 * float(np.mean([lab[i] == k for i in idx]))
# describe each morphotype
print('morphotype profiles: C0=most ramified ... C3=least (de-ramified)')
for k in range(4):
    m = lab == k
    f = lambda c: np.mean([float(rows[i][c]) for i in np.where(m)[0]])
    print(f'  C{k} (n={m.sum():>3}): branch/mm2={f("branch_per_mm2"):.0f} '
          f'ep/br={f("endpoint_branch_ratio"):.2f} thick={f("mean_thickness_um"):.2f} '
          f'rami={f("ramification"):.1f}')
print()
screen({f'C{k}_pct': (comp('F_WT_2', k), comp('F_HET_1', k), comp('F_HET_3', k)) for k in range(4)})
