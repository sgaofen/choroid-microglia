"""
Connected-component morphotype analysis (Stephen's idea, 2026-05-27): use each
CONNECTED skeleton component as the unit, instead of an arbitrary 200px grid.
Sparse region -> one component = one cell (free per-cell morphology where it's
reliable); dense region -> one big component = a clump of touching cells (which
Huixin says to treat as a unit; clumping is itself a signal). The component SIZE
distribution captures both fragmentation (many small) and clumping (few giant).

Caveat: "connected" is set by the binarization/skeleton (the connect-vs-separate
problem reappears as "one component or two"), but applied identically to all
images the distributions are comparable.

Outputs (experiments/full_morphology/out_cc/):
  cc_features.csv          per component: image, condition, length_um, span_um,
                           n_branches, n_endpoints, thickness_um, is_stub, cluster
  cc_morphotype.json       per-image summary + morphotype composition + WTvsHET
  <stem>_cc_map.png        each component colored by morphotype
  cc_size_distribution.png component-size distribution, WT vs HET
"""
import sys, json, csv
from pathlib import Path
import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage import filters, morphology
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = Path('/Users/stephenyu/choroid-microglia/project_A_morphology')
RAW = ROOT / 'data/raw'
V29 = ROOT / 'experiments/v29_short_spur_audit'
OUT = ROOT / 'experiments/full_morphology/out_cc'
sys.path.insert(0, str(ROOT / 'experiments'))
import clean_topology as ct

PIXEL_UM = 0.207
STEMS = ['F_WT_2', 'F_HET_1', 'F_HET_3']
COND = {'F_WT_2': 'WT', 'F_HET_1': 'HET', 'F_HET_3': 'HET'}
STRUCT = 13         # 8-connectivity
CLUMP_UM = 100.0    # component longer than this = clump (multi-cell) proxy
K = 4


def prep(stem):
    raw = tifffile.imread(next(RAW.glob(f'*{stem}*.tif'))).astype(np.float32)
    lo, hi = np.percentile(raw, [1.0, 99.5])
    norm = np.clip((raw - lo) / (hi - lo + 1e-9), 0, 1)
    sm = filters.gaussian(norm, 1.0)
    binary = sm > filters.threshold_otsu(sm) * 0.7
    binary = morphology.binary_closing(binary, morphology.disk(2))
    binary = morphology.remove_small_objects(binary, 20)
    dist = ndi.distance_transform_edt(binary)
    fg_mm2 = float(binary.sum()) * PIXEL_UM ** 2 / 1e6
    skel = np.load(V29 / f'{stem}_skel_pruned.npy').astype(bool)
    skel = ct.clean(skel); skel = ct.break_loops(skel, norm); skel = ct.prune_spurs(skel, 8)
    return binary, dist, skel, fg_mm2


def components(stem):
    binary, dist, skel, fg_mm2 = prep(stem)
    lab, n = ndi.label(skel, structure=np.ones((3, 3)))
    deg = ct.degree(skel)
    flat = lab[skel]
    npx = np.bincount(flat, minlength=n + 1)[1:]
    n_ep = np.bincount(lab[skel & (deg == 1)], minlength=n + 1)[1:]
    n_br = np.bincount(lab[skel & (deg >= 3)], minlength=n + 1)[1:]   # raw degree>=3 px
    th_sum = np.bincount(flat, weights=dist[skel], minlength=n + 1)[1:]
    slices = ndi.find_objects(lab)
    span = np.array([np.hypot(s[0].stop - s[0].start, s[1].stop - s[1].start)
                     if s else 0 for s in slices])
    rows = []
    for i in range(n):
        L = npx[i] * PIXEL_UM
        rows.append(dict(
            image=stem, condition=COND[stem], comp_id=i + 1,
            length_um=round(L, 2),
            span_um=round(span[i] * PIXEL_UM, 2),
            n_branches=int(n_br[i]),
            n_endpoints=int(n_ep[i]),
            thickness_um=round(th_sum[i] / npx[i] * 2 * PIXEL_UM, 3) if npx[i] else 0,
            branches_per_100um=round(100 * n_br[i] / (L + 1e-9), 2),
            is_stub=int(n_br[i] == 0),
        ))
    print(f'  {stem}: {n} components, fg={fg_mm2:.3f} mm2')
    return rows, fg_mm2, (lab, slices, skel.shape)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    fg = {}
    labinfo = {}
    print('labeling connected components...')
    for s in STEMS:
        rows, fg_mm2, li = components(s)
        all_rows += rows
        fg[s] = fg_mm2
        labinfo[s] = li

    # ---- per-image summary ----
    print('\n=== per-image component summary ===')
    print(f'{"image":>10}{"n_comp":>8}{"comp/mm2":>10}{"%stub":>8}'
          f'{"med_len":>9}{"p90_len":>9}{"%skel_clump":>12}')
    summ = {}
    for s in STEMS:
        r = [x for x in all_rows if x['image'] == s]
        L = np.array([x['length_um'] for x in r])
        stub = np.array([x['is_stub'] for x in r])
        clump_skel = L[L > CLUMP_UM].sum()
        summ[s] = dict(
            condition=COND[s], n_comp=len(r),
            comp_per_mm2=round(len(r) / fg[s], 1),
            pct_stub=round(100 * stub.mean(), 1),
            median_len_um=round(float(np.median(L)), 2),
            p90_len_um=round(float(np.percentile(L, 90)), 2),
            pct_skel_in_clumps=round(100 * clump_skel / L.sum(), 1),
        )
        d = summ[s]
        print(f'{s:>10}{d["n_comp"]:>8}{d["comp_per_mm2"]:>10}{d["pct_stub"]:>8}'
              f'{d["median_len_um"]:>9}{d["p90_len_um"]:>9}{d["pct_skel_in_clumps"]:>12}')

    # ---- replicate-consistency on summary metrics ----
    print('\n=== replicate-consistency (Stephen criterion) ===')
    print(f'{"metric":>20}{"WT":>10}{"HET_1":>10}{"HET_3":>10}{"verdict":>16}')
    for m in ['comp_per_mm2', 'pct_stub', 'median_len_um', 'p90_len_um', 'pct_skel_in_clumps']:
        wt, h1, h3 = summ['F_WT_2'][m], summ['F_HET_1'][m], summ['F_HET_3'][m]
        d1, d3 = h1 - wt, h3 - wt
        same = (d1 > 0) == (d3 > 0)
        gap = (abs(d1) + abs(d3)) / 2; spread = abs(d1 - d3)
        v = ('✓ CLEAN' if (same and spread < gap) else
             ('~ same-side' if same else '✗ HET disagree'))
        print(f'{m:>20}{wt:>10}{h1:>10}{h3:>10}{v:>16}')

    # ---- per-component morphotype clustering (drop stubs: cluster real arbors) ----
    feats = ['length_um', 'span_um', 'n_branches', 'n_endpoints',
             'thickness_um', 'branches_per_100um']
    X = np.array([[r[f] for f in feats] for r in all_rows], float)
    Xl = np.log1p(X)                       # heavy-tailed sizes -> log
    lab = KMeans(K, random_state=0, n_init=10).fit_predict(StandardScaler().fit_transform(Xl))
    # order morphotypes by mean length (small fragments -> big arbors/clumps)
    Lc = np.array([r['length_um'] for r in all_rows])
    order = np.argsort([Lc[lab == k].mean() for k in range(K)])
    remap = {o: i for i, o in enumerate(order)}
    lab = np.array([remap[l] for l in lab])
    for r, l in zip(all_rows, lab):
        r['cluster'] = int(l)

    print('\n=== component morphotype profiles (M0=smallest -> M3=largest) ===')
    for k in range(K):
        m = lab == k
        f = lambda c: np.mean([all_rows[i][c] for i in np.where(m)[0]])
        print(f'  M{k} (n={m.sum():>5}): len={f("length_um"):.1f}um span={f("span_um"):.1f}um '
              f'branch={f("n_branches"):.1f} ep={f("n_endpoints"):.1f} '
              f'thick={f("thickness_um"):.2f} stub%={100*np.mean([all_rows[i]["is_stub"] for i in np.where(m)[0]]):.0f}')

    print('\n=== morphotype composition (% of components) ===')
    print(f'{"morphotype":>10}{"WT":>8}{"HET_1":>8}{"HET_3":>8}{"verdict":>16}')
    comp = {}
    for k in range(K):
        def pct(s):
            idx = [i for i, r in enumerate(all_rows) if r['image'] == s]
            return round(100 * float(np.mean([lab[i] == k for i in idx])), 1)
        wt, h1, h3 = pct('F_WT_2'), pct('F_HET_1'), pct('F_HET_3')
        comp[f'M{k}'] = dict(WT=wt, HET_1=h1, HET_3=h3)
        d1, d3 = h1 - wt, h3 - wt
        same = (d1 > 0) == (d3 > 0)
        gap = (abs(d1) + abs(d3)) / 2; spread = abs(d1 - d3)
        v = ('✓ CLEAN' if (same and spread < gap) else ('~ same-side' if same else '✗ HET disagree'))
        print(f'{"M"+str(k):>10}{wt:>8}{h1:>8}{h3:>8}{v:>16}')

    # ---- save ----
    with open(OUT / 'cc_features.csv', 'w', newline='') as f:
        cols = ['image', 'condition', 'comp_id', 'length_um', 'span_um', 'n_branches',
                'n_endpoints', 'thickness_um', 'branches_per_100um', 'is_stub', 'cluster']
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in all_rows:
            w.writerow({c: r[c] for c in cols})
    (OUT / 'cc_morphotype.json').write_text(json.dumps(
        dict(clump_um=CLUMP_UM, k=K, summary=summ, composition=comp), indent=1))

    # ---- size distribution ----
    plt.figure(figsize=(9, 5))
    bins = np.logspace(np.log10(2), np.log10(max(r['length_um'] for r in all_rows) + 1), 40)
    for s, c in [('F_WT_2', 'C0'), ('F_HET_1', 'C3'), ('F_HET_3', 'C1')]:
        L = [r['length_um'] for r in all_rows if r['image'] == s]
        plt.hist(L, bins=bins, density=True, histtype='step', lw=2, label=f'{s}({COND[s]})', color=c)
    plt.xscale('log'); plt.xlabel('connected-component skeleton length (um)')
    plt.ylabel('density'); plt.legend(); plt.title('component size distribution (WT vs HET)')
    plt.savefig(OUT / 'cc_size_distribution.png', dpi=120, bbox_inches='tight'); plt.close()

    # ---- spatial morphotype maps ----
    cmap = ListedColormap(['#d7191c', '#fdae61', '#abd9e9', '#2c7bb6'][:K])  # M0 small/frag=red
    for s in STEMS:
        lab_img, slices, shape = labinfo[s]
        comp2clu = {r['comp_id']: r['cluster'] for r in all_rows if r['image'] == s}
        painted = np.full(shape, np.nan)
        idx = [(r['comp_id'], r['cluster']) for r in all_rows if r['image'] == s]
        lut = np.full(lab_img.max() + 1, np.nan)
        for cid, clu in idx:
            lut[cid] = clu
        m = lab_img > 0
        painted[m] = lut[lab_img[m]]
        plt.figure(figsize=(6, 6))
        plt.imshow(painted, cmap=cmap, vmin=-0.5, vmax=K - 0.5, interpolation='nearest')
        cb = plt.colorbar(ticks=range(K)); cb.set_label('morphotype (M0=small/frag -> M3=large/clump)')
        plt.title(f'{s} ({COND[s]}) component morphotypes'); plt.axis('off')
        plt.savefig(OUT / f'{s}_cc_map.png', dpi=130, bbox_inches='tight'); plt.close()

    print('\nsaved to', OUT)


if __name__ == '__main__':
    main()
