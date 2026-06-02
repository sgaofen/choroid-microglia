"""
Connected-component morphotype analysis — CORRECTED per GPT-Pro review.
Per-component branch/endpoint counts now come from the SAME global topology
(clean_topology.merged_branches exit>=3 + endpoints) assigned to each component,
not the loose dilation count. Unit = one connected skeleton component (sparse
region -> one cell; dense -> one clump, treated as a unit).
"""
import sys, json, csv
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).parent))
import pipeline as pl
from pipeline import PIXEL_UM, STEMS, COND

OUT = pl.ROOT / 'experiments/full_morphology/out_cc'
CLUMP_UM = 100.0
K = 4
SHAPE = ['length_um', 'span_um', 'branches_per_100um', 'endpoint_branch_ratio', 'thickness_um']


def coord_label(lab, y, x):
    yi, xi = int(round(y)), int(round(x))
    win = lab[max(0, yi-2):yi+3, max(0, xi-2):xi+3]
    nz = win[win > 0]
    return int(np.bincount(nz).argmax()) if nz.size else 0


def components(stem):
    norm, binary, dist, skel = pl.prep(stem)
    J, E = pl.global_topology(skel)
    midy, midx, seglen = pl.segments_df(skel)
    lab, n = ndi.label(skel, structure=np.ones((3, 3)))
    fg_mm2 = float(binary.sum()) * PIXEL_UM ** 2 / 1e6

    npx = np.bincount(lab[skel], minlength=n+1)[1:]
    th_sum = np.bincount(lab[skel], weights=dist[skel], minlength=n+1)[1:]
    # global topology -> per component
    lj = np.array([coord_label(lab, y, x) for y, x in J]) if len(J) else np.array([], int)
    nbr = np.bincount(lj[lj > 0], minlength=n+1)[1:] if lj.size else np.zeros(n)
    le = lab[E[:, 0], E[:, 1]] if len(E) else np.array([], int)
    nep = np.bincount(le[le > 0], minlength=n+1)[1:] if le.size else np.zeros(n)
    slices = ndi.find_objects(lab)
    span = np.array([np.hypot(s[0].stop-s[0].start, s[1].stop-s[1].start) if s else 0 for s in slices])

    rows = []
    for i in range(n):
        if npx[i] == 0:
            continue
        L = npx[i] * PIXEL_UM
        rows.append(dict(
            image=stem, condition=COND[stem], comp_id=i+1,
            length_um=round(L, 2), span_um=round(span[i]*PIXEL_UM, 2),
            n_branches=int(nbr[i]), n_endpoints=int(nep[i]),
            thickness_um=round(th_sum[i]/npx[i]*2*PIXEL_UM, 3),
            branches_per_100um=round(100*nbr[i]/(L+1e-9), 3),
            endpoint_branch_ratio=round(nep[i]/max(nbr[i], 1), 3),
            is_stub=int(nbr[i] == 0),
        ))
    print(f'  {stem}: {n} components, fg={fg_mm2:.3f} mm2, junctions={len(J)}')
    return rows, fg_mm2, (lab, skel.shape)


def verdict(vals):
    """vals: dict stem->value (any number of WT/HET images)."""
    return pl.group_verdict(vals)[0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []; fg = {}; labinfo = {}
    print('labeling components (corrected global-topology counts)...')
    for s in STEMS:
        rows, f, li = components(s)
        all_rows += rows; fg[s] = f; labinfo[s] = li

    # ---- per-image summary ----
    print('\n=== per-image summary (corrected) ===')
    summ = {}
    for s in STEMS:
        r = [x for x in all_rows if x['image'] == s]
        L = np.array([x['length_um'] for x in r])
        summ[s] = dict(condition=COND[s], n_comp=len(r),
                       comp_per_fg_mm2=round(len(r)/fg[s], 1),
                       pct_stub=round(100*np.mean([x['is_stub'] for x in r]), 1),
                       median_len_um=round(float(np.median(L)), 2),
                       p90_len_um=round(float(np.percentile(L, 90)), 2),
                       pct_skel_in_large=round(100*L[L > CLUMP_UM].sum()/L.sum(), 1),
                       mean_ep_br=round(float(np.mean([x['endpoint_branch_ratio'] for x in r])), 3))
    print(f'{"metric":>20}{("  ".join(STEMS)):>34}{"verdict":>16}')
    for m in ['comp_per_fg_mm2', 'pct_stub', 'median_len_um', 'p90_len_um', 'pct_skel_in_large', 'mean_ep_br']:
        vals = {s: summ[s][m] for s in STEMS}
        cells = '  '.join(f'{vals[s]:.4g}' for s in STEMS)
        print(f'{m:>20}{cells:>34}{verdict(vals):>16}')

    # ---- morphotype clustering (shape; component size IS morphology here) ----
    X = np.array([[r[k] for k in SHAPE] for r in all_rows], float)
    Xl = np.log1p(X)
    lab = KMeans(K, random_state=0, n_init=10).fit_predict(StandardScaler().fit_transform(Xl))
    Lc = np.array([r['length_um'] for r in all_rows])
    order = np.argsort([Lc[lab == k].mean() for k in range(K)])  # M0 small -> M3 large
    remap = {o: i for i, o in enumerate(order)}
    lab = np.array([remap[l] for l in lab])
    for r, l in zip(all_rows, lab):
        r['cluster'] = int(l)

    print('\n=== component morphotype profiles (M0 small/frag -> M3 large) ===')
    for k in range(K):
        m = lab == k
        f = lambda c: np.mean([all_rows[i][c] for i in np.where(m)[0]])
        print(f'  M{k} (n={m.sum():>5}): len={f("length_um"):.1f} span={f("span_um"):.1f} '
              f'branch={f("n_branches"):.1f} ep/br={f("endpoint_branch_ratio"):.2f} '
              f'thick={f("thickness_um"):.2f} stub%={100*np.mean([all_rows[i]["is_stub"] for i in np.where(m)[0]]):.0f}')

    print('\n=== morphotype composition (% of components) ===')
    print(f'{"morphotype":>10}{("  ".join(STEMS)):>34}{"verdict":>16}')
    comp = {}
    for k in range(K):
        def pct(s):
            idx = [i for i, r in enumerate(all_rows) if r['image'] == s]
            return round(100*float(np.mean([lab[i] == k for i in idx])), 1)
        vals = {s: pct(s) for s in STEMS}
        comp[f'M{k}'] = vals
        cells = '  '.join(f'{vals[s]:.4g}' for s in STEMS)
        print(f'{"M"+str(k):>10}{cells:>34}{verdict(vals):>16}')

    # ---- save ----
    with open(OUT / 'cc_features.csv', 'w', newline='') as f:
        cc = ['image', 'condition', 'comp_id', 'length_um', 'span_um', 'n_branches',
              'n_endpoints', 'thickness_um', 'branches_per_100um', 'endpoint_branch_ratio',
              'is_stub', 'cluster']
        w = csv.DictWriter(f, fieldnames=cc); w.writeheader()
        for r in all_rows:
            w.writerow({c: r[c] for c in cc})
    (OUT / 'cc_morphotype.json').write_text(json.dumps(
        dict(clump_um=CLUMP_UM, k=K, summary=summ, composition=comp), indent=1))

    # ---- size distribution ----
    plt.figure(figsize=(9, 5))
    bins = np.logspace(np.log10(2), np.log10(max(r['length_um'] for r in all_rows)+1), 40)
    for s in STEMS:
        c = 'C0' if COND[s] == 'WT' else 'C3'
        L = [r['length_um'] for r in all_rows if r['image'] == s]
        plt.hist(L, bins=bins, density=True, histtype='step', lw=2, label=f'{s}({COND[s]})', color=c)
    plt.xscale('log'); plt.xlabel('connected-component skeleton length (um)'); plt.ylabel('density')
    plt.legend(); plt.title('component size distribution (WT vs HET) — corrected')
    plt.savefig(OUT / 'cc_size_distribution.png', dpi=120, bbox_inches='tight'); plt.close()

    # ---- maps ----
    cmap = ListedColormap(['#d7191c', '#fdae61', '#abd9e9', '#2c7bb6'][:K])
    for s in STEMS:
        lab_img, shape = labinfo[s]
        lut = np.full(lab_img.max()+1, np.nan)
        for r, l in zip(all_rows, lab):
            if r['image'] == s:
                lut[r['comp_id']] = l
        painted = np.full(shape, np.nan); m = lab_img > 0
        painted[m] = lut[lab_img[m]]
        plt.figure(figsize=(6, 6)); plt.imshow(painted, cmap=cmap, vmin=-0.5, vmax=K-0.5, interpolation='nearest')
        cb = plt.colorbar(ticks=range(K)); cb.set_label('morphotype (M0 small/frag -> M3 large)')
        plt.title(f'{s} ({COND[s]}) component morphotypes — corrected'); plt.axis('off')
        plt.savefig(OUT / f'{s}_cc_map.png', dpi=130, bbox_inches='tight'); plt.close()

    print('\nsaved to', OUT)


if __name__ == '__main__':
    main()
