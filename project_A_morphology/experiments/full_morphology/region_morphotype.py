"""
Region-based morphotype analysis (no per-cell segmentation — that's the
unreliable connect-vs-separate problem). Stephen's idea (2026-05-27): tile the
image into MANY small regions, give each region a morphology fingerprint, then
cluster regions into morphotypes and compare their COMPOSITION between WT and
HET. The disease signal is FOCAL — a subset of regions de-ramify/swell — so the
distribution of region-types moves even when the whole-image MEAN does not.

Unit of analysis = a 200px (~41um) tile. Boundaries are ours, 100% reliable.

Outputs (experiments/full_morphology/out_region/):
  region_features.csv     every tile: image, condition, ty, tx, 6 features, cluster
  region_morphotype.json  cluster profiles + per-image morphotype composition
  <stem>_morphotype_map.png   spatial map: each tile colored by morphotype
  feature_distributions.png   per-feature tile histograms, WT vs HET
  composition_bar.png         morphotype composition WT vs HET
"""
import sys, json, csv
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.ndimage import binary_dilation
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
OUT = ROOT / 'experiments/full_morphology/out_region'
sys.path.insert(0, str(ROOT / 'experiments'))
import clean_topology as ct

PIXEL_UM = 0.207
STEMS = ['F_WT_2', 'F_HET_1', 'F_HET_3']
COND = {'F_WT_2': 'WT', 'F_HET_1': 'HET', 'F_HET_3': 'HET'}
TILE = 200                  # ~41 um; a few-cell neighborhood
MIN_FG_FRAC = 0.03          # skip near-empty tiles
K = 4                       # number of morphotypes
FEATURES = ['fg_fraction', 'skel_len_per_mm2', 'branch_per_mm2',
            'endpoint_per_mm2', 'endpoint_branch_ratio', 'mean_thickness_um']


def find_raw(stem):
    for p in RAW.glob('*.tif'):
        if stem in p.name:
            return p
    raise FileNotFoundError(stem)


def normalize(img, p_lo=1.0, p_hi=99.5):
    lo, hi = np.percentile(img, [p_lo, p_hi])
    return np.clip((img - lo) / (hi - lo + 1e-9), 0, 1)


def fg_and_dist(raw):
    norm = normalize(raw)
    sm = filters.gaussian(norm, 1.0)
    binary = sm > filters.threshold_otsu(sm) * 0.7
    binary = morphology.binary_closing(binary, morphology.disk(2))
    binary = morphology.remove_small_objects(binary, 20)
    dist = ndi.distance_transform_edt(binary)
    return norm, binary, dist


def tile_features(stem):
    """One feature vector per non-empty tile."""
    raw = tifffile.imread(find_raw(stem)).astype(np.float32)
    norm, binary, dist = fg_and_dist(raw)
    skel = np.load(V29 / f'{stem}_skel_pruned.npy').astype(bool)
    skel = ct.clean(skel); skel = ct.break_loops(skel, norm); skel = ct.prune_spurs(skel, 8)
    deg = ct.degree(skel)

    rows = []
    H, W = skel.shape
    for ty in range(0, H, TILE):
        for tx in range(0, W, TILE):
            sl = (slice(ty, ty+TILE), slice(tx, tx+TILE))
            tb = binary[sl]
            tile_px = tb.size
            fg_px = int(tb.sum())
            if fg_px / tile_px < MIN_FG_FRAC:
                continue
            tsk = skel[sl]
            tdeg = deg[sl]
            fg_mm2 = fg_px * PIXEL_UM ** 2 / 1e6
            skel_len_um = float(tsk.sum()) * PIXEL_UM
            # merged junction count: dilate degree>=3 pixels, count components
            bp = binary_dilation(tsk & (tdeg >= 3), iterations=4)
            _, n_branch = ndi.label(bp, structure=np.ones((3, 3)))
            n_endpoint = int((tsk & (tdeg == 1)).sum())
            thick = float(dist[sl][tsk].mean()) * 2 * PIXEL_UM if tsk.any() else 0.0
            rows.append(dict(
                image=stem, condition=COND[stem], ty=ty, tx=tx,
                fg_fraction=round(fg_px / tile_px, 4),
                skel_len_per_mm2=round(skel_len_um / fg_mm2, 1),
                branch_per_mm2=round(n_branch / fg_mm2, 1),
                endpoint_per_mm2=round(n_endpoint / fg_mm2, 1),
                endpoint_branch_ratio=round(n_endpoint / (n_branch + 1), 3),
                mean_thickness_um=round(thick, 3),
                ramification=round(100 * n_branch / (skel_len_um + 1e-9), 2),
            ))
    print(f'  {stem}: {len(rows)} tiles')
    return rows, skel.shape


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    shapes = {}
    print('extracting tile features...')
    for stem in STEMS:
        rows, shp = tile_features(stem)
        all_rows += rows
        shapes[stem] = shp

    X = np.array([[r[f] for f in FEATURES] for r in all_rows], float)
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=K, random_state=0, n_init=10).fit(Xs)
    labels = km.labels_

    # order clusters by ramification (most-branched -> least) for stable naming
    rami = np.array([r['ramification'] for r in all_rows])
    order = np.argsort([-rami[labels == k].mean() for k in range(K)])
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[l] for l in labels])
    for r, l in zip(all_rows, labels):
        r['cluster'] = int(l)

    # cluster profiles (mean feature value per morphotype, original units)
    print('\n=== morphotype profiles (mean feature per cluster) ===')
    prof_cols = FEATURES + ['ramification']
    print(f'{"cluster":>8} {"n":>5} ' + ' '.join(f'{c[:11]:>12}' for c in prof_cols))
    profiles = {}
    for k in range(K):
        m = labels == k
        vals = {c: round(float(np.mean([all_rows[i][c] for i in np.where(m)[0]])), 3) for c in prof_cols}
        profiles[k] = dict(n=int(m.sum()), **vals)
        print(f'{k:>8} {int(m.sum()):>5} ' + ' '.join(f'{vals[c]:>12}' for c in prof_cols))

    # composition per image
    print('\n=== morphotype composition (% of tiles) ===')
    comp = {}
    print(f'{"image":>10} {"n_tiles":>8} ' + ' '.join(f'{"C"+str(k):>7}' for k in range(K)))
    for stem in STEMS:
        idx = [i for i, r in enumerate(all_rows) if r['image'] == stem]
        n = len(idx)
        pct = [round(100 * np.mean([labels[i] == k for i in idx]), 1) for k in range(K)]
        comp[stem] = dict(condition=COND[stem], n_tiles=n,
                          pct={f'C{k}': pct[k] for k in range(K)})
        print(f'{stem:>10} {n:>8} ' + ' '.join(f'{p:>7}' for p in pct))

    # WT vs HET averaged composition
    print('\n=== composition WT vs HET ===')
    wt_pct = np.array([comp['F_WT_2']['pct'][f'C{k}'] for k in range(K)])
    het_pct = np.mean([[comp[s]['pct'][f'C{k}'] for k in range(K)]
                       for s in ['F_HET_1', 'F_HET_3']], axis=0)
    print(f'{"morphotype":>10} {"WT%":>8} {"HET%":>8} {"delta":>8}')
    for k in range(K):
        print(f'{"C"+str(k):>10} {wt_pct[k]:>8} {round(het_pct[k],1):>8} {round(het_pct[k]-wt_pct[k],1):>+8}')

    # ---- save data ----
    with open(OUT / 'region_features.csv', 'w', newline='') as f:
        cols = ['image', 'condition', 'ty', 'tx'] + FEATURES + ['ramification', 'cluster']
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in all_rows:
            w.writerow({c: r[c] for c in cols})
    (OUT / 'region_morphotype.json').write_text(json.dumps(dict(
        tile_um=TILE * PIXEL_UM, k=K, features=FEATURES,
        profiles=profiles, composition=comp,
        wt_vs_het={f'C{k}': dict(WT=float(wt_pct[k]), HET=round(float(het_pct[k]), 1),
                                 delta=round(float(het_pct[k] - wt_pct[k]), 1)) for k in range(K)},
    ), indent=1))

    # ---- spatial morphotype maps ----
    cmap = ListedColormap(['#2c7bb6', '#abd9e9', '#fdae61', '#d7191c'][:K])
    for stem in STEMS:
        H, W = shapes[stem]
        grid = np.full((H // TILE + 1, W // TILE + 1), np.nan)
        for i, r in enumerate(all_rows):
            if r['image'] == stem:
                grid[r['ty'] // TILE, r['tx'] // TILE] = labels[i]
        plt.figure(figsize=(6, 6))
        plt.imshow(grid, cmap=cmap, vmin=-0.5, vmax=K - 0.5)
        cb = plt.colorbar(ticks=range(K)); cb.set_label('morphotype (C0=most ramified -> C%d=least)' % (K-1))
        plt.title(f'{stem} ({COND[stem]}) morphotype map'); plt.axis('off')
        plt.savefig(OUT / f'{stem}_morphotype_map.png', dpi=110, bbox_inches='tight'); plt.close()

    # ---- per-feature tile distributions WT vs HET ----
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, feat in zip(axes.ravel(), FEATURES):
        wt_v = [r[feat] for r in all_rows if r['condition'] == 'WT']
        het_v = [r[feat] for r in all_rows if r['condition'] == 'HET']
        lo, hi = np.percentile(wt_v + het_v, [0, 99])
        bins = np.linspace(lo, hi, 30)
        ax.hist(wt_v, bins=bins, density=True, histtype='step', lw=2, color='C0', label='WT')
        ax.hist(het_v, bins=bins, density=True, histtype='step', lw=2, color='C3', label='HET')
        ax.set_title(feat); ax.legend(fontsize=8)
    fig.suptitle('per-region (tile) feature distributions, WT vs HET')
    fig.tight_layout(); fig.savefig(OUT / 'feature_distributions.png', dpi=110, bbox_inches='tight'); plt.close()

    # ---- composition bar ----
    plt.figure(figsize=(7, 5))
    x = np.arange(K); width = 0.35
    plt.bar(x - width/2, wt_pct, width, label='WT', color='C0')
    plt.bar(x + width/2, het_pct, width, label='HET (mean)', color='C3')
    plt.xticks(x, [f'C{k}' for k in range(K)]); plt.ylabel('% of regions')
    plt.title('morphotype composition: WT vs HET'); plt.legend()
    plt.savefig(OUT / 'composition_bar.png', dpi=120, bbox_inches='tight'); plt.close()

    print('\nsaved to', OUT)


if __name__ == '__main__':
    main()
