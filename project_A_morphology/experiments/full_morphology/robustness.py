"""
Robustness of the de-ramified-morphotype result to tile size and cluster count.
For each (tile, k): cluster regions on brightness-independent structural
features, pick the morphotype with the HIGHEST fragmentation (endpoint/branch
ratio) = the disease type, and check whether it is enriched in HET with the two
replicates agreeing (Stephen's criterion). If the conclusion only holds at one
(tile, k), it's a parameter artifact; if it holds across, it's real.
"""
import sys
from pathlib import Path
import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.ndimage import binary_dilation
from skimage import filters, morphology
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

ROOT = Path('/Users/stephenyu/choroid-microglia/project_A_morphology')
RAW = ROOT / 'data/raw'
V29 = ROOT / 'experiments/v29_short_spur_audit'
sys.path.insert(0, str(ROOT / 'experiments'))
import clean_topology as ct

PIXEL_UM = 0.207
STEMS = ['F_WT_2', 'F_HET_1', 'F_HET_3']
COND = {'F_WT_2': 'WT', 'F_HET_1': 'HET', 'F_HET_3': 'HET'}
MIN_FG_FRAC = 0.03
FEATS = ['skel_len_per_mm2', 'branch_per_mm2', 'endpoint_per_mm2',
         'endpoint_branch_ratio', 'mean_thickness_um', 'ramification']


def prep(stem):
    raw = tifffile.imread(next(RAW.glob(f'*{stem}*.tif'))).astype(np.float32)
    lo, hi = np.percentile(raw, [1.0, 99.5])
    norm = np.clip((raw - lo) / (hi - lo + 1e-9), 0, 1)
    sm = filters.gaussian(norm, 1.0)
    binary = sm > filters.threshold_otsu(sm) * 0.7
    binary = morphology.binary_closing(binary, morphology.disk(2))
    binary = morphology.remove_small_objects(binary, 20)
    dist = ndi.distance_transform_edt(binary)
    skel = np.load(V29 / f'{stem}_skel_pruned.npy').astype(bool)
    skel = ct.clean(skel); skel = ct.break_loops(skel, norm); skel = ct.prune_spurs(skel, 8)
    return binary, dist, skel, ct.degree(skel)


def tile_rows(stem, binary, dist, skel, deg, tile):
    rows = []
    H, W = skel.shape
    for ty in range(0, H, tile):
        for tx in range(0, W, tile):
            sl = (slice(ty, ty + tile), slice(tx, tx + tile))
            tb = binary[sl]; fg_px = int(tb.sum())
            if fg_px / tb.size < MIN_FG_FRAC:
                continue
            tsk = skel[sl]; tdeg = deg[sl]
            fg_mm2 = fg_px * PIXEL_UM ** 2 / 1e6
            skel_len_um = float(tsk.sum()) * PIXEL_UM
            bp = binary_dilation(tsk & (tdeg >= 3), iterations=4)
            _, nb = ndi.label(bp, structure=np.ones((3, 3)))
            ne = int((tsk & (tdeg == 1)).sum())
            thick = float(dist[sl][tsk].mean()) * 2 * PIXEL_UM if tsk.any() else 0.0
            rows.append(dict(
                image=stem,
                skel_len_per_mm2=skel_len_um / fg_mm2,
                branch_per_mm2=nb / fg_mm2,
                endpoint_per_mm2=ne / fg_mm2,
                endpoint_branch_ratio=ne / (nb + 1),
                mean_thickness_um=thick,
                ramification=100 * nb / (skel_len_um + 1e-9),
            ))
    return rows


# load each image once
print('loading 3 images (once)...')
data = {s: prep(s) for s in STEMS}

print(f'\n{"tile_um":>8}{"k":>4}{"WT%":>8}{"HET_1%":>8}{"HET_3%":>8}{"verdict":>16}')
for tile in (150, 200, 300):
    allrows = []
    for s in STEMS:
        allrows += tile_rows(s, *data[s], tile)
    X = np.array([[r[f] for f in FEATS] for r in allrows])
    Xs = StandardScaler().fit_transform(X)
    for k in (3, 4, 5):
        lab = KMeans(k, random_state=0, n_init=10).fit_predict(Xs)
        # disease morphotype = highest mean fragmentation (endpoint/branch)
        frag = np.array([r['endpoint_branch_ratio'] for r in allrows])
        dz = int(np.argmax([frag[lab == c].mean() for c in range(k)]))
        def pct(stem):
            idx = [i for i, r in enumerate(allrows) if r['image'] == stem]
            return 100 * float(np.mean([lab[i] == dz for i in idx]))
        wt, h1, h3 = pct('F_WT_2'), pct('F_HET_1'), pct('F_HET_3')
        d1, d3 = h1 - wt, h3 - wt
        same = (d1 > 0) == (d3 > 0)
        gap = (abs(d1) + abs(d3)) / 2; spread = abs(d1 - d3)
        v = '✓ HET enriched' if (same and d1 > 0 and spread < gap) else (
            '~ both up' if (same and d1 > 0) else '✗ inconsistent')
        print(f'{tile*PIXEL_UM:>8.0f}{k:>4}{wt:>8.1f}{h1:>8.1f}{h3:>8.1f}{v:>16}')
