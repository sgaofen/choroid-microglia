"""
Algorithm-first WT vs HET morphology comparison, on existing v30f cells +
v29 skeleton. Computes Huixin's features per cell, then aggregates per image
and compares WT (F_WT_2) vs HET (F_HET_1 + F_HET_3).

Per-cell features:
  primary       = n_trunks               (processes leaving soma; from v30f)
  secondary     = n_local_branches        (total branch points; from v30f)
  skel_len      = skel_len_local          (total skeleton length; size proxy)
  soma_radius   = r_soma                  (thickness of soma; from v30f)
  max_extent    = farthest skeleton pixel from soma within 45 px (longest axis)
  proc_thick    = mean distance-transform along local skeleton (process 粗细)
  fragments     = # disconnected skeleton components in 25 px disk (DAM/beaded)
  is_round      = n_trunks<=1 AND max_extent<14 (amoeboid; NOTE: skeleton-based,
                  so pure process-less round cells are under-counted)

Per image: density (cells/mm^2), per-feature mean AND coefficient-of-variation
(CV = heterogeneity, a disease signal per Huixin), and the 3-level morphology
distribution (round / branched-small / branched-large).
"""
import json
from pathlib import Path
from collections import Counter

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage import filters, morphology

ROOT = Path('/Users/stephenyu/choroid-microglia/project_A_morphology')
RAW = ROOT / 'data/raw'
V29 = ROOT / 'experiments/v29_short_spur_audit'
V30F = ROOT / 'experiments/v30f_trunk_gate'
OUT = ROOT / 'experiments/wt_het_features'
PIXEL_UM = 0.207

WT = ['F_WT_2']
HET = ['F_HET_1', 'F_HET_3']


def find_raw(stem):
    for p in RAW.glob('*.tif'):
        if stem in p.name:
            return p
    raise FileNotFoundError(stem)


def normalize(img, p_lo=1.0, p_hi=99.5):
    lo, hi = np.percentile(img, [p_lo, p_hi])
    return np.clip((img - lo) / (hi - lo + 1e-9), 0, 1)


def make_binary_dist(raw):
    norm = normalize(raw)
    smooth = filters.gaussian(norm, sigma=1.0)
    thr = filters.threshold_otsu(smooth) * 0.7
    binary = smooth > thr
    binary = morphology.binary_closing(binary, morphology.disk(2))
    dist = ndi.distance_transform_edt(binary)
    dist_s = filters.gaussian(dist, sigma=1.0)
    return binary, dist_s


def cell_features(stem):
    raw = tifffile.imread(find_raw(stem)).astype(np.float32)
    binary, dist_s = make_binary_dist(raw)
    skel = np.load(V29 / f'{stem}_skel_pruned.npy').astype(bool)
    H, W = skel.shape
    cells = json.loads((V30F / f'{stem}_trunk_metrics_v30f.json').read_text())

    fg_mm2 = float(binary.sum()) * (PIXEL_UM ** 2) / 1e6

    R_EXT = 45     # extent window
    R_LOC = 25     # local-stats disk
    yy_e, xx_e = np.ogrid[-R_EXT:R_EXT + 1, -R_EXT:R_EXT + 1]
    ext_disk = (yy_e ** 2 + xx_e ** 2) <= R_EXT ** 2
    yy_l, xx_l = np.ogrid[-R_LOC:R_LOC + 1, -R_LOC:R_LOC + 1]
    loc_disk = (yy_l ** 2 + xx_l ** 2) <= R_LOC ** 2

    out = []
    for c in cells:
        yc, xc = int(c['yc']), int(c['xc'])
        # --- max extent (longest reach of skeleton from soma) ---
        y0, y1 = max(0, yc - R_EXT), min(H, yc + R_EXT + 1)
        x0, x1 = max(0, xc - R_EXT), min(W, xc + R_EXT + 1)
        sk = skel[y0:y1, x0:x1]
        dyy = (np.arange(y0, y1) - yc)[:, None]
        dxx = (np.arange(x0, x1) - xc)[None, :]
        rad = np.sqrt(dyy ** 2 + dxx ** 2)
        sk_in = sk & (rad <= R_EXT)
        max_extent = float(rad[sk_in].max()) if sk_in.any() else 0.0

        # --- local disk for fragmentation + thickness ---
        yl0, yl1 = max(0, yc - R_LOC), min(H, yc + R_LOC + 1)
        xl0, xl1 = max(0, xc - R_LOC), min(W, xc + R_LOC + 1)
        skl = skel[yl0:yl1, xl0:xl1]
        ry = (np.arange(yl0, yl1) - yc)[:, None]
        rx = (np.arange(xl0, xl1) - xc)[None, :]
        diskmask = (ry ** 2 + rx ** 2) <= R_LOC ** 2
        skl_d = skl & diskmask
        # fragmentation = # disconnected skeleton components in the disk
        _, n_frag = ndi.label(skl_d, structure=np.ones((3, 3)))
        # process thickness = mean dist transform over local skeleton
        proc_thick = float(dist_s[yl0:yl1, xl0:xl1][skl_d].mean()) if skl_d.any() else 0.0

        primary = c['n_trunks']
        secondary = c['n_local_branches']
        skel_len = c['skel_len_local']
        soma_r = c.get('r_soma', 0.0)
        is_round = (primary <= 1) and (max_extent < 14)

        out.append(dict(
            primary=int(primary), secondary=int(secondary),
            skel_len=int(skel_len), soma_radius=float(soma_r),
            max_extent=round(max_extent, 1), proc_thick=round(proc_thick, 2),
            fragments=int(n_frag), is_round=bool(is_round),
        ))
    return out, fg_mm2


def morpho_tier(c, ext_median):
    if c['is_round']:
        return 'round/amoeboid'
    if c['max_extent'] < ext_median:
        return 'branched-small'
    return 'branched-large'


def cv(x):
    x = np.asarray(x, float)
    m = x.mean()
    return float(x.std() / m) if m else 0.0


def summarize(stem, feats, fg_mm2, ext_median):
    n = len(feats)
    density = n / fg_mm2
    tiers = Counter(morpho_tier(c, ext_median) for c in feats)
    F = lambda k: np.array([c[k] for c in feats], float)
    return dict(
        stem=stem, n=n, density=round(density, 0),
        pct_round=round(100 * tiers['round/amoeboid'] / n, 1),
        pct_small=round(100 * tiers['branched-small'] / n, 1),
        pct_large=round(100 * tiers['branched-large'] / n, 1),
        mean_primary=round(F('primary').mean(), 2),
        mean_secondary=round(F('secondary').mean(), 2),
        mean_extent=round(F('max_extent').mean(), 1),
        mean_thick=round(F('proc_thick').mean(), 2),
        mean_frag=round(F('fragments').mean(), 2),
        # heterogeneity (CV) — Huixin: disease tissue is more diverse
        cv_extent=round(cv(F('max_extent')), 3),
        cv_secondary=round(cv(F('secondary')), 3),
        cv_primary=round(cv(F('primary')), 3),
    )


def main():
    OUT.mkdir(exist_ok=True)
    raw_feats = {}
    fg = {}
    for stem in WT + HET:
        feats, fg_mm2 = cell_features(stem)
        raw_feats[stem] = feats
        fg[stem] = fg_mm2
        (OUT / f'{stem}_features.json').write_text(json.dumps(feats))

    # global extent median (consistent cutoff across ALL images — Huixin's rule)
    all_ext = np.array([c['max_extent'] for s in WT + HET for c in raw_feats[s]])
    ext_median = float(np.median(all_ext))

    summ = {s: summarize(s, raw_feats[s], fg[s], ext_median) for s in WT + HET}
    (OUT / 'summary.json').write_text(json.dumps(summ, indent=1))

    print(f'(consistent longest-axis cutoff = global median extent = {ext_median:.1f} px '
          f'= {ext_median*PIXEL_UM:.1f} um)\n')

    cols = ['n', 'density', 'pct_round', 'pct_small', 'pct_large',
            'mean_primary', 'mean_secondary', 'mean_extent', 'mean_thick',
            'mean_frag', 'cv_extent', 'cv_secondary']
    labels = {'n': 'cells', 'density': 'cells/mm2', 'pct_round': '%round',
              'pct_small': '%br-small', 'pct_large': '%br-large',
              'mean_primary': 'primary', 'mean_secondary': 'secondary',
              'mean_extent': 'extent_px', 'mean_thick': 'thick_px',
              'mean_frag': 'fragments', 'cv_extent': 'CV_extent',
              'cv_secondary': 'CV_second'}
    print(f'{"feature":12} {"WT_2":>9} {"HET_1":>9} {"HET_3":>9} {"HETavg":>9} {"HETvsWT":>10}')
    for col in cols:
        wt = summ['F_WT_2'][col]
        h1 = summ['F_HET_1'][col]; h3 = summ['F_HET_3'][col]
        hm = (h1 + h3) / 2
        if wt:
            delta = f'{100*(hm-wt)/wt:+.1f}%'
        else:
            delta = 'n/a'
        print(f'{labels[col]:12} {wt:>9} {h1:>9} {h3:>9} {hm:>9.1f} {delta:>10}')

    print('\nReading guide:')
    print('  %round under-counts (skeleton blind to process-less cells)')
    print('  CV_* = heterogeneity; HET higher CV would support the disease-')
    print('  diversity signal Huixin described.')


if __name__ == '__main__':
    main()
