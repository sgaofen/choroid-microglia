"""
Round / amoeboid cell detector — Stephen's logic (2026-05-26):
  a round cell = "a blob of mass with NO topology coming out of it".
  i.e. a compact bright soma-sized blob that does NOT radiate processes.

This is the INVERSE of the skeleton-based detector (which finds cells WITH
processes) and fixes the 0%-round blind spot. It is also better than v30g,
which required isolation from the skeleton and so caught vessel fragments —
here we make NO isolation requirement; we require mass + absence-of-topology.

Method:
  1. candidate soma blobs = local maxima of the distance transform (fat points)
  2. MASS test    : soma_radius >= MIN_R, bright, roundish (compact local fg)
  3. NO-TOPOLOGY  : annulus trunk count <= 1  AND  branch points <= 1 in 18 px
                    AND skeleton reach short (no long process radiating out)
  4. NMS so each blob is counted once; skip blobs already explained by a
     process-bearing v30f cell (those are handled elsewhere).
"""
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage import filters, morphology, measure
from skimage.feature import peak_local_max

ROOT = Path('/Users/stephenyu/choroid-microglia/project_A_morphology')
RAW = ROOT / 'data/raw'
V29 = ROOT / 'experiments/v29_short_spur_audit'
V30F = ROOT / 'experiments/v30f_trunk_gate'
OUT = ROOT / 'experiments/round_cell_detector'
PIXEL_UM = 0.207

MIN_SOMA_R = 3.0      # distance-transform value = soma half-width
MIN_INTENS = 0.20     # normalized intensity at center
NMS_RADIUS = 8
TOPO_RADIUS = 18      # look for processes/branches within this
MAX_TRUNKS_ROUND = 1  # round cell radiates <= 1 process
MAX_BP_ROUND = 1      # <= 1 branch point nearby
MAX_REACH_ROUND = 16  # skeleton reaches no farther than this (px)
ROUND_ECC_MAX = 0.80  # local fg blob must be roundish


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
    return binary, dist_s, norm


def skel_degree(skel):
    k = np.ones((3, 3), np.uint8); k[1, 1] = 0
    return ndi.convolve(skel.astype(np.uint8), k, mode='constant') * skel


def annulus_trunks(skel, y, x, r_soma):
    r_in = max(3, int(round(r_soma)) + 2); r_out = r_in + 6
    H, W = skel.shape
    y0, y1 = max(0, y - r_out - 1), min(H, y + r_out + 2)
    x0, x1 = max(0, x - r_out - 1), min(W, x + r_out + 2)
    patch = skel[y0:y1, x0:x1]
    yy, xx = np.indices(patch.shape)
    rsq = (yy - (y - y0)) ** 2 + (xx - (x - x0)) ** 2
    ann = (rsq >= r_in * r_in) & (rsq <= r_out * r_out) & patch
    if not ann.any():
        return 0
    _, n = ndi.label(ann, structure=np.ones((3, 3)))
    return int(n)


def detect(stem):
    raw = tifffile.imread(find_raw(stem)).astype(np.float32)
    binary, dist_s, norm = make_binary_dist(raw)
    skel = np.load(V29 / f'{stem}_skel_pruned.npy').astype(bool)
    deg = skel_degree(skel)
    bp_mask = skel & (deg >= 3)
    H, W = skel.shape

    # exclusion: process-bearing v30f cells (>=2 trunks) are handled already
    v30f = json.loads((V30F / f'{stem}_trunk_metrics_v30f.json').read_text())
    proc_yc = np.array([c['yc'] for c in v30f if c['n_trunks'] >= 2])
    proc_xc = np.array([c['xc'] for c in v30f if c['n_trunks'] >= 2])

    peaks = peak_local_max(dist_s, min_distance=NMS_RADIUS,
                           threshold_abs=MIN_SOMA_R)
    lbl = measure.label(binary)
    props = {p.label: p for p in measure.regionprops(lbl)}

    round_cells = []
    reasons = {'dim': 0, 'has_trunks': 0, 'has_branches': 0, 'long_reach': 0,
               'not_round': 0, 'near_proccell': 0, 'kept': 0}
    yy_t, xx_t = np.ogrid[-TOPO_RADIUS:TOPO_RADIUS + 1, -TOPO_RADIUS:TOPO_RADIUS + 1]
    tdisk = (yy_t ** 2 + xx_t ** 2) <= TOPO_RADIUS ** 2

    for (y, x) in peaks:
        d = float(dist_s[y, x])
        if norm[y, x] < MIN_INTENS:
            reasons['dim'] += 1; continue
        # NO-TOPOLOGY tests
        if annulus_trunks(skel, y, x, d) > MAX_TRUNKS_ROUND:
            reasons['has_trunks'] += 1; continue
        y0, y1 = max(0, y - TOPO_RADIUS), min(H, y + TOPO_RADIUS + 1)
        x0, x1 = max(0, x - TOPO_RADIUS), min(W, x + TOPO_RADIUS + 1)
        td = tdisk[(y0 - (y - TOPO_RADIUS)):(y1 - (y - TOPO_RADIUS)),
                   (x0 - (x - TOPO_RADIUS)):(x1 - (x - TOPO_RADIUS))]
        bp_here = int((bp_mask[y0:y1, x0:x1] & td).sum())
        if bp_here > MAX_BP_ROUND:
            reasons['has_branches'] += 1; continue
        sk_here = skel[y0:y1, x0:x1] & td
        if sk_here.any():
            ys, xs = np.where(sk_here)
            reach = np.sqrt((ys + y0 - y) ** 2 + (xs + x0 - x) ** 2).max()
            if reach > MAX_REACH_ROUND:
                reasons['long_reach'] += 1; continue
        # MASS test: roundish local fg blob
        comp_label = lbl[y, x]
        if comp_label > 0:
            pr = props[comp_label]
            if pr.eccentricity > ROUND_ECC_MAX and pr.area > 60:
                reasons['not_round'] += 1; continue
        # not already a process-bearing cell
        if len(proc_yc):
            if ((proc_yc - y) ** 2 + (proc_xc - x) ** 2).min() < NMS_RADIUS ** 2:
                reasons['near_proccell'] += 1; continue
        round_cells.append(dict(yc=int(y), xc=int(x), soma_r=round(d, 1),
                                intensity=round(float(norm[y, x]), 3)))
        reasons['kept'] += 1

    return round_cells, binary


def main():
    OUT.mkdir(exist_ok=True)
    summ = {}
    for stem in ['F_WT_2', 'F_HET_1', 'F_HET_3']:
        rc, binary = detect(stem)
        (OUT / f'{stem}_round_cells.json').write_text(json.dumps(rc, indent=1))
        fg_mm2 = float(binary.sum()) * (PIXEL_UM ** 2) / 1e6
        v30f = json.loads((V30F / f'{stem}_trunk_metrics_v30f.json').read_text())
        n_proc = len(v30f)
        n_round = len(rc)
        total = n_proc + n_round
        summ[stem] = dict(process_bearing=n_proc, round=n_round, total=total,
                          pct_round=round(100 * n_round / total, 1),
                          round_per_mm2=round(n_round / fg_mm2, 0))
        print(f'{stem}: process-bearing={n_proc}  round={n_round}  '
              f'total={total}  %round={summ[stem]["pct_round"]}  '
              f'round/mm2={summ[stem]["round_per_mm2"]:.0f}')

    print('\n=== %round WT vs HET (was 0% with skeleton-only) ===')
    wt = summ['F_WT_2']['pct_round']
    hm = (summ['F_HET_1']['pct_round'] + summ['F_HET_3']['pct_round']) / 2
    print(f'  WT_2={wt}%   HET_1={summ["F_HET_1"]["pct_round"]}%   '
          f'HET_3={summ["F_HET_3"]["pct_round"]}%   HETavg={hm:.1f}%   '
          f'delta={hm-wt:+.1f}pp')
    (OUT / 'summary.json').write_text(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
