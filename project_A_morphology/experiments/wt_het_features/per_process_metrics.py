"""
Per-process length & thickness (Stephen's point 2026-05-26): not aggregate
averages, but "is there a super-LONG process / a super-THICK process".

For each v30f process-bearing cell:
  longest_process_px = max GEODESIC distance soma->tip along the skeleton
                       (within 120 px window; fixes the old 45-px extent cap)
  thickest_process   = max distance-transform value along PROXIMAL process
                       skeleton (geodesic dist soma_r+2 .. soma_r+14), i.e.
                       how thick the fattest process trunk is
  proc_thick_mean    = mean dist-transform over the proximal process ring
Then WT vs HET on the per-cell MAX and the population spread (CV).
"""
import json
from collections import deque
from pathlib import Path

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
R_WIN = 120

WT = ['F_WT_2']; HET = ['F_HET_1', 'F_HET_3']


def find_raw(stem):
    for p in RAW.glob('*.tif'):
        if stem in p.name:
            return p
    raise FileNotFoundError(stem)


def normalize(img, p_lo=1.0, p_hi=99.5):
    lo, hi = np.percentile(img, [p_lo, p_hi])
    return np.clip((img - lo) / (hi - lo + 1e-9), 0, 1)


def make_dist(raw):
    norm = normalize(raw)
    smooth = filters.gaussian(norm, sigma=1.0)
    binary = smooth > filters.threshold_otsu(smooth) * 0.7
    binary = morphology.binary_closing(binary, morphology.disk(2))
    dist = ndi.distance_transform_edt(binary)
    return filters.gaussian(dist, sigma=1.0)


def geodesic_from_soma(skel, yc, xc, R):
    """BFS geodesic distance (px) from the soma's nearest skeleton pixel,
    over skeleton pixels within R. Returns {(y,x): dist}."""
    H, W = skel.shape
    y0, y1 = max(0, yc - R), min(H, yc + R + 1)
    x0, x1 = max(0, xc - R), min(W, xc + R + 1)
    sub = skel[y0:y1, x0:x1]
    if not sub.any():
        return {}, (y0, x0)
    ys, xs = np.where(sub)
    # nearest skeleton pixel to soma center
    d2 = (ys + y0 - yc) ** 2 + (xs + x0 - xc) ** 2
    si = int(np.argmin(d2))
    start = (int(ys[si]) + y0, int(xs[si]) + x0)
    sset = set(zip((ys + y0).tolist(), (xs + x0).tolist()))
    geo = {start: 0.0}
    q = deque([start])
    while q:
        cy, cx = q.popleft()
        d0 = geo[(cy, cx)]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nb = (cy + dy, cx + dx)
                if nb in sset and nb not in geo:
                    geo[nb] = d0 + (1.0 if dy == 0 or dx == 0 else 1.41421)
                    q.append(nb)
    return geo, (y0, x0)


def cell_metrics(stem):
    raw = tifffile.imread(find_raw(stem)).astype(np.float32)
    dist_s = make_dist(raw)
    skel = np.load(V29 / f'{stem}_skel_pruned.npy').astype(bool)
    cells = json.loads((V30F / f'{stem}_trunk_metrics_v30f.json').read_text())

    out = []
    for c in cells:
        yc, xc = int(c['yc']), int(c['xc'])
        soma_r = float(c.get('r_soma', 3.0))
        geo, _ = geodesic_from_soma(skel, yc, xc, R_WIN)
        if not geo:
            out.append(dict(longest=0.0, thickest=0.0, thick_mean=0.0)); continue
        longest = max(geo.values())
        lo, hi = soma_r + 2, soma_r + 14
        ring = [dist_s[p] for p, g in geo.items() if lo <= g <= hi]
        thickest = float(np.max(ring)) if ring else 0.0
        thick_mean = float(np.mean(ring)) if ring else 0.0
        out.append(dict(longest=round(longest, 1),
                        thickest=round(thickest, 2),
                        thick_mean=round(thick_mean, 2)))
    return out


def cv(x):
    x = np.asarray(x, float); m = x.mean()
    return float(x.std() / m) if m else 0.0


def main():
    per = {}
    for stem in WT + HET:
        per[stem] = cell_metrics(stem)
        (OUT / f'{stem}_per_process.json').write_text(json.dumps(per[stem]))

    def agg(stem):
        m = per[stem]
        L = np.array([c['longest'] for c in m])
        T = np.array([c['thickest'] for c in m])
        return dict(mean_longest=round(L.mean(), 1), p90_longest=round(np.percentile(L, 90), 1),
                    cv_longest=round(cv(L), 3), mean_thickest=round(T.mean(), 2),
                    cv_thickest=round(cv(T), 3))

    A = {s: agg(s) for s in WT + HET}
    print(f'{"metric":16}{"WT_2":>9}{"HET_1":>9}{"HET_3":>9}{"HETavg":>9}{"HETvsWT":>10}')
    for k in ['mean_longest', 'p90_longest', 'cv_longest', 'mean_thickest', 'cv_thickest']:
        wt = A['F_WT_2'][k]; h1 = A['F_HET_1'][k]; h3 = A['F_HET_3'][k]; hm = (h1 + h3) / 2
        d = f'{100*(hm-wt)/wt:+.1f}%' if wt else 'n/a'
        print(f'{k:16}{wt:>9}{h1:>9}{h3:>9}{hm:>9.1f}{d:>10}')
    print(f'\n(longest now in px, uncapped within {R_WIN}px window; '
          f'x{PIXEL_UM}=um. old extent was capped at 45px.)')
    (OUT / 'per_process_summary.json').write_text(json.dumps(A, indent=1))


if __name__ == '__main__':
    main()
