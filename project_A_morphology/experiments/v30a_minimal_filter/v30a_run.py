"""
v30a: minimal somaness filter on top of v29.

Reject h-max seeds that are process-peaks (eccentric tubes) before
counting cells. Re-run BFS endpoint attribution from accepted somas.

Rejection rules (GPT Pro round 3 minimal recipe):
  rule 1: core eccentricity > 0.85 AND skeleton exit dirs <= 2
  rule 2: tubeness > blobness AND skeleton degree near seed <= 2
"""
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage import morphology, filters, measure
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

ROOT = Path('/Users/stephenyu/Documents/choroid-microglia/project_A_morphology')
RAW = ROOT / 'data/raw'
V27 = ROOT / 'experiments/v27_clean_graph'
V29 = ROOT / 'experiments/v29_short_spur_audit'
OUT = ROOT / 'experiments/v30a_minimal_filter'

RAW_FILES = {
    'F_WT_2':  RAW / 'C2-MAX_2023 Nov_F_WT_2 20x tile unstitched #2.tif',
    'F_HET_1': RAW / 'C2-MAX_2023 Nov_F_HET_1 20x tile processed #2.tif',
    'F_HET_3': RAW / 'C2-MAX_2023 Nov_F_HET_3 20x tile processed #2.tif',
}

def find_raw(stem):
    for p in RAW.glob('*.tif'):
        if stem in p.name:
            return p
    raise FileNotFoundError(stem)


def load_inputs(stem):
    raw = tifffile.imread(find_raw(stem)).astype(np.float32)
    skel = np.load(V29 / f'{stem}_skel_pruned.npy')
    somas = np.load(V27 / f'{stem}_soma_cores.npy')
    return raw, skel, somas


def normalize(img, p_lo=1.0, p_hi=99.5):
    lo, hi = np.percentile(img, [p_lo, p_hi])
    out = np.clip((img - lo) / (hi - lo + 1e-9), 0, 1)
    return out


def make_binary(raw):
    norm = normalize(raw)
    smooth = filters.gaussian(norm, sigma=1.0)
    thr = filters.threshold_otsu(smooth) * 0.7
    binary = smooth > thr
    binary = morphology.binary_closing(binary, morphology.disk(2))
    return binary, smooth


def skeleton_exit_dirs(skel, yc, xc, r_in=5, r_out=14, n_bins=12):
    """Count how many angular sectors contain skeleton pixels in ring r_in..r_out."""
    H, W = skel.shape
    y0, y1 = max(0, yc - r_out), min(H, yc + r_out + 1)
    x0, x1 = max(0, xc - r_out), min(W, xc + r_out + 1)
    patch = skel[y0:y1, x0:x1]
    if not patch.any():
        return 0
    ys, xs = np.where(patch)
    ys = ys + y0 - yc
    xs = xs + x0 - xc
    rsq = ys * ys + xs * xs
    mask = (rsq >= r_in * r_in) & (rsq <= r_out * r_out)
    if mask.sum() == 0:
        return 0
    angs = np.arctan2(ys[mask], xs[mask])
    bins = ((angs + np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins
    return len(np.unique(bins))


def skeleton_degree_near(skel, yc, xc, radius=4):
    """Max degree (# neighbors) of skeleton pixels within a small box."""
    H, W = skel.shape
    y0, y1 = max(0, yc - radius), min(H, yc + radius + 1)
    x0, x1 = max(0, xc - radius), min(W, xc + radius + 1)
    patch = skel[y0:y1, x0:x1]
    if not patch.any():
        return 0
    # count neighbors for each skeleton pixel in patch
    k = np.ones((3, 3), dtype=np.uint8)
    k[1, 1] = 0
    nb = ndi.convolve(patch.astype(np.uint8), k, mode='constant', cval=0)
    return int(nb[patch].max())


def local_hessian_blobness(smooth, yc, xc, sigma=2.5, half=8):
    """Return (blobness, tubeness) at one point.

    blobness = |lam1| / |lam2|   (close to 1 = isotropic, blob-like)
    tubeness = 1 - blobness      (close to 1 = anisotropic, tube-like)
    Only meaningful if lam2 is negative (bright structure on dark bg).
    """
    H, W = smooth.shape
    y0, y1 = max(0, yc - half), min(H, yc + half + 1)
    x0, x1 = max(0, xc - half), min(W, xc + half + 1)
    patch = smooth[y0:y1, x0:x1]
    if patch.shape[0] < 5 or patch.shape[1] < 5:
        return 0.5, 0.5
    Hyy, Hxy, Hxx = hessian_matrix(patch, sigma=sigma, order='rc',
                                   use_gaussian_derivatives=True)
    lam1, lam2 = hessian_matrix_eigvals([Hyy, Hxy, Hxx])
    cy = yc - y0
    cx = xc - x0
    cy = min(max(cy, 0), patch.shape[0] - 1)
    cx = min(max(cx, 0), patch.shape[1] - 1)
    l1 = abs(lam1[cy, cx])
    l2 = abs(lam2[cy, cx])
    if l2 < 1e-9:
        return 0.5, 0.5
    blobness = l1 / l2
    tubeness = 1.0 - blobness
    return float(blobness), float(tubeness)


def score_seeds(raw, skel, somas):
    """For each soma label, compute scores and accept/reject flag."""
    binary, smooth = make_binary(raw)
    props = measure.regionprops(somas)
    records = []
    for p in props:
        yc, xc = (int(round(c)) for c in p.centroid)
        ecc = float(p.eccentricity) if p.major_axis_length > 0 else 0.0
        area = int(p.area)
        n_dirs = skeleton_exit_dirs(skel, yc, xc)
        deg = skeleton_degree_near(skel, yc, xc)
        blob, tube = local_hessian_blobness(smooth, yc, xc, sigma=2.5)

        # Minimal recipe rules
        reject_rule1 = (ecc > 0.85) and (n_dirs <= 2)
        reject_rule2 = (tube > blob) and (deg <= 2)
        accepted = not (reject_rule1 or reject_rule2)

        records.append(dict(
            label=int(p.label),
            yc=yc, xc=xc,
            area=area,
            eccentricity=ecc,
            n_exit_dirs=int(n_dirs),
            skel_degree=int(deg),
            blobness=blob,
            tubeness=tube,
            rule1=bool(reject_rule1),
            rule2=bool(reject_rule2),
            accepted=bool(accepted),
        ))
    return records, binary, smooth


def build_skel_graph(skel):
    """Return list of skeleton pixel coords + per-pixel neighbors dict."""
    ys, xs = np.where(skel)
    coords = list(zip(ys.tolist(), xs.tolist()))
    coord_set = set(coords)
    nbrs = {c: [] for c in coords}
    offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for y, x in coords:
        for dy, dx in offsets:
            n = (y+dy, x+dx)
            if n in coord_set:
                nbrs[(y, x)].append(n)
    return coords, nbrs


def endpoint_attribution(skel, somas, accepted_labels, collar=3):
    """For each skeleton endpoint, BFS along skeleton until hitting a dilated
    soma collar; attribute to that soma. Only somas with label in
    accepted_labels are sources/targets. Endpoints failing to reach any
    accepted soma are dropped.
    """
    coords, nbrs = build_skel_graph(skel)
    # endpoints = skeleton pixels with degree 1
    endpoints = [c for c in coords if len(nbrs[c]) == 1]

    # Build collar map: dilate accepted somas
    accepted_mask = np.isin(somas, list(accepted_labels))
    accepted_labelmap = np.where(accepted_mask, somas, 0)
    if collar > 0:
        # for each accepted soma, dilate independently
        # cheap approximation: dilate the labelmap with a disk, then
        # use nearest-label rule via distance transform of background.
        struct = morphology.disk(collar)
        # binary dilation
        bin_dil = ndi.binary_dilation(accepted_mask, structure=struct)
        # For pixels added by dilation, assign nearest accepted-soma label
        # via distance transform with indices
        _, inds = ndi.distance_transform_edt(~accepted_mask, return_indices=True)
        nearest = accepted_labelmap[inds[0], inds[1]]
        collar_label = np.where(bin_dil, nearest, 0)
    else:
        collar_label = accepted_labelmap

    counts = defaultdict(int)
    for ep in endpoints:
        if collar_label[ep] > 0:
            # endpoint already inside collar -> attribute and skip
            counts[int(collar_label[ep])] += 1
            continue
        # BFS along skeleton
        visited = {ep}
        q = deque([ep])
        target = None
        while q:
            cur = q.popleft()
            for nb in nbrs[cur]:
                if nb in visited:
                    continue
                if collar_label[nb] > 0:
                    target = int(collar_label[nb])
                    q.clear()
                    break
                visited.add(nb)
                q.append(nb)
            if target is not None:
                break
        if target is not None:
            counts[target] += 1
    # ensure all accepted somas appear
    for lab in accepted_labels:
        counts.setdefault(int(lab), 0)
    return dict(counts)


def run_image(stem):
    print(f'\n=== {stem} ===')
    raw, skel, somas = load_inputs(stem)
    print(f'  raw {raw.shape}  skel sum {int(skel.sum())}  soma labels {int(somas.max())}')

    records, binary, smooth = score_seeds(raw, skel, somas)
    n_total = len(records)
    n_acc = sum(r['accepted'] for r in records)
    n_rej_r1 = sum(r['rule1'] for r in records)
    n_rej_r2 = sum(r['rule2'] for r in records)
    print(f'  seeds: {n_total} total, {n_acc} accepted, '
          f'{n_total-n_acc} rejected  (rule1={n_rej_r1} rule2={n_rej_r2})')

    accepted_labels = [r['label'] for r in records if r['accepted']]
    counts = endpoint_attribution(skel, somas, accepted_labels, collar=3)

    arr = np.array(list(counts.values()))
    print(f'  endpoints per accepted cell: mean={arr.mean():.2f}  '
          f'median={np.median(arr):.1f}  '
          f'distribution {np.bincount(np.clip(arr,0,10))}')

    # save
    (OUT / f'{stem}_seed_scores.json').write_text(
        json.dumps(records, indent=1))
    (OUT / f'{stem}_endpoint_counts_v30a.json').write_text(
        json.dumps({str(k): int(v) for k, v in counts.items()}, indent=1))
    np.save(OUT / f'{stem}_accepted_labels.npy',
            np.array(accepted_labels, dtype=np.int64))
    return records, counts


if __name__ == '__main__':
    for stem in ['F_WT_2', 'F_HET_1', 'F_HET_3']:
        run_image(stem)
    print('\ndone.')
