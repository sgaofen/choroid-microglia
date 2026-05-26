"""
v31: skeleton-first cell detection.

For each connected component of the (pruned) skeleton:
  1) find soma center(s) as distance-transform peaks ALONG the skeleton
     (NMS with PEAK_NMS_RADIUS, accept if dist_at_peak >= MIN_SOMA_RADIUS)
  2) reject components with peak / median dist < MIN_PROMINENCE (vessels have
     ~uniform width — no prominent fat point)
  3) reject components with zero endpoints AND zero branch points
     (straight line piece, no morphology)
  4) attribute each skeleton pixel of a multi-soma component to its nearest
     soma via geodesic BFS, then count per-cell endpoints / branches / length

Output per cell: yc, xc, soma_radius, intensity, n_endpoints, n_branches,
skel_length, n_trunks (annulus method), component_id.
"""
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage import filters, morphology

ROOT = Path('/Users/stephenyu/Documents/choroid-microglia/project_A_morphology')
RAW = ROOT / 'data/raw'
V29 = ROOT / 'experiments/v29_short_spur_audit'
OUT = ROOT / 'experiments/v31_skeleton_first'

# --- thresholds ---
MIN_COMP_SKEL_PX = 8         # tiny components → noise
MIN_SOMA_RADIUS  = 4.0       # half-width at soma center via dist transform
                              # microglia soma typical r ≈ 4-7 px; processes r ≈ 1-3 px
MIN_SOMA_INTENS  = 0.20      # normalized intensity at soma center
PEAK_NMS_RADIUS  = 12        # px non-max-suppression for multi-soma in one comp
MIN_PROMINENCE   = 1.50      # peak_dist / median_dist along component
                              # real soma is much fatter than its processes
MIN_COMP_FOR_PROMINENCE = 25
MIN_ENDPOINTS_PER_CELL = 1   # post-BFS attribution — every cell must claim >= 1 endpoint

# Annulus trunk counter (same as v30f)
TRUNK_PAD_IN = 2
TRUNK_WIDTH  = 6


def find_raw(stem):
    for p in RAW.glob('*.tif'):
        if stem in p.name:
            return p
    raise FileNotFoundError(stem)


def normalize(img, p_lo=1.0, p_hi=99.5):
    lo, hi = np.percentile(img, [p_lo, p_hi])
    return np.clip((img - lo) / (hi - lo + 1e-9), 0, 1)


def make_binary_and_dist(raw):
    norm = normalize(raw)
    smooth = filters.gaussian(norm, sigma=1.0)
    thr = filters.threshold_otsu(smooth) * 0.7
    binary = smooth > thr
    binary = morphology.binary_closing(binary, morphology.disk(2))
    dist = ndi.distance_transform_edt(binary)
    dist_s = filters.gaussian(dist, sigma=1.0)
    return binary, dist_s, norm


def skel_degree(skel):
    k = np.ones((3, 3), dtype=np.uint8); k[1, 1] = 0
    return ndi.convolve(skel.astype(np.uint8), k, mode='constant') * skel


def find_soma_peaks(comp_pixels, dist_s, raw_norm):
    """NMS over skeleton component pixels, ranked by dist_s value.
    Returns list of (y, x, dist_value, intensity) tuples — at most one per
    PEAK_NMS_RADIUS neighborhood, all with dist >= MIN_SOMA_RADIUS."""
    ys = np.array([p[0] for p in comp_pixels])
    xs = np.array([p[1] for p in comp_pixels])
    dvals = dist_s[ys, xs]
    order = np.argsort(-dvals)
    peaks = []
    for idx in order:
        d = float(dvals[idx])
        if d < MIN_SOMA_RADIUS:
            break
        y, x = int(ys[idx]), int(xs[idx])
        too_close = any((y - py) ** 2 + (x - px) ** 2 < PEAK_NMS_RADIUS ** 2
                        for (py, px, _, _) in peaks)
        if too_close:
            continue
        ig = float(raw_norm[y, x])
        if ig < MIN_SOMA_INTENS:
            continue
        peaks.append((y, x, d, ig))
    return peaks


def neighbors8(y, x, H, W):
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0: continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                yield ny, nx


def attribute_pixels_to_somas(comp_pixels_set, somas_in_comp, skel, deg):
    """Multi-source BFS along skeleton. Returns dict pixel -> soma_index.
    All pixels in comp_pixels_set are assigned to their nearest soma index."""
    H, W = skel.shape
    src = {}
    q = deque()
    for i, (sy, sx, _, _) in enumerate(somas_in_comp):
        src[(sy, sx)] = i
        q.append((sy, sx))
    while q:
        y, x = q.popleft()
        i = src[(y, x)]
        for ny, nx in neighbors8(y, x, H, W):
            if (ny, nx) not in comp_pixels_set: continue
            if (ny, nx) in src: continue
            src[(ny, nx)] = i
            q.append((ny, nx))
    return src


def annulus_trunk_count(skel, yc, xc, soma_radius):
    r_in = max(3, int(round(soma_radius)) + TRUNK_PAD_IN)
    r_out = r_in + TRUNK_WIDTH
    H, W = skel.shape
    y0, y1 = max(0, yc - r_out - 1), min(H, yc + r_out + 2)
    x0, x1 = max(0, xc - r_out - 1), min(W, xc + r_out + 2)
    patch = skel[y0:y1, x0:x1]
    yy, xx = np.indices(patch.shape)
    cy = yc - y0; cx = xc - x0
    rsq = (yy - cy) ** 2 + (xx - cx) ** 2
    in_ann = (rsq >= r_in * r_in) & (rsq <= r_out * r_out) & patch
    if not in_ann.any():
        return 0, r_in, r_out
    _, n = ndi.label(in_ann, structure=np.ones((3, 3)))
    return int(n), int(r_in), int(r_out)


def run_image(stem):
    print(f'\n=== {stem} ===')
    raw = tifffile.imread(find_raw(stem)).astype(np.float32)
    skel = np.load(V29 / f'{stem}_skel_pruned.npy').astype(bool)
    binary, dist_s, raw_norm = make_binary_and_dist(raw)

    deg = skel_degree(skel)
    endpoint_mask = skel & (deg == 1)
    branch_mask = skel & (deg >= 3)

    # Connected components
    comp_labels, n_comp = ndi.label(skel, structure=np.ones((3, 3)))
    print(f'  {n_comp} skeleton components')

    reasons = defaultdict(int)
    all_cells = []

    for cid in range(1, n_comp + 1):
        comp_mask = comp_labels == cid
        ys, xs = np.where(comp_mask)
        comp_size = len(ys)
        if comp_size < MIN_COMP_SKEL_PX:
            reasons['rej_too_small'] += 1; continue

        n_ep = int(endpoint_mask[ys, xs].sum())
        n_br = int(branch_mask[ys, xs].sum())
        if n_ep + n_br == 0:
            reasons['rej_no_morphology'] += 1; continue

        # Find peak(s)
        comp_pixels = list(zip(ys.tolist(), xs.tolist()))
        peaks = find_soma_peaks(comp_pixels, dist_s, raw_norm)
        if not peaks:
            reasons['rej_no_fat_point'] += 1; continue

        # Prominence check on the BEST peak — vessels have ~uniform width
        if comp_size >= MIN_COMP_FOR_PROMINENCE:
            median_d = float(np.median(dist_s[ys, xs]))
            best_peak_d = peaks[0][2]
            if median_d > 0 and (best_peak_d / median_d) < MIN_PROMINENCE:
                reasons['rej_low_prominence'] += 1; continue

        # Per-cell attribution by multi-source BFS
        comp_pixels_set = set(comp_pixels)
        attr = attribute_pixels_to_somas(comp_pixels_set, peaks, skel, deg)

        # Count per-cell endpoints / branches / length
        per_cell = [dict(n_endpoints=0, n_branches=0, skel_length=0)
                    for _ in peaks]
        for (py, px), si in attr.items():
            per_cell[si]['skel_length'] += 1
            d = deg[py, px]
            if d == 1: per_cell[si]['n_endpoints'] += 1
            elif d >= 3: per_cell[si]['n_branches'] += 1

        for (i, (sy, sx, sd, si)) in enumerate(peaks):
            # post-BFS gate: every cell must claim at least 1 endpoint
            # (otherwise it's a spurious peak on a process joint)
            if per_cell[i]['n_endpoints'] < MIN_ENDPOINTS_PER_CELL:
                reasons['rej_no_attributed_endpoint'] += 1; continue
            n_tr, r_in, r_out = annulus_trunk_count(skel, sy, sx, sd)
            all_cells.append(dict(
                yc=int(sy), xc=int(sx),
                soma_radius=float(sd),
                intensity=float(si),
                component_id=int(cid),
                comp_skel_px=int(comp_size),
                n_endpoints=int(per_cell[i]['n_endpoints']),
                n_branches=int(per_cell[i]['n_branches']),
                skel_length=int(per_cell[i]['skel_length']),
                n_trunks=int(n_tr),
                trunk_r_in=int(r_in), trunk_r_out=int(r_out),
                somas_in_comp=len(peaks),
            ))

    print(f'  cells found: {len(all_cells)}')
    print(f'  rejection trace: {dict(reasons)}')
    distr_tr = np.bincount(np.clip([c['n_trunks'] for c in all_cells], 0, 10))
    distr_ep = np.bincount(np.clip([c['n_endpoints'] for c in all_cells], 0, 15))
    print(f'  n_trunks dist:    {distr_tr.tolist()}')
    print(f'  n_endpoints dist: {distr_ep.tolist()}')

    OUT.mkdir(exist_ok=True)
    (OUT / f'{stem}_cells_v31.json').write_text(json.dumps(all_cells, indent=1))
    print(f'  wrote {stem}_cells_v31.json')
    return all_cells


if __name__ == '__main__':
    summary = {}
    for stem in ['F_WT_2', 'F_HET_1', 'F_HET_3']:
        cells = run_image(stem)
        summary[stem] = len(cells)
    print('\n=== v31 summary ===')
    for stem, n in summary.items():
        print(f'  {stem}: {n} cells')
