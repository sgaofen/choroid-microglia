"""
v30b: full architecture.

For each seed (h-max proposal from v27/v29):
    somaness_score combining
        compact (solidity * minor/major axis ratio)
        blobness vs tubeness (Hessian at scale 2.5)
        radius prominence on the local skeleton
        sholl-like exit directions
    -> classify strong / weak / process_peak

Build skeleton graph. For pairs of strong/weak seeds in the same
foreground component, do a pairwise neck test on the skeleton path.

  weak seed connected to strong via mostly-thin path with no neck
      -> demote to process_peak, attribute to the strong seed
  two strong seeds with a real neck (low neck_ratio + thin segment)
      -> keep both as separate cells
  two strong seeds without a neck
      -> mark ambiguous; keep both but flag

Accepted = strong + weak that survived merging.
Endpoint attribution: BFS from each endpoint to nearest accepted soma collar.
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
OUT = ROOT / 'experiments/v30e_merge_fixes'

# --- thresholds ---
SCORE_STRONG = 0.40
SCORE_WEAK   = 0.22
HESSIAN_SIGMA = 2.5

# v30c additions
SNAP_RADIUS = 5            # snap centroid to local intensity peak in this radius
BLOB_HARD_MIN = 0.20       # blob < this -> hard reject regardless of score
FG_DENSITY_MIN = 0.45      # foreground fraction in 5-px disk around center; reject if lower

# v30d additions: local compact-component gate (Codex top fix)
# The local box must be SMALL — just large enough to cover the soma, not the
# processes — otherwise every microglia gets rejected because its processes
# extend past the box boundary and the connected component blows up.
LOCAL_BOX_HALF = 6         # 13x13 px box, ≈soma footprint, excludes radiating processes
LOCAL_COMP_ECC_MAX = 0.92  # local fg component eccentricity must be below this
LOCAL_COMP_SOL_MIN = 0.50  # local fg component solidity must be above this
LOCAL_COMP_AREA_MIN = 6    # too tiny -> debris
LOCAL_COMP_AREA_MAX = 200  # very loose; main job is to fire on debris/glow

# v30d round 2: stricter "strong" requirements informed by Codex's
# process-count critique and the audit:
# - rprom < 1.0 -> the seed is not locally thicker than its skeleton
#   neighborhood -> probably on uniform tube/vessel, demote to low-confidence
# - n_dirs < 3 -> doesn't look like a 3-process+ microglia, demote
STRONG_RPROM_MIN = 1.0
STRONG_N_DIRS_MIN = 3

# v30e: merging-bug fixes from Codex A code review.
# Two centers within this Euclidean distance are considered duplicate
# detections of the same soma (1 soma diameter ~= 6-10 px).
# Outside this distance, only merge with explicit evidence (thick bridge AND
# thin neck both) — otherwise keep both as ambiguous neighbors.
DUPLICATE_DISTANCE_PX = 8

# pairwise neck test thresholds
NECK_RATIO_DEMOTE = 0.60   # weak vs strong: thin path -> demote weak
NECK_RATIO_SPLIT  = 0.55   # both strong: low neck -> separate cells
THIN_LEN_REQUIRED = 5      # consecutive thin pixels to call a neck

GEODESIC_LIMIT = 80        # px, max pairwise geodesic to consider
PAIR_TRIM = 4              # ignore PAIR_TRIM px near each seed in neck stats


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
    return np.clip((img - lo) / (hi - lo + 1e-9), 0, 1)


def make_binary_and_dist(raw):
    norm = normalize(raw)
    smooth = filters.gaussian(norm, sigma=1.0)
    thr = filters.threshold_otsu(smooth) * 0.7
    binary = smooth > thr
    binary = morphology.binary_closing(binary, morphology.disk(2))
    dist = ndi.distance_transform_edt(binary)
    dist_s = filters.gaussian(dist, sigma=1.0)
    return binary, smooth, dist_s


def skeleton_exit_dirs(skel, yc, xc, r_in=5, r_out=14, n_bins=12):
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
    H, W = skel.shape
    y0, y1 = max(0, yc - radius), min(H, yc + radius + 1)
    x0, x1 = max(0, xc - radius), min(W, xc + radius + 1)
    patch = skel[y0:y1, x0:x1]
    if not patch.any():
        return 0
    k = np.ones((3, 3), dtype=np.uint8); k[1, 1] = 0
    nb = ndi.convolve(patch.astype(np.uint8), k, mode='constant', cval=0)
    return int(nb[patch].max())


def snap_to_local_peak(smooth, yc, xc, radius=SNAP_RADIUS):
    """Move (yc, xc) to the brightest pixel in a small window."""
    H, W = smooth.shape
    y0, y1 = max(0, yc - radius), min(H, yc + radius + 1)
    x0, x1 = max(0, xc - radius), min(W, xc + radius + 1)
    patch = smooth[y0:y1, x0:x1]
    if patch.size == 0:
        return yc, xc
    i = int(np.argmax(patch))
    dy, dx = np.unravel_index(i, patch.shape)
    return int(y0 + dy), int(x0 + dx)


def fg_density_around(binary, yc, xc, radius=5):
    """Fraction of foreground pixels in disk around (yc, xc)."""
    H, W = binary.shape
    y0, y1 = max(0, yc - radius), min(H, yc + radius + 1)
    x0, x1 = max(0, xc - radius), min(W, xc + radius + 1)
    patch = binary[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    yy, xx = np.indices(patch.shape)
    cy = yc - y0; cx = xc - x0
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius * radius
    if mask.sum() == 0:
        return 0.0
    return float(patch[mask].sum()) / float(mask.sum())


def local_component_stats(binary, yc, xc, half=LOCAL_BOX_HALF):
    """Take the connected fg component containing (yc, xc) within a local
    box, return (area, eccentricity, solidity, touches_border).

    A real soma in a local box should be a small compact island.
    A vessel/meshwork junction will show a large component that touches
    the box border (because it extends out).
    A beaded process will show an elongated component (high eccentricity).
    """
    H, W = binary.shape
    y0, y1 = max(0, yc - half), min(H, yc + half + 1)
    x0, x1 = max(0, xc - half), min(W, xc + half + 1)
    patch = binary[y0:y1, x0:x1].copy()
    if patch.size == 0 or not patch[yc - y0, xc - x0]:
        return dict(area=0, ecc=1.0, sol=0.0, touches_border=True)
    lab = measure.label(patch, connectivity=2)
    my_lab = int(lab[yc - y0, xc - x0])
    if my_lab == 0:
        return dict(area=0, ecc=1.0, sol=0.0, touches_border=True)
    comp = (lab == my_lab)
    touches_border = bool(
        comp[0, :].any() or comp[-1, :].any()
        or comp[:, 0].any() or comp[:, -1].any())
    props = measure.regionprops(comp.astype(np.uint8))
    if not props:
        return dict(area=0, ecc=1.0, sol=0.0, touches_border=touches_border)
    p = props[0]
    return dict(area=int(p.area),
                ecc=float(p.eccentricity),
                sol=float(p.solidity),
                touches_border=touches_border)


def hessian_blob_tube(smooth, yc, xc, sigma=HESSIAN_SIGMA, half=10):
    H, W = smooth.shape
    y0, y1 = max(0, yc - half), min(H, yc + half + 1)
    x0, x1 = max(0, xc - half), min(W, xc + half + 1)
    patch = smooth[y0:y1, x0:x1]
    if patch.shape[0] < 5 or patch.shape[1] < 5:
        return 0.5, 0.5
    Hyy, Hxy, Hxx = hessian_matrix(patch, sigma=sigma, order='rc',
                                   use_gaussian_derivatives=True)
    lam1, lam2 = hessian_matrix_eigvals([Hyy, Hxy, Hxx])
    cy = min(max(yc - y0, 0), patch.shape[0] - 1)
    cx = min(max(xc - x0, 0), patch.shape[1] - 1)
    a, b = lam1[cy, cx], lam2[cy, cx]
    # Need bright structure on dark bg: both eigenvalues should be negative.
    # If neither is negative enough, treat as ambiguous.
    if max(a, b) > -1e-6:
        return 0.0, 0.0          # neither bright tube nor bright blob
    abs_big = max(abs(a), abs(b))
    abs_small = min(abs(a), abs(b))
    if abs_big < 1e-9:
        return 0.5, 0.5
    blob = abs_small / abs_big   # ∈ [0, 1], close to 1 = isotropic blob
    tube = 1.0 - blob            # ∈ [0, 1], close to 1 = thin tube
    return float(blob), float(tube)


def radius_prominence(skel, dist_s, yc, xc, radius=25):
    """How much thicker is this seed than its local skeleton neighborhood?"""
    H, W = skel.shape
    y0, y1 = max(0, yc - radius), min(H, yc + radius + 1)
    x0, x1 = max(0, xc - radius), min(W, xc + radius + 1)
    patch_skel = skel[y0:y1, x0:x1]
    patch_dist = dist_s[y0:y1, x0:x1]
    if not patch_skel.any():
        return 0.0
    nearby = patch_dist[patch_skel]
    med = float(np.median(nearby))
    mad = float(np.median(np.abs(nearby - med))) + 1e-3
    r_here = float(dist_s[yc, xc])
    return (r_here - med) / mad


def sigmoid(x, scale=1.0):
    x = np.clip(x * scale, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


def sholl_score_from_dirs(n_dirs):
    if n_dirs >= 3: return 1.0
    if n_dirs == 2: return 0.40
    if n_dirs == 1: return 0.20
    return 0.05


def score_seed(prop, smooth, skel, dist_s, binary):
    yc0, xc0 = (int(round(c)) for c in prop.centroid)
    H, W = smooth.shape
    yc0 = min(max(yc0, 0), H - 1); xc0 = min(max(xc0, 0), W - 1)

    # snap to nearest local intensity peak so we score the actual bright spot
    yc, xc = snap_to_local_peak(smooth, yc0, xc0)

    # compactness from regionprops of the core (uses the v27 mask, not the snap point)
    if prop.major_axis_length > 0:
        axis_ratio = prop.minor_axis_length / prop.major_axis_length
        ecc = float(prop.eccentricity)
    else:
        axis_ratio = 1.0; ecc = 0.0
    solidity = float(prop.solidity) if hasattr(prop, 'solidity') else 1.0
    compact = float(axis_ratio) * solidity      # ∈ [0,1]

    blob, tube = hessian_blob_tube(smooth, yc, xc)
    n_dirs = skeleton_exit_dirs(skel, yc, xc)
    deg = skeleton_degree_near(skel, yc, xc)
    rprom = radius_prominence(skel, dist_s, yc, xc)
    sholl = sholl_score_from_dirs(n_dirs)
    fg_dens = fg_density_around(binary, yc, xc)
    lc = local_component_stats(binary, yc, xc)

    score = (
        0.25 * compact +
        0.20 * blob +
        0.20 * sigmoid(rprom, scale=1.0) +
        0.15 * sholl -
        0.20 * tube
    )

    # v30c hard gates: if there's no Hessian blob at all, this is not a cell.
    # Same for empty-ish neighborhoods (mostly background).
    hard_rejected = (blob < BLOB_HARD_MIN) or (fg_dens < FG_DENSITY_MIN)

    # v30d additional gate: in a SMALL local box (≈soma-size), the connected
    # foreground component containing the snapped center should be roundish.
    # Real soma -> compact island within 13x13 box.
    # Beaded process / tube section -> elongated component.
    # Junction -> low solidity (jagged).
    lc_ok = (
        LOCAL_COMP_AREA_MIN <= lc['area'] <= LOCAL_COMP_AREA_MAX and
        lc['ecc'] < LOCAL_COMP_ECC_MAX and
        lc['sol'] > LOCAL_COMP_SOL_MIN
    )
    if not lc_ok:
        hard_rejected = True

    # v30d: strong requires multi-process appearance + local thickness peak.
    # Anything below that bar is at best a weak/Kolmer candidate.
    strong_ok = (n_dirs >= STRONG_N_DIRS_MIN) and (rprom >= STRONG_RPROM_MIN)

    if hard_rejected:
        typ = 'process_peak'
    elif score >= SCORE_STRONG and strong_ok:
        typ = 'strong'
    elif score >= SCORE_WEAK:
        typ = 'weak'
    else:
        typ = 'process_peak'

    return dict(
        label=int(prop.label),
        yc=yc, xc=xc, yc_raw=yc0, xc_raw=xc0,
        area=int(prop.area), eccentricity=ecc, axis_ratio=axis_ratio,
        solidity=solidity, compact=compact,
        blob=blob, tube=tube, fg_dens=float(fg_dens),
        lc_area=int(lc['area']), lc_ecc=float(lc['ecc']),
        lc_sol=float(lc['sol']),
        lc_touches_border=bool(lc['touches_border']),
        n_dirs=int(n_dirs), deg=int(deg),
        rprom=float(rprom), sholl=float(sholl),
        score=float(score), type=typ,
        hard_rejected=bool(hard_rejected),
    )


# --- graph machinery ---

def neighbors8(y, x, H, W):
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0: continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                yield ny, nx


def skel_pix_dict(skel):
    ys, xs = np.where(skel)
    coords = list(zip(ys.tolist(), xs.tolist()))
    cset = set(coords)
    nbrs = {c: [] for c in coords}
    for c in coords:
        for n in neighbors8(c[0], c[1], *skel.shape):
            if n in cset:
                nbrs[c].append(n)
    return coords, nbrs


def nearest_skel_pix(skel, y, x, max_search=10):
    """Return the nearest skeleton pixel to (y,x). None if not found."""
    H, W = skel.shape
    for r in range(0, max_search + 1):
        y0, y1 = max(0, y - r), min(H, y + r + 1)
        x0, x1 = max(0, x - r), min(W, x + r + 1)
        patch = skel[y0:y1, x0:x1]
        if patch.any():
            ys, xs = np.where(patch)
            ys = ys + y0; xs = xs + x0
            d = (ys - y) ** 2 + (xs - x) ** 2
            i = int(np.argmin(d))
            return (int(ys[i]), int(xs[i]))
    return None


def bfs_path(start, end, nbrs):
    """Shortest path (in steps) along skeleton graph. Returns list of coords."""
    if start == end:
        return [start]
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        for n in nbrs[cur]:
            if n in prev: continue
            prev[n] = cur
            if n == end:
                path = [n]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                path.reverse()
                return path
            q.append(n)
    return None


def neck_stats(path, dist_s, trim=PAIR_TRIM):
    """Return (neck_ratio, thin_run_len, geo_len, ra, rb).
    neck_ratio = min middle radius / min(end_radii)."""
    if path is None or len(path) < 2:
        return None
    ra = float(dist_s[path[0]])
    rb = float(dist_s[path[-1]])
    if len(path) <= 2 * trim:
        middle = path[len(path)//2:len(path)//2+1]
    else:
        middle = path[trim:-trim]
    if not middle:
        middle = [path[len(path)//2]]
    radii = np.array([dist_s[p] for p in middle])
    neck = float(radii.min())
    rmin_end = min(ra, rb)
    neck_ratio = neck / (rmin_end + 1e-6)
    # longest consecutive thin run (radii < 0.5 * rmin_end OR < 2.5)
    thin_thr = min(0.5 * rmin_end, 2.5)
    runs = []
    cur = 0
    for r in radii:
        if r < thin_thr:
            cur += 1
        else:
            if cur: runs.append(cur)
            cur = 0
    if cur: runs.append(cur)
    thin_run = max(runs) if runs else 0
    return dict(neck_ratio=float(neck_ratio), thin_run=int(thin_run),
                geo_len=int(len(path)), ra=ra, rb=rb, neck=neck)


def resolve_seeds(seeds_info, skel, dist_s, binary):
    """Apply graph-based pairwise neck logic to demote/merge seeds."""
    coords, nbrs = skel_pix_dict(skel)
    cset = set(coords)
    # snap each seed to a skeleton pixel within 10 px
    for s in seeds_info:
        snap = nearest_skel_pix(skel, s['yc'], s['xc'])
        s['snap'] = snap
    # group by foreground component to limit pair search
    fg_lab = measure.label(binary)
    for s in seeds_info:
        s['comp'] = int(fg_lab[s['yc'], s['xc']])
    comps = defaultdict(list)
    for s in seeds_info:
        if s['comp'] > 0 and s['snap'] is not None and s['type'] != 'process_peak':
            comps[s['comp']].append(s)

    # In each component, iteratively merge weak into strong, then test strong-strong
    for comp_id, group in comps.items():
        # sort by score descending
        group.sort(key=lambda s: -s['score'])

        # Step A: for each weak, test against nearest strong
        strongs = [s for s in group if s['type'] == 'strong']
        weaks = [s for s in group if s['type'] == 'weak']

        if not strongs:
            # v30e fix #3: no strong in component -> KEEP ALL weak seeds as
            # low_confidence_soma. Previously only the best weak was kept,
            # collapsing real Kolmer clusters into a single cell.
            for w in weaks:
                w['type'] = 'low_confidence_soma'
            continue

        for w in weaks:
            # find closest strong by Euclidean (cheap)
            best_pair = None
            for s in strongs:
                if (s['yc']-w['yc'])**2 + (s['xc']-w['xc'])**2 > GEODESIC_LIMIT**2:
                    continue
                path = bfs_path(w['snap'], s['snap'], nbrs)
                if path is None or len(path) > GEODESIC_LIMIT:
                    continue
                stats = neck_stats(path, dist_s)
                if stats is None: continue
                if best_pair is None or stats['geo_len'] < best_pair[1]['geo_len']:
                    best_pair = (s, stats)
            if best_pair is None:
                # no strong within reach -> keep as low_confidence (real isolated cell)
                w['type'] = 'low_confidence_soma'
                continue
            s, stats = best_pair
            # v30e fix #2: weak demotion now also requires a real thin run,
            # not just a thin neck_ratio. Otherwise short thick bridges
            # between two separate small cells were incorrectly killing the
            # weaker one.
            dist_sq = (s['yc']-w['yc'])**2 + (s['xc']-w['xc'])**2
            very_close = dist_sq <= DUPLICATE_DISTANCE_PX ** 2
            thin_path = (stats['neck_ratio'] < NECK_RATIO_DEMOTE
                         and stats['thin_run'] >= THIN_LEN_REQUIRED)
            if very_close or thin_path:
                w['type'] = 'process_peak'
                w['demoted_to'] = s['label']
            else:
                w['type'] = 'low_confidence_soma'

        # Step B: strong-strong pair neck test
        kept_strong = list(strongs)
        changed = True
        while changed:
            changed = False
            for i, a in enumerate(kept_strong):
                for b in kept_strong[i+1:]:
                    if (a['yc']-b['yc'])**2 + (a['xc']-b['xc'])**2 > GEODESIC_LIMIT**2:
                        continue
                    path = bfs_path(a['snap'], b['snap'], nbrs)
                    if path is None or len(path) > GEODESIC_LIMIT:
                        continue
                    stats = neck_stats(path, dist_s)
                    if stats is None: continue
                    has_neck = (stats['neck_ratio'] < NECK_RATIO_SPLIT
                                and stats['thin_run'] >= THIN_LEN_REQUIRED)
                    dist_sq = ((a['yc']-b['yc'])**2 +
                               (a['xc']-b['xc'])**2)
                    very_close = dist_sq <= DUPLICATE_DISTANCE_PX ** 2
                    if very_close and not has_neck:
                        # v30e fix #1a: only merge strong-strong when centers
                        # are within a single soma diameter. This is strong
                        # evidence of duplicate detections of the same cell.
                        drop = a if a['score'] < b['score'] else b
                        keep = b if drop is a else a
                        drop['type'] = 'merged_into_strong'
                        drop['merged_to'] = keep['label']
                        kept_strong.remove(drop)
                        changed = True
                        break
                    elif not has_neck:
                        # v30e fix #1b: no clear neck, but centers are far
                        # enough apart that they could be two real cells in
                        # a dense region. Keep both and flag as ambiguous so
                        # downstream knows the partition is uncertain.
                        a.setdefault('ambiguous_neighbors',
                                     []).append(b['label'])
                        b.setdefault('ambiguous_neighbors',
                                     []).append(a['label'])
                    else:
                        a.setdefault('confirmed_neighbors',
                                     []).append(b['label'])
                        b.setdefault('confirmed_neighbors',
                                     []).append(a['label'])
                if changed:
                    break
    return seeds_info


def accepted_labels(seeds_info):
    keep = {'strong', 'weak', 'low_confidence_soma'}
    return [s['label'] for s in seeds_info if s['type'] in keep]


# --- endpoint attribution ---

def build_collar_label(somas, accepted, collar=3):
    accepted_mask = np.isin(somas, accepted)
    accepted_labelmap = np.where(accepted_mask, somas, 0)
    bin_dil = ndi.binary_dilation(accepted_mask, morphology.disk(collar))
    _, inds = ndi.distance_transform_edt(~accepted_mask, return_indices=True)
    nearest = accepted_labelmap[inds[0], inds[1]]
    return np.where(bin_dil, nearest, 0)


def endpoint_attribution(skel, somas, accepted, collar=3):
    coords, nbrs = skel_pix_dict(skel)
    endpoints = [c for c in coords if len(nbrs[c]) == 1]
    collar_lab = build_collar_label(somas, accepted, collar)

    counts = defaultdict(int)
    for ep in endpoints:
        if collar_lab[ep] > 0:
            counts[int(collar_lab[ep])] += 1; continue
        visited = {ep}; q = deque([ep]); target = None
        while q:
            cur = q.popleft()
            for nb in nbrs[cur]:
                if nb in visited: continue
                if collar_lab[nb] > 0:
                    target = int(collar_lab[nb])
                    q.clear(); break
                visited.add(nb); q.append(nb)
            if target is not None: break
        if target is not None:
            counts[target] += 1
    for lab in accepted: counts.setdefault(int(lab), 0)
    return dict(counts)


def run_image(stem):
    print(f'\n=== {stem} ===')
    raw, skel, somas = load_inputs(stem)
    binary, smooth, dist_s = make_binary_and_dist(raw)

    props = measure.regionprops(somas)
    print(f'  scoring {len(props)} seeds...')
    seeds = [score_seed(p, smooth, skel, dist_s, binary) for p in props]
    n_hard = sum(1 for s in seeds if s['hard_rejected'])
    print(f'  hard-rejected by Hessian/density gate: {n_hard}')

    # initial classification counts
    init = defaultdict(int)
    for s in seeds: init[s['type']] += 1
    print(f'  initial: {dict(init)}')

    print('  resolving graph...')
    seeds = resolve_seeds(seeds, skel, dist_s, binary)

    final = defaultdict(int)
    for s in seeds: final[s['type']] += 1
    print(f'  final  : {dict(final)}')

    accepted = accepted_labels(seeds)
    counts = endpoint_attribution(skel, somas, accepted, collar=3)
    arr = np.array(list(counts.values()))
    n_amb = sum(1 for s in seeds if 'confirmed_neighbors' in s)
    print(f'  accepted cells={len(accepted)}  '
          f'ambiguous_neighbor_pairs={n_amb//2}  '
          f'mean endpoints/cell={arr.mean():.2f}  '
          f'distribution {np.bincount(np.clip(arr, 0, 10))}')

    # save
    safe_seeds = []
    for s in seeds:
        d = dict(s); d.pop('snap', None); safe_seeds.append(d)
    (OUT / f'{stem}_seeds_v30e.json').write_text(json.dumps(safe_seeds, indent=1))
    (OUT / f'{stem}_endpoint_counts_v30e.json').write_text(
        json.dumps({str(k): int(v) for k, v in counts.items()}, indent=1))
    np.save(OUT / f'{stem}_accepted_labels.npy',
            np.array(accepted, dtype=np.int64))
    return seeds, counts


if __name__ == '__main__':
    for stem in ['F_WT_2', 'F_HET_1', 'F_HET_3']:
        run_image(stem)
    print('\ndone.')
