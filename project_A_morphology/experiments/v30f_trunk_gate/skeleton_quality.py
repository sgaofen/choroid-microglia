"""
Skeleton quality stats. We need to know whether v29's pruned skeleton is:
  (1) over-pruned (real processes lost)
  (2) under-pruned (noise spurs left)
  (3) broken (process gaps mid-line)
across the 3 images.

Metrics:
- Skeleton density per image (px/total_px) and per fg mask (px/fg_px)
- Endpoint density per image (count / fg_px)
- Branch point density (count / fg_px)
- Short-spur fraction (endpoint-rooted branches of len < 5)
- Mean branch length (segments between endpoints/junctions)
- Coverage of the binary fg: dist_transform of skel — what % of fg pixels
  are within K px of a skeleton pixel? Low % means skeleton has gaps.

Output a small table + per-image JSON.
"""
import json
from pathlib import Path
from collections import deque

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage import filters, morphology

ROOT = Path('/Users/stephenyu/Documents/choroid-microglia/project_A_morphology')
RAW = ROOT / 'data/raw'
V29 = ROOT / 'experiments/v29_short_spur_audit'
V30F = ROOT / 'experiments/v30f_trunk_gate'


def find_raw(stem):
    for p in RAW.glob('*.tif'):
        if stem in p.name: return p


def normalize(img, p_lo=1.0, p_hi=99.5):
    lo, hi = np.percentile(img, [p_lo, p_hi])
    return np.clip((img - lo) / (hi - lo + 1e-9), 0, 1)


def fg_mask(raw):
    norm = normalize(raw)
    smooth = filters.gaussian(norm, sigma=1.0)
    thr = filters.threshold_otsu(smooth) * 0.7
    binary = smooth > thr
    return morphology.binary_closing(binary, morphology.disk(2))


def skel_degree(skel):
    k = np.ones((3, 3), dtype=np.uint8); k[1, 1] = 0
    return ndi.convolve(skel.astype(np.uint8), k, mode='constant') * skel


def branch_lengths(skel):
    """Walk every branch from endpoints, record lengths."""
    deg = skel_degree(skel)
    endpoints = list(zip(*np.where((deg == 1) & skel)))
    visited = set()
    lengths = []
    for ep in endpoints:
        if ep in visited:
            continue
        path = [ep]; visited.add(ep)
        prev = None; cur = ep
        while True:
            nbrs = [(cur[0]+dy, cur[1]+dx)
                    for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                    if (dy or dx)
                    and 0 <= cur[0]+dy < skel.shape[0]
                    and 0 <= cur[1]+dx < skel.shape[1]
                    and skel[cur[0]+dy, cur[1]+dx]
                    and (cur[0]+dy, cur[1]+dx) != prev]
            if not nbrs:
                break
            if len(nbrs) > 1 or deg[cur] >= 3:
                # hit junction; stop walking
                break
            nxt = nbrs[0]
            if nxt in visited:
                break
            visited.add(nxt)
            path.append(nxt)
            prev = cur; cur = nxt
        lengths.append(len(path))
    return np.array(lengths)


def fg_coverage(skel, fg, k=4):
    """% of fg pixels within k px of a skeleton pixel."""
    dist = ndi.distance_transform_edt(~skel)
    within = (dist <= k) & fg
    return float(within.sum()) / max(1, fg.sum())


def compute(stem):
    raw = tifffile.imread(find_raw(stem)).astype(np.float32)
    skel = np.load(V29 / f'{stem}_skel_pruned.npy').astype(bool)
    fg = fg_mask(raw)
    deg = skel_degree(skel)
    n_endpoints = int(((deg == 1) & skel).sum())
    n_junctions = int(((deg >= 3) & skel).sum())
    n_skel = int(skel.sum())
    n_fg = int(fg.sum())
    n_px = int(skel.size)

    lengths = branch_lengths(skel)
    short_frac = float((lengths < 5).sum()) / max(1, len(lengths))
    mean_len = float(lengths.mean()) if len(lengths) else 0.0

    cov_k2 = fg_coverage(skel, fg, k=2)
    cov_k4 = fg_coverage(skel, fg, k=4)
    cov_k8 = fg_coverage(skel, fg, k=8)

    return dict(
        stem=stem,
        n_skel=n_skel, n_fg=n_fg, n_px=n_px,
        skel_per_fg=n_skel / max(1, n_fg),
        skel_per_px=n_skel / n_px,
        n_endpoints=n_endpoints,
        endpoint_per_fg=n_endpoints / max(1, n_fg),
        n_junctions=n_junctions,
        n_branches=len(lengths),
        short_spur_frac=short_frac,
        mean_branch_len=mean_len,
        fg_cov_within_2px=cov_k2,
        fg_cov_within_4px=cov_k4,
        fg_cov_within_8px=cov_k8,
    )


def main():
    all_out = []
    for stem in ['F_WT_2', 'F_HET_1', 'F_HET_3']:
        r = compute(stem)
        all_out.append(r)
        print(f'{stem}:')
        print(f'  skel/fg          : {r["skel_per_fg"]:.4f}')
        print(f'  endpoints/fg     : {r["endpoint_per_fg"]:.5f}')
        print(f'  short spur frac  : {r["short_spur_frac"]:.3f}')
        print(f'  mean branch len  : {r["mean_branch_len"]:.1f}')
        print(f'  fg cov within 2px: {r["fg_cov_within_2px"]:.3f}')
        print(f'  fg cov within 4px: {r["fg_cov_within_4px"]:.3f}')
        print(f'  fg cov within 8px: {r["fg_cov_within_8px"]:.3f}')
        print()

    (V30F / 'skeleton_quality.json').write_text(json.dumps(all_out, indent=1))
    print('saved skeleton_quality.json')


if __name__ == '__main__':
    main()
