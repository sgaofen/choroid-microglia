"""
v30g small-round-cell detector — conservative candidate proposer.

Designed from Codex F's blob triage criteria:
- CELL feature: compact oval body, bright core + faint halo, separation from
  linear vessel/wall structures, NOT tiny specks
- NOISE: 1-few px specks
- DEBRIS: asymmetric jagged dashes
- VESSEL_FRAGMENT: embedded in linear wall

This script proposes candidates that look like small round cells. It does NOT
yet integrate them into v30f's accepted set. Instead it samples 30 per image
and we will audit precision before adopting.

Criteria (conservative):
 1) area >= 15 (kills NOISE specks)
 2) area <= 80 (real microglia core)
 3) eccentricity <= 0.7 (kills DEBRIS dashes)
 4) solidity >= 0.85 (kills jagged shapes)
 5) peak/mean intensity >= 1.25 (must have bright core, not flat blob)
 6) distance to long-skeleton-branch >= 5 px (kills VESSEL_FRAGMENT)
 7) NOT within 8 px of any v30f accepted soma (avoid double-counting)
"""
import json, sys
from collections import Counter
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi
from skimage import filters, measure, morphology

ROOT = Path('/Users/stephenyu/Documents/choroid-microglia/project_A_morphology')
RAW = ROOT / 'data/raw'
V29 = ROOT / 'experiments/v29_short_spur_audit'
V30F = ROOT / 'experiments/v30f_trunk_gate'
V30G = ROOT / 'experiments/v30g_small_cells'

# Conservative criteria
AREA_MIN = 15
AREA_MAX = 80
ECC_MAX = 0.70
SOL_MIN = 0.85
PEAK_MEAN_MIN = 1.25
DIST_TO_LONG_BRANCH_MIN = 5
EXCLUSION_FROM_V30F_PX = 8

SAMPLE_N = 30
RNG_SEED = 20260524
CROP_SIZE = 60


def find_raw(stem):
    for p in RAW.glob('*.tif'):
        if stem in p.name: return p


def normalize(img, p_lo=1.0, p_hi=99.5):
    lo, hi = np.percentile(img, [p_lo, p_hi])
    return np.clip((img - lo) / (hi - lo + 1e-9), 0, 1)


def make_binary(raw):
    norm = normalize(raw)
    smooth = filters.gaussian(norm, sigma=1.0)
    thr = filters.threshold_otsu(smooth) * 0.7
    binary = smooth > thr
    return morphology.binary_closing(binary, morphology.disk(2))


def skel_degree(skel):
    k = np.ones((3, 3), dtype=np.uint8); k[1, 1] = 0
    return ndi.convolve(skel.astype(np.uint8), k, mode='constant') * skel


def neighbors8(y, x, h, w):
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0: continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                yield ny, nx


def edge_key(a, b):
    return (a, b) if a <= b else (b, a)


def long_branch_mask(skel, min_len=15):
    skel = skel.astype(bool)
    deg = skel_degree(skel)
    node_mask = skel & (deg != 2)
    long = np.zeros_like(skel, dtype=bool)
    visited = set()
    ys, xs = np.where(node_mask)
    for start in zip(ys.tolist(), xs.tolist()):
        for nb in neighbors8(start[0], start[1], *skel.shape):
            if not skel[nb]: continue
            key = edge_key(start, nb)
            if key in visited: continue
            path = [start, nb]; visited.add(key)
            prev, cur = start, nb
            while not node_mask[cur] and deg[cur] == 2:
                nexts = [n for n in neighbors8(cur[0], cur[1], *skel.shape)
                         if skel[n] and n != prev]
                if not nexts: break
                nxt = nexts[0]
                key = edge_key(cur, nxt)
                if key in visited: break
                visited.add(key); path.append(nxt)
                prev, cur = cur, nxt
            if len(path) > min_len:
                yy, xx = zip(*path)
                long[yy, xx] = True
    remaining = skel & ~node_mask & ~long
    labels = measure.label(remaining, connectivity=2)
    for prop in measure.regionprops(labels):
        if prop.area > min_len:
            coords = prop.coords
            long[coords[:, 0], coords[:, 1]] = True
    return long


def crop60(arr, yc, xc):
    h, w = arr.shape
    half = CROP_SIZE // 2
    y0 = min(max(int(round(yc)) - half, 0), h - CROP_SIZE)
    x0 = min(max(int(round(xc)) - half, 0), w - CROP_SIZE)
    return arr[y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE]


def save_png(path, crop_norm):
    Image.fromarray((np.clip(crop_norm, 0, 1) * 255).astype(np.uint8)).save(path)


def find_candidates(stem):
    raw = tifffile.imread(find_raw(stem)).astype(np.float32)
    norm = normalize(raw)
    binary = make_binary(raw)
    skel = np.load(V29 / f'{stem}_skel_pruned.npy').astype(bool)
    long_skel = long_branch_mask(skel, min_len=15)
    long_dist = ndi.distance_transform_edt(~long_skel)

    # v30f accepted somas → exclusion mask
    recs = json.loads((V30F / f'{stem}_seeds_v30f.json').read_text())
    v30f_yc = [r['yc'] for r in recs
               if r['type'] in ('strong', 'weak', 'low_confidence_soma')]
    v30f_xc = [r['xc'] for r in recs
               if r['type'] in ('strong', 'weak', 'low_confidence_soma')]
    H, W = norm.shape
    excl = np.zeros((H, W), dtype=bool)
    for yc, xc in zip(v30f_yc, v30f_xc):
        y0, y1 = max(0, yc - EXCLUSION_FROM_V30F_PX), min(H, yc + EXCLUSION_FROM_V30F_PX + 1)
        x0, x1 = max(0, xc - EXCLUSION_FROM_V30F_PX), min(W, xc + EXCLUSION_FROM_V30F_PX + 1)
        excl[y0:y1, x0:x1] = True

    labels = measure.label(binary, connectivity=2)
    accepted = []
    n_seen = 0
    cnt = Counter()
    for prop in measure.regionprops(labels, intensity_image=norm):
        n_seen += 1
        if prop.area < AREA_MIN: cnt['fail_area_low'] += 1; continue
        if prop.area > AREA_MAX: cnt['fail_area_high'] += 1; continue
        if prop.eccentricity > ECC_MAX: cnt['fail_ecc'] += 1; continue
        sol = float(prop.solidity)
        if sol < SOL_MIN: cnt['fail_sol'] += 1; continue
        coords = prop.coords
        # touching/very close to long skeleton branches → vessel fragment
        if long_skel[coords[:, 0], coords[:, 1]].any(): cnt['fail_on_long'] += 1; continue
        min_dist_to_long = long_dist[coords[:, 0], coords[:, 1]].min()
        if min_dist_to_long < DIST_TO_LONG_BRANCH_MIN: cnt['fail_near_long'] += 1; continue
        # bright core requirement
        vals = norm[coords[:, 0], coords[:, 1]]
        peak = float(vals.max()); mean = float(vals.mean())
        if mean <= 0: cnt['fail_mean_zero'] += 1; continue
        if peak / mean < PEAK_MEAN_MIN: cnt['fail_no_bright_core'] += 1; continue
        # not within exclusion zone of v30f accepted somas
        if excl[coords[:, 0], coords[:, 1]].any(): cnt['fail_near_v30f'] += 1; continue

        yc, xc = prop.centroid
        accepted.append(dict(
            label=int(prop.label),
            yc=float(yc), xc=float(xc),
            area=int(prop.area),
            ecc=float(prop.eccentricity),
            sol=float(sol),
            peak=peak, mean=mean,
            peak_mean=peak / mean,
            min_dist_to_long=float(min_dist_to_long),
        ))

    print(f'  {stem}: {n_seen} components total -> {len(accepted)} candidates')
    print(f'    filter trace: {dict(cnt)}')
    return accepted, norm


def main():
    V30G.mkdir(exist_ok=True)
    summary = {}
    for stem in ['F_WT_2', 'F_HET_1', 'F_HET_3']:
        cands, norm = find_candidates(stem)
        summary[stem] = len(cands)
        (V30G / f'{stem}_small_cell_candidates_v30g.json').write_text(
            json.dumps(cands, indent=1))
        # sample crops for audit
        crop_dir = V30G / f'small_cell_crops_{stem}'
        crop_dir.mkdir(exist_ok=True)
        if len(cands) >= SAMPLE_N:
            rng = np.random.default_rng(RNG_SEED)
            idx = rng.choice(len(cands), size=SAMPLE_N, replace=False)
        else:
            idx = list(range(len(cands)))
        for out_i, ci in enumerate(idx, 1):
            c = cands[int(ci)]
            crop = crop60(norm, c['yc'], c['xc'])
            save_png(crop_dir / f'small_cell_{out_i:02d}_L{c["label"]}.png', crop)

    print('\n=== v30g small-cell proposer summary ===')
    print(f'{"image":>10}  cands')
    for stem, n in summary.items():
        print(f'{stem:>10}  {n}')


if __name__ == '__main__':
    main()
