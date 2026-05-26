"""
Annulus-based primary trunk counter (Codex's recommended replacement for
greedy distal-endpoint attribution).

Definition: for each accepted soma at (yc, xc) with soma-collar radius r_soma,
count the number of CONNECTED COMPONENTS of skeleton pixels lying in the
annulus r_soma+2 ≤ r ≤ r_soma+8. Each connected component is one process
trunk leaving the soma.

Compared to current endpoint_attribution:
- This is local — doesn't sweep through the whole skeleton network
- It counts what biologists mean by "primary processes" — the major
  branches that leave the cell body. Distal tips don't inflate the count.
- A real microglia: 3-6 typical
- A bead with passing process: 2
- An isolated cell with no processes: 0
- A vessel-following cell: typically 2 (in and out along vessel)

For each accepted cell, we save both metrics so WT/HET stats can use whichever
is more meaningful. We also save: total_endpoints (per cell, via the existing
greedy BFS), total_branch_points (skeleton degree ≥ 3 within 30 px),
total_skeleton_length (px within 30 px). All four signals lets us see which
is most stable across WT vs HET.
"""
import json
import sys
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from skimage import measure

ROOT = Path('/Users/stephenyu/Documents/choroid-microglia/project_A_morphology')
V27 = ROOT / 'experiments/v27_clean_graph'
V29 = ROOT / 'experiments/v29_short_spur_audit'
V30E = ROOT / 'experiments/v30e_merge_fixes'

sys.path.insert(0, str(V30E))
from v30e_run import load_inputs, make_binary_and_dist


def soma_collar_radius(dist_s, yc, xc):
    """Approximate soma radius from distance transform at the snapped center."""
    return float(dist_s[yc, xc])


def annulus_trunk_count(skel, yc, xc, r_in, r_out):
    H, W = skel.shape
    y0, y1 = max(0, yc - r_out - 1), min(H, yc + r_out + 2)
    x0, x1 = max(0, xc - r_out - 1), min(W, xc + r_out + 2)
    patch_skel = skel[y0:y1, x0:x1]
    yy, xx = np.indices(patch_skel.shape)
    cy = yc - y0; cx = xc - x0
    rsq = (yy - cy) ** 2 + (xx - cx) ** 2
    in_ann = (rsq >= r_in * r_in) & (rsq <= r_out * r_out) & patch_skel
    if not in_ann.any():
        return 0
    lab, n = ndi.label(in_ann, structure=np.ones((3, 3)))
    return int(n)


def local_skel_length(skel, yc, xc, radius=30):
    H, W = skel.shape
    y0, y1 = max(0, yc - radius), min(H, yc + radius + 1)
    x0, x1 = max(0, xc - radius), min(W, xc + radius + 1)
    patch = skel[y0:y1, x0:x1]
    yy, xx = np.indices(patch.shape)
    cy = yc - y0; cx = xc - x0
    rsq = (yy - cy) ** 2 + (xx - cx) ** 2
    mask = rsq <= radius * radius
    return int((patch & mask).sum())


def local_branch_count(skel, yc, xc, radius=30):
    H, W = skel.shape
    y0, y1 = max(0, yc - radius), min(H, yc + radius + 1)
    x0, x1 = max(0, xc - radius), min(W, xc + radius + 1)
    patch = skel[y0:y1, x0:x1]
    k = np.ones((3, 3), dtype=np.uint8); k[1, 1] = 0
    nb = ndi.convolve(patch.astype(np.uint8), k, mode='constant') * patch
    yy, xx = np.indices(patch.shape)
    cy = yc - y0; cx = xc - x0
    rsq = (yy - cy) ** 2 + (xx - cx) ** 2
    mask = rsq <= radius * radius
    return int(((nb >= 3) & mask).sum())


def compute_for_image(stem):
    raw, skel, somas = load_inputs(stem)
    _, _, dist_s = make_binary_and_dist(raw)
    recs = json.loads((V30E / f'{stem}_seeds_v30e.json').read_text())
    accepted = [r for r in recs
                if r['type'] in ('strong', 'weak', 'low_confidence_soma')]

    out = []
    for r in accepted:
        yc, xc = r['yc'], r['xc']
        r_soma = soma_collar_radius(dist_s, yc, xc)
        r_in = max(3, int(round(r_soma)) + 2)
        r_out = r_in + 6
        n_trunks = annulus_trunk_count(skel, yc, xc, r_in, r_out)
        n_branches = local_branch_count(skel, yc, xc, radius=30)
        skel_len = local_skel_length(skel, yc, xc, radius=30)
        out.append(dict(
            label=r['label'], yc=yc, xc=xc, type=r['type'],
            score=float(r['score']),
            r_soma=float(r_soma), r_in=int(r_in), r_out=int(r_out),
            n_trunks=int(n_trunks),
            n_local_branches=int(n_branches),
            skel_len_local=int(skel_len),
        ))
    return out


def summarize(stem, recs):
    trunks = np.array([r['n_trunks'] for r in recs])
    branches = np.array([r['n_local_branches'] for r in recs])
    skel_len = np.array([r['skel_len_local'] for r in recs])
    print(f'{stem}: n={len(recs)}')
    print(f'  trunks   mean={trunks.mean():.2f} median={np.median(trunks):.1f} '
          f'dist={np.bincount(np.clip(trunks, 0, 10)).tolist()}')
    print(f'  branches mean={branches.mean():.2f} median={np.median(branches):.1f}')
    print(f'  skel_len mean={skel_len.mean():.0f} median={np.median(skel_len):.0f}')


if __name__ == '__main__':
    all_recs = {}
    for stem in ['F_WT_2', 'F_HET_1', 'F_HET_3']:
        recs = compute_for_image(stem)
        all_recs[stem] = recs
        (V30E / f'{stem}_trunk_metrics_v30e.json').write_text(
            json.dumps(recs, indent=1))
        summarize(stem, recs)
    # cross-genotype quick table
    print('\nCross-genotype (annulus primary trunks):')
    print(f'{"image":10}  n  mean_trunks  median_trunks  '
          f'mean_branches  mean_skel_len')
    for stem in ['F_WT_2', 'F_HET_1', 'F_HET_3']:
        recs = all_recs[stem]
        t = np.array([r['n_trunks'] for r in recs])
        b = np.array([r['n_local_branches'] for r in recs])
        s = np.array([r['skel_len_local'] for r in recs])
        print(f'{stem:10}  {len(recs):4d}  {t.mean():.2f}         '
              f'{np.median(t):.1f}            '
              f'{b.mean():.2f}           {s.mean():.0f}')
