"""
Merged branch-point count — fixes skeleton over-counting that Stephen caught
(2026-05-26, vision-verified on a 150px crop: 61 raw degree>=3 pixels were
really only 14 junctions; each true bifurcation yields several adjacent
degree>=3 pixels).

FIX: dilate the degree>=3 mask by 2 px, connected-component label, count one
junction per cluster. Use THIS instead of raw (skel & deg>=3).sum() anywhere
branch points are counted (compute_features.py, morphology classifier).

Result per cell (30px nbhd): WT 11.5->3.4, HET 10.4->3.1 raw->merged.
WT-vs-HET secondary-branch signal holds: -9.4% (raw) -> -8.8% (merged), now on
biologically sensible counts (~3 junctions/cell, not ~11).
"""
import json, numpy as np
from pathlib import Path
from scipy import ndimage as ndi
from scipy.ndimage import binary_dilation

ROOT = Path('/Users/stephenyu/choroid-microglia/project_A_morphology')
V29 = ROOT / 'experiments/v29_short_spur_audit'
V30F = ROOT / 'experiments/v30f_trunk_gate'


def merged_junction_label(skel):
    """Return a labeled image where each cluster of adjacent degree>=3
    skeleton pixels (within ~2px) is one junction id."""
    k = np.ones((3, 3), np.uint8); k[1, 1] = 0
    deg = ndi.convolve(skel.astype(np.uint8), k, mode='constant') * skel
    bp = binary_dilation(skel & (deg >= 3), iterations=2)
    lab, _ = ndi.label(bp, structure=np.ones((3, 3)))
    return lab


def merged_branches_in_nbhd(lab, yc, xc, R=30):
    H, W = lab.shape
    y0, y1 = max(0, yc - R), min(H, yc + R + 1)
    x0, x1 = max(0, xc - R), min(W, xc + R + 1)
    sub = lab[y0:y1, x0:x1]
    yy, xx = np.indices(sub.shape)
    rsq = (yy - (yc - y0)) ** 2 + (xx - (xc - x0)) ** 2
    return len(np.unique(sub[(rsq <= R * R) & (sub > 0)]))


if __name__ == '__main__':
    for stem in ['F_WT_2', 'F_HET_1', 'F_HET_3']:
        skel = np.load(V29 / f'{stem}_skel_pruned.npy').astype(bool)
        lab = merged_junction_label(skel)
        cells = json.loads((V30F / f'{stem}_trunk_metrics_v30f.json').read_text())
        m = np.array([merged_branches_in_nbhd(lab, int(c['yc']), int(c['xc'])) for c in cells])
        print(f'{stem}: merged branch/cell mean={m.mean():.2f} median={np.median(m):.0f}')
