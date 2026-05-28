"""
Clean skeleton topology — keep the GOOD v29 skeleton, only clean the topology
noise Stephen flagged (2026-05-27, vision-verified on detail crops):
  - faint isolated fragments -> drop small connected components
  - "branch after branch" (clustered junctions) -> merge junctions within ~8px
  - "ending with a branch" / tiny terminal stubs -> prune terminal spurs <=10px
  - over-counted junctions: count merged clusters, not raw degree>=3 pixels

Do NOT lower the binarization threshold to connect faint gaps — that pulls in
noise (empirically: endpoints 120->202). Connect-vs-separate of faint signal is
a semantic decision for a trained instance model (Cellpose), not thresholding.

Result on a 170px crop: branches 91 -> 20 (clean, one per real junction),
endpoints stable (~25). Use clean()+merged_branches() everywhere topology is
counted.
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import binary_dilation


def degree(sk):
    k = np.ones((3, 3), np.uint8); k[1, 1] = 0
    return ndi.convolve(sk.astype(np.uint8), k, mode='constant') * sk


def prune_spurs(sk, min_len=10):
    sk = sk.copy()
    for _ in range(min_len + 3):
        d = degree(sk); sset = set(zip(*np.where(sk)))
        rm = []
        for (y, x) in zip(*np.where(sk & (d == 1))):
            prev, cur, path = None, (y, x), [(y, x)]
            while True:
                nb = [(cur[0]+a, cur[1]+b) for a in (-1,0,1) for b in (-1,0,1)
                      if (a or b) and (cur[0]+a, cur[1]+b) in sset and (cur[0]+a, cur[1]+b) != prev]
                if len(nb) != 1:
                    break
                nxt = nb[0]
                if d[nxt] >= 3:
                    break
                path.append(nxt); prev, cur = cur, nxt
                if len(path) > min_len:
                    break
            if len(path) <= min_len and d[cur] >= 3:
                rm += path[:-1]
        if not rm:
            break
        for p in rm:
            sk[p] = False
    return sk


def clean(sk, min_component=12, spur_len=10):
    """Drop tiny isolated fragments, then prune short terminal spurs."""
    lab, n = ndi.label(sk, structure=np.ones((3, 3)))
    if n:
        sizes = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
        keep = np.where(sizes >= min_component)[0] + 1
        sk = np.isin(lab, list(keep)) & sk
    return prune_spurs(sk, spur_len)


def break_loops(sk, raw_norm, max_hole=60):
    """Microglia are TREES — loops are over-connection artifacts (the skeleton
    went around a dark gap on both sides). Break each small enclosed loop at its
    DIMMEST pixel (the spurious faint bridge), per Stephen's rule: don't glue
    across a clear gap. Loops are rare (~9 in a full F_WT_2)."""
    sk = sk.copy()
    for _ in range(8):
        holes = ndi.binary_fill_holes(sk) & ~sk
        hl, hn = ndi.label(holes)
        broke = False
        for i in range(1, hn + 1):
            hole = (hl == i)
            if hole.sum() > max_hole:
                continue
            loop = binary_dilation(hole, iterations=1) & sk
            ly, lx = np.where(loop)
            if len(ly) < 3:
                continue
            j = int(np.argmin(raw_norm[ly, lx]))
            sk[ly[j], lx[j]] = False
            broke = True
        if not broke:
            break
    return sk


def merged_branches(sk):
    """A junction = a skeleton pixel with degree >= 3 (>=3 lines meeting). On a
    1-px pruned skeleton that IS the definition (Stephen, 2026-05-28). Directly-
    adjacent (8-connected) junction pixels are the same physical junction, so
    count = number of 8-connected components of the degree>=3 mask; return one
    centroid per component. NO dilation-merge and NO arm-count validation — both
    were over-engineering that massively under-counted (18 vs the correct ~125
    on a WT crop). Spurs are already removed by prune_spurs upstream, so degree>=3
    pixels are real junctions."""
    jp = sk & (degree(sk) >= 3)
    l, n = ndi.label(jp, structure=np.ones((3, 3)))
    if n == 0:
        return np.empty((0, 2))
    return np.array(ndi.center_of_mass(jp, l, range(1, n + 1)))


def endpoints(sk):
    return np.column_stack(np.where(sk & (degree(sk) == 1)))
