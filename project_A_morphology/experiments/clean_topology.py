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


def merged_branches(sk, merge_radius=4):
    """One point per REAL junction. Dilate degree>=3 by merge_radius to fuse
    'branch after branch' clusters, then keep a cluster only if >=3 skeleton
    paths actually leave it. Enforces Stephen's rule (2026-05-27): a path's last
    point must be an ENDPOINT, never a branch — terminal cluster has 1 exit,
    pass-through has 2, real junction >=3."""
    d = degree(sk)
    bpd = binary_dilation(sk & (d >= 3), iterations=merge_radius)
    l, nn = ndi.label(bpd, structure=np.ones((3, 3)))
    out = []
    for i in range(1, nn + 1):
        clust = (l == i)
        ring = binary_dilation(clust, iterations=1) & sk & ~clust
        _, n_exits = ndi.label(ring, structure=np.ones((3, 3)))
        if n_exits >= 3:
            ys, xs = np.where(clust & sk)
            if len(ys):
                out.append((ys.mean(), xs.mean()))
    return np.array(out) if out else np.empty((0, 2))


def endpoints(sk):
    return np.column_stack(np.where(sk & (degree(sk) == 1)))
