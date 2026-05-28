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


def merged_branches(sk, merge_radius=3, min_arms=3):
    """One point per REAL junction, where a real junction has >= min_arms (3)
    distinct skeleton arms leaving it (Stephen's rule, 2026-05-27 & 05-28: a
    branch must have >=3 paths; a path's last point is an ENDPOINT, never a
    branch). Two-step so SHORT arms are NOT eaten:
      1. GROUP nearby junction pixels into one node (dilate by merge_radius only
         to fuse 'branch after branch' clusters that are really one junction).
      2. COUNT arms by removing ONLY the actual junction pixels (not the dilated
         blob), labelling the rest, and counting distinct components 8-adjacent
         to those pixels. Removing only the junction pixels (vs the fat blob)
         means an arm a few px long still survives to be counted."""
    d = degree(sk)
    jp = sk & (d >= 3)                                   # actual junction pixels
    grp = binary_dilation(jp, iterations=merge_radius)   # group nearby ones into nodes
    l, nn = ndi.label(grp, structure=np.ones((3, 3)))
    out = []
    pad = merge_radius + 2
    objs = ndi.find_objects(l)
    H, W = sk.shape
    for i, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        y0 = max(0, sl[0].start - pad); y1 = min(H, sl[0].stop + pad)
        x0 = max(0, sl[1].start - pad); x1 = min(W, sl[1].stop + pad)
        node = (l[y0:y1, x0:x1] == i)
        skl = sk[y0:y1, x0:x1]
        jpl = jp[y0:y1, x0:x1] & node                    # the junction pixels of this node
        rest = skl & ~jpl                                # remove ONLY junction px, keep short arms
        rl, _ = ndi.label(rest, structure=np.ones((3, 3)))
        touch = rl[binary_dilation(jpl, iterations=1) & rest]
        n_arms = len(np.unique(touch[touch > 0]))
        if n_arms >= min_arms:
            ys, xs = np.where(jpl)
            if len(ys):
                out.append((ys.mean() + y0, xs.mean() + x0))
    return np.array(out) if out else np.empty((0, 2))


def endpoints(sk):
    return np.column_stack(np.where(sk & (degree(sk) == 1)))
