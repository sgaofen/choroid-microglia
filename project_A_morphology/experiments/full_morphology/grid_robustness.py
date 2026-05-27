"""
Grid robustness (P5, GPT-Pro review): does the HET fragmentation signal depend
on WHERE the tile grid is drawn? A focus could be split by a tile boundary. Re-
tile at several grid ORIGIN offsets (and a sliding/overlapping stride) and check
the fragmentation-hotspot enrichment in HET is stable. Uses the clustering-free
fragmentation score so there is no re-fitting between configurations.

Each image's skeleton + global topology + segments are computed ONCE, then
re-binned at each offset/stride.
"""
import sys
from pathlib import Path
import numpy as np
import tifffile
sys.path.insert(0, str(Path(__file__).parent))
import pipeline as pl
from pipeline import PIXEL_UM, STEMS, COND

TILE = 200
MIN_FG_FRAC = 0.03


def cache(stem):
    norm, binary, dist, skel = pl.prep(stem)
    J, E = pl.global_topology(skel)
    midy, midx, seglen = pl.segments_df(skel)
    sy, sx = np.where(skel)
    return dict(binary=binary, shape=skel.shape, J=J, E=E,
                midy=midy, midx=midx, seglen=seglen, sy=sy, sx=sx)


def tile_frag(c, oy, ox, stride):
    """Per-tile fragmentation features at grid origin (oy,ox) with given stride."""
    H, W = c['shape']
    binary = c['binary']
    rows = []
    for ty in range(oy, H, stride):
        for tx in range(ox, W, stride):
            y1, x1 = min(ty+TILE, H), min(tx+TILE, W)
            if ty < 0 or tx < 0:
                continue
            tb = binary[ty:y1, tx:x1]
            if tb.size == 0 or tb.sum()/tb.size < MIN_FG_FRAC:
                continue
            def inb(ys, xs):
                return (ys >= ty) & (ys < y1) & (xs >= tx) & (xs < x1)
            nb = int(inb(c['J'][:, 0], c['J'][:, 1]).sum()) if len(c['J']) else 0
            ne = int(inb(c['E'][:, 0], c['E'][:, 1]).sum()) if len(c['E']) else 0
            segm = inb(c['midy'], c['midx'])
            skel_um = int(inb(c['sy'], c['sx']).sum()) * PIXEL_UM
            if skel_um < 1:
                continue
            rows.append((ne/max(nb, 1), 100*ne/skel_um, 100*nb/skel_um,
                         c['seglen'][segm].mean() if segm.any() else 0.0))
    return rows  # (ep_br, ep100, br100, seglen)


def main():
    print('caching skeleton + topology per image (once)...')
    C = {s: cache(s) for s in STEMS}

    configs = [('origin 0,0', 0, 0, TILE), ('origin 100,100', 100, 100, TILE),
               ('origin 50,50', 50, 50, TILE), ('origin 0,100', 0, 100, TILE),
               ('origin 100,0', 100, 0, TILE), ('sliding stride100', 0, 0, 100)]

    print(f'\n{"config":>20}{"WT":>9}{"HET_1":>9}{"HET_3":>9}{"verdict":>14}')
    for name, oy, ox, stride in configs:
        per = {s: tile_frag(C[s], oy, ox, stride) for s in STEMS}
        allr = [r for s in STEMS for r in per[s]]
        A = np.array(allr)  # cols: ep_br, ep100, br100, seglen
        z = (A - A.mean(0)) / (A.std(0) + 1e-9)
        frag = z[:, 0] + z[:, 1] - z[:, 2] - z[:, 3]
        thr = np.percentile(frag, 75)
        # split frag back per image
        hot = {}
        i = 0
        for s in STEMS:
            k = len(per[s])
            hot[s] = round(100*float((frag[i:i+k] > thr).mean()), 1)
            i += k
        wt, h1, h3 = hot['F_WT_2'], hot['F_HET_1'], hot['F_HET_3']
        d1, d3 = h1-wt, h3-wt
        same = (d1 > 0) == (d3 > 0)
        gap = (abs(d1)+abs(d3))/2; spread = abs(d1-d3)
        v = '✓ HET↑' if (same and d1 > 0 and spread < gap) else ('~ both up' if (same and d1 > 0) else '✗')
        print(f'{name:>20}{wt:>9}{h1:>9}{h3:>9}{v:>14}')
    print('\n(values = % of tiles that are fragmentation hotspots, pooled-p75 threshold)')


if __name__ == '__main__':
    main()
