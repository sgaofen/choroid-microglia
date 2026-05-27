"""
Compare a human endpoint annotation (napari CSV: index,axis-0,axis-1) against
the v29 skeleton's endpoints, and calibrate a spur-prune threshold.

Lesson (2026-05-26): the first pass (131 endpoints, sparsely marked over a big
region) did NOT yield a clean spur-length threshold — accepted spurs median
19px vs ignored 11px, heavily overlapping — because sparse marking means
"unmarked != spurious" (many unmarked endpoints are real, just not marked).
=> For calibration, mark ALL endpoints EXHAUSTIVELY in one small region.

Usage:
  python validate_endpoints.py "/path/to/ENDPOINT.csv" F_WT_2
"""
import sys, csv, numpy as np
from pathlib import Path
from scipy import ndimage as ndi

ROOT = Path('/Users/stephenyu/choroid-microglia/project_A_morphology')
V29 = ROOT / 'experiments/v29_short_spur_audit'


def spur_len(sset, deg, ey, ex):
    prev, cur, L = None, (ey, ex), 0.0
    for _ in range(80):
        nb = [(cur[0]+dy, cur[1]+dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
              if (dy or dx) and (cur[0]+dy, cur[1]+dx) in sset and (cur[0]+dy, cur[1]+dx) != prev]
        if not nb:
            break
        nxt = nb[0]
        L += 1.0 if (nxt[0] == cur[0] or nxt[1] == cur[1]) else 1.414
        if deg[nxt] >= 3:
            break
        prev, cur = cur, nxt
    return L


def main():
    csv_path, stem = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else 'F_WT_2')
    rows = list(csv.reader(open(csv_path)))[1:]
    his = np.array([[float(r[1]), float(r[2])] for r in rows])
    skel = np.load(V29 / f'{stem}_skel_pruned.npy').astype(bool)
    k = np.ones((3, 3), np.uint8); k[1, 1] = 0
    deg = ndi.convolve(skel.astype(np.uint8), k, mode='constant') * skel
    sset = set(zip(*[a.tolist() for a in np.where(skel)]))
    pad = 15
    y0, y1 = int(his[:, 0].min()-pad), int(his[:, 0].max()+pad)
    x0, x1 = int(his[:, 1].min()-pad), int(his[:, 1].max()+pad)
    ey, ex = np.where(skel & (deg == 1))
    m = (ey >= y0) & (ey <= y1) & (ex >= x0) & (ex <= x1)
    algo = np.column_stack([ey[m], ex[m]])
    kept, rej = [], []
    for yy, xx in algo:
        L = spur_len(sset, deg, int(yy), int(xx))
        (kept if np.linalg.norm(his - [yy, xx], axis=1).min() <= 6 else rej).append(L)
    kept, rej = np.array(kept), np.array(rej)
    print(f'human={len(his)} algo={len(algo)} matched={len(kept)} unmarked={len(rej)}')
    print(f'accepted spur median={np.median(kept):.1f}px  ignored median={np.median(rej):.1f}px')
    for thr in (5, 8, 10, 12, 15):
        print(f'  thr {thr}px: keep {(kept>=thr).mean()*100:.0f}% of accepted, '
              f'cut {(rej<thr).mean()*100:.0f}% of ignored')


if __name__ == '__main__':
    main()
