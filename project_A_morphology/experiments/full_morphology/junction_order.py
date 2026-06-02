import numpy as np
from pathlib import Path
from scipy import ndimage as ndi

PIX = 0.207
PIX_AREA = PIX**2
# auto-discover cached images (relative to this file) — works on any machine
CACHE = Path(__file__).resolve().parent / 'cache_bg'
paths = {p.stem: str(p) for p in sorted(CACHE.glob('*.npz'))}

STRUCT8 = np.ones((3,3), bool)

def tissue_area_um2(binary):
    # tissue = full image area in um2 (we'll also compute fg area for ref)
    H,W = binary.shape
    return H*W*PIX_AREA

def junction_order(skel, jy, jx, r=3):
    """Recount arms: remove a disk of radius r around junction, label remaining skel
    8-connected, count distinct components that touch the disk's annulus boundary."""
    H,W = skel.shape
    y0,y1 = max(0,jy-r-2), min(H, jy+r+3)
    x0,x1 = max(0,jx-r-2), min(W, jx+r+3)
    patch = skel[y0:y1, x0:x1].copy()
    if patch.sum() == 0:
        return 0
    cy, cx = jy-y0, jx-x0
    yy, xx = np.ogrid[:patch.shape[0], :patch.shape[1]]
    disk = (yy-cy)**2 + (xx-cx)**2 <= r*r
    cut = patch.copy()
    cut[disk] = False
    lab, n = ndi.label(cut, structure=STRUCT8)
    if n == 0:
        return 0
    # arms = components that touch the removed-disk border (i.e. adjacent to disk)
    # dilate disk by 1, intersect with labeled skel
    disk_dil = ndi.binary_dilation(disk, structure=STRUCT8)
    ring = disk_dil & ~disk
    touching = np.unique(lab[ring & (lab>0)])
    return int(len(touching))

rows = {}
extra = {}
for name, p in paths.items():
    d = np.load(p, allow_pickle=True)
    skel = d['skel']
    J = d['J']
    E = d['E']
    binary = d['binary']
    cond = str(d['cond'])

    skel_len_um = skel.sum() * PIX   # 1px skeleton -> length approx in px; convert
    # better skeleton length: total skeleton px * pixel_um (each px ~1 unit length)
    tissue_um2 = tissue_area_um2(binary)
    tissue_mm2 = tissue_um2 / 1e6
    fg_um2 = binary.sum()*PIX_AREA

    orders = np.array([junction_order(skel, int(round(y)), int(round(x)), r=3)
                       for y,x in J])
    valid = orders >= 3
    nv = int(valid.sum())
    o = orders[valid]
    n3 = int((o==3).sum())
    n4 = int((o==4).sum())
    n5p = int((o>=5).sum())
    total = nv if nv>0 else 1

    n_junc = nv
    n_end = len(E)

    rows[name] = dict(
        cond=cond,
        n_junc_raw=len(J), n_junc_valid=nv,
        pct3=100*n3/total, pct4=100*n4/total, pct5p=100*n5p/total,
        n3=n3,n4=n4,n5p=n5p,
        junc_per_mm2 = nv / tissue_mm2,
        junc_per_100um_skel = 100*nv / skel_len_um,
        end_junc_ratio = n_end / nv if nv>0 else np.nan,
        mean_order = float(o.mean()) if len(o) else np.nan,
        pct_highorder = 100*(n4+n5p)/total,
        skel_len_um=skel_len_um, tissue_mm2=tissue_mm2, n_end=n_end,
        fg_um2=fg_um2,
    )
    # robustness: also r=2 and r=4
    for rr in (2,4):
        o2 = np.array([junction_order(skel, int(round(y)), int(round(x)), r=rr) for y,x in J])
        o2v = o2[o2>=3]
        t2 = len(o2v) if len(o2v)>0 else 1
        extra[(name,rr)] = (100*(o2v==3).sum()/t2, 100*(o2v==4).sum()/t2, 100*(o2v>=5).sum()/t2, len(o2v))

import json
print(json.dumps(rows, indent=2, default=float))
print("\n--- robustness (pct3,pct4,pct5p,nvalid) at r=2,4 ---")
for k,v in extra.items():
    print(k, [round(x,2) for x in v])
