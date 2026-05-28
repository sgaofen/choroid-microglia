"""Cache the single source of truth for the multi-angle workflow: for each image
on the BACKGROUND-NORMALIZED baseline (+ fixed branch detection), save binary,
skeleton, distance, global junctions/endpoints, connected-component labels, and
per-segment (midpoint, length, branch-type). Downstream workflow agents read
these arrays so every angle shares the SAME extraction (no re-skeletonizing, no
inconsistent choices)."""
import sys
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from skan import Skeleton, summarize
sys.path.insert(0, str(Path(__file__).parent))
import pipeline as pl

CACHE = pl.ROOT / 'experiments/full_morphology/cache_bg'
CACHE.mkdir(parents=True, exist_ok=True)
PX = pl.PIXEL_UM


def seg_table(skel):
    sko = Skeleton(skel)
    df = summarize(sko, separator='-')
    def col(sub):
        for c in df.columns:
            if c.endswith(sub):
                return c
        raise KeyError(sub)
    sy = df[col('coord-src-0')].to_numpy(); sx = df[col('coord-src-1')].to_numpy()
    dy = df[col('coord-dst-0')].to_numpy(); dx = df[col('coord-dst-1')].to_numpy()
    return ((sy+dy)/2.0, (sx+dx)/2.0,
            df['branch-distance'].to_numpy()*PX, df['branch-type'].to_numpy())


for s in pl.STEMS:
    norm, binary, dist, skel = pl.prep(s)
    J, E = pl.global_topology(skel)
    cc, ncc = ndi.label(skel, structure=np.ones((3, 3)))
    midy, midx, seglen, segtype = seg_table(skel)
    np.savez_compressed(
        CACHE / f'{s}.npz',
        cond=pl.COND[s], pixel_um=PX,
        binary=binary, skel=skel, dist=dist.astype(np.float16),
        J=J, E=E, cc=cc.astype(np.int32),
        midy=midy.astype(np.float32), midx=midx.astype(np.float32),
        seglen=seglen.astype(np.float32), segtype=segtype.astype(np.int8),
        shape=np.array(skel.shape, np.int32))
    print(f'{s} ({pl.COND[s]}): skel={int(skel.sum())} J={len(J)} E={len(E)} '
          f'cc={ncc} seg={len(seglen)} fg={float(binary.mean()):.3f}')
print('cached to', CACHE)
