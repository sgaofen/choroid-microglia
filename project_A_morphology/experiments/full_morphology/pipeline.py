"""
Shared pipeline for the corrected morphology analysis (addresses GPT-Pro review,
2026-05-27). Single source of truth for: segmentation (several methods, for the
threshold sensitivity analysis), skeleton, and GLOBAL topology (endpoints +
merged junctions extracted ONCE per image with the validated clean_topology
rules, then assigned to tiles/components — so tile/component counts are
consistent with the whole image and tile borders never split a junction cluster).
"""
import sys
from pathlib import Path
import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.ndimage import white_tophat
from skimage import filters, morphology
from skan import Skeleton, summarize

ROOT = Path('/Users/stephenyu/choroid-microglia/project_A_morphology')
RAW = ROOT / 'data/raw'
V29 = ROOT / 'experiments/v29_short_spur_audit'
sys.path.insert(0, str(ROOT / 'experiments'))
import clean_topology as ct

PIXEL_UM = 0.207
STEMS = ['F_WT_2', 'F_HET_1', 'F_HET_3']
COND = {'F_WT_2': 'WT', 'F_HET_1': 'HET', 'F_HET_3': 'HET'}


def find_raw(stem):
    return next(RAW.glob(f'*{stem}*.tif'))


def normalize(raw, p_lo=1.0, p_hi=99.5):
    lo, hi = np.percentile(raw, [p_lo, p_hi])
    return np.clip((raw - lo) / (hi - lo + 1e-9), 0, 1)


def segment(norm, method='bg', raw=None):
    """Binarize. Default 'bg' = rolling-ball background subtraction on the RAW
    image (per Huixin: make empty space equally black across images BEFORE
    thresholding, removing the acquisition-brightness confound) + otsu*0.7.
    Other methods kept for the P4 segmentation-sensitivity comparison."""
    if method == 'bg':
        src = raw if raw is not None else norm
        th = white_tophat(filters.gaussian(src, 1.0), size=31)
        b = th > filters.threshold_otsu(th) * 0.7
        b = morphology.binary_closing(b, morphology.disk(2))
        b = morphology.remove_small_objects(b, 20)
        return b
    sm = filters.gaussian(norm, 1.0)
    if method == 'otsu07':
        b = sm > filters.threshold_otsu(sm) * 0.7
    elif method == 'otsu':
        b = sm > filters.threshold_otsu(sm)
    elif method == 'hysteresis':                       # bright seeds, grow down toward faint
        t = filters.threshold_otsu(sm)                 # NOTE: low must stay ABOVE background (~0.6t here);
        b = filters.apply_hysteresis_threshold(sm, t * 0.7, t)  # going lower FLOODS (faint processes sit at bg level)
    elif method == 'frangi':                           # tubeness / ridge enhancement for thin processes
        f = filters.frangi(sm, sigmas=range(1, 5), black_ridges=False)
        b = f > filters.threshold_otsu(f)
    elif method == 'adaptive':                         # local adaptive threshold (uneven background)
        b = sm > filters.threshold_local(sm, block_size=101, method='gaussian', offset=-0.01)
    else:
        raise ValueError(method)
    b = morphology.binary_closing(b, morphology.disk(2))
    b = morphology.remove_small_objects(b, 20)
    return b


def clean_skel(raw_skel, norm, spur=8):
    sk = ct.clean(raw_skel)
    sk = ct.break_loops(sk, norm)
    sk = ct.prune_spurs(sk, spur)
    return sk


def prep(stem, method='bg'):
    """Return norm, binary, dist, cleaned skeleton. Default baseline is now the
    background-normalized segmentation (Huixin's brightness correction), always
    re-skeletonized from THIS segmentation (the old v29 skeleton came from the
    acquisition-confounded otsu07 and must not be reused)."""
    raw = tifffile.imread(find_raw(stem)).astype(np.float32)
    norm = normalize(raw)
    binary = segment(norm, method, raw=raw)
    dist = ndi.distance_transform_edt(binary)
    raw_skel = morphology.skeletonize(binary)
    skel = clean_skel(raw_skel, norm)
    return norm, binary, dist, skel


def global_topology(skel):
    """Validated junctions (merged, exit>=3) and endpoints, extracted ONCE."""
    J = ct.merged_branches(skel)            # (n,2) float cluster centroids
    E = ct.endpoints(skel)                  # (m,2) int
    return J, E


def segments_df(skel):
    """Per skan branch: midpoint (y,x) and length_um. Midpoint = mean of the two
    branch ends (good enough to bin a segment into a tile)."""
    sko = Skeleton(skel)
    df = summarize(sko, separator='-')
    # locate the src/dst coordinate columns regardless of skan version naming
    def col(sub):
        for c in df.columns:
            if c.endswith(sub):
                return c
        raise KeyError(sub)
    sy = df[col('coord-src-0')].to_numpy(); sx = df[col('coord-src-1')].to_numpy()
    dy = df[col('coord-dst-0')].to_numpy(); dx = df[col('coord-dst-1')].to_numpy()
    midy = (sy + dy) / 2.0; midx = (sx + dx) / 2.0
    length_um = df['branch-distance'].to_numpy() * PIXEL_UM
    return midy, midx, length_um
