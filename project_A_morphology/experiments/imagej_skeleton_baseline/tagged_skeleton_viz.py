"""
Tagged-skeleton visualization in ImageJ AnalyzeSkeleton colors:
  endpoints  = blue
  junctions  = magenta/purple
  slab/branch= orange
Background = original normalized image. One tight crop per genotype so WT and
HET can be eyeballed side by side. This is the exact picture Huixin sees in
the ImageJ AnalyzeSkeleton "tagged skeleton" output.
"""
from pathlib import Path

import numpy as np
import tifffile
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.ndimage import binary_dilation

ROOT = Path('/Users/stephenyu/choroid-microglia/project_A_morphology')
RAW = ROOT / 'data/raw'
V29 = ROOT / 'experiments/v29_short_spur_audit'
OUT = ROOT / 'experiments/imagej_skeleton_baseline'


def find_raw(stem):
    for p in RAW.glob('*.tif'):
        if stem in p.name:
            return p
    raise FileNotFoundError(stem)


def normalize(img, p_lo=1.0, p_hi=99.5):
    lo, hi = np.percentile(img, [p_lo, p_hi])
    return np.clip((img - lo) / (hi - lo + 1e-9), 0, 1)


def degree(skel):
    k = np.ones((3, 3), dtype=np.uint8); k[1, 1] = 0
    return ndi.convolve(skel.astype(np.uint8), k, mode='constant') * skel


def render(stem, yc, xc, half=300):
    raw = tifffile.imread(find_raw(stem)).astype(np.float32)
    norm = normalize(raw)
    skel = np.load(V29 / f'{stem}_skel_pruned.npy').astype(bool)
    H, W = norm.shape
    y0, y1 = max(0, yc - half), min(H, yc + half)
    x0, x1 = max(0, xc - half), min(W, xc + half)

    crop = norm[y0:y1, x0:x1]
    sk = skel[y0:y1, x0:x1]
    deg = degree(sk)
    ep = sk & (deg == 1)
    jt = sk & (deg >= 3)
    sl = sk & (deg == 2)

    # dilate for visibility
    ep_d = binary_dilation(ep, iterations=1)
    jt_d = binary_dilation(jt, iterations=1)
    sl_d = sl  # keep slab thin

    overlay = np.zeros((*sk.shape, 4))
    overlay[sl_d] = (1.0, 0.55, 0.0, 0.9)    # orange slabs
    overlay[jt_d] = (0.8, 0.0, 1.0, 1.0)     # purple junctions
    overlay[ep_d] = (0.0, 0.6, 1.0, 1.0)     # blue endpoints

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    axes[0].imshow(crop, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f'{stem}  ORIG  crop=({y0},{x0})', fontsize=12)
    axes[0].axis('off')
    axes[1].imshow(crop, cmap='gray', vmin=0, vmax=1)
    axes[1].imshow(overlay)
    n_ep = int(ep.sum()); n_jt = int(jt.sum())
    axes[1].set_title(
        f'{stem}  tagged skeleton  '
        f'(blue endpoints={n_ep}, purple junctions={n_jt}, orange slab)',
        fontsize=12)
    axes[1].axis('off')
    fig.tight_layout()
    out = OUT / f'tagged_{stem}.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out.name}  ({out.stat().st_size//1024} KB)  ep={n_ep} jt={n_jt}')


if __name__ == '__main__':
    for stem, yc, xc in [('F_WT_2', 1100, 1100),
                         ('F_HET_1', 2400, 1900),
                         ('F_HET_3', 1500, 1500)]:
        render(stem, yc, xc, half=300)
