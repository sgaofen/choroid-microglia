"""
Segmentation sensitivity analysis (P4, GPT-Pro review). The core worry: the
"HET is more fragmented" signal could be a thresholding artifact (faint HET
processes broken by the threshold), not biology. Test it: re-binarize + re-
skeletonize each image with several segmentation methods and check whether the
fragmentation signal (whole-image endpoint/branch ratio up, branch-per-100um
down in HET) SURVIVES across methods. If it holds regardless of segmentation,
it is not a single-threshold artifact.

Methods: otsu07 (current), otsu (stricter), hysteresis (keep faint only if
connected to a bright core), frangi (tubeness/ridge), adaptive (local).
Also renders a raw-vs-skeleton crop under each method to eyeball process
continuity (does a method break faint continuations?).
"""
import sys
from pathlib import Path
import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage import morphology
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import pipeline as pl
from pipeline import PIXEL_UM, STEMS, COND

OUT = pl.ROOT / 'experiments/full_morphology/out_seg'
METHODS = ['otsu07', 'otsu', 'hysteresis', 'frangi', 'adaptive']


def metrics_for(norm, method):
    binary = pl.segment(norm, method)
    skel = pl.clean_skel(morphology.skeletonize(binary), norm)
    J, E = pl.global_topology(skel)
    skel_um = float(skel.sum()) * PIXEL_UM
    n_br, n_ep = len(J), len(E)
    H, W = skel.shape
    return dict(
        n_branch=n_br, n_endpoint=n_ep,
        endpoint_branch_ratio=round(n_ep / max(n_br, 1), 3),
        branch_per_100um=round(100 * n_br / (skel_um + 1e-9), 3),
        endpoint_per_100um=round(100 * n_ep / (skel_um + 1e-9), 3),
        skel_per_tissue_mm2=round(skel_um / (H * W * PIXEL_UM**2 / 1e6), 1),
    )


def verdict(wt, h1, h3, want_up):
    d1, d3 = h1 - wt, h3 - wt
    same = (d1 > 0) == (d3 > 0)
    ok = same and ((d1 > 0) == want_up)
    return ('✓ ' + ('HET↑' if (d1 > 0) else 'HET↓')) if ok and same else (
        '~ both ' + ('up' if d1 > 0 else 'down') if same else '✗ disagree')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    norms = {}
    res = {m: {} for m in METHODS}
    for s in STEMS:
        raw = tifffile.imread(pl.find_raw(s)).astype(np.float32)
        norms[s] = pl.normalize(raw)
        for m in METHODS:
            res[m][s] = metrics_for(norms[s], m)
            print(f'  {s} [{m}]: ep/br={res[m][s]["endpoint_branch_ratio"]} '
                  f'br/100um={res[m][s]["branch_per_100um"]} '
                  f'skel/tissue={res[m][s]["skel_per_tissue_mm2"]}')

    # ---- does the fragmentation signal survive across methods? ----
    print('\n=== fragmentation signal vs segmentation method ===')
    print('(expect HET: endpoint/branch ratio UP, branch/100um DOWN, ep/100um UP)')
    print(f'{"method":>11}{"metric":>22}{"WT":>9}{"HET_1":>9}{"HET_3":>9}{"verdict":>14}')
    checks = [('endpoint_branch_ratio', True), ('branch_per_100um', False),
              ('endpoint_per_100um', True), ('skel_per_tissue_mm2', False)]
    for m in METHODS:
        for key, want_up in checks:
            wt = res[m]['F_WT_2'][key]; h1 = res[m]['F_HET_1'][key]; h3 = res[m]['F_HET_3'][key]
            print(f'{m:>11}{key:>22}{wt:>9}{h1:>9}{h3:>9}{verdict(wt, h1, h3, want_up):>14}')
        print()

    # ---- visual: a HET_1 window under each method (raw + skeleton) ----
    s = 'F_HET_1'
    binb = pl.segment(norms[s], 'otsu07')
    # pick the 700px window with most foreground
    TILE = 700
    H, W = norms[s].shape
    best, bn = (0, 0), -1
    for ty in range(0, H-TILE, 350):
        for tx in range(0, W-TILE, 350):
            c = binb[ty:ty+TILE, tx:tx+TILE].sum()
            if c > bn:
                bn, best = c, (ty, tx)
    ty, tx = best
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    rc = norms[s][ty:ty+TILE, tx:tx+TILE]
    axes[0, 0].imshow(rc, cmap='gray', vmin=0, vmax=np.percentile(rc, 99))
    axes[0, 0].set_title('raw'); axes[0, 0].axis('off')
    for ax, m in zip(axes.ravel()[1:], METHODS):
        b = pl.segment(norms[s], m)
        sk = pl.clean_skel(morphology.skeletonize(b), norms[s])
        skc = sk[ty:ty+TILE, tx:tx+TILE]
        ax.imshow(rc, cmap='gray', vmin=0, vmax=np.percentile(rc, 99))
        yy, xx = np.where(skc)
        ax.scatter(xx, yy, s=0.4, c='#ff3b3b', marker='.')
        ax.set_title(f'{m}: ep/br={res[m][s]["endpoint_branch_ratio"]}'); ax.axis('off')
    fig.suptitle(f'{s} (HET) same window under each segmentation (~145um)', fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / 'segmentation_compare_crop.png', dpi=110, bbox_inches='tight')
    plt.close()
    print('saved to', OUT)


if __name__ == '__main__':
    main()
