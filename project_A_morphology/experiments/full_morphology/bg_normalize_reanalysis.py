"""
Background-normalized re-analysis (per Huixin, 2026-05-28): the WT and HET images
were acquired at different brightness/contrast, which inflates apparent cell
size/area. Fix: instead of a per-image otsu (which bakes in the acquisition
difference), threshold each RAW image at  background_mode + N * background_noise
with the SAME N for all images. This makes the empty space equally black AND
makes "signal" a consistent statistical criterion (robust to both offset and
gain). Then re-skeletonize and recompute, comparing OLD (otsu07) vs NEW.
"""
import sys
from pathlib import Path
import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.ndimage import white_tophat
from skimage import filters, morphology
sys.path.insert(0, str(Path(__file__).parent))
import pipeline as pl
from pipeline import PIXEL_UM, STEMS, COND
import clean_topology as ct

TOPHAT = 31  # rolling-ball radius (px); removes background broader than this


def bg_threshold(raw):
    """FIJI-style rolling-ball background subtraction (white tophat) flattens the
    diffuse background to ~0 in EVERY image (empty space equally black), then a
    consistent otsu*0.7 on the background-free image. Acquisition brightness no
    longer biases the foreground."""
    sm = filters.gaussian(raw, 1.0)
    th = white_tophat(sm, size=TOPHAT)
    t = float(filters.threshold_otsu(th))
    b = th > t * 0.7
    b = morphology.binary_closing(b, morphology.disk(2))
    b = morphology.remove_small_objects(b, 20)
    return b, t, float((th > t * 0.7).mean())


def metrics(binary, norm):
    H, W = binary.shape
    img_mm2 = H * W * PIXEL_UM ** 2 / 1e6
    total_area = float(binary.sum()) * PIXEL_UM ** 2 / 1e6
    skel = ct.prune_spurs(ct.break_loops(ct.clean(morphology.skeletonize(binary)), norm), 8)
    skel_um = float(skel.sum()) * PIXEL_UM
    J = ct.merged_branches(skel); E = ct.endpoints(skel)
    nb, ne = len(J), len(E)
    lab, ncc = ndi.label(binary, structure=np.ones((3, 3)))
    areas = np.bincount(lab.ravel())[1:] * PIXEL_UM ** 2
    sl = np.bincount(ndi.label(skel, structure=np.ones((3, 3)))[0][skel],
                     minlength=1)[1:] * PIXEL_UM if skel.any() else np.array([0.0])
    return dict(
        fg_frac=round(total_area / img_mm2, 4),
        total_area_mm2=round(total_area, 4),
        skel_per_img_mm2=round(skel_um / img_mm2, 0),
        ep_br=round(ne / max(nb, 1), 3),
        branch_per_100um=round(100 * nb / (skel_um + 1e-9), 3),
        n_components=int(ncc),
        median_cell_area_um2=round(float(np.median(areas)), 1),
        pct_skel_large=round(100 * float(sl[sl > 100].sum()) / (sl.sum() + 1e-9), 1),
    )


def verdict(wt, h1, h3):
    d1, d3 = h1 - wt, h3 - wt
    same = (d1 > 0) == (d3 > 0)
    gap = (abs(d1) + abs(d3)) / 2; spread = abs(d1 - d3)
    return '✓ CLEAN' if (same and spread < gap and gap > 0) else ('~ same-side' if same else '✗ disagree')


def main():
    OLD, NEW = {}, {}
    for s in STEMS:
        raw = tifffile.imread(pl.find_raw(s)).astype(np.float32)
        norm = pl.normalize(raw)
        # OLD: per-image otsu*0.7 (the acquisition-confounded way)
        b_old = pl.segment(norm, 'otsu07')
        OLD[s] = metrics(b_old, norm)
        # NEW: rolling-ball background subtraction + consistent otsu*0.7
        b_new, t, fg_raw = bg_threshold(raw)
        NEW[s] = metrics(b_new, norm)
        print(f'{s}: tophat-otsu t={t:.1f}  fg OLD={OLD[s]["fg_frac"]} NEW={NEW[s]["fg_frac"]}')

    keys = [('total_area_mm2', 'total Iba1 area (mm2)'),
            ('fg_frac', 'foreground fraction'),
            ('skel_per_img_mm2', 'skeleton / image area'),
            ('median_cell_area_um2', 'median object area (um2)'),
            ('n_components', 'n connected pieces'),
            ('pct_skel_large', '% skel in >100um pieces (connectivity)'),
            ('branch_per_100um', 'branch / 100um'),
            ('ep_br', 'endpoint/branch ratio (fragmentation)')]

    for tag, D in [('OLD (otsu*0.7, per-image — acquisition-confounded)', OLD),
                   (f'NEW (rolling-ball bg subtraction, tophat={TOPHAT}px + otsu*0.7)', NEW)]:
        print(f'\n================ {tag} ================')
        print(f'{"metric":<42}{"WT":>10}{"HET_1":>10}{"HET_3":>10}{"verdict":>14}')
        for k, label in keys:
            wt, h1, h3 = D['F_WT_2'][k], D['F_HET_1'][k], D['F_HET_3'][k]
            print(f'{label:<42}{wt:>10}{h1:>10}{h3:>10}{verdict(wt, h1, h3):>14}')


if __name__ == '__main__':
    main()
