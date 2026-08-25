"""Cell Skeleton Tuning Workbench — backend (microglia / macrophage skeleton extraction)

Data: the tif_max directory named by config.SRC_TIF_MAX (uncompressed TIF, memmap tiling)
      Override without editing this file: CHP_SRC=/path/to/tif_max ./.venv/bin/python server.py
Recipe is based on pipeline_v10 (local background subtraction -> PSF smoothing -> local noise z
-> lee skeleton, no post-processing), but every parameter is exposed for tuning, and alternative
paths (Sato tubeness enhancement / Otsu / percentile threshold) have been added.

Usage: .venv/bin/python server.py   -> http://localhost:8901
"""
import base64
import io
import json
import os
import re
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold, sato, threshold_otsu
from skimage.measure import label as sklabel
from skimage.morphology import closing, disk, remove_small_objects, skeletonize

from config import IMG_PX, PIX_UM, SRC_TIF_MAX  # noqa: F401  (IMG_PX re-exported for the batch scripts)

TOOL = os.path.dirname(os.path.abspath(__file__))
SRC = SRC_TIF_MAX

# INPUT FILENAME CONTRACT. Source TIFs must be named
#   <prefix>_<sample>_<stain>_<channel>.tif   e.g. 10x_M-WT1_CCR2-CD45_C0.tif -> sample "M-WT1"
# Anything that does not match FILE_RE is invisible to the tuner: the sample dropdown simply
# comes up empty, with no error. Rename the files, or override without editing this file:
#   CHP_FILE_RE='20x_(.+)_MYSTAIN_C0\.tif$' CHP_FILE_FMT='20x_{sample}_MYSTAIN_{ch}.tif' ...
FILE_RE = os.environ.get("CHP_FILE_RE", r"10x_(.+)_CCR2-CD45_C0\.tif$")
FILE_FMT = os.environ.get("CHP_FILE_FMT", "10x_{sample}_CCR2-CD45_{ch}.tif")

DS = 4  # downsampling factor for background / local statistics (same as v10)
K8 = np.ones((3, 3), bool)
MAX_ROI_PX = 4096

# Nothing else creates this directory, and global_sigma() / the experiment scripts write into it.
os.makedirs(f"{TOOL}/cache", exist_ok=True)

_memmaps = {}
_mm_lock = threading.Lock()
_sigma_cache = {}


def list_samples():
    names = []
    for p in sorted(os.listdir(SRC)):
        m = re.match(FILE_RE, p)
        if m:
            names.append(m.group(1))
    return names


def get_mm(sample, ch):
    key = (sample, ch)
    with _mm_lock:
        if key not in _memmaps:
            path = f"{SRC}/" + FILE_FMT.format(sample=sample, ch=ch)
            _memmaps[key] = tifffile.memmap(path, mode="r")
        return _memmaps[key]


def get_clean(sample, ch):
    """Clean image (fjff flat-field + whole-image uniform background cleanup, produced by
    preprocess_clean_images.py). Returns None if that step has not been run yet."""
    key = (sample, ch, "clean")
    tp, jp = f"{TOOL}/clean/{sample}_{ch}_clean.tif", f"{TOOL}/clean/{sample}_{ch}_clean.json"
    if not (os.path.exists(tp) and os.path.exists(jp)):
        return None, None
    with _mm_lock:
        if key not in _memmaps:
            _memmaps[key] = (tifffile.memmap(tp, mode="r"), json.load(open(jp))["sigma"])
        return _memmaps[key]


def global_sigma(sample, ch):
    """Sample-level noise floor (v10's sg_glob): 8x subsampling + tissue mask + lower-70% MAD.
    Cached to disk."""
    key = f"{sample}_{ch}"
    if key in _sigma_cache:
        return _sigma_cache[key]
    cpath = f"{TOOL}/cache/sigma_{key}.json"
    if os.path.exists(cpath):
        v = json.load(open(cpath))["sigma"]
        _sigma_cache[key] = v
        return v
    a = np.asarray(get_mm(sample, ch)[::8, ::8], np.float32)
    sm = ndi.gaussian_filter(a, 8)
    tm = sm > max(np.percentile(sm, 20), threshold_otsu(sm) * 0.25)
    vt = a[tm] if tm.any() else a.ravel()
    lo = vt[vt < np.percentile(vt, 70)]
    sg = max(1.4826 * float(np.median(np.abs(lo - np.median(lo)))), 1e-6)
    json.dump({"sigma": sg}, open(cpath, "w"))
    _sigma_cache[key] = sg
    return sg


OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def prune_spurs_by_width(sk, dist_px, k, min_um=0.0):
    """Delete terminal twigs (tip -> junction) shorter than k x (local radius at the junction).

    Skeletonizing thick blobs inevitably grows short spurs off edge bumps, on the same scale as
    the local radius (geometric artifact, not a real branch). On thin structures the local radius
    is small (~1px), so the threshold automatically drops to removing only 1-2px noise spurs ->
    real branches are unaffected.
    Isolated line-segment components with no junction are kept whole.
    """
    if k <= 0:
        return sk
    sk = sk.copy()
    nc = ndi.convolve(sk.astype(np.uint8), K8.astype(np.uint8), mode="constant") - sk
    H, W = sk.shape
    for y0, x0 in np.argwhere(sk & (nc == 1)):
        py, px = -1, -1
        cy, cx = int(y0), int(x0)
        path, length, hit_junction = [(cy, cx)], 0.0, False
        for _ in range(300):
            nxt = None
            multi = False
            for dy, dx in OFFS:
                ny, nx_ = cy + dy, cx + dx
                if 0 <= ny < H and 0 <= nx_ < W and sk[ny, nx_] and (ny, nx_) != (py, px):
                    if nxt is not None:
                        multi = True
                        break
                    nxt = (ny, nx_, 1.4142 if (dy and dx) else 1.0)
            if multi:                        # this pixel is itself a fork -> twig ends here
                path.pop()
                hit_junction = True
                break
            if nxt is None:                  # dead end: isolated segment, keep it
                break
            ny, nx_, step = nxt
            if nc[ny, nx_] >= 3:             # reached a junction pixel -> twig ends
                cy, cx = ny, nx_
                length += step
                hit_junction = True
                break
            py, px, cy, cx = cy, cx, ny, nx_
            path.append((cy, cx))
            length += step
        if not hit_junction or not path:
            continue
        # threshold = max(k x local radius at the junction, absolute floor): on thin structures the
        # radius is ~1px so the floor takes over (else all 2-4um jitter twigs survive, i.e. the
        # source of "a pile of forks in small places")
        if length < max(k * max(float(dist_px[cy, cx]), 1.0), min_um / PIX_UM):
            for qy, qx in path:
                sk[qy, qx] = False
    return sk


def sato_threaded(img, s, n_threads=None):
    """Sato with row-slice multithreading: scipy's gaussian/Hessian release the GIL, so all cores
    can be saturated."""
    from concurrent.futures import ThreadPoolExecutor
    n = n_threads or min(8, os.cpu_count() or 4)
    H = img.shape[0]
    if H < 512 or n < 2:
        return sato(img, sigmas=[s], black_ridges=False, mode="reflect").astype(np.float32)
    ov = int(4 * s + 8)
    step = (H + n - 1) // n
    out = np.empty(img.shape, np.float32)

    def work(i):
        a0, a1 = i * step, min(H, (i + 1) * step)
        if a0 >= a1:
            return
        b0, b1 = max(0, a0 - ov), min(H, a1 + ov)
        r = sato(img[b0:b1], sigmas=[s], black_ridges=False, mode="reflect")
        out[a0:a1] = r[a0 - b0: a0 - b0 + (a1 - a0)]

    with ThreadPoolExecutor(n) as ex:
        list(ex.map(work, range(n)))
    return out


def break_loops(sk, img, max_iter=3):
    """Break small skeleton loops (repo clean_topology rule): cut the loop at its darkest
    degree-2 pixel.

    Ragged mask edges -> the lee skeleton grows small parallel loops, each adding +2 phantom
    junctions. Once broken, a loop becomes an ordinary bend; the short stub left behind is
    cleaned up by spur pruning. Only degree-2 pixels are removed, so connectivity is never broken.
    """
    sk = sk.copy()
    for _ in range(max_iter):
        holes = ndi.binary_fill_holes(sk) & ~sk
        hl, nh = ndi.label(holes, structure=K8)
        if not nh:
            break
        nc = ndi.convolve(sk.astype(np.uint8), K8.astype(np.uint8), mode="constant") - sk
        changed = False
        for sl_ in ndi.find_objects(hl):
            if sl_ is None:
                continue
            sl2 = (slice(max(0, sl_[0].start - 2), min(sk.shape[0], sl_[0].stop + 2)),
                   slice(max(0, sl_[1].start - 2), min(sk.shape[1], sl_[1].stop + 2)))
            hole = hl[sl2] > 0
            ring = ndi.binary_dilation(hole, structure=K8) & sk[sl2] & (nc[sl2] == 2)
            if not ring.any():
                continue
            ys, xs = np.nonzero(ring)
            j = int(np.argmin(img[sl2][ys, xs]))
            sk[sl2[0].start + ys[j], sl2[1].start + xs[j]] = False
            changed = True
        if not changed:
            break
    return sk



def _gap_frac_rel(v_sk, a_view):
    """Gap fraction v2: a skeleton pixel counts as a gap only if its intensity is < 20% of the
    median intensity of its own component. Each cell is compared against itself -> insensitive to
    overall brightness differences between images (uncalibrated, sigma spans 6x)."""
    n = int(v_sk.sum())
    if n == 0:
        return 0.0
    lab_g, ng = ndi.label(v_sk, structure=K8)
    med = ndi.labeled_comprehension(a_view, lab_g, np.arange(1, ng + 1), np.median, np.float64, 0.0)
    lut = np.concatenate([[np.inf], med])          # label 0 = background gets inf, never misjudged
    gap = v_sk & (a_view < 0.2 * lut[lab_g])
    return round(float(gap.sum()) / n, 4)


# ---------------- core processing (on a crop with padding) ----------------

def process(sample, ch, x, y, w, h, P):
    """Returns a dict: the image layers (ndarray) + metrics. x,y,w,h are full-resolution pixels."""
    clean_mm, clean_sigma = (get_clean(sample, ch) if P["use_clean"] else (None, None))
    if P["use_clean"] and clean_mm is None:
        # Falling back to the raw tif_max here would silently change almost every metric
        # (measured on F-WT1 ROI#1: 16 of 17 metrics move). Fail loudly instead.
        raise FileNotFoundError(
            f"use_clean=1 but the clean base image is missing:\n"
            f"    {TOOL}/clean/{sample}_{ch}_clean.tif  (+ the matching _clean.json)\n"
            f"Either symlink the lab drive's clean/ into this folder\n"
            f"    ln -s <lab drive>/ChP_morphometry_2026-08/images/clean {TOOL}/clean\n"
            f"or rebuild it with:  ./.venv/bin/python preprocess_clean_images.py {ch}\n"
            f"(Set use_clean=0 only if you deliberately want raw tif_max numbers - they do NOT "
            f"match any published value.)")
    mm = clean_mm if clean_mm is not None else get_mm(sample, ch)
    sg_all = clean_sigma if clean_mm is not None else global_sigma(sample, ch)
    H, W = mm.shape
    w, h = min(int(w), MAX_ROI_PX), min(int(h), MAX_ROI_PX)
    x = min(max(0, int(x)), max(0, W - w))
    y = min(max(0, int(y)), max(0, H - h))
    w, h = min(w, W - x), min(h, H - y)

    pad = int(min(256, max(64, P["bg_um"] / PIX_UM * 1.5, P["local_win_um"] / PIX_UM * 0.75)))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    a = np.asarray(mm[y0:y1, x0:x1], np.float32)
    vy, vx = y - y0, x - x0  # offset of the view region within the crop

    # 1) local background subtraction (grey opening, computed at DS downsampling); the clean image
    #    already had a uniform whole-image subtraction, so skip it
    flat = a
    if P["bg_um"] > 0 and clean_mm is None:
        small = a[::DS, ::DS]
        r = max(2, int(round(P["bg_um"] / (PIX_UM * DS))))
        bg = ndi.gaussian_filter(ndi.grey_opening(ndi.uniform_filter(small, 3), footprint=disk(r)), r / 2)
        up = np.empty(a.shape, np.float32)
        ndi.zoom(bg, (a.shape[0] / bg.shape[0], a.shape[1] / bg.shape[1]), output=up, order=1)
        flat = np.clip(a - up, 0, None)

    # the defocus-gate sample must be taken BEFORE smoothing (smoothing wipes out fine-scale
    # energy, driving clarity to zero everywhere)
    d2_pre = flat[::2, ::2].copy() if P["focus_thr"] > 0 else None

    # 2) PSF-scale smoothing
    if P["smooth_um"] > 0:
        flat = ndi.gaussian_filter(flat, P["smooth_um"] / PIX_UM)

    # 3) optional: Sato tubeness enhancement (sensitive to thin processes, insensitive to haze);
    #    row-slice multithreading (scipy releases the GIL)
    enh = flat
    if P["enhance"] == "sato":
        s = max(0.6, P["sato_um"] / PIX_UM)
        enh = sato_threaded(flat, s)

    def local_z(img, floor):
        """z map from local median/MAD (estimated at 8x downsampling — the statistics of a 40um
        window are accurate enough at 8x, and 8x faster)."""
        DSS = 8
        fs = img[::DSS, ::DSS]
        t = max(3, int(round(P["local_win_um"] / (PIX_UM * DSS))))
        med = ndi.median_filter(fs, size=t)
        mad = ndi.median_filter(np.abs(fs - med), size=t)
        umed = np.empty(img.shape, np.float32)
        ndi.zoom(med, (img.shape[0] / med.shape[0], img.shape[1] / med.shape[1]), output=umed, order=1)
        umad = np.empty(img.shape, np.float32)
        ndi.zoom(mad, (img.shape[0] / mad.shape[0], img.shape[1] / mad.shape[1]), output=umad, order=1)
        return (img - umed) / np.maximum(1.4826 * umad, floor)

    # 4) threshold
    if P["thr_mode"] in ("zlocal", "zhyst"):
        def thr(zm):
            if P["thr_mode"] == "zhyst":
                # hysteresis threshold: the high threshold sets seeds, the low threshold only lets
                # in dim segments connected to a seed (keeps continuity, does not swallow haze)
                return apply_hysteresis_threshold(zm, min(P["z_lo"], P["z_thr"] - 0.1), P["z_thr"])
            return zm > P["z_thr"]

        z_int = local_z(flat, sg_all)
        if P["enhance"] == "sato":
            v = enh.ravel()
            lo = v[v < np.percentile(v, 70)]
            floor = max(1.4826 * float(np.median(np.abs(lo - np.median(lo)))), 1e-6)
            m0 = thr(local_z(enh, floor))
            # brightness gate: beyond shape (sato), also require the raw intensity local z to pass a
            # loose threshold; this specifically kills "speckle haze strung into chains"
            if P["gate_z"] > 0:
                m0 &= z_int > P["gate_z"]
            # union: on thick bright structures sato only responds at the edges (they get cut into
            # parallel strips), so the intensity mask fills the whole block back in; thin dim
            # processes rely on sato.
            if P["sato_union"]:
                m0 |= thr(z_int)
        else:
            m0 = thr(z_int)
    elif P["thr_mode"] == "otsu":
        pos = enh[enh > 0]
        thr = threshold_otsu(pos) * P["otsu_k"] if pos.size > 100 else np.inf
        m0 = enh > thr
    else:  # pct
        m0 = enh > np.percentile(enh, P["pct"])

    # 4b) defocus map (clarity = fine-scale 0.5-2um energy / haze-scale 4-16um energy, same as
    #     clarity_select); not cut per pixel (that would hurt dim segments of real processes) but
    #     filtered after the skeleton is built, by "component mean clarity" (see 6a3)
    focus_ratio = None
    if P["focus_thr"] > 0:
        d2 = d2_pre
        px2 = PIX_UM * 2

        def gb(s_um):
            return ndi.gaussian_filter(d2, s_um / px2 * 0.5)

        fine = ndi.gaussian_filter(np.abs(gb(0.5) - gb(2.0)), 5.0 / px2)
        coarse = ndi.gaussian_filter(np.abs(gb(4.0) - gb(16.0)), 5.0 / px2)
        rat = fine / np.maximum(coarse, 1e-6)
        focus_ratio = np.empty(flat.shape, np.float32)
        ndi.zoom(rat, (flat.shape[0] / rat.shape[0], flat.shape[1] / rat.shape[1]),
                 output=focus_ratio, order=1)

    # Debug layer keys (only populated when P["dbg"] is set). Kept in sync with
    # historical reasons and are an exact contract with diagnose_missed_round_cells.py -
    # rename them in both files or not at all. Glossary:
    #   1_raw_mask_after_threshold          stage 1: raw mask straight after thresholding
    #   2_mask_after_cleanup        stage 2: mask after closing / small-object removal / hole fill
    #   3_skeleton_after_length_filter      stage 3: skeleton after the tiered min-length filter
    #   4_skeleton_after_round_and_web_filters  stage 4: skeleton after the round-cell and speckle-web filters
    _dbg = {}
    if P.get("dbg"):
        _dbg["1_raw_mask_after_threshold"] = m0.copy()          # stage 1: after threshold

    # 5) cleanup
    if P["close_px"] > 0:
        m0 = closing(m0, np.ones((int(P["close_px"]),) * 2, bool))
    if P["min_area_um2"] > 0:
        m0 = remove_small_objects(m0, min_size=int(round(P["min_area_um2"] / PIX_UM ** 2)), connectivity=2)
    mask = ndi.binary_fill_holes(m0) if P["fill_holes"] else m0

    if P.get("dbg"):
        _dbg["2_mask_after_cleanup"] = mask.copy()      # stage 2: after cleanup

    # 6) skeleton (raw lee) + loop breaking + width-adaptive spur pruning + tiered
    #    minimum-skeleton-length filter
    sk = skeletonize(mask, method="lee").astype(bool)
    dist = ndi.distance_transform_edt(mask)
    if P["loop_break"]:
        sk = break_loops(sk, flat)
    if P["spur_k"] > 0:
        sk = prune_spurs_by_width(sk, dist, P["spur_k"], P["spur_min_um"])
    soma = mask & (dist * PIX_UM >= P["soma_r_um"])
    if P["soma_merge_um"] > 0:
        # soma merge: two real soma centres can never be < ~6um apart; a dumbbell whose waist got
        # split into two is rejoined by closing
        rm = max(1, int(round(P["soma_merge_um"] / 2 / PIX_UM)))
        soma = ndi.binary_closing(soma, structure=disk(rm))
    lab, ncomp = ndi.label(sk, structure=K8)
    if ncomp:
        # components with a soma (real cells / branches hanging off a cell) use min_skel_um;
        # soma-free isolated fragments must meet min_skel_free_um (debris cannot be a cell)
        sizes = ndi.sum_labels(sk, lab, index=np.arange(1, ncomp + 1)) * PIX_UM
        anchored = np.zeros(ncomp + 1, bool)
        near_soma = ndi.binary_dilation(soma, structure=K8, iterations=6)
        anchored[np.unique(lab[sk & near_soma])] = True
        anchored[0] = False
        keep = np.zeros(ncomp + 1, bool)
        # has a soma -> always keep (a round cell's skeleton is inherently only a few pixels;
        # killing by length would slaughter every round-dot cell);
        # no soma -> still require min_skel_free_um (debris cannot be a cell)
        keep[1:] = anchored[1:] | (sizes >= max(P["min_skel_free_um"], P["min_skel_um"]))
        sk &= keep[lab]
        lab, ncomp = ndi.label(sk, structure=K8)

    if P.get("dbg"):
        _dbg["3_skeleton_after_length_filter"] = sk.copy()      # stage 3: after tiered size filter

    # 6a3) defocus component filter: a component whose mean clarity along the skeleton is below the
    #      threshold = an out-of-focus chain of blurry dots, delete the whole component.
    #      Averaged per component, not per pixel: a real cell's sharp trunk lifts the mean, so its
    #      dim segments are not dragged down
    if focus_ratio is not None and ncomp:
        mean_f = ndi.labeled_comprehension(focus_ratio, lab, np.arange(1, ncomp + 1),
                                           np.mean, np.float64, 1e9)
        keepf = np.ones(ncomp + 1, bool)
        keepf[1:] = mean_f >= P["focus_thr"]
        sk &= keepf[lab]
        lab, ncomp = ndi.label(sk, structure=K8)

    # 6a2) round-dot filter: skeleton almost entirely shrunk around the soma and very short = a
    #      bright dot with no processes, not a cell
    #      speckle-web filter: a very short component yet packed with junction pixels = a small web
    #      of connected speckles, not a cell
    # (round dots are no longer deleted; classification and counting are unified in the metrics section)
    if ncomp and (P["dot_filter"] or P["web_filter"]):
        sizes = ndi.sum_labels(sk, lab, index=np.arange(1, ncomp + 1)) * PIX_UM
        drop = np.zeros(ncomp + 1, bool)
        # dot_filter is now pure classification (a round dot is a cell, not junk); nothing is
        # deleted any more, counting happens in the metrics section
        if P["web_filter"]:
            # measured calibration (F-Het1 dense region): junk fragment = no soma + short + carries
            # junction clusters (23.7um with 3 clusters); real cells never reach that branch
            # density, and they all carry a soma
            near6 = ndi.binary_dilation(soma, structure=K8, iterations=6)
            anch = np.zeros(ncomp + 1, bool)
            anch[np.unique(lab[sk & near6])] = True
            anch[0] = False
            ncmap = ndi.convolve(sk.astype(np.uint8), K8.astype(np.uint8), mode="constant") - sk
            jpix = sk & (ncmap >= 3)
            njc = np.zeros(ncomp + 1, np.int64)
            if jpix.any():
                jr = max(1, int(round(P["junc_merge_um"] / 2 / PIX_UM))) if P["junc_merge_um"] > 0 else 1
                jlab, _ = ndi.label(ndi.binary_dilation(jpix, structure=disk(jr)), structure=K8)
                sel = sk & (jlab > 0)
                pairs = np.unique(jlab[sel].astype(np.int64) * (ncomp + 1) + lab[sel])
                njc = np.bincount(pairs % (ncomp + 1), minlength=ncomp + 1)
            # v3 (2026-08-13 second calibration): only kill indisputable junk — short + forked +
            # no soma + dim along its length. Anything of doubtful brightness or length is kept
            # (v2 wrongly killed 30-50um real cells in dense bands)
            sgf = sg_all
            med_b = ndi.labeled_comprehension(
                np.clip(flat - 2 * sgf, 0, None), lab, np.arange(1, ncomp + 1),
                np.median, np.float64, 0.0) / max(sgf, 1e-9)
            drop[1:] |= ~anch[1:] & (sizes < 25.0) & (njc[1:] >= 1) & (med_b < 10.0)
        if drop.any():
            sk &= ~drop[lab]
            lab, ncomp = ndi.label(sk, structure=K8)

    if P.get("dbg"):
        _dbg["4_skeleton_after_round_and_web_filters"] = sk.copy()  # stage 4: after round-cell + speckle-web filters

    # 6b) discard whole mask blobs that have no skeleton: the skeleton being filtered out means the
    #     system judged it not a cell, so the blue marking and the foreground % should no longer
    #     include it (noise dots are no longer tinted)
    if P["drop_nosk"]:
        lab_m, nm = ndi.label(mask, structure=K8)
        if nm:
            has_sk = np.zeros(nm + 1, bool)
            has_sk[np.unique(lab_m[sk])] = True
            has_sk[0] = False
            mask &= has_sk[lab_m]

    # 7) metrics (view region only, padding excluded)
    if P["drop_nosk"]:
        soma &= mask
    # background-cleaned display (affects display only, not extraction): after background
    # subtraction, anything below clean_floor x sigma is crushed to black — i.e. the repo's rule
    # that "blank areas are equally black in every image" (Huixin's brightness normalization)
    if P["clean_bg"]:
        disp = np.clip(flat - P["clean_floor"] * sg_all, 0, None)
    else:
        disp = a

    sl = (slice(vy, vy + h), slice(vx, vx + w))
    v_sk, v_mask, v_soma = sk[sl], mask[sl], soma[sl]
    nc = ndi.convolve(sk.astype(np.uint8), K8.astype(np.uint8), mode="constant") - sk
    v_nc = nc[sl]
    jpix = v_sk & (v_nc >= 3)
    if P["junc_merge_um"] > 0 and jpix.any():
        # junction merge (same rule as the repo): a cluster of junction points within 3um counts as one
        jr = max(1, int(round(P["junc_merge_um"] / 2 / PIX_UM)))
        _, n_j = ndi.label(ndi.binary_dilation(jpix, structure=disk(jr)), structure=K8)
        n_j = int(n_j)
    else:
        n_j = int(sklabel(jpix, connectivity=2).max())
    n_t = int(sklabel(v_sk & (v_nc == 1), connectivity=2).max())
    n_soma = int(sklabel(v_soma, connectivity=2).max())
    n_comp = int(sklabel(v_sk, connectivity=2).max())

    # round-dot / amoeboid count (first step of classification): kept components that are "short
    # with most of the skeleton shrunk inside the soma", plus ones removed by the round-dot filter
    # whose centroid lies in the view. round_frac = round cells / (round cells + branched components)
    round_marks = []
    n_round = 0
    lab_v, ncv = ndi.label(v_sk, structure=K8)
    n_kept_round = 0
    if ncv:
        near_v = ndi.binary_dilation(v_soma, structure=K8, iterations=3)
        sz_v = ndi.sum_labels(v_sk, lab_v, index=np.arange(1, ncv + 1)) * PIX_UM
        in_v = ndi.sum_labels(v_sk & near_v, lab_v, index=np.arange(1, ncv + 1)) * PIX_UM
        cand_ids = np.where((sz_v < 20.0) & (in_v / np.maximum(sz_v, 1e-9) > 0.6))[0] + 1
        if len(cand_ids):
            # hard brightness gate: real round dots are "very, very bright" (candidate brightness
            # distribution p10 ~ 50 sigma; dim blurry blobs sit in the low tail)
            a_view = a[sl]
            objs_v = ndi.find_objects(lab_v)
            for i in cand_ids:
                o = objs_v[i - 1]
                oy0, oy1 = max(0, o[0].start - 8), min(a_view.shape[0], o[0].stop + 8)
                ox0, ox1 = max(0, o[1].start - 8), min(a_view.shape[1], o[1].stop + 8)
                reg = v_mask[oy0:oy1, ox0:ox1]
                pk = float(a_view[oy0:oy1, ox0:ox1][reg].max()) / max(sg_all, 1e-9) if reg.any() else 0.0
                if pk >= P["round_bright_z"]:
                    n_kept_round += 1
                    cm = ndi.center_of_mass(lab_v == i)
                    round_marks.append(cm)
    n_round += n_kept_round
    area_mm2 = w * h * (PIX_UM / 1e3) ** 2
    skel_um = float(v_sk.sum()) * PIX_UM
    width_um = float((2 * dist * PIX_UM)[sl][v_sk].mean()) if v_sk.any() else 0.0

    metrics = dict(
        area_mm2=round(area_mm2, 5),
        fg_pct=round(100 * float(v_mask.mean()), 3),
        skel_len_um=round(skel_um, 1),
        skel_mm_per_mm2=round(skel_um / 1e3 / area_mm2, 3),
        n_junctions=n_j, n_tips=n_t, n_comp=n_comp, n_soma=n_soma,
        junc_per_mm2=round(n_j / area_mm2, 1),
        tips_per_mm2=round(n_t / area_mm2, 1),
        soma_per_mm2=round(n_soma / area_mm2, 1),
        junc_per_100um=round(100 * n_j / skel_um, 3) if skel_um else 0.0,
        mean_width_um=round(width_um, 3),
        n_round=n_round,
        round_per_mm2=round(n_round / area_mm2, 1),
        round_frac=round(n_round / max(n_round + (n_comp - n_kept_round), 1), 4),
        skel_gap_frac=_gap_frac_rel(v_sk, a[sl]),
    )
    out = dict(a=disp[sl], sk=v_sk, mask=v_mask, soma=v_soma, nc=v_nc, metrics=metrics, sg=sg_all,
               round_marks=round_marks)
    if P.get("dbg"):
        out["dbg"] = {k: v[sl].copy() for k, v in _dbg.items()}
    return out


# ---------------- rendering ----------------

def stretch(a, lo_p, hi_p, gamma):
    lo, hi = np.percentile(a, [lo_p, hi_p])
    g = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    if abs(gamma - 1.0) > 1e-3:
        g = g ** (1.0 / gamma)
    return (g * 255).astype(np.uint8)


def stretch_uni(a, sg, hi_sigma, gamma):
    """Uniform-scale display: 0 = background black, white point = fixed hi_sigma x sigma — one
    brightness yardstick for the whole batch, so a weak image is supposed to look weak."""
    g = np.clip(a / max(hi_sigma * sg, 1e-6), 0, 1)
    if abs(gamma - 1.0) > 1e-3:
        g = g ** (1.0 / gamma)
    return (g * 255).astype(np.uint8)


def disp_gray(a, sg, P):
    return (stretch_uni(a, sg, P["uni_hi"], P["gamma"]) if P["uni_disp"]
            else stretch(a, P["disp_lo"], P["disp_hi"], P["gamma"]))


def render(r, P):
    g = disp_gray(r["a"], r.get("sg", 1.0), P)
    orig = np.stack([g] * 3, -1)

    ov = orig.copy().astype(np.int16)
    al = P["mask_alpha"]
    m = r["mask"]
    ov[m] = ov[m] * (1 - al) + np.array([40, 90, 200]) * al  # mask = blue
    sk_d = ndi.binary_dilation(r["sk"], K8) if P["fat_skel"] else r["sk"]
    ov[sk_d] = [255, 60, 60]                                  # skeleton = red
    jpix = r["sk"] & (r["nc"] >= 3)
    if P["junc_merge_um"] > 0 and jpix.any():
        # draw one green dot at the centroid of each merged cluster (same as repo: within 3um = one junction)
        jr = max(1, int(round(P["junc_merge_um"] / 2 / PIX_UM)))
        jlab, nj = ndi.label(ndi.binary_dilation(jpix, structure=disk(jr)), structure=K8)
        jd = np.zeros_like(jpix)
        for cy, cx in ndi.center_of_mass(jlab > 0, jlab, range(1, nj + 1)):
            jd[int(round(cy)), int(round(cx))] = True
        jd = ndi.binary_dilation(jd, disk(3))
    else:
        jd = ndi.binary_dilation(jpix, K8, iterations=2)
    td = ndi.binary_dilation(r["sk"] & (r["nc"] == 1), K8, iterations=2)
    ov[jd] = [80, 255, 80]                                    # junction = green
    ov[td] = [80, 230, 255]                                   # tip = cyan
    som = r["soma"] & ~ndi.binary_erosion(r["soma"])
    ov[ndi.binary_dilation(som, K8)] = [255, 220, 60]         # soma outline = yellow
    # round-dot cell = magenta circle (detection marker; drawn in a local window)
    H_, W_ = r["sk"].shape
    R_ = 19
    for cy_, cx_ in r.get("round_marks", []):
        y0_, y1_ = max(0, int(cy_) - R_ - 1), min(H_, int(cy_) + R_ + 2)
        x0_, x1_ = max(0, int(cx_) - R_ - 1), min(W_, int(cx_) + R_ + 2)
        yy0, xx0 = np.ogrid[y0_:y1_, x0_:x1_]
        d2 = (yy0 - cy_) ** 2 + (xx0 - cx_) ** 2
        ov[y0_:y1_, x0_:x1_][(d2 >= 16 ** 2) & (d2 <= 18.5 ** 2)] = [255, 70, 255]
    overlay = np.clip(ov, 0, 255).astype(np.uint8)

    lab, n = ndi.label(r["sk"], structure=K8)
    comp = np.stack([g // 3] * 3, -1)
    if n:
        rng = np.random.default_rng(0)
        lut = np.concatenate([[[0, 0, 0]], rng.integers(70, 256, (n, 3))]).astype(np.uint8)
        cd = ndi.grey_dilation(lab, footprint=K8)
        on = cd > 0
        comp[on] = lut[cd[on]]
    return orig, overlay, comp


def to_b64(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "png", optimize=False, compress_level=3)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------- parameter parsing ----------------

DEFAULTS = dict(
    ch="C0", disp_lo=1.0, disp_hi=99.7, gamma=1.0, mask_alpha=0.35, fat_skel=1,
    clean_bg=1, clean_floor=2.0, use_clean=1, uni_disp=0, uni_hi=25.0,
    bg_um=8.0, smooth_um=0.6, enhance="sato", sato_um=0.6, sato_union=1,
    thr_mode="zlocal", z_thr=3.5, z_lo=2.0, gate_z=2.0, local_win_um=40.0, otsu_k=0.7, pct=99.0,
    close_px=2, min_area_um2=6.0, fill_holes=1, min_skel_um=5.0, min_skel_free_um=15.0,
    soma_r_um=2.4, spur_k=1.2, spur_min_um=4.0, drop_nosk=1, junc_merge_um=3.0, soma_merge_um=6.0,
    round_bright_z=50.0,
    dot_filter=1, web_filter=1, loop_break=1, focus_thr=0.0,
)


def parse_params(q):
    P = dict(DEFAULTS)
    for k, v in q.items():
        if k in P and v:
            P[k] = v[0] if isinstance(P[k], str) else type(P[k])(float(v[0]))
    return P


# ---------------- ROI load / save ----------------

def roi_path(sample):
    return f"{TOOL}/rois/{sample}.json"


def load_rois(sample):
    p = roi_path(sample)
    return json.load(open(p)) if os.path.exists(p) else []


def save_rois(sample, rois):
    json.dump(rois, open(roi_path(sample), "w"), ensure_ascii=False, indent=1)


# ---------------- HTTP ----------------

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        try:
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/":
                self._send(200, open(f"{TOOL}/index.html", "rb").read(), "text/html; charset=utf-8")
            elif u.path == "/api/samples":
                ss = list_samples()
                self._send(200, dict(samples=ss, pix_um=PIX_UM, img_px=IMG_PX,
                                     roi_counts={s: len(load_rois(s)) for s in ss}))
            elif u.path == "/api/overview":
                s, ch = q["sample"][0], q.get("ch", ["C0"])[0]
                cp = f"{TOOL}/clean/{s}_{ch}_clean_overview.png"
                if q.get("use_clean", ["0"])[0] == "1" and os.path.exists(cp):
                    self._send(200, open(cp, "rb").read(), "image/png")
                    return
                p = f"{SRC}/" + FILE_FMT.format(sample=s, ch=ch)[:-4] + "_overview.png"
                if os.path.exists(p):
                    self._send(200, open(p, "rb").read(), "image/png")
                else:
                    a = np.asarray(get_mm(s, ch)[::8, ::8], np.float32)
                    self._send(200, to_png_bytes(stretch(a, 1, 99.7, 1.0)), "image/png")
            elif u.path == "/api/crop":
                s, ch = q["sample"][0], q.get("ch", ["C0"])[0]
                P = parse_params(q)
                x, y = int(q["x"][0]), int(q["y"][0])
                w, h = int(q["w"][0]), int(q["h"][0])
                w, h = min(w, MAX_ROI_PX), min(h, MAX_ROI_PX)
                t0 = time.time()
                if q.get("quick", ["0"])[0] == "1":
                    # quick preview: fetch the raw image only (clean base + display floor), no extraction
                    cm, cs = (get_clean(s, ch) if P["use_clean"] else (None, None))
                    mm2 = cm if cm is not None else get_mm(s, ch)
                    sg2 = cs if cm is not None else global_sigma(s, ch)
                    Hh, Ww = mm2.shape
                    x = min(max(0, x), max(0, Ww - w)); y = min(max(0, y), max(0, Hh - h))
                    aq = np.asarray(mm2[y:y + min(h, Hh - y), x:x + min(w, Ww - x)], np.float32)
                    if P["clean_bg"]:
                        aq = np.clip(aq - P["clean_floor"] * sg2, 0, None)
                    g = disp_gray(aq, sg2, P)
                    self._send(200, dict(orig=to_b64(np.stack([g] * 3, -1)), quick=1,
                                         ms=int(1000 * (time.time() - t0))))
                    return
                r = process(s, ch, x, y, w, h, P)
                orig, overlay, comp = render(r, P)
                self._send(200, dict(orig=to_b64(orig), overlay=to_b64(overlay), comp=to_b64(comp),
                                     metrics=r["metrics"], ms=int(1000 * (time.time() - t0))))
            elif u.path == "/api/rois":
                self._send(200, load_rois(q["sample"][0]))
            elif u.path == "/api/roi_metrics":
                s, ch = q["sample"][0], q.get("ch", ["C0"])[0]
                P = parse_params(q)
                rid = q["id"][0]
                rois = load_rois(s)
                ro = next((r for r in rois if str(r["id"]) == rid), None)
                if not ro:
                    self._send(404, dict(error="no such roi"))
                    return
                r = process(s, ch, ro["x"], ro["y"], ro["w"], ro["h"], P)
                self._send(200, r["metrics"])
            elif u.path == "/api/export.csv":
                P = parse_params(q)
                ch = q.get("ch", ["C0"])[0]
                rows, cols = [], None
                # Recomputes EVERY ROI of EVERY sample synchronously - minutes, not seconds.
                # The browser has no progress bar for it, so log to the console instead.
                for s in list_samples():
                    for ro in load_rois(s):
                        print(f"export: {s} R{ro['id']}", flush=True)
                        m = process(s, ch, ro["x"], ro["y"], ro["w"], ro["h"], P)["metrics"]
                        if cols is None:
                            cols = list(m)
                        rows.append([s, ro["id"], ro.get("note", ""), ro["x"], ro["y"], ro["w"], ro["h"]]
                                    + [m[c] for c in cols])
                head = ["sample", "roi_id", "note", "x", "y", "w", "h"] + (cols or [])
                csv = "\n".join(",".join(str(v) for v in row) for row in [head] + rows)
                self._send(200, csv.encode("utf-8-sig"), "text/csv; charset=utf-8")
            else:
                self._send(404, dict(error="not found"))
        except BrokenPipeError:
            pass
        except Exception:
            traceback.print_exc()
            self._send(500, dict(error=traceback.format_exc()[-800:]))

    def do_POST(self):
        try:
            u = urlparse(self.path)
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0) or b"{}")
            if u.path == "/api/rois":
                s = body["sample"]
                rois = load_rois(s)
                if body.get("delete") is not None:
                    rois = [r for r in rois if r["id"] != body["delete"]]
                else:
                    nid = max([r["id"] for r in rois], default=0) + 1
                    rois.append(dict(id=nid, x=int(body["x"]), y=int(body["y"]),
                                     w=int(body["w"]), h=int(body["h"]), note=body.get("note", "")))
                save_rois(s, rois)
                self._send(200, rois)
            else:
                self._send(404, dict(error="not found"))
        except Exception:
            traceback.print_exc()
            self._send(500, dict(error=traceback.format_exc()[-800:]))


def to_png_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "png")
    return buf.getvalue()


if __name__ == "__main__":
    print("Samples:", list_samples(), flush=True)
    srv = ThreadingHTTPServer(("127.0.0.1", 8901), H)
    print("http://localhost:8901", flush=True)
    srv.serve_forever()
