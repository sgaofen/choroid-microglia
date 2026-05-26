from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi
from skimage import filters, measure, morphology


ROOT = Path("/Users/stephenyu/Documents/choroid-microglia/project_A_morphology")
RAW = ROOT / "data/raw"
V29 = ROOT / "experiments/v29_short_spur_audit"
OUT = ROOT / "experiments/v30e_merge_fixes"
CROP_OUT = OUT / "blob_candidates"

STEM = "F_WT_2"
SAMPLE_N = 20
RNG_SEED = 20260524
CROP_SIZE = 60


def find_raw(stem):
    for p in RAW.glob("*.tif"):
        if stem in p.name:
            return p
    raise FileNotFoundError(stem)


def normalize(img, p_lo=1.0, p_hi=99.5):
    lo, hi = np.percentile(img, [p_lo, p_hi])
    return np.clip((img - lo) / (hi - lo + 1e-9), 0, 1)


def make_binary_and_dist(raw):
    norm = normalize(raw)
    smooth = filters.gaussian(norm, sigma=1.0)
    thr = filters.threshold_otsu(smooth) * 0.7
    binary = smooth > thr
    binary = morphology.binary_closing(binary, morphology.disk(2))
    dist = ndi.distance_transform_edt(binary)
    dist_s = filters.gaussian(dist, sigma=1.0)
    return binary, smooth, dist_s


def neighbors8(y, x, shape):
    h, w = shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                yield ny, nx


def skeleton_degree(skel):
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    return ndi.convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)


def edge_key(a, b):
    return (a, b) if a <= b else (b, a)


def long_branch_mask(skel, min_len=15):
    skel = skel.astype(bool)
    deg = skeleton_degree(skel)
    node_mask = skel & (deg != 2)
    long = np.zeros_like(skel, dtype=bool)
    visited_edges = set()

    node_ys, node_xs = np.where(node_mask)
    for start in zip(node_ys.tolist(), node_xs.tolist()):
        for nb in neighbors8(start[0], start[1], skel.shape):
            if not skel[nb]:
                continue
            key = edge_key(start, nb)
            if key in visited_edges:
                continue
            path = [start, nb]
            visited_edges.add(key)
            prev, cur = start, nb
            while not node_mask[cur] and deg[cur] == 2:
                nexts = [
                    n for n in neighbors8(cur[0], cur[1], skel.shape)
                    if skel[n] and n != prev
                ]
                if not nexts:
                    break
                nxt = nexts[0]
                key = edge_key(cur, nxt)
                if key in visited_edges:
                    break
                visited_edges.add(key)
                path.append(nxt)
                prev, cur = cur, nxt
            if len(path) > min_len:
                yy, xx = zip(*path)
                long[yy, xx] = True

    # Handle rare closed loops with no endpoint/junction nodes.
    remaining = skel & ~node_mask & ~long
    labels = measure.label(remaining, connectivity=2)
    for prop in measure.regionprops(labels):
        if prop.area > min_len:
            coords = prop.coords
            long[coords[:, 0], coords[:, 1]] = True

    return long


def crop60(arr, yc, xc):
    h, w = arr.shape
    half = CROP_SIZE // 2
    y0 = min(max(int(round(yc)) - half, 0), h - CROP_SIZE)
    x0 = min(max(int(round(xc)) - half, 0), w - CROP_SIZE)
    return arr[y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE]


def save_png(path, crop_norm):
    # crop_norm is float in [0,1]; convert to uint8 so it's viewable
    img = Image.fromarray((np.clip(crop_norm, 0, 1) * 255).astype(np.uint8))
    img.save(path)


def main():
    raw_path = find_raw(STEM)
    raw_orig = tifffile.imread(raw_path)
    if raw_orig.ndim != 2:
        raise ValueError(f"Expected 2D raw image, got shape {raw_orig.shape}")

    raw = raw_orig.astype(np.float32)
    norm = normalize(raw)
    binary, _, _ = make_binary_and_dist(raw)
    skel = np.load(V29 / f"{STEM}_skel_pruned.npy").astype(bool)
    long_skel = long_branch_mask(skel, min_len=15)

    labels = measure.label(binary, connectivity=2)
    candidates = []
    for prop in measure.regionprops(labels, intensity_image=norm):
        if not (6 <= prop.area <= 80):
            continue
        if prop.mean_intensity <= 0.4:
            continue
        coords = prop.coords
        if long_skel[coords[:, 0], coords[:, 1]].any():
            continue
        candidates.append(prop)

    if len(candidates) < SAMPLE_N:
        raise RuntimeError(f"Only {len(candidates)} candidates; need {SAMPLE_N}")

    rng = np.random.default_rng(RNG_SEED)
    sample_idx = rng.choice(len(candidates), size=SAMPLE_N, replace=False)

    CROP_OUT.mkdir(exist_ok=True)
    for out_i, cand_i in enumerate(sample_idx, start=1):
        prop = candidates[int(cand_i)]
        crop = crop60(norm, prop.centroid[0], prop.centroid[1])
        save_png(CROP_OUT / f"blob_{out_i:02d}.png", crop)

    print(f"raw={raw_path}")
    print(f"skeleton={V29 / f'{STEM}_skel_pruned.npy'}")
    print(f"candidate_count={len(candidates)}")
    print(f"sample_seed={RNG_SEED}")
    for out_i, cand_i in enumerate(sample_idx, start=1):
        prop = candidates[int(cand_i)]
        y, x = prop.centroid
        print(
            f"blob_{out_i:02d}: label={prop.label} "
            f"centroid=({y:.1f},{x:.1f}) area={prop.area:.0f} "
            f"mean_norm={prop.mean_intensity:.3f}"
        )


if __name__ == "__main__":
    main()
