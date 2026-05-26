"""
Faithful Python replication of the Young & Morrison 2018 microglia-morphology
protocol (the workflow Huixin says ImageJ already extracts).

ImageJ recipe:  binarize -> skeletonize -> AnalyzeSkeleton (2D/3D) plugin
                -> # branches / # junctions / # endpoints / branch length.

We reproduce AnalyzeSkeleton with `skan` (same skeleton-graph analysis as the
Arganda-Carreras algorithm ImageJ uses). Output is AGGREGATE per image — the
standard ROI-level ramification readout, NOT per-cell. NO machine-learning
model involved anywhere.

Branch-type codes (skan, matching AnalyzeSkeleton):
  0 = endpoint-to-endpoint (isolated)
  1 = junction-to-endpoint  (terminal twig)
  2 = junction-to-junction  (internal)
  3 = isolated cycle
"""
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage import filters, morphology
from skan import Skeleton, summarize

ROOT = Path('/Users/stephenyu/choroid-microglia/project_A_morphology')
RAW = ROOT / 'data/raw'
V29 = ROOT / 'experiments/v29_short_spur_audit'
OUT = ROOT / 'experiments/imagej_skeleton_baseline'
PIXEL_UM = 0.207  # µm/px from acquisition metadata


def find_raw(stem):
    for p in RAW.glob('*.tif'):
        if stem in p.name:
            return p
    raise FileNotFoundError(stem)


def normalize(img, p_lo=1.0, p_hi=99.5):
    lo, hi = np.percentile(img, [p_lo, p_hi])
    return np.clip((img - lo) / (hi - lo + 1e-9), 0, 1)


def binarize(raw):
    norm = normalize(raw)
    smooth = filters.gaussian(norm, sigma=1.0)
    thr = filters.threshold_otsu(smooth) * 0.7
    binary = smooth > thr
    binary = morphology.binary_closing(binary, morphology.disk(2))
    binary = morphology.remove_small_objects(binary, min_size=20)
    return binary


def analyze(stem, use_v29_skeleton=True):
    raw = tifffile.imread(find_raw(stem)).astype(np.float32)
    binary = binarize(raw)
    if use_v29_skeleton:
        # our validated, short-spur-pruned skeleton (cleaner than a fresh
        # skeletonize; avoids trivial-spur inflation of endpoint counts)
        skel = np.load(V29 / f'{stem}_skel_pruned.npy').astype(bool)
    else:
        skel = morphology.skeletonize(binary)

    sk = Skeleton(skel, spacing=PIXEL_UM)
    df = summarize(sk, separator='-')

    deg = _degree(skel)
    n_endpoints = int(((deg == 1) & skel).sum())
    n_junctions = int(((deg >= 3) & skel).sum())

    branch_types = df['branch-type'].value_counts().to_dict()
    total_len_um = float(df['branch-distance'].sum())
    mean_len_um = float(df['branch-distance'].mean())

    fg_px = int(binary.sum())
    fg_area_mm2 = fg_px * (PIXEL_UM ** 2) / 1e6

    return dict(
        stem=stem,
        n_branches=int(len(df)),
        n_endpoints=n_endpoints,
        n_junctions=n_junctions,
        total_branch_len_um=round(total_len_um, 1),
        mean_branch_len_um=round(mean_len_um, 3),
        branch_type_counts={int(k): int(v) for k, v in branch_types.items()},
        fg_area_mm2=round(fg_area_mm2, 4),
        endpoints_per_mm2=round(n_endpoints / fg_area_mm2, 1),
        junctions_per_mm2=round(n_junctions / fg_area_mm2, 1),
        branch_len_um_per_mm2=round(total_len_um / fg_area_mm2, 1),
    )


def _degree(skel):
    k = np.ones((3, 3), dtype=np.uint8); k[1, 1] = 0
    return ndi.convolve(skel.astype(np.uint8), k, mode='constant') * skel


def main():
    OUT.mkdir(exist_ok=True)
    recs = [analyze(s) for s in ['F_WT_2', 'F_HET_1', 'F_HET_3']]
    for rec in recs:
        print(f'=== {rec["stem"]} ===')
        for k, v in rec.items():
            if k != 'stem':
                print(f'  {k:24}: {v}')
        print()

    (OUT / 'young_morrison_metrics.json').write_text(json.dumps(recs, indent=1))

    print('=== AGGREGATE WT vs HET (Young & Morrison, density-normalized) ===')
    wt, h1, h3 = recs
    cols = ['endpoints_per_mm2', 'junctions_per_mm2',
            'branch_len_um_per_mm2', 'mean_branch_len_um']
    print(f'{"metric":24} {"WT_2":>12} {"HET_1":>12} {"HET_3":>12}  {"HETvsWT":>9}')
    for c in cols:
        het_mean = (h1[c] + h3[c]) / 2
        delta = 100 * (het_mean - wt[c]) / wt[c]
        print(f'{c:24} {wt[c]:>12} {h1[c]:>12} {h3[c]:>12}  {delta:>+8.1f}%')
    print('\nsaved young_morrison_metrics.json')


if __name__ == '__main__':
    main()
