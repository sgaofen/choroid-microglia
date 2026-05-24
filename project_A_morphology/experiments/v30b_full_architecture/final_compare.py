"""
Final 3-way compare: v29 vs v30a vs v30b.
1) Cross-genotype summary table.
2) Side-by-side crops on the 3 elongated-cell examples Stephen flagged
   (these are documented as approximate coords in v30_gpt_pro_consult).
"""
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path('/Users/stephenyu/Documents/choroid-microglia/project_A_morphology')
V27 = ROOT / 'experiments/v27_clean_graph'
V28 = ROOT / 'experiments/v28_endpoint_attr'
V29 = ROOT / 'experiments/v29_short_spur_audit'
V30A = ROOT / 'experiments/v30a_minimal_filter'
V30B = ROOT / 'experiments/v30b_full_architecture'
RAW = ROOT / 'data/raw'

import tifffile
from skimage import filters, morphology
from scipy import ndimage as ndi

sys.path.insert(0, str(V30B))
from v30b_run import load_inputs, normalize


def find_raw(stem):
    for p in RAW.glob('*.tif'):
        if stem in p.name:
            return p
    raise FileNotFoundError(stem)


def get_v29_counts(stem):
    """v29 had per-cell endpoint counts with all 3000-ish cells."""
    d = json.loads((V29 / f'{stem}_endpoint_counts_v29.json').read_text())
    return {int(k): int(v) for k, v in d.items()}


def get_v30a_counts(stem):
    d = json.loads((V30A / f'{stem}_endpoint_counts_v30a.json').read_text())
    return {int(k): int(v) for k, v in d.items()}


def get_v30b_counts(stem):
    d = json.loads((V30B / f'{stem}_endpoint_counts_v30b.json').read_text())
    return {int(k): int(v) for k, v in d.items()}


def summarize(counts):
    arr = np.array(list(counts.values()))
    if len(arr) == 0:
        return dict(n=0, mean=0.0, median=0.0)
    return dict(n=len(arr), mean=float(arr.mean()),
                median=float(np.median(arr)),
                dist=np.bincount(np.clip(arr, 0, 10)).tolist())


def print_summary():
    print('\n=== Cross-genotype summary (cells | mean endpoints) ===')
    print(f'{"image":8} {"v29":>20} {"v30a":>20} {"v30b":>20}')
    for stem in ['F_WT_2', 'F_HET_1', 'F_HET_3']:
        c29 = summarize(get_v29_counts(stem))
        c30a = summarize(get_v30a_counts(stem))
        c30b = summarize(get_v30b_counts(stem))
        print(f'{stem:8}  {c29["n"]:>6}|{c29["mean"]:>6.2f}     '
              f'{c30a["n"]:>6}|{c30a["mean"]:>6.2f}     '
              f'{c30b["n"]:>6}|{c30b["mean"]:>6.2f}')


def make_compare_panel(stem, yc, xc, tag, half=70):
    raw = tifffile.imread(find_raw(stem)).astype(np.float32)
    norm = normalize(raw)
    skel = np.load(V29 / f'{stem}_skel_pruned.npy')
    somas = np.load(V27 / f'{stem}_soma_cores.npy')

    # all v27 soma centroids
    from skimage import measure
    props = measure.regionprops(somas)
    centroids = [(int(round(p.centroid[0])), int(round(p.centroid[1])),
                  int(p.label)) for p in props]

    # v30a accepted
    v30a_acc = set(int(x) for x in
                   np.load(V30A / f'{stem}_accepted_labels.npy').tolist())
    # v30b accepted
    v30b_acc = set(int(x) for x in
                   np.load(V30B / f'{stem}_accepted_labels.npy').tolist())

    H, W = raw.shape
    y0, y1 = max(0, yc - half), min(H, yc + half)
    x0, x1 = max(0, xc - half), min(W, xc + half)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    titles = ['ORIG', f'v29 (all seeds)', 'v30a', 'v30b']
    for k, t in enumerate(titles):
        axes[k].imshow(norm[y0:y1, x0:x1], cmap='gray', vmin=0, vmax=1)
        axes[k].set_title(f'{stem} {tag}  {t}')
        axes[k].axis('off')

    # overlay skeleton on cols 1..3
    skel_crop = skel[y0:y1, x0:x1]
    ov = np.zeros((*skel_crop.shape, 4)); ov[skel_crop] = (0, 0.9, 1, 0.4)
    for k in [1, 2, 3]:
        axes[k].imshow(ov)

    n29 = n30a = n30b = 0
    for cy, cx, lab in centroids:
        if not (y0 <= cy < y1 and x0 <= cx < x1):
            continue
        pos = (cx - x0, cy - y0)
        # v29: all somas count
        axes[1].plot(*pos, 'o', ms=12, mfc='none', mec='red', mew=2)
        n29 += 1
        if lab in v30a_acc:
            axes[2].plot(*pos, 'o', ms=12, mfc='none', mec='lime', mew=2)
            n30a += 1
        else:
            axes[2].plot(*pos, 'x', ms=10, mec='red', mew=2)
        if lab in v30b_acc:
            axes[3].plot(*pos, 'o', ms=12, mfc='none', mec='lime', mew=2)
            n30b += 1
        else:
            axes[3].plot(*pos, 'x', ms=10, mec='red', mew=2)
    axes[1].set_title(f'v29 (all={n29})')
    axes[2].set_title(f'v30a (kept={n30a})')
    axes[3].set_title(f'v30b (kept={n30b})')
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    print_summary()

    # Make compare panels on a few elongated-cell hot spots.
    # We don't have exact coords for example_1/2/3 from before, so I'll
    # pick locations where v29 has dense seeds and v30b has few accepted.
    hot_spots = {
        'F_WT_2': [(2875, 564, 'cluster1'),
                   (1909, 2019, 'long_vessel'),  # from earlier samples
                   (1760, 1047, 'horizontal')],
        'F_HET_1': [(3061, 1916, 'cluster1'),
                    (2304, 2591, 'compact_blob'),
                    (2827, 1511, 'dense_region')],
        'F_HET_3': [(1968, 689, 'ramified'),
                    (874, 2390, 'vessel_following'),
                    (104, 899, 'edge_cells')],
    }
    for stem, spots in hot_spots.items():
        for (yc, xc, tag) in spots:
            fig = make_compare_panel(stem, yc, xc, tag, half=80)
            out = V30B / f'compare_{stem}_{tag}.png'
            fig.savefig(out, dpi=110, bbox_inches='tight')
            plt.close(fig)
            print(f'wrote {out.name}')
