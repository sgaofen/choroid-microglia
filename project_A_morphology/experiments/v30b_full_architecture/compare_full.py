"""
2x2 comparison with full type labeling:
  panel 0: ORIG (clean)
  panel 1: ORIG + skeleton
  panel 2: ORIG + skeleton + v29 (every v27 soma in red)
  panel 3: ORIG + skeleton + v30b 5-color markers
            lime    = strong accepted
            cyan    = weak / low_confidence accepted
            magenta = ambiguous neighbor (strong but flagged)
            yellow  = merged_into_strong rejected
            red     = process_peak rejected
"""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tifffile

ROOT = Path('/Users/stephenyu/Documents/choroid-microglia/project_A_morphology')
RAW = ROOT / 'data/raw'
V27 = ROOT / 'experiments/v27_clean_graph'
V29 = ROOT / 'experiments/v29_short_spur_audit'
V30B = ROOT / 'experiments/v30b_full_architecture'

sys.path.insert(0, str(V30B))
from v30b_run import normalize

from skimage import measure


def find_raw(stem):
    for p in RAW.glob('*.tif'):
        if stem in p.name:
            return p
    raise FileNotFoundError(stem)


def color_for(rec):
    t = rec['type']
    if t == 'strong':
        if 'confirmed_neighbors' in rec and rec['confirmed_neighbors']:
            return 'magenta', 'amb'
        return 'lime', 'S'
    if t in ('weak', 'low_confidence_soma'):
        return 'cyan', 'L'
    if t == 'merged_into_strong':
        return 'yellow', 'M'
    return 'red', 'P'


def make_full(stem, yc, xc, tag, half=80):
    raw = tifffile.imread(find_raw(stem)).astype(np.float32)
    norm = normalize(raw)
    skel = np.load(V29 / f'{stem}_skel_pruned.npy')
    somas = np.load(V27 / f'{stem}_soma_cores.npy')
    recs = json.loads((V30B / f'{stem}_seeds_v30b.json').read_text())

    H, W = raw.shape
    y0, y1 = max(0, yc - half), min(H, yc + half)
    x0, x1 = max(0, xc - half), min(W, xc + half)

    props = measure.regionprops(somas)
    cents = [(int(round(p.centroid[0])), int(round(p.centroid[1])), int(p.label))
             for p in props
             if y0 <= int(round(p.centroid[0])) < y1
             and x0 <= int(round(p.centroid[1])) < x1]

    skel_crop = skel[y0:y1, x0:x1]
    ov = np.zeros((*skel_crop.shape, 4))
    ov[skel_crop] = (0, 0.9, 1, 0.55)

    rec_by_label = {r['label']: r for r in recs}

    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.ravel()
    titles = [f'{stem} {tag}  ORIG',
              f'{stem} {tag}  ORIG + skeleton',
              f'v29 all = {len(cents)}',
              'v30b 5-color']
    for k, t in enumerate(titles):
        axes[k].imshow(norm[y0:y1, x0:x1], cmap='gray', vmin=0, vmax=1)
        axes[k].set_title(t, fontsize=13)
        axes[k].axis('off')
    for k in [1, 2, 3]:
        axes[k].imshow(ov)

    counts = {'S': 0, 'L': 0, 'amb': 0, 'M': 0, 'P': 0}
    for cy, cx, lab in cents:
        pos = (cx - x0, cy - y0)
        axes[2].plot(*pos, 'o', ms=18, mfc='none', mec='red', mew=2.3)
        rec = rec_by_label.get(lab)
        if rec is None:
            continue
        col, tag_ = color_for(rec)
        counts[tag_] += 1
        axes[3].plot(*pos, 'o', ms=18, mfc='none', mec=col, mew=2.3)
        axes[3].text(pos[0] + 8, pos[1] - 8, tag_, color=col,
                     fontsize=9, weight='bold')

    legend = (f'lime S={counts["S"]}   cyan L={counts["L"]}   '
              f'magenta amb={counts["amb"]}   '
              f'yellow M={counts["M"]}   red P={counts["P"]}')
    axes[3].set_title(f'v30b 5-color\n{legend}', fontsize=11)

    fig.tight_layout()
    return fig


if __name__ == '__main__':
    pairs = [
        ('F_WT_2', 1909, 2019, 'long_vessel'),
        ('F_HET_1', 3061, 1916, 'cluster1'),
        ('F_HET_3', 874, 2390, 'vessel_following'),
        ('F_HET_3', 1968, 689, 'ramified'),
    ]
    for (s, y, x, t) in pairs:
        fig = make_full(s, y, x, t)
        out = V30B / f'compare_full_{s}_{t}.png'
        fig.savefig(out, dpi=110, bbox_inches='tight')
        plt.close(fig)
        print('wrote', out.name)
