"""Stack 4 panels vertically into a tall single PNG."""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tifffile

ROOT = Path('/Users/stephenyu/Documents/choroid-microglia/project_A_morphology')
RAW = ROOT / 'data/raw'
V27 = ROOT / 'experiments/v27_clean_graph'
V29 = ROOT / 'experiments/v29_short_spur_audit'
V30A = ROOT / 'experiments/v30a_minimal_filter'
V30B = ROOT / 'experiments/v30b_full_architecture'

sys.path.insert(0, str(V30B))
from v30b_run import normalize

from skimage import measure


def find_raw(stem):
    for p in RAW.glob('*.tif'):
        if stem in p.name:
            return p


def make_single(stem, yc, xc, tag, half=80):
    raw = tifffile.imread(find_raw(stem)).astype(np.float32)
    norm = normalize(raw)
    skel = np.load(V29 / f'{stem}_skel_pruned.npy')
    somas = np.load(V27 / f'{stem}_soma_cores.npy')
    v30a_acc = set(int(x) for x in np.load(V30A / f'{stem}_accepted_labels.npy').tolist())
    v30b_acc = set(int(x) for x in np.load(V30B / f'{stem}_accepted_labels.npy').tolist())

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
    ov[skel_crop] = (0, 0.9, 1, 0.45)

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.ravel()
    titles = ['ORIG', f'v29 all={len(cents)}',
              'v30a kept', 'v30b kept']
    for k, t in enumerate(titles):
        axes[k].imshow(norm[y0:y1, x0:x1], cmap='gray', vmin=0, vmax=1)
        axes[k].set_title(f'{stem} {tag}  {t}', fontsize=14)
        axes[k].axis('off')
    for k in [1, 2, 3]:
        axes[k].imshow(ov)
    n30a = n30b = 0
    for cy, cx, lab in cents:
        pos = (cx - x0, cy - y0)
        axes[1].plot(*pos, 'o', ms=18, mfc='none', mec='red', mew=2.5)
        if lab in v30a_acc:
            axes[2].plot(*pos, 'o', ms=18, mfc='none', mec='lime', mew=2.5); n30a += 1
        else:
            axes[2].plot(*pos, 'x', ms=14, mec='red', mew=2)
        if lab in v30b_acc:
            axes[3].plot(*pos, 'o', ms=18, mfc='none', mec='lime', mew=2.5); n30b += 1
        else:
            axes[3].plot(*pos, 'x', ms=14, mec='red', mew=2)
    axes[2].set_title(f'v30a kept={n30a}', fontsize=14)
    axes[3].set_title(f'v30b kept={n30b}', fontsize=14)
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
        fig = make_single(s, y, x, t)
        out = V30B / f'compare2x2_{s}_{t}.png'
        fig.savefig(out, dpi=100, bbox_inches='tight')
        plt.close(fig)
        print('wrote', out.name)
