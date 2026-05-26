"""
v31 master 3-panel visualizer.

  panel 0: ORIG
  panel 1: ORIG + skeleton + endpoints/branches
  panel 2: ORIG + skeleton + v31 accepted cells (lime circles sized to
           soma_radius, plus n_endpoints / n_branches / n_trunks labels)
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tifffile
from scipy.ndimage import binary_dilation, convolve

ROOT = Path('/Users/stephenyu/Documents/choroid-microglia/project_A_morphology')
RAW = ROOT / 'data/raw'
V29 = ROOT / 'experiments/v29_short_spur_audit'
V31 = ROOT / 'experiments/v31_skeleton_first'

sys.path.insert(0, str(V31))
from v31_run import normalize


def find_raw(stem):
    for p in RAW.glob('*.tif'):
        if stem in p.name: return p


def make_master(stem, yc, xc, half=300, tag='tight'):
    raw = tifffile.imread(find_raw(stem)).astype(np.float32)
    norm = normalize(raw)
    skel = np.load(V29 / f'{stem}_skel_pruned.npy')
    cells = json.loads((V31 / f'{stem}_cells_v31.json').read_text())

    H, W = raw.shape
    y0, y1 = max(0, yc - half), min(H, yc + half)
    x0, x1 = max(0, xc - half), min(W, xc + half)

    raw_crop = norm[y0:y1, x0:x1]
    skel_crop = skel[y0:y1, x0:x1]
    skel_thick = binary_dilation(skel_crop, iterations=1)
    ov = np.zeros((*skel_crop.shape, 4))
    ov[skel_thick] = (1.0, 0.35, 0.75, 0.95)

    nbr = convolve(skel_crop.astype(np.uint8),
                   np.ones((3, 3), dtype=np.uint8),
                   mode='constant') - skel_crop.astype(np.uint8)
    ep_mask = skel_crop & (nbr == 1)
    br_mask = skel_crop & (nbr >= 3)
    ep_y, ep_x = np.where(ep_mask)
    br_y, br_x = np.where(br_mask)

    visible = [c for c in cells if y0 <= c['yc'] < y1 and x0 <= c['xc'] < x1]

    fig, axes = plt.subplots(1, 3, figsize=(28, 10))
    titles = [
        f'{stem} crop=({y0},{x0})  ORIG',
        f'skeleton + endpoints(yellow) + branches(orange)  '
        f'[{len(ep_y)} ep, {len(br_y)} br in view]',
        f'v31 cells: {len(visible)} accepted (lime circle = soma, size = radius)',
    ]
    for k, t in enumerate(titles):
        axes[k].imshow(raw_crop, cmap='gray', vmin=0, vmax=1)
        axes[k].set_title(t, fontsize=12)
        axes[k].axis('off')
    for k in (1, 2):
        axes[k].imshow(ov)
        axes[k].scatter(ep_x, ep_y, s=12, c='yellow',
                        edgecolors='black', linewidths=0.3, zorder=3)
        axes[k].scatter(br_x, br_y, s=12, c='orange',
                        edgecolors='black', linewidths=0.3, zorder=3)

    for c in visible:
        cy = c['yc'] - y0; cx = c['xc'] - x0
        r = c['soma_radius']
        # circle scaled to soma radius (visual)
        axes[2].plot(cx, cy, 'o', ms=2 + 2 * r, mfc='none', mec='lime',
                     mew=2.0)
        axes[2].text(cx + r + 2, cy - r - 2,
                     f'ep{c["n_endpoints"]}/br{c["n_branches"]}/t{c["n_trunks"]}',
                     color='lime', fontsize=7, weight='bold')

    fig.tight_layout()
    out = V31 / f'v31_master_{stem}_{tag}.png'
    fig.savefig(out, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out.name}  ({out.stat().st_size // 1024} KB)')


if __name__ == '__main__':
    targets = [
        ('F_WT_2',  1100, 1100, 300, 'tight'),
        ('F_HET_1', 2400, 1900, 300, 'tight'),
        ('F_HET_3', 1500, 1500, 300, 'tight'),
    ]
    for s, y, x, h, t in targets:
        make_master(s, y, x, half=h, tag=t)
