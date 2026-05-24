"""
v30a visualization:
For each image, dump 6-7 sample crops showing original + skeleton + seeds.
Color code:  green = accepted soma, red = rejected (rule1 ecc),
             orange = rejected (rule2 tube), yellow = rejected by both
"""
import json
import sys
import random
from pathlib import Path

import numpy as np
import tifffile
import matplotlib.pyplot as plt
from skimage import morphology, filters

sys.path.insert(0, str(Path(__file__).parent))
from v30a_run import (RAW, V27, V29, OUT, find_raw,
                      load_inputs, make_binary, normalize)

random.seed(42)


def color_for(rec):
    if rec['accepted']:
        return 'lime'
    if rec['rule1'] and rec['rule2']:
        return 'yellow'
    if rec['rule1']:
        return 'red'
    return 'orange'


def crop_around(yc, xc, half, shape):
    H, W = shape
    y0 = max(0, yc - half)
    y1 = min(H, yc + half)
    x0 = max(0, xc - half)
    x1 = min(W, xc + half)
    return y0, y1, x0, x1


def make_panel(stem, sample_centers, half=70):
    raw, skel, somas = load_inputs(stem)
    norm = normalize(raw)
    records = json.loads((OUT / f'{stem}_seed_scores.json').read_text())

    n = len(sample_centers)
    fig, axes = plt.subplots(2, n, figsize=(3*n, 6))
    if n == 1:
        axes = axes[:, None]

    for i, (yc_focus, xc_focus, tag) in enumerate(sample_centers):
        y0, y1, x0, x1 = crop_around(yc_focus, xc_focus, half, raw.shape)

        # row 0: raw only
        axes[0, i].imshow(norm[y0:y1, x0:x1], cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'{tag}\n({yc_focus},{xc_focus})', fontsize=8)
        axes[0, i].axis('off')

        # row 1: raw + skeleton + colored seeds
        axes[1, i].imshow(norm[y0:y1, x0:x1], cmap='gray', vmin=0, vmax=1)
        skel_crop = skel[y0:y1, x0:x1]
        skel_overlay = np.zeros((*skel_crop.shape, 4))
        skel_overlay[skel_crop] = (0, 0.6, 1, 0.7)  # cyan
        axes[1, i].imshow(skel_overlay)

        # plot all seeds in crop
        for r in records:
            if y0 <= r['yc'] < y1 and x0 <= r['xc'] < x1:
                c = color_for(r)
                axes[1, i].plot(r['xc']-x0, r['yc']-y0, 'o',
                                ms=5, mfc='none', mec=c, mew=1.5)
        # legend in first cell only
        axes[1, i].axis('off')

    fig.suptitle(
        f'{stem}   green=accepted  red=rule1(ecc)  '
        f'orange=rule2(tube)  yellow=both',
        fontsize=10)
    fig.tight_layout()
    return fig


def pick_random_accepted(records, k):
    accepted = [r for r in records if r['accepted']]
    sample = random.sample(accepted, min(k, len(accepted)))
    return [(r['yc'], r['xc'], f"acc_lab{r['label']}") for r in sample]


def pick_random_rejected(records, k):
    rejected = [r for r in records if not r['accepted']]
    sample = random.sample(rejected, min(k, len(rejected)))
    return [(r['yc'], r['xc'], f"rej_lab{r['label']}") for r in sample]


if __name__ == '__main__':
    for stem in ['F_WT_2', 'F_HET_1', 'F_HET_3']:
        records = json.loads((OUT / f'{stem}_seed_scores.json').read_text())
        # 4 random accepted + 3 random rejected = 7
        centers = (pick_random_accepted(records, 4) +
                   pick_random_rejected(records, 3))
        fig = make_panel(stem, centers, half=70)
        out = OUT / f'{stem}_v30a_samples.png'
        fig.savefig(out, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'wrote {out}')
