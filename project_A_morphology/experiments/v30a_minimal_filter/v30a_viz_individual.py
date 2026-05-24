"""
Per-cell individual crops, large size, so I can read with Read tool one at a time.
For each image: 4 random accepted + 3 random rejected = 7 cells.
Each gets a 2-panel PNG: raw original | raw+skel+seed overlay.
"""
import json
import sys
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from v30a_run import OUT, load_inputs, normalize

random.seed(42)
HALF = 60  # crop half-size, px


def color_for(rec):
    if rec['accepted']:
        return 'lime'
    if rec['rule1'] and rec['rule2']:
        return 'yellow'
    if rec['rule1']:
        return 'red'
    return 'orange'


def make_one(stem, rec, all_records, tag, half=HALF):
    raw, skel, somas = load_inputs(stem)
    norm = normalize(raw)
    H, W = raw.shape
    yc, xc = rec['yc'], rec['xc']
    y0, y1 = max(0, yc - half), min(H, yc + half)
    x0, x1 = max(0, xc - half), min(W, xc + half)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(norm[y0:y1, x0:x1], cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f'{stem} {tag} L{rec["label"]} @ ({yc},{xc}) ORIG')
    axes[0].axis('off')

    axes[1].imshow(norm[y0:y1, x0:x1], cmap='gray', vmin=0, vmax=1)
    skel_crop = skel[y0:y1, x0:x1]
    overlay = np.zeros((*skel_crop.shape, 4))
    overlay[skel_crop] = (0, 0.9, 1, 0.6)
    axes[1].imshow(overlay)

    for r in all_records:
        if y0 <= r['yc'] < y1 and x0 <= r['xc'] < x1:
            c = color_for(r)
            axes[1].plot(r['xc']-x0, r['yc']-y0, 'o',
                         ms=14, mfc='none', mec=c, mew=2.0)
    # mark focus seed
    axes[1].plot(xc-x0, yc-y0, '+',
                 ms=20, mec='white', mew=2)

    info = (f"ecc={rec['eccentricity']:.2f}  "
            f"exit_dirs={rec['n_exit_dirs']}  "
            f"deg={rec['skel_degree']}  "
            f"blob={rec['blobness']:.2f} tube={rec['tubeness']:.2f}\n"
            f"rule1={rec['rule1']} rule2={rec['rule2']}  "
            f"-> {'ACCEPT' if rec['accepted'] else 'REJECT'}")
    axes[1].set_title(info, fontsize=9)
    axes[1].axis('off')

    fig.tight_layout()
    return fig


if __name__ == '__main__':
    for stem in ['F_WT_2', 'F_HET_1', 'F_HET_3']:
        recs = json.loads((OUT / f'{stem}_seed_scores.json').read_text())
        accepted = [r for r in recs if r['accepted']]
        rejected = [r for r in recs if not r['accepted']]
        picks = (random.sample(accepted, 4) + random.sample(rejected, 3))
        for i, r in enumerate(picks):
            tag = 'A' if r['accepted'] else 'R'
            fig = make_one(stem, r, recs, f'{tag}{i+1}')
            out = OUT / f'{stem}_cell_{i+1}_{tag}.png'
            fig.savefig(out, dpi=110, bbox_inches='tight')
            plt.close(fig)
        print(f'{stem}: wrote 7 cells')
