"""
Per-cell individual crops, big size. 4 colors:
  lime = strong accepted soma
  cyan = weak / low_confidence accepted
  magenta = ambiguous neighbor pair member
  red    = process_peak rejected
  yellow = merged_into_strong rejected
"""
import json
import sys
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from v30b_run import OUT, load_inputs, normalize, V29

random.seed(42)
HALF = 60


def color_for(rec):
    t = rec['type']
    if t == 'strong':
        if 'confirmed_neighbors' in rec:
            return 'magenta'
        return 'lime'
    if t in ('weak', 'low_confidence_soma'):
        return 'cyan'
    if t == 'merged_into_strong':
        return 'yellow'
    return 'red'


def make_one(stem, rec, all_records, tag, half=HALF):
    raw = load_inputs(stem)[0]
    skel = np.load(V29 / f'{stem}_skel_pruned.npy')
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
    overlay[skel_crop] = (0, 0.9, 1, 0.5)
    axes[1].imshow(overlay)

    for r in all_records:
        if y0 <= r['yc'] < y1 and x0 <= r['xc'] < x1:
            c = color_for(r)
            axes[1].plot(r['xc']-x0, r['yc']-y0, 'o',
                         ms=14, mfc='none', mec=c, mew=2.0)
    axes[1].plot(xc-x0, yc-y0, '+', ms=20, mec='white', mew=2)

    info = (f"type={rec['type']} score={rec['score']:.2f}  "
            f"compact={rec['compact']:.2f} blob={rec['blob']:.2f} tube={rec['tube']:.2f}\n"
            f"rprom={rec['rprom']:.1f} dirs={rec['n_dirs']} deg={rec['deg']}")
    axes[1].set_title(info, fontsize=9)
    axes[1].axis('off')
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    for stem in ['F_WT_2', 'F_HET_1', 'F_HET_3']:
        recs = json.loads((OUT / f'{stem}_seeds_v30b.json').read_text())
        strong = [r for r in recs if r['type'] == 'strong']
        weak = [r for r in recs if r['type'] in ('weak', 'low_confidence_soma')]
        rejected = [r for r in recs if r['type'] in ('process_peak', 'merged_into_strong')]
        # 3 strong, 2 weak/lowconf, 2 rejected
        picks = (random.sample(strong, min(3, len(strong))) +
                 random.sample(weak, min(2, len(weak))) +
                 random.sample(rejected, min(2, len(rejected))))
        for i, r in enumerate(picks):
            tag = {'strong': 'S', 'weak': 'W', 'low_confidence_soma': 'L',
                   'process_peak': 'P', 'merged_into_strong': 'M'}.get(r['type'], '?')
            fig = make_one(stem, r, recs, f'{tag}{i+1}')
            out = OUT / f'{stem}_cell_{i+1}_{tag}.png'
            fig.savefig(out, dpi=110, bbox_inches='tight')
            plt.close(fig)
        print(f'{stem}: wrote {len(picks)} cells')
