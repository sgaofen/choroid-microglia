"""
Check v30a vs v29 on the 3 specific examples from Stephen.
The example crops were saved in v30_gpt_pro_consult as PNGs, but the
crop coordinates aren't stored. Approximate them by visual matching:
    example_1: vertical long process — search WT_2 dense area
For simplicity, scan all images for high-rejection clusters
(clusters where multiple seeds within 30 px were all rejected as
process peaks) and dump 6 of the best examples.
"""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from v30a_run import OUT, load_inputs, normalize

HALF = 70


def find_oversplit_clusters(records, neighbor_r=25, min_rejected_in_cluster=2):
    """Return seeds that have >=2 other rejected seeds within neighbor_r."""
    arr = np.array([(r['yc'], r['xc']) for r in records])
    flags = np.array([r['accepted'] for r in records])
    out = []
    for i, r in enumerate(records):
        d = np.sqrt(((arr - arr[i]) ** 2).sum(1))
        local = (d <= neighbor_r) & (np.arange(len(arr)) != i)
        n_rej_local = int(((~flags) & local).sum())
        if n_rej_local >= min_rejected_in_cluster:
            out.append((n_rej_local, i, r))
    out.sort(reverse=True)
    return out


def draw(stem, rec, all_records, idx, half=HALF):
    raw, skel, somas = load_inputs(stem)
    norm = normalize(raw)
    yc, xc = rec['yc'], rec['xc']
    H, W = raw.shape
    y0, y1 = max(0, yc - half), min(H, yc + half)
    x0, x1 = max(0, xc - half), min(W, xc + half)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(norm[y0:y1, x0:x1], cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f'{stem} cluster {idx} @ ({yc},{xc}) ORIG')
    axes[0].axis('off')
    axes[1].imshow(norm[y0:y1, x0:x1], cmap='gray', vmin=0, vmax=1)
    skel_crop = skel[y0:y1, x0:x1]
    ov = np.zeros((*skel_crop.shape, 4))
    ov[skel_crop] = (0, 0.9, 1, 0.6)
    axes[1].imshow(ov)

    n_acc = 0
    n_rej = 0
    for r in all_records:
        if y0 <= r['yc'] < y1 and x0 <= r['xc'] < x1:
            if r['accepted']:
                c = 'lime'; n_acc += 1
            else:
                c = 'red' if r['rule1'] else 'orange'; n_rej += 1
            axes[1].plot(r['xc']-x0, r['yc']-y0, 'o',
                         ms=14, mfc='none', mec=c, mew=2.0)
    axes[1].set_title(f'v30a: {n_acc} accepted, {n_rej} rejected')
    axes[1].axis('off')
    return fig


if __name__ == '__main__':
    for stem in ['F_WT_2', 'F_HET_1', 'F_HET_3']:
        recs = json.loads((OUT / f'{stem}_seed_scores.json').read_text())
        clusters = find_oversplit_clusters(recs)
        print(f'{stem}: found {len(clusters)} clusters with >=2 rejected nbrs')
        for i, (_, _, rec) in enumerate(clusters[:3]):
            fig = draw(stem, rec, recs, i+1)
            out = OUT / f'{stem}_oversplit_cluster_{i+1}.png'
            fig.savefig(out, dpi=110, bbox_inches='tight')
            plt.close(fig)
            print(f'  wrote {out.name}')
