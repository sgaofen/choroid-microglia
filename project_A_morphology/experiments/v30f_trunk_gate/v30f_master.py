"""
v30f_master.py — 3-panel format showing original + skeleton + cell markers
+ process endpoints/branches. Adapted from v30b_master_tight.

  panel 0: ORIG (raw normalized)
  panel 1: ORIG + skeleton (magenta) + endpoints (yellow) + branches (orange)
           + ALL v27 raw seeds (red, no labels)
  panel 2: ORIG + skeleton + endpoints/branches + v30f ACCEPTED cells only
           (S=lime strong, L=cyan low_confidence, A=magenta ambiguous,
           number after letter = endpoint count attributed to that cell)
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tifffile
from scipy.ndimage import binary_dilation, convolve
from skimage import measure

ROOT = Path('/Users/stephenyu/Documents/choroid-microglia/project_A_morphology')
RAW = ROOT / 'data/raw'
V27 = ROOT / 'experiments/v27_clean_graph'
V29 = ROOT / 'experiments/v29_short_spur_audit'
V30F = ROOT / 'experiments/v30f_trunk_gate'

sys.path.insert(0, str(V30F))
from v30f_run import normalize


def find_raw(stem):
    for p in RAW.glob('*.tif'):
        if stem in p.name:
            return p


def color_for(rec):
    t = rec['type']
    if t == 'strong':
        if rec.get('confirmed_neighbors'):
            return 'magenta', 'A'
        return 'lime', 'S'
    if t in ('weak', 'low_confidence_soma'):
        return 'cyan', 'L'
    if t == 'merged_into_strong':
        return 'yellow', 'M'
    return 'red', 'P'


def make_master(stem, yc, xc, half=300, tag='tight'):
    raw = tifffile.imread(find_raw(stem)).astype(np.float32)
    norm = normalize(raw)
    skel = np.load(V29 / f'{stem}_skel_pruned.npy')
    somas = np.load(V27 / f'{stem}_soma_cores.npy')
    recs = json.loads((V30F / f'{stem}_seeds_v30f.json').read_text())
    endpoints = json.loads(
        (V30F / f'{stem}_endpoint_counts_v30f.json').read_text())
    endpoints = {int(k): int(v) for k, v in endpoints.items()}

    H, W = raw.shape
    y0, y1 = max(0, yc - half), min(H, yc + half)
    x0, x1 = max(0, xc - half), min(W, xc + half)

    raw_crop = norm[y0:y1, x0:x1]
    skel_crop = skel[y0:y1, x0:x1]
    skel_thick = binary_dilation(skel_crop, iterations=1)

    skel_overlay = np.zeros((*skel_crop.shape, 4))
    skel_overlay[skel_thick] = (1.0, 0.35, 0.75, 0.95)

    nbr = convolve(skel_crop.astype(np.uint8),
                   np.ones((3, 3), dtype=np.uint8),
                   mode='constant') - skel_crop.astype(np.uint8)
    endpoint_mask = skel_crop & (nbr == 1)
    branch_mask = skel_crop & (nbr >= 3)
    ep_y, ep_x = np.where(endpoint_mask)
    br_y, br_x = np.where(branch_mask)

    props = measure.regionprops(somas)
    rec_by_label = {r['label']: r for r in recs}
    visible = []
    for p in props:
        cy = int(round(p.centroid[0])); cx = int(round(p.centroid[1]))
        if y0 <= cy < y1 and x0 <= cx < x1:
            lab = int(p.label)
            rec = rec_by_label.get(lab)
            if rec is None: continue
            visible.append((cy - y0, cx - x0, lab, rec))

    fig, axes = plt.subplots(1, 3, figsize=(28, 10))
    titles = [
        f'{stem} crop=({y0},{x0})  ORIG (raw normalized)',
        'BEFORE: skeleton + endpoints(yellow) + branches(orange) + all v27 raw seeds(red)',
        'AFTER (v30f kept): skeleton + accepted cells only',
    ]
    for k, t in enumerate(titles):
        axes[k].imshow(raw_crop, cmap='gray', vmin=0, vmax=1)
        axes[k].set_title(t, fontsize=12)
        axes[k].axis('off')
    for k in (1, 2):
        axes[k].imshow(skel_overlay)
        axes[k].scatter(ep_x, ep_y, s=12, c='yellow',
                        edgecolors='black', linewidths=0.3, zorder=3)
        axes[k].scatter(br_x, br_y, s=12, c='orange',
                        edgecolors='black', linewidths=0.3, zorder=3)

    n_raw = 0
    for (cy, cx, lab, rec) in visible:
        axes[1].plot(cx, cy, 'o', ms=14, mfc='none', mec='red', mew=1.6)
        n_raw += 1
    axes[1].set_title(
        f'BEFORE: skeleton + endpoints/branches + all v27 raw seeds = {n_raw} red circles',
        fontsize=12)

    counts = {'S': 0, 'L': 0, 'A': 0, 'M': 0, 'P': 0}
    for (cy, cx, lab, rec) in visible:
        col, code = color_for(rec)
        counts[code] += 1
        if code in ('S', 'L', 'A'):
            axes[2].plot(cx, cy, 'o', ms=16, mfc='none', mec=col, mew=2.0)
            ep = endpoints.get(lab, '')
            axes[2].text(cx + 9, cy - 9, f'{code}{ep}',
                         color=col, fontsize=8, weight='bold')

    kept = counts['S'] + counts['L'] + counts['A']
    rejected = counts['M'] + counts['P']
    legend = (f'kept={kept} (S={counts["S"]} L={counts["L"]} A={counts["A"]}) '
              f'rejected={rejected} (M={counts["M"]} P={counts["P"]} not drawn)')
    axes[2].set_title(f'AFTER (v30f kept): {legend}', fontsize=12)

    fig.tight_layout()
    out = V30F / f'v30f_master_{stem}_{tag}.png'
    fig.savefig(out, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out.name}  ({out.stat().st_size // 1024} KB)')


if __name__ == '__main__':
    targets = [
        ('F_WT_2',  1100, 1100, 300, 'tight'),
        ('F_HET_1', 2400, 1900, 300, 'tight'),
        ('F_HET_3', 1500, 1500, 300, 'tight'),
    ]
    for s, y, x, half, tag in targets:
        make_master(s, y, x, half=half, tag=tag)
