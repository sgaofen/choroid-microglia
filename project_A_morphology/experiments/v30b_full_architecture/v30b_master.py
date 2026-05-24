"""
v30b_master.py — single-version master 3-panel like v29_master, but
huge (6000+ px wide) and v30b only:

  panel 0: ORIG (raw normalized)
  panel 1: ORIG + skeleton (cyan)
  panel 2: ORIG + skeleton + v30b 5-color cell markers + endpoint count

Color scheme (consistent with v30b):
  lime    S  = strong accepted
  cyan    L  = weak / low_confidence accepted
  magenta A  = strong but flagged ambiguous neighbor
  yellow  M  = merged_into_strong (rejected, kept for transparency)
  red     P  = process_peak (rejected)
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
        if rec.get('confirmed_neighbors'):
            return 'magenta', 'A'
        return 'lime', 'S'
    if t in ('weak', 'low_confidence_soma'):
        return 'cyan', 'L'
    if t == 'merged_into_strong':
        return 'yellow', 'M'
    return 'red', 'P'


def make_master(stem, yc, xc, half=550, tag='master'):
    raw = tifffile.imread(find_raw(stem)).astype(np.float32)
    norm = normalize(raw)
    skel = np.load(V29 / f'{stem}_skel_pruned.npy')
    somas = np.load(V27 / f'{stem}_soma_cores.npy')
    recs = json.loads((V30B / f'{stem}_seeds_v30b.json').read_text())
    endpoints = json.loads(
        (V30B / f'{stem}_endpoint_counts_v30b.json').read_text())
    endpoints = {int(k): int(v) for k, v in endpoints.items()}

    H, W = raw.shape
    y0, y1 = max(0, yc - half), min(H, yc + half)
    x0, x1 = max(0, xc - half), min(W, xc + half)

    raw_crop = norm[y0:y1, x0:x1]
    skel_crop = skel[y0:y1, x0:x1]
    # dilate skeleton so a 1-px line is visible after rasterization
    from scipy.ndimage import binary_dilation, convolve
    skel_thick = binary_dilation(skel_crop, iterations=1)

    skel_overlay = np.zeros((*skel_crop.shape, 4))
    skel_overlay[skel_thick] = (1.0, 0.35, 0.75, 0.95)  # bright magenta

    # endpoint / branch point detection on the 1-pixel skeleton
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
        cy = int(round(p.centroid[0]))
        cx = int(round(p.centroid[1]))
        if y0 <= cy < y1 and x0 <= cx < x1:
            lab = int(p.label)
            rec = rec_by_label.get(lab)
            if rec is None:
                continue
            visible.append((cy - y0, cx - x0, lab, rec))

    fig, axes = plt.subplots(1, 3, figsize=(36, 13))
    titles = [
        f'{stem} crop=({y0},{x0})  ORIG (raw normalized)',
        f'BEFORE: skeleton + endpoints (yellow) + branches (orange) + all v27 raw seeds (red)',
        f'AFTER (v30b kept): skeleton + accepted cells only',
    ]
    for k, t in enumerate(titles):
        axes[k].imshow(raw_crop, cmap='gray', vmin=0, vmax=1)
        axes[k].set_title(t, fontsize=14)
        axes[k].axis('off')
    for k in [1, 2]:
        axes[k].imshow(skel_overlay)
    # endpoint + branch dots on BOTH skeleton panels
    for k in [1, 2]:
        axes[k].scatter(ep_x, ep_y, s=12, c='yellow',
                        edgecolors='black', linewidths=0.3, zorder=3)
        axes[k].scatter(br_x, br_y, s=12, c='orange',
                        edgecolors='black', linewidths=0.3, zorder=3)
    # ALL v27 raw seeds (red, no labels) on middle "BEFORE" panel
    n_raw = 0
    for (cy, cx, lab, rec) in visible:
        axes[1].plot(cx, cy, 'o', ms=14, mfc='none', mec='red', mew=1.6)
        n_raw += 1
    axes[1].set_title(
        f'BEFORE: skeleton + endpoints/branches + all v27 raw seeds = {n_raw} red circles',
        fontsize=14)

    counts = {'S': 0, 'L': 0, 'A': 0, 'M': 0, 'P': 0}
    for (cy, cx, lab, rec) in visible:
        col, code = color_for(rec)
        counts[code] += 1
        # only DRAW circles for accepted cells (S / L / A) on right panel
        if code in ('S', 'L', 'A'):
            axes[2].plot(cx, cy, 'o', ms=16, mfc='none', mec=col, mew=2.0)
            ep = endpoints.get(lab, '')
            axes[2].text(cx + 9, cy - 9, f'{code}{ep}',
                         color=col, fontsize=8, weight='bold')

    kept = counts['S'] + counts['L'] + counts['A']
    rejected = counts['M'] + counts['P']
    legend = (f'kept={kept}  (S={counts["S"]} lime  L={counts["L"]} cyan  '
              f'A={counts["A"]} magenta)   '
              f'rejected={rejected}  (M={counts["M"]} merged  '
              f'P={counts["P"]} process_peak — not drawn)')
    axes[2].set_title(
        f'AFTER (v30b kept): skeleton + accepted cells only\n{legend}',
        fontsize=13)

    fig.tight_layout()
    out = V30B / f'v30b_master_{stem}_{tag}.png'
    fig.savefig(out, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print('wrote', out.name, '   ', out.stat().st_size // 1024, 'KB')
    return out


if __name__ == '__main__':
    targets = [
        ('F_WT_2', 1100, 1100, 550, 'wide'),
        ('F_HET_1', 2400, 1900, 550, 'wide'),
        ('F_HET_3', 1500, 1500, 550, 'wide'),
    ]
    for (s, y, x, half, tag) in targets:
        make_master(s, y, x, half=half, tag=tag)
