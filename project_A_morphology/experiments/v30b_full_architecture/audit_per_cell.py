"""
Per-cell audit. For each accepted v30b cell in a given region,
write a tiny ORIG | ORIG+marker side-by-side PNG so the human
(or Claude) can verify whether the marked center actually lands
on the bright soma.

We sort by score and emit up to N per region so output stays manageable.
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


def make_cell_crops(stem, y_lo, y_hi, x_lo, x_hi, tag, half=40, max_cells=20):
    raw = tifffile.imread(find_raw(stem)).astype(np.float32)
    norm = normalize(raw)
    recs = json.loads((V30B / f'{stem}_seeds_v30b.json').read_text())
    accepted = [r for r in recs
                if r['type'] in ('strong', 'weak', 'low_confidence_soma')
                and y_lo <= r['yc'] < y_hi
                and x_lo <= r['xc'] < x_hi]
    accepted.sort(key=lambda r: -r['score'])
    accepted = accepted[:max_cells]
    print(f'{stem} {tag}: {len(accepted)} accepted cells in region')

    H, W = raw.shape
    outs = []
    for i, r in enumerate(accepted):
        yc, xc = r['yc'], r['xc']
        y0, y1 = max(0, yc - half), min(H, yc + half)
        x0, x1 = max(0, xc - half), min(W, xc + half)

        crop = norm[y0:y1, x0:x1]
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(crop, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f'{stem} #{i} L{r["label"]} ({yc},{xc})', fontsize=8)
        axes[0].axis('off')

        axes[1].imshow(crop, cmap='gray', vmin=0, vmax=1)
        col, code = color_for(r)
        axes[1].plot(xc - x0, yc - y0, 'o', ms=20, mfc='none',
                     mec=col, mew=2.0)
        axes[1].plot(xc - x0, yc - y0, '+', ms=14, mec=col, mew=1.5)
        axes[1].set_title(
            f'{code} score={r["score"]:.2f} '
            f'compact={r["compact"]:.2f} '
            f'blob={r["blob"]:.2f} tube={r["tube"]:.2f}',
            fontsize=8)
        axes[1].axis('off')
        fig.tight_layout()
        out = V30B / f'audit_{stem}_{tag}_cell{i:02d}_L{r["label"]}.png'
        fig.savefig(out, dpi=140, bbox_inches='tight')
        plt.close(fig)
        outs.append(out)
    return outs


if __name__ == '__main__':
    targets = [
        ('F_WT_2',  850, 1350, 850, 1350, 'WTtight', 15),
        ('F_HET_1', 2150, 2650, 1650, 2150, 'HET1tight', 15),
        ('F_HET_3', 1250, 1750, 1250, 1750, 'HET3tight', 15),
        ('F_HET_3',  624, 1124, 2140, 2640, 'HET3vessel', 15),
    ]
    for (s, y_lo, y_hi, x_lo, x_hi, tag, n) in targets:
        outs = make_cell_crops(s, y_lo, y_hi, x_lo, x_hi, tag, max_cells=n)
        print(f'  -> wrote {len(outs)} PNGs')
