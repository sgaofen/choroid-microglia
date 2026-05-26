"""Per-cell audit for v30d outputs — same WT_2 region as v30c."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tifffile

ROOT = Path('/Users/stephenyu/Documents/choroid-microglia/project_A_morphology')
RAW = ROOT / 'data/raw'
V30D = ROOT / 'experiments/v30d_compact_gate'

sys.path.insert(0, str(V30D))
from v30d_run import normalize


def find_raw(stem):
    for p in RAW.glob('*.tif'):
        if stem in p.name: return p


def color_for(rec):
    t = rec['type']
    if t == 'strong':
        if rec.get('confirmed_neighbors'): return 'magenta', 'A'
        return 'lime', 'S'
    if t in ('weak', 'low_confidence_soma'): return 'cyan', 'L'
    if t == 'merged_into_strong': return 'yellow', 'M'
    return 'red', 'P'


def make_cell_crops(stem, y_lo, y_hi, x_lo, x_hi, tag, half=40, max_cells=15):
    raw = tifffile.imread(find_raw(stem)).astype(np.float32)
    norm = normalize(raw)
    recs = json.loads((V30D / f'{stem}_seeds_v30d.json').read_text())
    accepted = [r for r in recs
                if r['type'] in ('strong', 'weak', 'low_confidence_soma')
                and y_lo <= r['yc'] < y_hi and x_lo <= r['xc'] < x_hi]
    accepted.sort(key=lambda r: -r['score'])
    accepted = accepted[:max_cells]
    print(f'{stem} {tag}: {len(accepted)} accepted cells')

    H, W = raw.shape
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
        axes[1].plot(xc - x0, yc - y0, 'o', ms=20, mfc='none', mec=col, mew=2.0)
        axes[1].plot(xc - x0, yc - y0, '+', ms=14, mec=col, mew=1.5)
        if 'yc_raw' in r and (r['yc_raw'] != r['yc'] or r['xc_raw'] != r['xc']):
            axes[1].plot(r['xc_raw'] - x0, r['yc_raw'] - y0, 'x',
                         ms=10, mec='red', mew=1.5)
        axes[1].set_title(
            f'{code} s={r["score"]:.2f} blob={r["blob"]:.2f} '
            f'lc:a={r["lc_area"]} e={r["lc_ecc"]:.2f} s={r["lc_sol"]:.2f}',
            fontsize=8)
        axes[1].axis('off')
        fig.tight_layout()
        out = V30D / f'audit_{stem}_{tag}_cell{i:02d}_L{r["label"]}.png'
        fig.savefig(out, dpi=140, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    make_cell_crops('F_WT_2', 850, 1350, 850, 1350, 'WTtight', max_cells=15)
