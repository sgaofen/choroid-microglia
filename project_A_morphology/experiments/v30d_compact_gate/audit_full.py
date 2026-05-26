"""
Generate four kinds of v30d audit crops:
  A. WT_2 tight region (15 cells, top by score)  — also re-runs for round 2 retest
  B. HET_1 tight region (15 cells, top by score)
  C. HET_3 tight region (15 cells, top by score)
  D. dense_cluster regions (one per image): full 200x200 ORIG + v30d markers
  E. merged-into-strong sample: 10 cells per image that were absorbed by a
     strong neighbor, so a human/Codex can verify they weren't real cells
"""
import json, sys, random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tifffile

ROOT = Path('/Users/stephenyu/Documents/choroid-microglia/project_A_morphology')
RAW = ROOT / 'data/raw'
V27 = ROOT / 'experiments/v27_clean_graph'
V29 = ROOT / 'experiments/v29_short_spur_audit'
V30D = ROOT / 'experiments/v30d_compact_gate'

sys.path.insert(0, str(V30D))
from v30d_run import normalize

random.seed(42)


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


def per_cell_crop(stem, recs_subset, norm, tag, half=40, prefix='cell'):
    H, W = norm.shape
    for i, r in enumerate(recs_subset):
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
            f'{code} s={r.get("score",0):.2f} blob={r.get("blob",0):.2f} '
            f'n_dirs={r.get("n_dirs","?")} rprom={r.get("rprom",0):.1f}',
            fontsize=8)
        axes[1].axis('off')
        fig.tight_layout()
        out = V30D / f'audit_{stem}_{tag}_{prefix}{i:02d}_L{r["label"]}.png'
        fig.savefig(out, dpi=140, bbox_inches='tight')
        plt.close(fig)


def dense_region_crop(stem, norm, recs, all_recs, yc, xc, half=100, tag='dense'):
    """Full neighborhood snap, all cells visible labeled by type."""
    H, W = norm.shape
    y0, y1 = max(0, yc - half), min(H, yc + half)
    x0, x1 = max(0, xc - half), min(W, xc + half)
    crop = norm[y0:y1, x0:x1]

    # skeleton overlay
    from scipy.ndimage import binary_dilation
    skel = np.load(V29 / f'{stem}_skel_pruned.npy')[y0:y1, x0:x1]
    skel_thick = binary_dilation(skel, iterations=1)
    ov = np.zeros((*skel.shape, 4))
    ov[skel_thick] = (1.0, 0.35, 0.75, 0.85)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    titles = ['ORIG', 'ORIG + skeleton',
              'ORIG + skel + ALL v30d markers (incl rejected)']
    for k, t in enumerate(titles):
        axes[k].imshow(crop, cmap='gray', vmin=0, vmax=1)
        axes[k].set_title(f'{stem} {tag}  {t}', fontsize=11)
        axes[k].axis('off')
    for k in (1, 2):
        axes[k].imshow(ov)
    for r in all_recs:
        if y0 <= r['yc'] < y1 and x0 <= r['xc'] < x1:
            col, code = color_for(r)
            axes[2].plot(r['xc'] - x0, r['yc'] - y0, 'o',
                         ms=14, mfc='none', mec=col, mew=1.8)
            axes[2].text(r['xc'] - x0 + 5, r['yc'] - y0 - 5,
                         code, color=col, fontsize=8, weight='bold')
    fig.tight_layout()
    out = V30D / f'audit_{stem}_{tag}_region.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote dense {out.name}')


def merged_crop(stem, norm, all_recs, n_per=10):
    """Show each merged_into_strong cell PLUS its assigned strong, side by side."""
    H, W = norm.shape
    merged = [r for r in all_recs if r['type'] == 'merged_into_strong']
    random.shuffle(merged)
    merged = merged[:n_per]
    by_lab = {r['label']: r for r in all_recs}
    for i, r in enumerate(merged):
        keep_lab = r.get('merged_to')
        keep = by_lab.get(keep_lab)
        if keep is None:
            continue
        # bounding box covering both
        ys = [r['yc'], keep['yc']]; xs = [r['xc'], keep['xc']]
        cy = (min(ys) + max(ys)) // 2
        cx = (min(xs) + max(xs)) // 2
        half = max(40, max(abs(ys[0]-ys[1]), abs(xs[0]-xs[1])) // 2 + 30)
        y0, y1 = max(0, cy - half), min(H, cy + half)
        x0, x1 = max(0, cx - half), min(W, cx + half)
        crop = norm[y0:y1, x0:x1]
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(crop, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f'{stem} merged L{r["label"]} into L{keep_lab}', fontsize=9)
        axes[0].axis('off')
        axes[1].imshow(crop, cmap='gray', vmin=0, vmax=1)
        axes[1].plot(r['xc'] - x0, r['yc'] - y0, 'o', ms=20, mfc='none',
                     mec='yellow', mew=2.0)
        axes[1].text(r['xc'] - x0 + 8, r['yc'] - y0 - 8,
                     f'M score={r["score"]:.2f}', color='yellow', fontsize=8)
        axes[1].plot(keep['xc'] - x0, keep['yc'] - y0, 'o', ms=20, mfc='none',
                     mec='lime', mew=2.0)
        axes[1].text(keep['xc'] - x0 + 8, keep['yc'] - y0 - 8,
                     f'S score={keep["score"]:.2f}', color='lime', fontsize=8)
        axes[1].set_title('yellow=merged victim, lime=kept strong', fontsize=9)
        axes[1].axis('off')
        fig.tight_layout()
        out = V30D / f'audit_{stem}_merged{i:02d}_L{r["label"]}_into_L{keep_lab}.png'
        fig.savefig(out, dpi=140, bbox_inches='tight')
        plt.close(fig)


def main():
    targets = [
        ('F_WT_2',  850, 1350, 850, 1350, 'WTtight',
         (1909, 2019, 'WTdense')),
        ('F_HET_1', 2150, 2650, 1650, 2150, 'HET1tight',
         (3061, 1916, 'HET1dense')),
        ('F_HET_3', 1250, 1750, 1250, 1750, 'HET3tight',
         (874, 2390, 'HET3vessel_dense')),
    ]
    for (stem, y_lo, y_hi, x_lo, x_hi, tight_tag, (yc_d, xc_d, dense_tag)) in targets:
        print(f'=== {stem} ===')
        raw = tifffile.imread(find_raw(stem)).astype(np.float32)
        norm = normalize(raw)
        recs = json.loads((V30D / f'{stem}_seeds_v30d.json').read_text())

        # A/B/C: per-cell tight region top 15
        accepted = [r for r in recs
                    if r['type'] in ('strong', 'weak', 'low_confidence_soma')
                    and y_lo <= r['yc'] < y_hi and x_lo <= r['xc'] < x_hi]
        accepted.sort(key=lambda r: -r['score'])
        per_cell_crop(stem, accepted[:15], norm, tight_tag)
        print(f'  wrote {len(accepted[:15])} tight per-cell')

        # D: dense region
        dense_region_crop(stem, norm, recs, recs, yc_d, xc_d, half=100,
                          tag=dense_tag)

        # E: merged victims
        merged_crop(stem, norm, recs, n_per=10)
        merged_count = sum(1 for r in recs if r['type'] == 'merged_into_strong')
        print(f'  total merged_into_strong in image: {merged_count}')


if __name__ == '__main__':
    main()
