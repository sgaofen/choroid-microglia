"""
Side-by-side dense-region comparator: v30e markers vs v30f markers on the
same image patches Codex D audited. Lets Codex G judge whether v30f's
tighter thresholds and trunk-gate killed the right cells (Codex D's reported
over-acceptances) without losing legitimate ones.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tifffile

ROOT = Path('/Users/stephenyu/Documents/choroid-microglia/project_A_morphology')
RAW = ROOT / 'data/raw'
V29 = ROOT / 'experiments/v29_short_spur_audit'
V30E = ROOT / 'experiments/v30e_merge_fixes'
V30F = ROOT / 'experiments/v30f_trunk_gate'

sys.path.insert(0, str(V30F))
from v30f_run import normalize


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


def render_side_by_side(stem, yc, xc, half=100, tag='dense'):
    raw = tifffile.imread(find_raw(stem)).astype(np.float32)
    norm = normalize(raw)
    H, W = norm.shape
    y0, y1 = max(0, yc - half), min(H, yc + half)
    x0, x1 = max(0, xc - half), min(W, xc + half)
    crop = norm[y0:y1, x0:x1]

    from scipy.ndimage import binary_dilation
    skel = np.load(V29 / f'{stem}_skel_pruned.npy')[y0:y1, x0:x1]
    skel_thick = binary_dilation(skel, iterations=1)
    ov = np.zeros((*skel.shape, 4))
    ov[skel_thick] = (1.0, 0.35, 0.75, 0.65)

    recs_e = json.loads((V30E / f'{stem}_seeds_v30e.json').read_text())
    recs_f = json.loads((V30F / f'{stem}_seeds_v30f.json').read_text())

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    titles = ['ORIG',
              f'v30e (accepted only)',
              f'v30f (accepted only)']
    for k in range(3):
        axes[k].imshow(crop, cmap='gray', vmin=0, vmax=1)
        axes[k].set_title(f'{stem} {tag}  {titles[k]}', fontsize=11)
        axes[k].axis('off')

    for k in (1, 2):
        axes[k].imshow(ov)

    accepted_types = ('strong', 'weak', 'low_confidence_soma')
    for ax, recs in [(axes[1], recs_e), (axes[2], recs_f)]:
        n = 0
        for r in recs:
            if r['type'] not in accepted_types: continue
            if not (y0 <= r['yc'] < y1 and x0 <= r['xc'] < x1): continue
            col, code = color_for(r)
            ax.plot(r['xc'] - x0, r['yc'] - y0, 'o',
                    ms=14, mfc='none', mec=col, mew=1.8)
            ax.text(r['xc'] - x0 + 5, r['yc'] - y0 - 5,
                    code, color=col, fontsize=8, weight='bold')
            n += 1
        ax.set_title(ax.get_title() + f' n={n}', fontsize=11)

    fig.tight_layout()
    out = V30F / f'cmp_{stem}_{tag}.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out.name}')


def main():
    # Use SAME dense centers Codex D audited
    targets = [
        ('F_WT_2',  1909, 2019, 'WTdense'),
        ('F_HET_1', 3061, 1916, 'HET1dense'),
        ('F_HET_3', 874,  2390, 'HET3vessel_dense'),
        # tight per-genotype top regions too
        ('F_WT_2',  1100, 1100, 'WTtight'),
        ('F_HET_1', 2400, 1900, 'HET1tight'),
        ('F_HET_3', 1500, 1500, 'HET3tight'),
    ]
    for stem, yc, xc, tag in targets:
        render_side_by_side(stem, yc, xc, half=100, tag=tag)


if __name__ == '__main__':
    main()
