"""
Coarse per-cell morphology classification on ONE image (F_WT_2), using the
already-computed v30f per-cell skeleton metrics — NO new segmentation.

Each accepted cell has:
  n_trunks          = primary processes leaving the soma (annulus count)
  n_local_branches  = total branch points (bifurcations) within 30 px
                      -> captures secondary / higher-order branching
  skel_len_local    = total skeleton length within 30 px

Complexity tiers (simple -> ramified), combining primary + secondary branching:
  I   amoeboid/简单   : trunks <= 1                       (barely any processes)
  II  双极/低分支     : trunks == 2 and branches <= 8
  IV  高度ramified    : trunks >= 4 and branches >= 16, or trunks >= 5
  III 中度ramified    : everything in between

Caveat: skeleton-based detection under-represents PURE amoeboid (round, no
process) microglia — they have no skeleton, so Tier I here is a floor, not the
full amoeboid fraction.
"""
import json
from pathlib import Path
from collections import Counter

import numpy as np
import tifffile
import matplotlib.pyplot as plt

ROOT = Path('/Users/stephenyu/choroid-microglia/project_A_morphology')
RAW = ROOT / 'data/raw'
V30F = ROOT / 'experiments/v30f_trunk_gate'
OUT = ROOT / 'experiments/morphology_distribution'
STEM = 'F_WT_2'

TIERS = ['I 类amoeboid/简单', 'II 双极/低分支', 'III 中度ramified', 'IV 高度ramified']
COLORS = {TIERS[0]: '#d62728', TIERS[1]: '#ff7f0e',
          TIERS[2]: '#2ca02c', TIERS[3]: '#1f77b4'}


def find_raw(stem):
    for p in RAW.glob('*.tif'):
        if stem in p.name:
            return p
    raise FileNotFoundError(stem)


def normalize(img, p_lo=1.0, p_hi=99.5):
    lo, hi = np.percentile(img, [p_lo, p_hi])
    return np.clip((img - lo) / (hi - lo + 1e-9), 0, 1)


def classify(trunks, branches):
    if trunks <= 1:
        return TIERS[0]
    if trunks == 2 and branches <= 8:
        return TIERS[1]
    if (trunks >= 4 and branches >= 16) or trunks >= 5:
        return TIERS[3]
    return TIERS[2]


def main():
    OUT.mkdir(exist_ok=True)
    recs = json.loads((V30F / f'{STEM}_trunk_metrics_v30f.json').read_text())
    for r in recs:
        r['tier'] = classify(r['n_trunks'], r['n_local_branches'])

    n = len(recs)
    counts = Counter(r['tier'] for r in recs)

    print(f'=== {STEM}: {n} process-bearing cells ===')
    print(f'{"tier":22} {"count":>6} {"pct":>7}  {"mean_primary":>12} {"mean_branchpts":>14}')
    for t in TIERS:
        sub = [r for r in recs if r['tier'] == t]
        c = len(sub)
        mp = np.mean([r['n_trunks'] for r in sub]) if sub else 0
        mb = np.mean([r['n_local_branches'] for r in sub]) if sub else 0
        print(f'{t:22} {c:>6} {100*c/n:>6.1f}%  {mp:>12.2f} {mb:>14.1f}')

    # ---- figure: cells colored by tier on the image + proportion bar ----
    raw = tifffile.imread(find_raw(STEM)).astype(np.float32)
    norm = normalize(raw)
    fig, axes = plt.subplots(1, 2, figsize=(22, 11),
                             gridspec_kw={'width_ratios': [3, 1]})
    axes[0].imshow(norm, cmap='gray', vmin=0, vmax=1)
    for t in TIERS:
        xs = [r['xc'] for r in recs if r['tier'] == t]
        ys = [r['yc'] for r in recs if r['tier'] == t]
        axes[0].scatter(xs, ys, s=14, c=COLORS[t], label=f'{t} ({len(xs)})',
                        edgecolors='none', alpha=0.85)
    axes[0].set_title(f'{STEM} — {n} cells colored by morphology complexity',
                      fontsize=13)
    axes[0].axis('off')
    axes[0].legend(loc='upper right', fontsize=10, framealpha=0.9)

    pcts = [100 * counts[t] / n for t in TIERS]
    bars = axes[1].barh(range(len(TIERS)), pcts,
                        color=[COLORS[t] for t in TIERS])
    axes[1].set_yticks(range(len(TIERS)))
    axes[1].set_yticklabels([t.split()[0] for t in TIERS], fontsize=11)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('% of cells', fontsize=11)
    axes[1].set_title('morphology distribution', fontsize=13)
    for i, (b, p) in enumerate(zip(bars, pcts)):
        axes[1].text(p + 1, i, f'{p:.1f}%', va='center', fontsize=11)
    axes[1].set_xlim(0, max(pcts) * 1.25)

    fig.suptitle(f'{STEM} microglia morphology complexity '
                 f'(process-bearing cells; pure amoeboid under-counted)',
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = OUT / f'{STEM}_morphology_classes.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
