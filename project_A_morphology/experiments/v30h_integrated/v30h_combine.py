"""
v30h: combine v30f accepted set with v30g small-cell candidates.

Pipeline:
  v30f -> primary microglia (process-bearing, with >=1 trunk annulus)
  v30g -> small round cells (Kolmer-like, no trunk required)
  v30h -> union with dedup (no v30g within R_DEDUP of any v30f)

This script does NOT re-score. It just merges existing JSON outputs into a
final accepted-cell list, recomputes endpoint-distribution and trunk metrics
treating the combined set as the cohort.

ONLY ACTIVATE WHEN v30g precision (per Codex H) >= 0.4 ish.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path('/Users/stephenyu/Documents/choroid-microglia/project_A_morphology')
V30F = ROOT / 'experiments/v30f_trunk_gate'
V30G = ROOT / 'experiments/v30g_small_cells'
V30H = ROOT / 'experiments/v30h_integrated'

R_DEDUP = 8  # px


def load(stem):
    f = json.loads((V30F / f'{stem}_seeds_v30f.json').read_text())
    g = json.loads((V30G / f'{stem}_small_cell_candidates_v30g.json').read_text())
    return f, g


def main():
    summary_rows = []
    for stem in ['F_WT_2', 'F_HET_1', 'F_HET_3']:
        f, g = load(stem)
        f_acc = [r for r in f
                 if r['type'] in ('strong', 'weak', 'low_confidence_soma')]
        f_yc = np.array([r['yc'] for r in f_acc])
        f_xc = np.array([r['xc'] for r in f_acc])

        kept_g = []
        for c in g:
            d2 = (f_yc - c['yc']) ** 2 + (f_xc - c['xc']) ** 2
            if d2.min() < R_DEDUP * R_DEDUP:
                continue
            kept_g.append(c)

        combined = []
        for r in f_acc:
            combined.append(dict(
                source='v30f', label=int(r['label']),
                yc=r['yc'], xc=r['xc'], type=r['type'],
                score=r.get('score', 0.0)))
        for c in kept_g:
            combined.append(dict(
                source='v30g_small', label=int(c['label']),
                yc=c['yc'], xc=c['xc'], type='small_round_cell',
                area=c['area'], ecc=c['ecc'], sol=c['sol'],
                peak_mean=c['peak_mean']))

        (V30H / f'{stem}_combined_v30h.json').write_text(
            json.dumps(combined, indent=1))

        print(f'{stem}: v30f={len(f_acc)} + v30g={len(g)} '
              f'(deduped to {len(kept_g)}) = v30h={len(combined)}')
        summary_rows.append((stem, len(f_acc), len(g), len(kept_g), len(combined)))

    print('\n=== v30h summary ===')
    print(f'{"image":>10}  v30f  v30g_raw  v30g_kept  v30h_total')
    for s, vf, vg, vgk, vh in summary_rows:
        print(f'{s:>10}  {vf:4d}  {vg:8d}  {vgk:9d}  {vh:10d}')


if __name__ == '__main__':
    main()
