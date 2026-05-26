"""
Compute primary-trunk / local-branch / local-skel metrics for ALL v30f
accepted cells (strong, weak, low_confidence_soma). Then dump per-image
distributions so the WT vs HET comparison can use whichever signal is most
stable.
"""
import json, sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path('/Users/stephenyu/Documents/choroid-microglia/project_A_morphology')
V27 = ROOT / 'experiments/v27_clean_graph'
V29 = ROOT / 'experiments/v29_short_spur_audit'
V30F = ROOT / 'experiments/v30f_trunk_gate'

sys.path.insert(0, str(V30F))
from v30f_run import load_inputs, make_binary_and_dist, annulus_trunk_count_seed


def local_skel_length(skel, yc, xc, radius=30):
    H, W = skel.shape
    y0, y1 = max(0, yc - radius), min(H, yc + radius + 1)
    x0, x1 = max(0, xc - radius), min(W, xc + radius + 1)
    patch = skel[y0:y1, x0:x1]
    yy, xx = np.indices(patch.shape)
    cy = yc - y0; cx = xc - x0
    rsq = (yy - cy) ** 2 + (xx - cx) ** 2
    mask = rsq <= radius * radius
    return int((patch & mask).sum())


def local_branch_count(skel, yc, xc, radius=30):
    H, W = skel.shape
    y0, y1 = max(0, yc - radius), min(H, yc + radius + 1)
    x0, x1 = max(0, xc - radius), min(W, xc + radius + 1)
    patch = skel[y0:y1, x0:x1]
    k = np.ones((3, 3), dtype=np.uint8); k[1, 1] = 0
    nb = ndi.convolve(patch.astype(np.uint8), k, mode='constant') * patch
    yy, xx = np.indices(patch.shape)
    cy = yc - y0; cx = xc - x0
    rsq = (yy - cy) ** 2 + (xx - cx) ** 2
    mask = rsq <= radius * radius
    return int(((nb >= 3) & mask).sum())


def main():
    summary_rows = []
    for stem in ['F_WT_2', 'F_HET_1', 'F_HET_3']:
        raw, skel, _ = load_inputs(stem)
        _, _, dist_s = make_binary_and_dist(raw)
        recs = json.loads((V30F / f'{stem}_seeds_v30f.json').read_text())
        accepted = [r for r in recs
                    if r['type'] in ('strong', 'weak', 'low_confidence_soma')]

        out = []
        for r in accepted:
            yc, xc = r['yc'], r['xc']
            r_soma = float(dist_s[yc, xc])
            r_in = max(3, int(round(r_soma)) + 2)
            r_out = r_in + 6
            n_trunks = annulus_trunk_count_seed(skel, yc, xc, r_in, r_out)
            n_branches = local_branch_count(skel, yc, xc, radius=30)
            skel_len = local_skel_length(skel, yc, xc, radius=30)
            out.append(dict(
                label=r['label'], yc=yc, xc=xc, type=r['type'],
                score=float(r['score']),
                r_soma=float(r_soma), r_in=int(r_in), r_out=int(r_out),
                n_trunks=int(n_trunks),
                n_local_branches=int(n_branches),
                skel_len_local=int(skel_len),
            ))

        (V30F / f'{stem}_trunk_metrics_v30f.json').write_text(json.dumps(out, indent=1))

        trunks = np.array([r['n_trunks'] for r in out])
        branches = np.array([r['n_local_branches'] for r in out])
        skel_len = np.array([r['skel_len_local'] for r in out])

        n_strong = sum(1 for r in out if r['type'] == 'strong')
        n_weak = sum(1 for r in out if r['type'] in ('weak', 'low_confidence_soma'))
        print(f'{stem}: n={len(out)} (strong={n_strong} weak={n_weak})')
        print(f'  trunks   mean={trunks.mean():.2f} median={np.median(trunks):.1f} '
              f'dist={np.bincount(np.clip(trunks, 0, 10)).tolist()}')
        print(f'  branches mean={branches.mean():.2f} median={np.median(branches):.1f}')
        print(f'  skel_len mean={skel_len.mean():.0f} median={np.median(skel_len):.0f}')

        summary_rows.append((stem, len(out), n_strong, n_weak,
                             trunks.mean(), branches.mean(), skel_len.mean()))

    print('\nCross-genotype v30f:')
    print(f'{"image":10}  n   strong weak   mean_trunks  mean_branches  mean_skel_len')
    for stem, n, ns, nw, mt, mb, ms in summary_rows:
        print(f'{stem:10}  {n:4d}  {ns:4d}  {nw:4d}    {mt:.2f}         '
              f'{mb:.2f}          {ms:.0f}')


if __name__ == '__main__':
    main()
