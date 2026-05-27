"""
Topology annotator: mark CELL CENTER (中心) + BRANCH POINT (分叉点) + ENDPOINT.

Pre-filled from the algorithm so you CORRECT instead of mark-from-scratch:
  center   (white, big)   <- v30f detected soma centers
  branch   (magenta, mid) <- skeleton junctions (degree >= 3)
  endpoint (yellow, small)<- skeleton tips      (degree == 1)
Reference (look only): raw image, cyan skeleton.

This validates the skeleton's topology against your eye — delete spurious
branch/endpoints, add missed ones, fix centers.

RUN (default = clean 400x400 region of F_WT_2 at 1000,1000):
  ~/.micromamba/envs/micro-sam/bin/python3.13 annotate_topology.py
  ~/.micromamba/envs/micro-sam/bin/python3.13 annotate_topology.py F_WT_2 1000 1000 400
  ~/.micromamba/envs/micro-sam/bin/python3.13 annotate_topology.py F_HET_1 2300 1800 400

SAVE: press  s  (writes *_topology_annotations.json next to this script).
"""
import sys, json
from pathlib import Path
from datetime import datetime

import numpy as np
import tifffile
from scipy import ndimage as ndi

ROOT = Path('/Users/stephenyu/choroid-microglia/project_A_morphology')
RAW = ROOT / 'data/raw'
V29 = ROOT / 'experiments/v29_short_spur_audit'
V30F = ROOT / 'experiments/v30f_trunk_gate'
OUT = ROOT / 'experiments/annotation_tool'


def find_raw(stem):
    for p in RAW.glob('*.tif'):
        if stem in p.name:
            return p
    raise FileNotFoundError(stem)


def normalize(img, p_lo=1.0, p_hi=99.5):
    lo, hi = np.percentile(img, [p_lo, p_hi])
    return np.clip((img - lo) / (hi - lo + 1e-9), 0, 1)


def main():
    stem = sys.argv[1] if len(sys.argv) > 1 else 'F_WT_2'
    if len(sys.argv) >= 5:
        y0, x0, sz = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    else:
        y0, x0, sz = 1000, 1000, 400

    import napari

    raw = tifffile.imread(find_raw(stem)).astype(np.float32)
    norm = normalize(raw)[y0:y0 + sz, x0:x0 + sz]
    skel = np.load(V29 / f'{stem}_skel_pruned.npy').astype(bool)[y0:y0 + sz, x0:x0 + sz]

    # branch points (deg>=3) and endpoints (deg==1) from the skeleton
    k = np.ones((3, 3), np.uint8); k[1, 1] = 0
    deg = ndi.convolve(skel.astype(np.uint8), k, mode='constant') * skel
    by, bx = np.where(skel & (deg >= 3))
    ey, ex = np.where(skel & (deg == 1))
    branch_pts = np.column_stack([by, bx]) if len(by) else np.empty((0, 2))
    end_pts = np.column_stack([ey, ex]) if len(ey) else np.empty((0, 2))

    # centers from v30f, cropped + offset to local coords
    cells = json.loads((V30F / f'{stem}_trunk_metrics_v30f.json').read_text())
    ctr = [[c['yc'] - y0, c['xc'] - x0] for c in cells
           if y0 <= c['yc'] < y0 + sz and x0 <= c['xc'] < x0 + sz]
    center_pts = np.array(ctr) if ctr else np.empty((0, 2))

    v = napari.Viewer(title=f'topology {stem} [{y0},{x0}] {sz}x{sz}')
    v.add_image(norm, name='raw', colormap='gray', contrast_limits=(0, 1))
    v.add_image(skel.astype(float), name='skeleton (ref)', colormap='cyan',
                blending='additive', opacity=0.55)
    p_end = v.add_points(end_pts, name='endpoint', face_color='yellow',
                         size=8, border_color='black')
    p_br = v.add_points(branch_pts, name='分叉点 branch', face_color='magenta',
                        size=11, border_color='white')
    p_ctr = v.add_points(center_pts, name='中心 center', face_color='white',
                         size=18, border_color='red', opacity=0.9)
    p_ctr.mode = 'add'

    @v.bind_key('s')
    def save(viewer):
        def gpts(layer):
            return [[float(p[0] + y0), float(p[1] + x0)] for p in layer.data]
        rec = dict(stem=stem, crop=[y0, x0, sz],
                   saved=datetime.now().isoformat(timespec='seconds'),
                   center=gpts(p_ctr), branch=gpts(p_br), endpoint=gpts(p_end),
                   counts=dict(center=len(p_ctr.data), branch=len(p_br.data),
                               endpoint=len(p_end.data)))
        OUT.mkdir(exist_ok=True)
        fp = OUT / f'{stem}_{y0}_{x0}_topology_annotations.json'
        fp.write_text(json.dumps(rec, indent=1))
        print(f'[saved] {fp}  center={rec["counts"]["center"]} '
              f'branch={rec["counts"]["branch"]} endpoint={rec["counts"]["endpoint"]}')

    print('=' * 64)
    print(f'TOPOLOGY  {stem}  region [{y0},{x0}] {sz}x{sz}')
    print(f'pre-filled: {len(ctr)} centers, {len(by)} branch points, {len(ey)} endpoints')
    print('Layers (left list): 中心 center(white) / 分叉点 branch(magenta) / endpoint(yellow)')
    print('Mark: pick a layer -> click "Add points" tool (top-left) -> click image')
    print('Fix : "Select points" tool -> click a point -> drag to move / Delete to remove')
    print('Save: press  s')
    print('=' * 64)
    napari.run()


if __name__ == '__main__':
    main()
