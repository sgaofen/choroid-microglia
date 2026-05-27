"""
Topology annotator (BLANK start): mark CELL CENTER (中心) + BRANCH POINT (分叉点)
+ ENDPOINT yourself, from scratch. Layers start EMPTY (unbiased ground truth).
Reference only (look, don't edit): raw image + cyan skeleton.

Points are small so they don't cover the feature you're clicking; use the
"point size" slider in the layer controls if you want them even smaller/bigger.

SWITCH IMAGE = change the first argument:
  ...annotate_topology.py F_WT_2          # whole image (default)
  ...annotate_topology.py F_HET_1         # whole image
  ...annotate_topology.py F_HET_3
WHOLE image vs a REGION:
  ...annotate_topology.py F_WT_2                 # WHOLE image
  ...annotate_topology.py F_WT_2 1000 1000 600   # region: y0 x0 size

SAVE: press  s   (writes *_topology_annotations.json next to this script).
"""
import sys, json
from pathlib import Path
from datetime import datetime

import numpy as np
import tifffile

ROOT = Path('/Users/stephenyu/choroid-microglia/project_A_morphology')
RAW = ROOT / 'data/raw'
V29 = ROOT / 'experiments/v29_short_spur_audit'
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
    full = len(sys.argv) < 5
    if full:
        y0, x0, sz = 0, 0, None
    else:
        y0, x0, sz = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

    import napari

    raw = tifffile.imread(find_raw(stem)).astype(np.float32)
    norm = normalize(raw)
    skel = np.load(V29 / f'{stem}_skel_pruned.npy').astype(float)
    if not full:
        norm = norm[y0:y0 + sz, x0:x0 + sz]
        skel = skel[y0:y0 + sz, x0:x0 + sz]

    v = napari.Viewer(title=f'topology {stem}' + ('' if full else f' [{y0},{x0}] {sz}'))
    v.add_image(norm, name='raw', colormap='gray', contrast_limits=(0, 1))
    v.add_image(skel, name='skeleton (ref)', colormap='cyan',
                blending='additive', opacity=0.55)
    # EMPTY layers, SMALL points
    p_end = v.add_points(np.empty((0, 2)), name='endpoint',
                         face_color='yellow', size=4, border_width=0)
    p_br = v.add_points(np.empty((0, 2)), name='分叉点 branch',
                        face_color='magenta', size=5, border_width=0)
    p_ctr = v.add_points(np.empty((0, 2)), name='中心 center',
                         face_color='red', size=7, border_width=0)
    p_ctr.mode = 'add'

    @v.bind_key('s')
    def save(viewer):
        def gpts(layer):
            return [[float(p[0] + y0), float(p[1] + x0)] for p in layer.data]
        rec = dict(stem=stem, crop=(None if full else [y0, x0, sz]),
                   saved=datetime.now().isoformat(timespec='seconds'),
                   center=gpts(p_ctr), branch=gpts(p_br), endpoint=gpts(p_end),
                   counts=dict(center=len(p_ctr.data), branch=len(p_br.data),
                               endpoint=len(p_end.data)))
        OUT.mkdir(exist_ok=True)
        tag = 'full' if full else f'{y0}_{x0}'
        fp = OUT / f'{stem}_{tag}_topology_annotations.json'
        fp.write_text(json.dumps(rec, indent=1))
        print(f'[saved] {fp}  center={rec["counts"]["center"]} '
              f'branch={rec["counts"]["branch"]} endpoint={rec["counts"]["endpoint"]}')

    print('=' * 60)
    print(f'TOPOLOGY  {stem}  {"WHOLE image" if full else f"region [{y0},{x0}] {sz}"}')
    print('Layers start EMPTY. Pick a layer (中心/分叉点/endpoint), the')
    print('"Add points" tool is on -> click image to drop a point.')
    print('Select tool (arrow): click a point -> drag=move, Delete=remove.')
    print('Point-size slider is in the layer controls if dots feel too big.')
    print('Save: press  s')
    print('=' * 60)
    napari.run()


if __name__ == '__main__':
    main()
