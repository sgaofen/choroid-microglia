"""
Topology annotator (blank): mark CELL CENTER (中心) + BRANCH POINT (分叉点)
+ ENDPOINT yourself, from scratch. Only the raw image is shown (no skeleton).

CONTROLS
  add a point      : pick a layer (左侧), click on the image
  DELETE nearest   : RIGHT-CLICK near a point (in that layer)  <-- new
  clear ALL points : press  c
  move a point     : Select tool (arrow) -> drag
  zoom / pan       : scroll = zoom ; hold SPACE + drag = pan (works while adding)
  point size       : slider in layer controls (already tiny)
  SAVE             : press  s

SWITCH IMAGE = first arg:  F_WT_2 | F_HET_1 | F_HET_3
WHOLE vs REGION:
  ...annotate_topology.py F_WT_2                 # whole image (default)
  ...annotate_topology.py F_WT_2 1000 1000 600   # region y0 x0 size
"""
import sys, json
from pathlib import Path
from datetime import datetime

import numpy as np
import tifffile

ROOT = Path('/Users/stephenyu/choroid-microglia/project_A_morphology')
RAW = ROOT / 'data/raw'
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
    if not full:
        norm = norm[y0:y0 + sz, x0:x0 + sz]

    v = napari.Viewer(title=f'topology {stem}' + ('' if full else f' [{y0},{x0}] {sz}'))
    v.add_image(norm, name='raw', colormap='gray', contrast_limits=(0, 1))
    # EMPTY layers, very small points
    p_end = v.add_points(np.empty((0, 2)), name='endpoint',
                         face_color='yellow', size=3, border_width=0)
    p_br = v.add_points(np.empty((0, 2)), name='分叉点 branch',
                        face_color='magenta', size=3, border_width=0)
    p_ctr = v.add_points(np.empty((0, 2)), name='中心 center',
                         face_color='red', size=4, border_width=0)
    p_ctr.mode = 'add'

    # right-click deletes the nearest point in that layer
    def attach_rclick(layer):
        def cb(lyr, event):
            if event.button == 2 and len(lyr.data):
                try:
                    pos = np.asarray(lyr.world_to_data(event.position))[-2:]
                except Exception:
                    pos = np.asarray(event.position)[-2:]
                data = np.asarray(lyr.data)[:, -2:]
                d = np.linalg.norm(data - pos, axis=1)
                i = int(np.argmin(d))
                if d[i] < 30:
                    lyr.data = np.delete(lyr.data, i, axis=0)
        layer.mouse_drag_callbacks.append(cb)
    for lyr in (p_ctr, p_br, p_end):
        attach_rclick(lyr)

    @v.bind_key('c')
    def clear_all(viewer):
        for lyr in (p_ctr, p_br, p_end):
            lyr.data = np.empty((0, 2))
        print('[cleared all points]')

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
    print(f'TOPOLOGY {stem} {"WHOLE" if full else f"[{y0},{x0}] {sz}"} — blank, no skeleton')
    print('add: pick layer + click | RIGHT-CLICK: delete nearest | c: clear all')
    print('zoom=scroll, pan=space+drag, move=Select tool. Save: s')
    print('=' * 60)
    napari.run()


if __name__ == '__main__':
    main()
