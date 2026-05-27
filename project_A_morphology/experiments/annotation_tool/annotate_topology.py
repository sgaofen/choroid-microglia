"""
Topology annotator — mark cell CENTER + BRANCH point + ENDPOINT.

Simple workflow (legend is shown on-screen too):
  press q  -> now marking 中心 CENTER   (red),   click image to drop
  press w  -> now marking 分叉 BRANCH   (magenta)
  press e  -> now marking ENDPOINT      (yellow)
  (number keys 1/2/3 are NOT used — they clash with napari's tool shortcuts)
  d        -> DELETE the point nearest the cursor (use THIS, NOT the Delete key)
  c        -> clear all points
  s        -> SAVE to JSON
  scroll   -> zoom        space+drag -> pan
  (right-click also deletes nearest. Never press Delete/Backspace or the trash
   icon — those remove the whole layer, a napari quirk.)

SWITCH IMAGE = first arg:  F_WT_2 | F_HET_1 | F_HET_3
WHOLE vs REGION:
  annotate_topology.py F_WT_2                  # whole image
  annotate_topology.py F_WT_2 1000 1000 600    # region y0 x0 size
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
    p_end = v.add_points(np.empty((0, 2)), name='E = ENDPOINT (yellow)',
                         face_color='yellow', size=3, border_width=0)
    p_br = v.add_points(np.empty((0, 2)), name='W = BRANCH 分叉 (magenta)',
                        face_color='magenta', size=3, border_width=0)
    p_ctr = v.add_points(np.empty((0, 2)), name='Q = CENTER 中心 (red)',
                         face_color='red', size=4, border_width=0)
    layers = {'q': p_ctr, 'w': p_br, 'e': p_end}

    # on-screen legend (always visible)
    v.text_overlay.visible = True
    v.text_overlay.font_size = 13
    v.text_overlay.color = 'white'
    v.text_overlay.text = (
        'Q=CENTER中心(red)   W=BRANCH分叉(magenta)   E=ENDPOINT(yellow)\n'
        'click=add point    d=delete nearest    c=clear all    s=SAVE\n'
        'scroll=zoom   space+drag=pan    (do NOT press Delete — it kills the layer)')

    def activate(key):
        lyr = layers[key]
        v.layers.selection.active = lyr
        lyr.mode = 'add'
        v.status = f'marking: {lyr.name}'

    for kk in ('q', 'w', 'e'):
        v.bind_key(kk, (lambda k: (lambda viewer: activate(k)))(kk), overwrite=True)

    def attach_rclick(layer):
        def cb(lyr, event):
            if event.button == 2 and len(lyr.data):
                try:
                    pos = np.asarray(lyr.world_to_data(event.position))[-2:]
                except Exception:
                    pos = np.asarray(event.position)[-2:]
                d = np.linalg.norm(np.asarray(lyr.data)[:, -2:] - pos, axis=1)
                i = int(np.argmin(d))
                if d[i] < 30:
                    lyr.data = np.delete(lyr.data, i, axis=0)
        layer.mouse_drag_callbacks.append(cb)
    for lyr in layers.values():
        attach_rclick(lyr)

    @v.bind_key('d', overwrite=True)
    def del_nearest(viewer):
        pos = np.asarray(viewer.cursor.position)[-2:]
        best = None
        for lyr in layers.values():
            if len(lyr.data):
                d = np.linalg.norm(np.asarray(lyr.data)[:, -2:] - pos, axis=1)
                i = int(np.argmin(d))
                if best is None or d[i] < best[2]:
                    best = (lyr, i, float(d[i]))
        if best and best[2] < 40:
            lyr, i, _ = best
            lyr.data = np.delete(lyr.data, i, axis=0)

    @v.bind_key('c', overwrite=True)
    def clear_all(viewer):
        for lyr in layers.values():
            lyr.data = np.empty((0, 2))

    @v.bind_key('s', overwrite=True)
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
        v.status = (f'SAVED center={rec["counts"]["center"]} '
                    f'branch={rec["counts"]["branch"]} endpoint={rec["counts"]["endpoint"]}')
        print(f'[saved] {fp}  {rec["counts"]}')

    activate('q')  # start on CENTER, add mode
    napari.run()


if __name__ == '__main__':
    main()
