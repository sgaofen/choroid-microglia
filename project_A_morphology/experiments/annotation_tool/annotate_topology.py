"""
Topology annotator — Q/W/E hotkeys (proven to work) + AUTO-SAVE (no data loss).
Whole image by default (zoom in; nothing is cropped at the edges).

WORKFLOW
  press  q  -> mark 中心 CENTER   (red)    then LEFT-CLICK on image
  press  w  -> mark 分叉 BRANCH   (magenta)
  press  e  -> mark ENDPOINT      (yellow)
  press  d  -> delete the point nearest the cursor   (hover, then d)
  zoom = scroll      pan = hold SPACE + drag
  AUTO-SAVES on every add/delete -> no need to press save, nothing is ever lost.
  (Do NOT press Delete/Backspace or the trash icon — those remove a whole layer.)

SWITCH IMAGE:  annotate_topology.py F_WT_2 | F_HET_1 | F_HET_3
REGION (optional): annotate_topology.py F_WT_2 1000 1000 400
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
    y0, x0, sz = (0, 0, None) if full else (int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))

    import napari

    raw = tifffile.imread(find_raw(stem)).astype(np.float32)
    norm = normalize(raw)
    if not full:
        norm = norm[y0:y0 + sz, x0:x0 + sz]

    v = napari.Viewer(title=f'topology {stem}' + ('' if full else f' [{y0},{x0}] {sz}'))
    v.add_image(norm, name='raw', colormap='gray', contrast_limits=(0, 1))
    p_end = v.add_points(np.empty((0, 2)), name='ENDPOINT (yellow)',
                         face_color='yellow', size=4, border_width=0)
    p_br = v.add_points(np.empty((0, 2)), name='BRANCH 分叉 (magenta)',
                        face_color='magenta', size=4, border_width=0)
    p_ctr = v.add_points(np.empty((0, 2)), name='CENTER 中心 (red)',
                         face_color='red', size=5, border_width=0)
    order = [p_ctr, p_br, p_end]

    fp = OUT / (f'{stem}_full_topology_annotations.json' if full
               else f'{stem}_{y0}_{x0}_topology_annotations.json')

    def do_save(*_):
        rec = dict(stem=stem, crop=(None if full else [y0, x0, sz]),
                   saved=datetime.now().isoformat(timespec='seconds'),
                   center=[[float(p[0] + y0), float(p[1] + x0)] for p in p_ctr.data],
                   branch=[[float(p[0] + y0), float(p[1] + x0)] for p in p_br.data],
                   endpoint=[[float(p[0] + y0), float(p[1] + x0)] for p in p_end.data],
                   counts=dict(center=len(p_ctr.data), branch=len(p_br.data),
                               endpoint=len(p_end.data)))
        OUT.mkdir(exist_ok=True)
        fp.write_text(json.dumps(rec, indent=1))
        v.status = (f'auto-saved {fp.name}: center={rec["counts"]["center"]} '
                    f'branch={rec["counts"]["branch"]} endpoint={rec["counts"]["endpoint"]}')

    # AUTO-SAVE on any point edit (proven reliable)
    for lyr in order:
        lyr.events.data.connect(lambda e=None: do_save())

    def activate(layer):
        v.layers.selection.active = layer
        layer.mode = 'add'
        v.status = f'marking: {layer.name}'

    # Q/W/E switch (keys keep canvas focus, unlike buttons) — proven to work
    v.bind_key('q', lambda viewer: activate(p_ctr), overwrite=True)
    v.bind_key('w', lambda viewer: activate(p_br), overwrite=True)
    v.bind_key('e', lambda viewer: activate(p_end), overwrite=True)

    @v.bind_key('d', overwrite=True)
    def del_nearest(viewer):
        pos = np.asarray(v.cursor.position)[-2:]
        best = None
        for lyr in order:
            if len(lyr.data):
                dd = np.linalg.norm(np.asarray(lyr.data)[:, -2:] - pos, axis=1)
                i = int(np.argmin(dd))
                if best is None or dd[i] < best[2]:
                    best = (lyr, i, float(dd[i]))
        if best and best[2] < 40:
            lyr, i, _ = best
            lyr.data = np.delete(lyr.data, i, axis=0)

    v.text_overlay.visible = True
    v.text_overlay.font_size = 14
    v.text_overlay.text = ('q=CENTER中心(red)   w=BRANCH分叉(magenta)   e=ENDPOINT(yellow)   '
                           'then click.   d=delete nearest.   AUTO-SAVES (never lost).')

    activate(p_ctr)
    napari.run()


if __name__ == '__main__':
    main()
