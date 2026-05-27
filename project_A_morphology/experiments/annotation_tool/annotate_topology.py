"""
Topology annotator (BUTTON version — robust, no reliance on shadowed hotkeys).

Right-side panel has real buttons:
  [Mark CENTER 中心] / [Mark BRANCH 分叉] / [Mark ENDPOINT]  -> pick what to add
  [Undo last point]   -> remove the last point you added in the active layer
  [Clear ALL]         -> wipe all points
  [SAVE]              -> write JSON  (also auto-saves every change)
Then just LEFT-CLICK on the image to drop a point of the chosen type.
Hover a point + press 'd' (or right-click) also deletes the nearest point.
Zoom = scroll,  pan = hold SPACE + drag.

SWITCH IMAGE = first arg:  F_WT_2 | F_HET_1 | F_HET_3
WHOLE vs REGION:  annotate_topology.py F_WT_2 [y0 x0 size]
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
    from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout, QLabel

    raw = tifffile.imread(find_raw(stem)).astype(np.float32)
    norm = normalize(raw)
    if not full:
        norm = norm[y0:y0 + sz, x0:x0 + sz]

    v = napari.Viewer(title=f'topology {stem}' + ('' if full else f' [{y0},{x0}] {sz}'))
    v.add_image(norm, name='raw', colormap='gray', contrast_limits=(0, 1))
    p_end = v.add_points(np.empty((0, 2)), name='ENDPOINT (yellow)',
                         face_color='yellow', size=3, border_width=0)
    p_br = v.add_points(np.empty((0, 2)), name='BRANCH 分叉 (magenta)',
                        face_color='magenta', size=3, border_width=0)
    p_ctr = v.add_points(np.empty((0, 2)), name='CENTER 中心 (red)',
                         face_color='red', size=4, border_width=0)
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
        v.status = (f'SAVED -> {fp.name}  center={rec["counts"]["center"]} '
                    f'branch={rec["counts"]["branch"]} endpoint={rec["counts"]["endpoint"]}')

    def activate(layer):
        v.layers.selection.active = layer
        layer.mode = 'add'
        v.status = f'marking: {layer.name}  (click to add)'
        # auto-save whenever this layer's points change
    # auto-save on any point edit
    for lyr in order:
        lyr.events.data.connect(lambda e: do_save())

    def do_undo(*_):
        lyr = v.layers.selection.active
        if lyr in order and len(lyr.data):
            lyr.data = lyr.data[:-1]

    def do_clear(*_):
        for lyr in order:
            lyr.data = np.empty((0, 2))

    def do_del_nearest(*_):
        pos = np.asarray(v.cursor.position)[-2:]
        best = None
        for lyr in order:
            if len(lyr.data):
                d = np.linalg.norm(np.asarray(lyr.data)[:, -2:] - pos, axis=1)
                i = int(np.argmin(d))
                if best is None or d[i] < best[2]:
                    best = (lyr, i, float(d[i]))
        if best and best[2] < 40:
            lyr, i, _ = best
            lyr.data = np.delete(lyr.data, i, axis=0)

    # ---- button panel ----
    panel = QWidget(); lay = QVBoxLayout(panel)
    lay.addWidget(QLabel('Click a button to choose what to mark,\nthen LEFT-CLICK on the image.'))
    for name, lyr in [('● Mark CENTER 中心 (red)', p_ctr),
                      ('● Mark BRANCH 分叉 (magenta)', p_br),
                      ('● Mark ENDPOINT (yellow)', p_end)]:
        b = QPushButton(name); b.clicked.connect(lambda _, L=lyr: activate(L)); lay.addWidget(b)
    lay.addWidget(QLabel('—'))
    for name, fn in [('Undo last point', do_undo),
                     ('Delete nearest to cursor (or key d)', do_del_nearest),
                     ('Clear ALL', do_clear),
                     ('💾 SAVE NOW', do_save)]:
        b = QPushButton(name); b.clicked.connect(fn); lay.addWidget(b)
    lay.addWidget(QLabel('(auto-saves on every edit too)'))
    lay.addStretch()
    v.window.add_dock_widget(panel, area='right', name='annotate')

    v.text_overlay.visible = True
    v.text_overlay.font_size = 13
    v.text_overlay.text = ('use the buttons on the right -> pick CENTER/BRANCH/ENDPOINT, '
                           'then click image.  d=delete nearest. zoom=scroll, pan=space+drag.')
    for k, fn in [('d', do_del_nearest), ('s', do_save), ('c', do_clear)]:
        try:
            v.bind_key(k, (lambda f: (lambda viewer: f()))(fn), overwrite=True)
        except Exception:
            pass

    activate(p_ctr)
    napari.run()


if __name__ == '__main__':
    main()
