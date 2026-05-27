"""
Lightweight napari cell annotator for choroid-plexus microglia.

WHAT YOU DO:
  - 4 point layers (pick one in the layer list, then click to drop a point):
        round        (red)    — round / amoeboid, no real processes
        simple       (orange) — branched but simple (short, few branches)
        complex      (lime)   — branched, large / many secondary branches
  - 1 shapes layer "clump_exclude" (cyan): draw polygons/rectangles around
    dense clumps to EXCLUDE from analysis.
  - ALL THREE layers are PRE-FILLED with the algorithm's guesses so you correct
    (move / delete / re-categorize) instead of starting from scratch.
  - 'round' is pre-filled from the round-cell detector (mass-without-topology);
    it is sparse and likely incomplete, so ADDING the round cells it missed is
    the most valuable thing you do.

REFERENCE LAYERS (don't edit, just look): raw image, skeleton (cyan),
v30f detected centers (faint yellow).

SAVE: press  s  (writes JSON next to this script). Press it again anytime.

RUN:
  ~/.micromamba/envs/micro-sam/bin/python3.13 annotate_cells.py            # F_WT_2 full
  ~/.micromamba/envs/micro-sam/bin/python3.13 annotate_cells.py F_HET_1
  ~/.micromamba/envs/micro-sam/bin/python3.13 annotate_cells.py F_WT_2 1000 1000 600
                                                  # stem  y0 x0 size  (clean sub-region)
TIP (Huixin): start with ONE clean sub-region, not the whole messy image.
"""
import sys, json
from pathlib import Path
from datetime import datetime

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage import filters, morphology

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
    crop = None
    if len(sys.argv) >= 5:
        y0, x0, sz = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        crop = (y0, x0, sz)

    import napari

    raw = tifffile.imread(find_raw(stem)).astype(np.float32)
    norm = normalize(raw)
    skel = np.load(V29 / f'{stem}_skel_pruned.npy').astype(float)
    cells = json.loads((V30F / f'{stem}_trunk_metrics_v30f.json').read_text())

    oy, ox = 0, 0
    if crop:
        y0, x0, sz = crop
        norm = norm[y0:y0 + sz, x0:x0 + sz]
        skel = skel[y0:y0 + sz, x0:x0 + sz]
        oy, ox = y0, x0
        cells = [c for c in cells
                 if y0 <= c['yc'] < y0 + sz and x0 <= c['xc'] < x0 + sz]

    # pre-fill guesses: split detected cells by secondary-branch median
    sec = np.array([c['n_local_branches'] for c in cells]) if cells else np.array([0])
    med = float(np.median(sec))
    simple_pts = [[c['yc'] - oy, c['xc'] - ox] for c in cells
                  if c['n_local_branches'] < med]
    complex_pts = [[c['yc'] - oy, c['xc'] - ox] for c in cells
                   if c['n_local_branches'] >= med]
    ref_pts = [[c['yc'] - oy, c['xc'] - ox] for c in cells]

    # pre-fill round layer from the round-cell detector (mass-without-topology)
    round_pts = []
    rc_path = ROOT / 'experiments/round_cell_detector' / f'{stem}_round_cells.json'
    if rc_path.exists():
        rc = json.loads(rc_path.read_text())
        if crop:
            y0, x0, sz = crop
            rc = [r for r in rc if y0 <= r['yc'] < y0 + sz and x0 <= r['xc'] < x0 + sz]
        round_pts = [[r['yc'] - oy, r['xc'] - ox] for r in rc]

    v = napari.Viewer(title=f'annotate {stem}' + (f' crop{crop}' if crop else ''))
    v.add_image(norm, name='raw', colormap='gray', contrast_limits=(0, 1))
    v.add_image(skel, name='skeleton (ref)', colormap='cyan',
                blending='additive', opacity=0.6)
    if ref_pts:
        v.add_points(np.array(ref_pts), name='v30f detected (ref)',
                     face_color='yellow', opacity=0.35, size=6)
    pr = v.add_points(np.array(round_pts) if round_pts else np.empty((0, 2)),
                      name='round', face_color='red', size=14,
                      border_color='white')
    ps = v.add_points(np.array(simple_pts) if simple_pts else np.empty((0, 2)),
                      name='simple', face_color='orange', size=14,
                      border_color='white')
    pc = v.add_points(np.array(complex_pts) if complex_pts else np.empty((0, 2)),
                      name='complex', face_color='lime', size=14,
                      border_color='white')
    sh = v.add_shapes(name='clump_exclude', edge_color='cyan',
                      face_color='transparent', edge_width=4)

    pr.mode = 'add'  # round layer ready to click

    @v.bind_key('s')
    def save(viewer):
        def pts(layer):
            return [[float(p[0] + oy), float(p[1] + ox)] for p in layer.data]
        rec = dict(
            stem=stem, crop=crop, saved=datetime.now().isoformat(timespec='seconds'),
            round=pts(pr), simple=pts(ps), complex=pts(pc),
            clump_exclude=[np.asarray(s + [oy, ox]).tolist() for s in sh.data],
            counts=dict(round=len(pr.data), simple=len(ps.data),
                        complex=len(pc.data), clumps=len(sh.data)),
        )
        OUT.mkdir(exist_ok=True)
        tag = f'_crop{crop[0]}_{crop[1]}' if crop else ''
        fp = OUT / f'{stem}{tag}_annotations.json'
        fp.write_text(json.dumps(rec, indent=1))
        print(f'[saved] {fp}  round={rec["counts"]["round"]} '
              f'simple={rec["counts"]["simple"]} complex={rec["counts"]["complex"]} '
              f'clumps={rec["counts"]["clumps"]}')

    print('=' * 64)
    print(f'Annotating {stem}{" crop"+str(crop) if crop else " (full image)"}')
    print('Pick a point layer (round/simple/complex) in the left list, then')
    print('click to drop points. "round" is active and pre-filled from the')
    print(f'detector ({len(round_pts)} candidates) — add the ones it missed,')
    print('delete wrong ones. Draw clump_exclude shapes around dense blobs.')
    print('Press  s  to save (re-press anytime). Close window when done.')
    print('=' * 64)
    napari.run()


if __name__ == '__main__':
    main()
