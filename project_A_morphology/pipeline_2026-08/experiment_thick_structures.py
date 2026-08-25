"""EXPLORATORY one-off, kept for the record - not part of the pipeline.

Compares four parameter sets (baseline / brightness union / union+spur1.2 / union+spur2.0)
on four hand-picked windows to check the thick-structure repair. The quality-control (QC)
statistic is the tree-identity residual |tips - junctions - 2*components|, which should
stay near 0; it blows up when skeletonizing thick blobs invents false branch points.

The four WINDOWS are pixel coordinates in the 2026-08 12-animal images (one of them is the
problem spot originally reported by the author, M-WT1 8388,5668). They mean nothing on a
different image set - edit WINDOWS before reusing.

Writes: cache/exp2_<sample>_<cy>.png   |  Comparison table -> stdout
Usage:  ./.venv/bin/python experiment_thick_structures.py
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server as S

os.makedirs(f"{os.path.dirname(os.path.abspath(__file__))}/cache", exist_ok=True)

WINDOWS = [  # (sample, cx, cy, size, note)   cx,cy=centre
    ("M-WT1", 8388, 5668, 512, "user issue area"),
    ("M-WT1", 4200, 7800, 600, "bestwin sharp"),
    ("F-Het3", 8400, 3000, 600, "dense folded"),
    ("M-Het5", 10800, 600, 600, "pure haze ctrl"),
]

SETS = {
    "G_old_default": dict(sato_union=0, spur_k=0.0),
    "U_union": dict(sato_union=1, spur_k=0.0),
    "UP_union+spur1.2": dict(sato_union=1, spur_k=1.2),
    "UP2_union+spur2": dict(sato_union=1, spur_k=2.0),
}

rows = []
for sample, cx, cy, size, tag in WINDOWS:
    panels, labels = [], []
    x, y = cx - size // 2, cy - size // 2
    for name, over in SETS.items():
        P = dict(S.DEFAULTS)
        P.update(over)
        r = S.process(sample, "C0", x, y, size, size, P)
        m = r["metrics"]
        resid = m["n_tips"] - m["n_junctions"] - 2 * m["n_comp"]
        rows.append((sample, tag, name, m, resid))
        if not panels:
            g = S.stretch(r["a"], 1, 99.7, 1.0)
            panels.append(np.stack([g] * 3, -1))
            labels.append(f"{sample} ({tag}) raw")
        orig, ov, _ = S.render(r, P)
        panels.append(ov)
        labels.append(f"{name} J={m['n_junctions']} E={m['n_tips']} comp={m['n_comp']} skel={m['skel_len_um']:.0f}µm")

    strip = Image.new("RGB", (size * len(panels), size + 22), (10, 10, 14))
    d = ImageDraw.Draw(strip)
    for i, (p, lb) in enumerate(zip(panels, labels)):
        strip.paste(Image.fromarray(p), (i * size, 22))
        d.text((i * size + 6, 4), lb, fill=(255, 220, 120))
    out = f"{os.path.dirname(os.path.abspath(__file__))}/cache/exp2_{sample}_{cy}.png"
    strip.save(out)
    print("saved", out, flush=True)

print(f"\n{'window':<22}{'params':<18}{'J':>5}{'E':>5}{'comp':>6}{'skel_um':>9}{'fg%':>7}{'|E-J-2C|':>9}")
for sample, tag, name, m, resid in rows:
    print(f"{sample}-{tag:<16}{name:<18}{m['n_junctions']:>5}{m['n_tips']:>5}{m['n_comp']:>6}"
          f"{m['skel_len_um']:>9.0f}{m['fg_pct']:>7.2f}{abs(resid):>9}")
