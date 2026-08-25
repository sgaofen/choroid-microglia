"""EXPLORATORY one-off: connectivity tuning. Compares 6 parameter sets on sharp windows,
with haze windows as false-positive controls.
Output: one horizontal comparison strip per window (cache/exp_*.png) + metric table (stdout).

The five WINDOWS are pixel coordinates in the 2026-08 12-animal images; edit them before
reusing on a different set.
Usage: ./.venv/bin/python experiment_connectivity.py
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server as S

os.makedirs(f"{os.path.dirname(os.path.abspath(__file__))}/cache", exist_ok=True)

WINDOWS = [  # (sample, x, y, note)
    ("M-WT1", 3600, 7200, "sharp-frag"),  # sharp, but was previously over-fragmented
    ("M-WT2", 1200, 9000, "sharp"),
    ("M-Het1", 7800, 3000, "sharp"),
    ("F-Het3", 7800, 2400, "haze ctrl"),
    ("M-Het5", 10200, 0, "haze ctrl"),
]
W = 1200

SETS = {
    "A_v10base": {},
    "C_sato": dict(enhance="sato", sato_um=0.6),
    "G_sato_gate2": dict(enhance="sato", sato_um=0.6, gate_z=2.0),
    "H_sato_skel15": dict(enhance="sato", sato_um=0.6, min_skel_um=15.0),
    "I_sato_g2_s12": dict(enhance="sato", sato_um=0.6, gate_z=2.0, min_skel_um=12.0),
    "K_sato_hy_g2": dict(enhance="sato", sato_um=0.6, thr_mode="zhyst", z_thr=3.5, z_lo=2.0, gate_z=2.0),
}

PANEL = 600  # per-panel display size (native 600px crop from the window centre)


def overlay_img(r, P):
    orig, ov, _ = S.render(r, P)
    return ov


rows = []
for sample, x, y, tag in WINDOWS:
    panels = []
    labels = []
    # centre 600² sub-window (to inspect detail)
    cx, cy = x + (W - PANEL) // 2, y + (W - PANEL) // 2
    for name, over in SETS.items():
        P = dict(S.DEFAULTS)
        P.update(over)
        r = S.process(sample, "C0", cx, cy, PANEL, PANEL, P)
        m = r["metrics"]
        rows.append((sample, tag, name, m))
        if not panels:  # first column holds the raw image
            g = S.stretch(r["a"], 1, 99.7, 1.0)
            panels.append(np.stack([g] * 3, -1))
            labels.append(f"{sample} raw ({tag})")
        panels.append(overlay_img(r, P))
        labels.append(f"{name} comp={m['n_comp']} skel={m['skel_len_um']:.0f}µm fg={m['fg_pct']:.1f}%")

    strip = Image.new("RGB", (PANEL * len(panels), PANEL + 22), (10, 10, 14))
    d = ImageDraw.Draw(strip)
    for i, (p, lb) in enumerate(zip(panels, labels)):
        strip.paste(Image.fromarray(p), (i * PANEL, 22))
        d.text((i * PANEL + 6, 4), lb, fill=(255, 220, 120))
    out = f"{os.path.dirname(os.path.abspath(__file__))}/cache/exp_{sample}.png"
    strip.save(out)
    print("saved", out, flush=True)

print(f"\n{'window':<10}{'type':<10}{'params':<14}{'comp':>5}{'skel_um':>9}{'fg%':>7}{'junc':>5}{'tips':>5}{'soma':>5}{'comp/mm':>9}")
for sample, tag, name, m in rows:
    cpm = m["n_comp"] / max(m["skel_len_um"] / 1e3, 1e-9)
    print(f"{sample:<10}{tag:<10}{name:<14}{m['n_comp']:>5}{m['skel_len_um']:>9.0f}{m['fg_pct']:>7.2f}"
          f"{m['n_junctions']:>5}{m['n_tips']:>5}{m['n_soma']:>5}{cpm:>9.1f}")
