"""Render the full WT and HET images AFTER background normalization (Huixin's
step: rolling-ball / white-tophat so empty space is equally black across images),
all displayed on a COMMON intensity scale so brightness is comparable, not per-
image stretched. One big panel per image (full ~3168px) + a 3-up comparison."""
import sys
from pathlib import Path
import numpy as np
import tifffile
from scipy.ndimage import white_tophat
from skimage import filters
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
import pipeline as pl
from pipeline import STEMS, COND
TOPHAT = 31
OUT = pl.ROOT / 'experiments/full_morphology/out_bg'

# background-subtract every image, collect for a COMMON display range
bg = {}
for s in STEMS:
    raw = tifffile.imread(pl.find_raw(s)).astype(np.float32)
    bg[s] = white_tophat(filters.gaussian(raw, 1.0), size=TOPHAT)
    print(f'{s}: bg-subtracted, p99.5={np.percentile(bg[s],99.5):.1f}')

pooled = np.concatenate([v.ravel()[::40] for v in bg.values()])
vhi = float(np.percentile(pooled, 99.5))   # common upper display, empty space already ~0

# full single-image panels
for s in STEMS:
    plt.figure(figsize=(11, 11))
    plt.imshow(bg[s], cmap='gray', vmin=0, vmax=vhi)
    plt.title(f'{s} ({COND[s]}) — background-normalized, common scale', fontsize=13)
    plt.axis('off')
    plt.savefig(OUT / f'full_bg_{s}.png', dpi=150, bbox_inches='tight'); plt.close()
    print('saved full_bg_' + s + '.png')

# 3-up holistic comparison
fig, ax = plt.subplots(1, 3, figsize=(22, 8))
for a, s in zip(ax, STEMS):
    a.imshow(bg[s], cmap='gray', vmin=0, vmax=vhi)
    a.set_title(f'{s} ({COND[s]})', fontsize=14); a.axis('off')
fig.suptitle('Full images, background-normalized (empty space equally black) on a COMMON intensity scale — '
             'WT vs HET holistic view', fontsize=15)
fig.tight_layout(); fig.savefig(OUT / 'full_bg_compare.png', dpi=130, bbox_inches='tight'); plt.close()
print('saved full_bg_compare.png')
