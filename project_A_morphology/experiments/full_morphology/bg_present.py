"""Show the background-normalized result: (1) visual check raw->bg-subtracted->
new mask (proves empty space is equally black now), (2) OLD vs NEW key metrics
bars (what flipped vs survived), (3) data CSV."""
import sys, csv
from pathlib import Path
import numpy as np
import tifffile
from scipy.ndimage import white_tophat
from skimage import filters, morphology
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
import pipeline as pl
from pipeline import STEMS, COND
OUT = pl.ROOT / 'experiments/full_morphology/out_bg'
OUT.mkdir(parents=True, exist_ok=True)
TOPHAT = 31

# ---- numbers from bg_normalize_reanalysis.py (committed run) ----
OLD = {'total area (mm2)': (0.0617, 0.0532, 0.0545), 'median cell area (um2)': (23.9, 20.6, 19.6),
       'connectivity %skel in big webs': (44.3, 23.6, 30.5), 'endpoint/branch (fragmentation)': (4.373, 5.457, 4.845)}
NEW = {'total area (mm2)': (0.0445, 0.0371, 0.033), 'median cell area (um2)': (10.6, 11.1, 14.4),
       'connectivity %skel in big webs': (26.8, 6.7, 5.8), 'endpoint/branch (fragmentation)': (5.245, 6.125, 6.469)}
SURV = {'total area (mm2)': 'WT higher — survives', 'median cell area (um2)': 'FLIPPED — was brightness artifact',
        'connectivity %skel in big webs': 'WT higher — survives (stronger)', 'endpoint/branch (fragmentation)': 'HET higher — survives (cleaner)'}

# ---- (1) visual check ----
raws = {s: tifffile.imread(pl.find_raw(s)).astype(np.float32) for s in STEMS}
pooled = np.concatenate([r.ravel()[::50] for r in raws.values()])
vlo, vhi = np.percentile(pooled, [40, 99.7])          # COMMON raw display range -> brightness diffs show
fig, ax = plt.subplots(3, 3, figsize=(13, 13))
for i, s in enumerate(STEMS):
    sm = filters.gaussian(raws[s], 1.0)
    th = white_tophat(sm, size=TOPHAT)
    t = filters.threshold_otsu(th)
    b = morphology.remove_small_objects(morphology.binary_closing(th > t*0.7, morphology.disk(2)), 20)
    ax[i, 0].imshow(sm, cmap='gray', vmin=vlo, vmax=vhi)
    ax[i, 0].set_title(f'{s} ({COND[s]}) raw — COMMON range', fontsize=10); ax[i, 0].axis('off')
    ax[i, 1].imshow(th, cmap='gray', vmin=0, vmax=np.percentile(th, 99.7))
    ax[i, 1].set_title('rolling-ball bg subtracted (empty=black)', fontsize=10); ax[i, 1].axis('off')
    ax[i, 2].imshow(b, cmap='gray'); ax[i, 2].set_title('new mask', fontsize=10); ax[i, 2].axis('off')
fig.suptitle('Background normalization (FIJI-style): raw (note brightness differs) -> bg-subtracted (equally black) -> mask', fontsize=12)
fig.tight_layout(); fig.savefig(OUT / 'bg_check.png', dpi=95, bbox_inches='tight'); plt.close()
print('saved bg_check.png')

# ---- (2) OLD vs NEW bars ----
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colA = ['#2c7bb6', '#d7191c', '#fd8d3c']
for ax_, metric in zip(axes.ravel(), OLD):
    x = np.array([0, 1, 2, 4, 5, 6])
    vals = list(OLD[metric]) + list(NEW[metric])
    ax_.bar(x, vals, color=colA*2)
    ax_.set_xticks(x); ax_.set_xticklabels(['WT', 'H1', 'H3', 'WT', 'H1', 'H3'])
    ax_.axvline(3, color='k', lw=0.8, ls=':')
    ax_.text(1, ax_.get_ylim()[1], 'OLD (confounded)', ha='center', va='bottom', fontsize=9)
    ax_.text(5, ax_.get_ylim()[1], 'NEW (bg-normalized)', ha='center', va='bottom', fontsize=9)
    for xi, v in zip(x, vals):
        ax_.text(xi, v, f'{v:g}', ha='center', va='bottom', fontsize=8)
    ax_.set_title(f'{metric}\n{SURV[metric]}', fontsize=11); ax_.margins(y=0.2)
fig.suptitle('OLD vs background-normalized — size flipped (artifact); fragmentation & connectivity survived', fontsize=13)
fig.tight_layout(); fig.savefig(OUT / 'bg_oldnew_bars.png', dpi=115, bbox_inches='tight'); plt.close()
print('saved bg_oldnew_bars.png')

# ---- (3) CSV ----
with open(OUT / 'bg_oldnew.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['metric', 'normalization', 'WT', 'HET_1', 'HET_3', 'survives_bg_normalization'])
    for m in OLD:
        w.writerow([m, 'OLD_otsu', *OLD[m], SURV[m]])
        w.writerow([m, 'NEW_bgmatched', *NEW[m], SURV[m]])
print('saved bg_oldnew.csv'); print('saved to', OUT)
