"""Render the 3 raw 16-bit TIFs as viewable grayscale PNGs, normalized the same
way the pipeline sees them (1-99.5 percentile per image), so Stephen can eyeball
whether the 'WT bigger/more complete vs HET small/fragmented' read holds."""
import sys
from pathlib import Path
import numpy as np
import tifffile
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
import pipeline as pl

OUT = pl.ROOT / 'experiments/full_morphology/out_region'
for s in pl.STEMS:
    raw = tifffile.imread(pl.find_raw(s)).astype(np.float32)
    norm = pl.normalize(raw)
    plt.imsave(OUT / f'raw_{s}.png', norm, cmap='gray', vmin=0, vmax=1)
    print(f'saved raw_{s}.png  {raw.shape}  (each independently contrast-normalized)')
