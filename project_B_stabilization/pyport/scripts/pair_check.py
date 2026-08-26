#!/usr/bin/env python
"""Content pairing check: which finished zproj does a raw run correspond to?

Background: finished filenames aren't reliable (a WT run is labeled FAD-F;
the txt frame counts for FAD-F_2/WT-F_3 are swapped), so pairing must go by
image content. Method: take the z-mean projection of the first few raw
timepoints (unregistered, tissue drift is limited on a few-frame scale) and
run phase cross-correlation against frame 1 of each candidate zproj; peak
response + post-alignment Pearson r decide the winner. A correct pairing
usually scores r >0.5; a wrong pairing (different mouse/field of view) is
near 0.

Usage: pair_check.py <raw_subset.tif> <candidate1.tif> [candidate2.tif ...]
  raw_subset.tif = the first few T (all z, at least ch1) pulled from an OIR
  via bfconvert
"""
import sys
import numpy as np
import tifffile


def load_raw_template(path, proj_lo=9, proj_hi=31):
    """raw subset -> ch1 (red) z[proj_lo:proj_hi] mean then T mean, returns 2D float64."""
    with tifffile.TiffFile(path) as tf:
        s = tf.series[0]
        ax, arr = s.axes, s.asarray()
    # normalize to (T, C, Z, Y, X)
    order = {a: i for i, a in enumerate(ax)}
    full = "TCZYX"
    for a in full:
        if a not in order:
            arr = np.expand_dims(arr, 0)
            ax = a + ax
            order = {a: i for i, a in enumerate(ax)}
    arr = np.transpose(arr, [order[a] for a in full])
    vol = arr[:, 0].astype(np.float64)          # ch1 = red (vasculature)
    return vol[:, proj_lo:proj_hi].mean(axis=(0, 1))


def load_candidate_frame(path):
    """Candidate zproj (TCYX) frame 1 ch1."""
    with tifffile.TiffFile(path) as tf:
        pg = tf.pages[0]                        # TCYX page order = T0C0
        return pg.asarray().astype(np.float64)


def phase_corr_score(a, b):
    """Phase cross-correlation: returns (peak sharpness, aligned Pearson r, (dy,dx))."""
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    F = np.fft.fft2(a) * np.conj(np.fft.fft2(b))
    F /= np.abs(F) + 1e-12
    r = np.real(np.fft.ifft2(F))
    peak = np.unravel_index(np.argmax(r), r.shape)
    sharp = float(r[peak] / (r.std() + 1e-12))
    dy = peak[0] if peak[0] <= a.shape[0] // 2 else peak[0] - a.shape[0]
    dx = peak[1] if peak[1] <= a.shape[1] // 2 else peak[1] - a.shape[1]
    br = np.roll(np.roll(b, dy, 0), dx, 1)
    pear = float(np.corrcoef(a.ravel(), br.ravel())[0, 1])
    return sharp, pear, (dy, dx)


def main():
    raw, cands = sys.argv[1], sys.argv[2:]
    tpl = load_raw_template(raw)
    print("raw template: %s  mean=%.2f" % (tpl.shape, tpl.mean()))
    best = None
    for c in cands:
        fr = load_candidate_frame(c)
        sharp, pear, (dy, dx) = phase_corr_score(tpl, fr)
        print("%-60s peak_sharp=%7.1f  r=%.4f  shift=(%d,%d)" % (c.split("/")[-1], sharp, pear, dy, dx))
        if best is None or pear > best[0]:
            best = (pear, c)
    print("\n==> paired: %s (r=%.4f)" % (best[1], best[0]))


if __name__ == "__main__":
    main()
