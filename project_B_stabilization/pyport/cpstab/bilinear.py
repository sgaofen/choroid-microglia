"""Bit-exact, faster stand-in for scipy.ndimage.shift at order 1.

One function, `bilinear_shift2`, used by both interpolation call sites in
the pipeline (apply_project._imtranslate_float and shifts2d.imtranslate).
It lives in its own module so neither of those has to import the other.
"""
import numpy as np


# -------------------------------------------------------------------------
# Bit-exact stand-in for scipy.ndimage.shift(order=1, mode='grid-constant')
#
# The apply stage spends most of its wall clock in that one call (82 per
# volume: 2 channels x 41 planes), and scipy pays a general N-D
# coordinate/spline machinery per output pixel for what is, at order 1 and
# a pure translation, four fixed taps with per-row / per-column weights.
# Rewriting it as four strided numpy terms is ~2.5x faster on a contiguous
# 512x512 double, and the win is bigger on the strided plane views the
# caller has (measured 0.727 -> 0.158 s/volume once the caller also hands
# over contiguous planes; see _apply_shifts_volume).
#
# BIT-EXACTNESS is the whole point, so the two things scipy does that a
# naive rewrite gets wrong are both reproduced here, and both were found by
# probing scipy 1.13.1, not by reading it:
#
#   1. THE SECOND WEIGHT IS 1 - w0, NOT frac(cc).  scipy's
#      get_spline_interpolation_weights sets w[0] = 1 - frac and then closes
#      with "the last weight is one minus the others".  For a fractional
#      part below eps(1.0) (e.g. 1e-17) that yields exactly 0 where frac
#      would yield 1e-17 — a 1-ulp difference in the result, reproducible on
#      demand.
#   2. THE ACCUMULATION IS SEQUENTIAL, t += (v*wy)*wx over the taps in
#      (0,0), (0,1), (1,0), (1,1) order.  Pairing the sum as (A+B)+(C+D), or
#      factoring it as (v00*wy0 + v10*wy1)*wx0 + ..., disagrees with scipy on
#      ~30-50% of random pixels.  Probed over 4000 random interior pixels:
#      the sequential form matched 4000/4000, the other three 1900-2900.
#
# Out-of-grid taps contribute cval=0 and are therefore SKIPPED rather than
# materialized: the accumulator starts at +0.0 and +0.0 + (-0.0) == +0.0, so
# it can never hold -0.0, which is the only value a skipped `+= 0.0` could
# have changed.
#
# Validated bitwise against scipy over 8928 cases: shapes 1x1 .. 512x512 and
# non-square, magnitudes 1e-6 .. 1e9, shifts spanning fractional, exact
# integer, whole-image, 1e6, +-1e-17 and +-nextafter(0,1), on contiguous
# arrays and on the strided (Y, X, Nz) plane views the pipeline actually
# passes.  Zero mismatches.  float32 input is NOT routed here: scipy
# accumulates in double and rounds once into a float32 output, which numpy
# float32 arithmetic does not reproduce, so fast mode keeps the scipy call.
# -------------------------------------------------------------------------
def _taps(n, s):
    """(start index, w0, w1) per output index for a pure translation."""
    cc = np.arange(n, dtype=np.float64) - s
    start = np.floor(cc)
    w0 = 1.0 - (cc - start)
    return start.astype(np.intp), w0, 1.0 - w0


def bilinear_shift2(img, r_shift, c_shift):
    n0, n1 = img.shape
    y0, wy0, wy1 = _taps(n0, r_shift)
    x0, wx0, wx1 = _taps(n1, c_shift)
    wy, wx = (wy0, wy1), (wx0, wx1)
    out = np.zeros((n0, n1), dtype=np.float64)

    # floor(i - s) is normally i + floor(-s), so both tap sets are plain
    # slices; it is not for shifts whose fractional part underflows the
    # index (the rounding of i - s can land exactly on the integer), and
    # those fall through to the gather form below.
    uniform = (n0 and n1
               and np.array_equal(y0, np.arange(n0) + y0[0])
               and np.array_equal(x0, np.arange(n1) + x0[0]))
    if uniform:
        ky, kx = int(y0[0]), int(x0[0])
        for a in (0, 1):
            oy = ky + a
            i0, i1 = max(0, -oy), min(n0, n0 - oy)
            if i1 <= i0:
                continue
            for b in (0, 1):
                ox = kx + b
                j0, j1 = max(0, -ox), min(n1, n1 - ox)
                if j1 <= j0:
                    continue
                term = img[i0 + oy:i1 + oy, j0 + ox:j1 + ox] * wy[a][i0:i1, None]
                term *= wx[b][None, j0:j1]
                out[i0:i1, j0:j1] += term
        return out

    for a in (0, 1):
        ya = y0 + a
        oky = (ya >= 0) & (ya < n0)
        rows = img[np.clip(ya, 0, n0 - 1)]
        for b in (0, 1):
            xb = x0 + b
            okx = (xb >= 0) & (xb < n1)
            v = rows[:, np.clip(xb, 0, n1 - 1)]
            v = np.where(oky[:, None] & okx[None, :], v, 0.0)
            out += (v * wy[a][:, None]) * wx[b][None, :]
    return out


