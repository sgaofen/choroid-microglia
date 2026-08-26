"""Exact (band-limited) 2-D translation by a Fourier phase ramp.

PORT EXTENSION — no MATLAB counterpart. Used only on the APPLY side of the
'improved' mode (cpstab/improved.py, correction 2); the replicate path never
calls into this module.

WHY
---
A translation by a non-integer amount is a resampling, and the pipeline does
one per plane, per volume, per channel — on the order of 10^5 of them for a
real run. MATLAB's imtranslate (and the port's scipy `order=1` mirror) does
that resampling with BILINEAR interpolation, which is a low-pass filter: its
transfer function is sinc^2(f), so at half-Nyquist it has already thrown away
~19% of the amplitude, and at Nyquist 100%. That loss is applied once per
plane and it is not recoverable downstream.

Shifting by a phase ramp instead is exact for any band-limited signal:

    out(y, x) = in(y - dy, x - dx)
    OUT(ky, kx) = IN(ky, kx) * exp(-2*pi*i * (ky*dy/M + kx*dx/N))

The transfer function is unit modulus, so nothing is attenuated — the
operation is a pure rotation of phase. Measured modulus for a 0.25 px shift on
a 64-row frame, this function versus the bilinear it replaces (1.0 = nothing
lost):

    frequency (cycles/px)   0.016   0.125   0.250   0.375   0.484   0.500
    fshift2 ..............  1.0000  1.0000  1.0000  1.0000  1.0000  0.7071
    bilinear .............  0.9991  0.9435  0.7906  0.5999  0.5018  1.0000

Read the last column carefully — it is the one honest exception, and it is not
a defect of this implementation. At EXACTLY the Nyquist frequency of an
even-length axis, a real-valued result forces the surviving amplitude to
|cos(pi * dy)|: a Nyquist cosine shifted by half a sample IS a Nyquist sine,
which samples to identically zero on the integer grid. No interpolator can
recover it, and 0.7071 = |cos(pi/4)| is exactly the information-theoretic
limit for a 0.25 px shift. (Bilinear's 1.0 there is not a win: its Nyquist
response is |1 - 2*dy|, which happens to peak at dy = 0 and 1 and collapses to
0 at dy = 0.5 — it is not preserving that component either, it is aliasing
it.) One consequence worth naming: fshift2 is exactly reversible for content
below Nyquist, but a round trip multiplies the Nyquist bin by cos^2, so
`fshift2(fshift2(x, d), -d)` restores white noise only in the band the grid
can actually represent.

Verified against an analytically-known band-limited test image
(tests/test_improved.py): reconstruction error 7e-15 versus the closed-form
shifted signal, against 5.9e-2 for bilinear on the same input.

THE PRICE, MEASURED
-------------------
The DFT treats the frame as PERIODIC, and a microscope frame is not: the top
and bottom rows are unrelated, so the periodic extension has a step
discontinuity at the wrap seam. Two consequences, both real:

1. WRAPAROUND. Content shifted off one edge reappears at the opposite edge.
   This is not an approximation, it is wrong data, and `fshift2` zeroes it:
   ceil(|dy|) rows and ceil(|dx|) columns on the side the content came FROM
   (dy > 0 moves content toward larger y, so the TOP rows are the ones filled
   by wrap; dy < 0 fills the BOTTOM).

   That is ONE row/column MORE than the bilinear path it replaces leaves at
   exactly zero. For dy = 2.3 scipy `mode='grid-constant'` zeroes rows 0-1
   outright but leaves row 2 as 0.7*in(0) — a real pixel darkened 30% by the
   fill it was blended with. fshift2 cannot leave that row: its equivalent is
   blended with WRAPPED content from the far edge, which is worse than a
   darkened row, so ceil() clears it. Net effect on the deliverable: the valid
   region of an improved-mode frame is up to 1 px narrower per axis than the
   replicate-mode frame, and correspondingly free of a partially-attenuated
   edge row that replicate keeps.

2. GIBBS RINGING off the wrap seam, which is NOT confined to the cleared band.
   Measured on a real frame from the validation subset (FAD-F_1 volume 0
   plane 20, Gaussian-smoothed so a 5th-order spline is a near-exact
   reference, shifted by dy=2.37 dx=-3.62; frame mean 22.2 counts, std 5.9,
   mean |row_0 - row_last| wrap jump 9.5 counts) as |fshift2 - spline5| at a
   given distance d from the border:

       d =   4 px   mean 0.65 counts   max 3.14
       d =   8 px   mean 0.24          max 1.14
       d =  20 px   mean 0.082         max 0.39
       d =  40 px   mean 0.042         max 0.19
       deep interior mean 0.017        (bilinear, same region: 0.050)

   So the ringing decays roughly as 1/d and is back to interior level by
   ~40 px. Even at its worst (3 counts at d=4) it is below this data's
   per-pixel shot noise (sigma = sqrt(22) = 4.7 counts), and the outer ring of
   these frames is already invalid in BOTH modes — the shifts themselves open
   a black border of |shift| pixels that no analysis uses. In the deep
   interior, where the science is, fshift2 is 3x closer to the spline
   reference than bilinear is.

   NOT MITIGATED, deliberately. The clean fix is to make the extension
   continuous by mirroring the frame to 2M x 2N before the transform, which
   costs 4x the FFT work on the single hottest loop in the pipeline. Given the
   measurement above — ringing under the noise floor, in a border region that
   is already discarded — that trade was not taken. If a future dataset has a
   large frame-to-frame wrap jump (bright structure running off one edge),
   re-measure with the profile above before trusting the border.

3. RINGING OFF EVERY SHARP FEATURE, NOT JUST THE SEAM — and it is two orders
   of magnitude bigger than #2. THIS is what `fshift2_vst` below exists for,
   and the reason point 2's numbers are so reassuring is that they were taken
   on a GAUSSIAN-SMOOTHED frame, i.e. on a surrogate from which the content
   that actually rings had already been removed. On the raw frame:

       FAD-F_1 t=3 z=20 c=1 (mean 22.1 counts, median 6, p99.9 453),
       shifted by dy=0.37 dx=-1.62, interior only (>= 40 px from any border):

           pixels with a NEGATIVE value ......... 25.6%
           most negative value .................. -373 counts
           median of the negative population .... -13.6 counts
           |negative mass| / total intensity .... 20.0%
           93% of those negatives sit more than 20 px from the cleared band

   That is not the wrap seam. It is the ordinary Gibbs ringing of an ideal
   (brick-wall) reconstruction around single-pixel-scale content, and this
   data is full of it: photon counts on a near-zero background, so the samples
   carry real energy at and above Nyquist. The separable sinc kernel for a
   half-pixel shift has its first sidelobe at -0.212, so a 137-count local
   peak on a 6-count background drives its neighbours to roughly -13 — exactly
   the measured median. The phase ramp is doing its job correctly; the input
   simply is not band-limited, and a band-limited reconstruction of aliased
   photon counts is not a physical image.

   WHY IT MATTERS ON THE DELIVERABLE. The ringing is coherent — shift back
   and it cancels to 1e-13 — so it survives a round trip, and it largely
   cancels in the z-projection too (25.6% of plane pixels negative -> 0.145%
   of projection pixels negative, min -22.3 counts). What does not cancel
   lands in a uint16 TIFF, where every negative becomes a 0 with real data all
   around it. Measured on the 40-volume subset (2 channels, 80 frames), zeros
   whose 4-connected zero component does NOT touch the frame border:

       replicate ................................. 3
       improved, corrections 1/3/4 only .......... 1220     (525 >3 px, 39 >10 px
                                                             from the black band)
       improved with a RAW phase ramp ......... 33480   (26931 >3 px, 18398 >10 px)
       improved with fshift2_vst ................ 5712    (2925 >3 px, 340 >10 px)

   i.e. the raw ramp put ~230 black speckles per frame into the middle of the
   image, 98% of which the VST removes. The ~1220 floor is NOT this module's:
   it is the ragged edge of the black band, produced by corrections 1/3/4
   moving each plane by a different amount, and it is there with the Fourier
   shift switched off entirely.

WHY THE FIX IS A DOMAIN CHANGE AND NOT A CLIP
---------------------------------------------
Nonnegativity cannot be bolted on afterwards, and the reason is worth stating
because three obvious repairs were tried and measured first. A translation
must not change the photon count; every clipping guard does, because it
deletes the negative half of a ±-balanced oscillation:

    guard                                proj. intensity   deep interior zeros
    clip the shifted plane at 0 ........ +14.0%            348
    clamp to the 2x2 source bracket .... +10.7%            343
    clip + local mean restoration ......  +0.1%             94   (sharpness
                                                                   0.0176 -> 0.0077)
    LINEAR spectral taper .............. exact, but a Gaussian taper wide
                                         enough to remove the negatives
                                         (sigma 0.7 px) is a WORSE low-pass
                                         than the bilinear it replaces
                                         (gain at 0.45 cyc/px: 0.14 vs 0.45)
    shift sqrt(x), square back .........  -0.1%            340   (sharpness
                                                                   0.0176 -> 0.0126)

The taper row is the general statement: a convolution kernel that cannot
produce a negative from nonnegative input is a nonnegative kernel, and a
nonnegative kernel has |H(f)| <= H(0) everywhere — it is a low-pass, which is
the thing correction 2 exists to stop doing. Positivity and a unit passband
are mutually exclusive for any LINEAR shift, so the fix has to be nonlinear,
and the cheapest honest nonlinearity is to change the variable.

`fshift2_vst` shifts sqrt(x) and squares the result. For Poisson data that is
the variance-stabilizing (Anscombe) transform, so it is the domain in which
the noise is homoscedastic; more to the point here, it compresses the local
contrast that drives the ringing (a 137:6 peak-to-background ratio becomes
11.7:2.4) and the inverse is a square, so the output cannot be negative no
matter what the ramp does. Measured cost on the subset, against the raw ramp:

    resid px median  0.0224 -> 0.0316   (replicate 0.0636)
    resid px p95     0.0500 -> 0.0672   (replicate 0.1203)
    sharpness        0.0176 -> 0.0126   (replicate 0.0063)
    total intensity  1.0000 -> 0.9992

so roughly 70% of correction 2's headline gain over replicate survives, and
all of the black speckle goes away. Read the sharpness drop next to
metrics.py's field_noise_ratio, which falls 0.985 -> 0.955 on channel 1 and
1.137 -> 0.935 on channel 2: part of what the raw ramp was scoring as
"sharpness" was the ringing itself.

KNOWN ARTIFACT OF THE VST, stated so nobody rediscovers it as a bug: where
the ramp drives the sqrt-domain value below zero (17.6% of interior pixels on
the frame above) the square FOLDS it back up instead of leaving a hole. The
folded values are small — median 1.6 counts, p99 34.6, max 116 on a 6-count
background — and they are a positive artifact where the raw ramp had a -373
count hole, which is the trade being made. `max(y, 0) ** 2` would turn each
fold into a zero instead, i.e. back into the speckle this module is removing.

CONVENTION
----------
`fshift2(img, dy, dx)` matches apply_project._imtranslate_float(img, dx, dy)
and MATLAB imtranslate(img, [dx, dy]) — note MATLAB's [x, y] argument order
versus this function's (dy, dx) = (row, col), which follows the package's
[Y, X] array convention and scipy.ndimage.shift. `fshift2_vst` takes the same
arguments and returns the same shape and class.
"""

import numpy as np

from .precision import as_float

__all__ = ["fshift2", "fshift2_vst", "wrap_margin"]


def wrap_margin(dy, dx):
    """Width of the wraparound-contaminated band, as (rows, cols).

    ceil(|d|) on each axis: a shift of 2.3 px pulls content from 3 rows away
    at the extreme, so 3 rows on the incoming side are built (partly or
    wholly) from wrapped data. Integer shifts give exactly |d|.

    Deliberately NOT clamped to any frame size — it answers a question about
    the shift alone and has no array to clamp against, so `wrap_margin(20, 0)`
    is 20 even for a 16-row frame. Callers that index with it must clamp;
    _clear_wrap does.
    """
    return int(np.ceil(abs(float(dy)))), int(np.ceil(abs(float(dx))))


def _clear_wrap(out, dy, dx):
    """Zero the bands that the circular wrap filled with far-edge content.

    out(y, x) = in(y - dy, x - dx), so for dy > 0 the content moves toward
    larger y and rows [0, ceil(dy)) are the ones with nothing legitimate to
    come from — they receive in(negative) which the DFT wraps to the BOTTOM of
    the input. For dy < 0 the same argument puts the damaged band at the
    bottom. Columns follow with dx. Both bands are cleared when both shifts
    are nonzero (their intersection is a corner block, cleared twice).

    The margin is CLAMPED to the axis length. Once |dy| >= M every output row
    is wrapped content and the whole frame must go; without the clamp the
    dy < 0 branch would index `out[M - mr:]` with a NEGATIVE start, which
    Python reads from the far end and which therefore clears only mr - M rows
    — leaving pure wraparound in place and calling it data. See DESIGN NOTES
    #6; the dy > 0 branch never had the bug because `out[:mr]` clamps on its
    own.
    """
    mr, mc = wrap_margin(dy, dx)
    nrow, ncol = out.shape
    mr = min(mr, nrow)
    mc = min(mc, ncol)
    if mr:
        if dy > 0:
            out[:mr, :] = 0.0
        else:
            out[nrow - mr:, :] = 0.0
    if mc:
        if dx > 0:
            out[:, :mc] = 0.0
        else:
            out[:, ncol - mc:] = 0.0
    return out


def fshift2(img, dy, dx, clear_wrap=True):
    """Translate a 2-D real image by (dy, dx) with an exact phase ramp.

    out(y, x) = in(y - dy, x - dx), i.e. positive dy moves content DOWN
    (toward larger row index) and positive dx moves it RIGHT — the same
    convention as apply_project._imtranslate_float(img, c_shift=dx,
    r_shift=dy) and scipy.ndimage.shift(img, (dy, dx)).

    Parameters
    ----------
    img : 2-D real array [Y, X]. Promoted to the compute class
        (cpstab/precision.py: float64 by default, float32 in fast mode), which
        is also the class of the returned array — a float32 run keeps single
        precision through the transform pair.
    dy, dx : float — row / column translation in pixels.
    clear_wrap : bool, default True — zero the wraparound-contaminated bands
        (see _clear_wrap). False returns the raw circular shift; that is not a
        valid image, and exists so tests can separate the ramp's correctness
        from the clearing policy.

    Returns
    -------
    ndarray, same shape and compute class as the promoted input.

    Notes
    -----
    Integer shifts take an exact np.roll fast path: no transform, no rounding,
    no ringing. The phase ramp agrees with it to ~1e-13 relative, so this is a
    speed and exactness convenience, not a change of definition.

    The real transform pair (rfft2/irfft2) is used rather than the complex
    one. irfft2 performs the Hermitian projection that keeps the Nyquist
    row/column correct for a real result — verified against the analytic
    band-limited truth including deliberate Nyquist-frequency terms (error
    1.5e-14), so no special-casing of the even-size Nyquist bin is needed.
    """
    a = as_float(img)
    if a.ndim != 2:
        raise ValueError("fshift2: expected a 2-D image [Y, X], got shape %r"
                         % (a.shape,))
    dy = float(dy)
    dx = float(dx)
    if not (np.isfinite(dy) and np.isfinite(dx)):
        raise ValueError("fshift2: non-finite shift (%r, %r)" % (dy, dx))

    if dy == 0.0 and dx == 0.0:
        return a.copy()

    if dy == int(dy) and dx == int(dx):
        out = np.roll(a, (int(dy), int(dx)), axis=(0, 1))
    else:
        m, n = a.shape
        ramp_y = np.exp(-2j * np.pi * np.fft.fftfreq(m) * dy)[:, None]
        ramp_x = np.exp(-2j * np.pi * np.fft.rfftfreq(n) * dx)[None, :]
        spec = np.fft.rfft2(a) * (ramp_y * ramp_x)
        out = np.fft.irfft2(spec, s=a.shape).astype(a.dtype, copy=False)

    if clear_wrap:
        _clear_wrap(out, dy, dx)
    return out


def fshift2_vst(img, dy, dx, clear_wrap=True):
    """Translate a NONNEGATIVE 2-D image by (dy, dx) in the sqrt domain.

    This is what the 'improved' apply path actually calls (cpstab/improved.py
    correction 2, apply_project._apply_shifts_volume and zproj_reg).  It is
    `fshift2` applied to sqrt(img), squared back:

        out = fshift2(sqrt(img), dy, dx) ** 2

    and the whole of the reasoning — why the raw ramp cannot be shipped on
    photon-count data, why no clip or spectral taper is an acceptable
    substitute, and what the square-back fold costs — is in the module
    docstring under "RINGING OFF EVERY SHARP FEATURE" and "WHY THE FIX IS A
    DOMAIN CHANGE AND NOT A CLIP".  The short version: sqrt is the
    variance-stabilizing transform for Poisson data, it compresses the local
    contrast that drives Gibbs ringing, and squaring makes a negative output
    impossible, so the uint16 deliverable can no longer contain a black
    speckle where a bright neighbour rang.

    Parameters
    ----------
    img : 2-D real array [Y, X], NONNEGATIVE.  A negative sample raises —
        sqrt() would otherwise return NaN, which this pipeline's uint16 cast
        silently turns into a 0 (pipeline.matlab_uint16), i.e. exactly the
        wrong-data-without-an-exception failure the package refuses to have.
        Every call site satisfies this by construction: the apply stage's
        planes are uint16-valued (apply_project._quantize_u16) and the
        projection zproj_reg refines is a mean of nonnegative planes.
    dy, dx, clear_wrap : as `fshift2`.

    Returns
    -------
    ndarray, same shape and compute class as the promoted input, >= 0
    elementwise.

    Notes
    -----
    INTEGER shifts delegate to `fshift2` unchanged.  An integer shift is an
    exact np.roll with no ringing to suppress, so putting it through the VST
    round trip would buy nothing and cost the bit-exactness that
    tests/test_improved.py test_2(b) pins (sqrt then square is not the
    identity in floating point).  dy = dx = 0 goes the same way.

    Total intensity is preserved to ~0.1% (measured: 0.9984 per plane, 0.9992
    on the 40-volume subset projection), which no clipping guard manages —
    those inflate by 11-14%.  It is NOT exact, and cannot be: the transform is
    nonlinear, so DC is only approximately conserved.
    """
    a = as_float(img)
    if a.ndim != 2:
        raise ValueError("fshift2_vst: expected a 2-D image [Y, X], got "
                         "shape %r" % (a.shape,))
    dy = float(dy)
    dx = float(dx)
    if not (np.isfinite(dy) and np.isfinite(dx)):
        raise ValueError("fshift2_vst: non-finite shift (%r, %r)" % (dy, dx))

    if dy == int(dy) and dx == int(dx):
        return fshift2(a, dy, dx, clear_wrap=clear_wrap)

    if a.size and float(a.min()) < 0.0:
        raise ValueError(
            "fshift2_vst: input has a negative sample (min = %g); this is a "
            "square-root-domain shift and only accepts nonnegative photon "
            "counts. Use fshift2 for signed data." % (float(a.min()),))

    root = fshift2(np.sqrt(a), dy, dx, clear_wrap=False)
    out = root * root
    if clear_wrap:
        _clear_wrap(out, dy, dx)
    return out


# DESIGN NOTES
# ------------
# 1. Why ceil(|d|) and not floor/round for the cleared band: a band-limited
#    sample at output row y draws on input row y - d, which for fractional d
#    straddles floor(y-d) and ceil(y-d). Row ceil(|d|) - 1 is the last one
#    whose support reaches outside the frame, so ceil() is the exact count of
#    damaged rows. floor() would leave one contaminated row for every
#    non-integer d (that is precisely what the bilinear path does — see the
#    module docstring — and there the contaminant is merely the zero fill,
#    while here it is wrapped far-edge content); round() would leave one for
#    d = 2.4 and clear a clean one for d = 2.6.
# 2. The bands are cleared on the side the content came FROM, not the side it
#    went to. Getting this backwards is silent and survives casual inspection
#    (both choices produce a plausible black border), which is why
#    tests/test_improved.py asserts the side explicitly for all four sign
#    combinations rather than only checking that "some" band is zero.
# 3. No quantization. The bilinear path this replaces
#    (apply_project._imtranslate_u16) rounds each translated plane back to
#    uint16 because MATLAB's arrays were uint16 there; that costs up to 0.5
#    counts per plane on data whose mean is ~22 counts. Returning floats is
#    part of correction 2, not an oversight — see apply_project's
#    _apply_shifts_volume, which is where the two paths diverge.
# 4. Follows the compute dtype rather than pinning float64. This is a PIXEL
#    operation, not a correlation that decides a shift, so precision.py's rule
#    puts it on the compute side; a float32 run gets a complex64 transform
#    pair. That is a ~1e-6 relative round-trip error, i.e. ~0.06 counts on a
#    65535-count range — below the quantization the replicate path applies
#    anyway.
# 5. Not vectorized over a stack. The call sites (_apply_shifts_volume,
#    zproj_reg) shift every plane by a DIFFERENT amount, so a batched
#    transform would need a per-plane ramp anyway and would only save the
#    Python loop overhead, which is <1% of the transform cost at 512x512.
# 6. REVIEW FIX — the margin is clamped to the axis length inside
#    _clear_wrap. The original wrote `out[out.shape[0] - mr:, :] = 0` for
#    dy < 0, which for |dy| > M gives a negative slice start; Python then
#    counts from the far end and the statement clears mr - M rows instead of
#    all M. Measured before the fix on a 16-row frame: dy = -20 left 12 of 16
#    rows untouched and dx = -20 left 12 of 16 columns, every one of them pure
#    wraparound returned as if it were data — and dy = -33 happened to look
#    correct again (mr - M >= M wraps the slice back over the whole axis),
#    which is exactly the kind of non-monotone symptom that hides in a spot
#    check. The dy > 0 branch was always safe: `out[:mr]` clamps by itself,
#    so only the negative half was wrong and only past |d| >= M.
#
#    UNREACHABLE on this pipeline — shifts are median-centred and run ~15 px
#    on a 512 px frame — so no shipped number changes and the iron-law
#    regressions cannot see it (replicate never calls this module at all).
#    Fixed anyway because the failure mode is silent wrong data rather than an
#    exception, which is the one thing DESIGN NOTE #2 above says this file is
#    trying not to do. wrap_margin() itself stays unclamped: it is a question
#    about the shift, not about an array.
# 7. `fshift2` KEEPS the raw ramp and stays the tested primitive, even though
#    nothing in the shipped pipeline calls it directly any more. It is what
#    the module's exactness claim is ABOUT — tests/test_improved.py test_2
#    proves 7e-15 against a closed-form band-limited truth, and that proof
#    cannot be written against fshift2_vst, whose sqrt() of a band-limited
#    signal is not band-limited. Two functions, two honest claims: fshift2 is
#    the exact band-limited translation, fshift2_vst is the one you can point
#    at a photon-count image. Making the VST a keyword of fshift2 was the
#    other option and was rejected for exactly this reason — a default that
#    changes what the function's own test proves is a trap.
# 8. The VST is NOT a fifth entry in improved.FEATURES, for the same reason
#    the chain-refine gate is not (cpstab/improved.py DESIGN NOTE 6): there is
#    no "correction 2 without the VST" worth shipping, since that is the
#    33480-speckle bug measured in the module docstring. Ablation is still
#    one line away — call fshift2 instead — which is how the comparison table
#    up there was produced.
