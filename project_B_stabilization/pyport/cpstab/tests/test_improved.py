# -*- coding: utf-8 -*-
"""'improved' mode (cfg.mode) — the switchboard, the four corrections, the
iron laws, and the measurement that says whether any of it helped.

Runs standalone (``python test_improved.py``) or under pytest.

WHAT IS BEING PROVEN
--------------------
  test_1  the switchboard works and DEFAULTS TO REPLICATE: every use_*() is
          False out of the box, 'improved' turns all four on, both scopes
          restore (including on an exception), and the per-feature overrides
          beat the mode in both directions — which is what the ablation in
          test_6 relies on.
  test_2  fshift2 (correction 2) is an EXACT translation: on a band-limited
          image it reproduces the analytic shifted signal to ~1e-14 while
          bilinear is off by ~1e-1; integer shifts are bit-identical to
          np.roll + clear AND to the _imtranslate_float they replace (the
          convention check); it is reversible in the interior; and the
          wraparound band is cleared on the correct side, at width ceil(|d|),
          for all four sign combinations — and, past |d| >= the frame size,
          that the WHOLE frame clears (review regression: the negative branch
          used to under-clear there, see fourier_shift DESIGN NOTES #6).
  test_2b what correction 2 actually SHIPS: fshift2_vst, the same ramp carried
          out in the sqrt domain. On a Poisson field with nearly unresolved
          puncta — the input class that makes an ideal reconstruction ring —
          the raw ramp drives 17% of the interior negative and fshift2_vst
          cannot go negative at all, while holding the interior intensity to
          within 3% (every clipping guard moves it 11-14%, which is why none
          of them was taken) and keeping the mid-band amplitude that bilinear
          throws away. Integer shifts still delegate to fshift2 bit for bit,
          and a negative input RAISES rather than becoming a NaN that the
          uint16 cast would silently turn into a 0.
  test_3  corrections 1 and 4 as unit facts:
            * the per-timepoint median ALGEBRAICALLY ANNIHILATES any
              plane-constant shift term — proved on a constructed matrix and
              then on the real .dftshifts payload, where it is what discards
              RS2/CS2 and RS_chunk/CS_chunk;
            * the sub-plane Z shift interpolates between neighbours, reduces
              to a zero-filled plane shift at integer z, is the identity at
              z = 0, and drops the original's off-by-one clear.
  test_4  correction 3 does what it claims: after the refinement pass, the
          planes of a chain-registered volume sit closer to the volume mean
          than the raw chain leaves them.
  test_4b correction 3's TRUST GATE, negative half: a plane with no content in
          common with the volume mean (pure noise) keeps its chain value
          EXACTLY. The same volume with the gate disabled is the shipped bug —
          corrections of 37-43 grid px, i.e. the ±N/2 uniform draw off a flat
          correlation surface — so this test also demonstrates what the gate
          is for, and that it leaves the honest planes bit-identical.
  test_4c correction 3's TRUST GATE, positive half: a real ~1 px residual is
          still measured and folded in. The gate must not be a way of turning
          correction 3 off.
  test_5  IRON LAW (a): tests/test_synthetic.py still passes 7/7.
  test_6  IRON LAW (b) + the improved-mode measurement, from a pair of full
          40-volume pipeline runs:
            * replicate output bit-identical to the archived reference
              reportA/port_run/FAD-F_1_T0-39_mean_zproj.tif;
            * improved runs end to end and differs (as it must);
            * cpstab/metrics.py evaluated on both.

The big-data tests SKIP (not fail) when the workspace files are absent.

MEASURED (macOS 15 / Apple silicon, numpy 2.0.2, scipy 1.13.1, python 3.9,
single process, FAD-F_1_T0-39.tif, refchannel=1 scale=4 chunksize=20
proj_range=quarter, metrics at stride 1 / 80% central crop) — recorded here so
a future change that moves them shows up in the diff:

    run                 ch   resid px med   resid px p95   sharpness   field
    replicate            1        0.0636         0.1203     0.00632    0.743
    replicate            2        0.1962         3.3905     0.00565    0.836
    improved (all)       1        0.0316         0.0672     0.01259    0.955
    improved (all)       2        0.0685         0.1506     0.01138    0.935

    per-correction, each added to replicate ALONE (median / p95, px):
                          channel 1            channel 2
      +1 global_median    0.0620 / 0.1082      0.1718 / 2.6441
      +2 fourier_shift    0.0316 / 0.0547      0.0851 / 0.1433
      +3 chain_refine     0.0505 / 0.1267      0.1601 / 3.1191
      +4 subplane_z       0.0620 / 0.1273      0.2031 / 0.4321

    THE TWO CORRECTION-2 ROWS MOVED when the phase ramp was put into the SQRT
    domain (fourier_shift.fshift2_vst; the argument is in that module and in
    improved.py correction 2). With the RAW ramp they read improved (all)
    0.0224 / 0.0500, 0.01758, 0.985 and 0.0671 / 0.1204, 0.01406, 1.137, and
    +2 alone 0.0283 / 0.0447, 0.01779 and 0.0583 / 0.1348, 0.01426 -- i.e. the
    fix costs about 30% of correction 2's headline gain over replicate. What
    it buys is the deliverable not being speckled with black pixels. Isolated
    internal zeros (a zero whose 4-connected zero component does not reach the
    frame border), 40-volume subset, 80 frames, with the count more than 10 px
    from the black band in brackets:

      replicate ................................  3   (0)
      +2 alone, RAW ramp ................... 18637   (16037)
      +2 alone, fshift2_vst ...................  49   (4)
      improved (all), RAW ramp ............. 33480   (18398)
      improved (all), fshift2_vst ...........  5712   (340)
      improved minus correction 2 ............ 1220   (39)

    Read the last two rows together: correction 2's OWN contribution is fixed
    (18637 -> 49, replicate's own magnitude), and what is left in full
    improved mode is the ragged edge of the black band, which corrections
    1/3/4 produce by moving every plane by a different amount and which is
    there with the Fourier shift switched off. It concentrates in the same six
    high-motion timepoints (t = 11, 15, 18, 19, 20, 33) in every variant, and
    it sits within a few px of a border region no analysis uses. Widening the
    per-plane cleared band by 1 or 2 px was tried against it and made it worse
    (5712 -> 6022 -> 6580), so it is not the wrap band's own edge.

    (Rows +1/+3/+4 do not touch correction 2 and are unchanged by the sqrt
    domain. The 'improved (all)' sharpness of 0.01259 is still 2.0x
    replicate's and its field_noise_ratio FELL 0.985 -> 0.955 on ch1 and
    1.137 -> 0.935 on ch2, which is metrics.py's control saying that part of
    what the raw ramp scored as sharpness was the ringing itself.)

    The two chain_refine-bearing rows ('improved (all)' and '+3') were
    re-measured after the orchestrator.py:242 dtype-chain fix (the refinement
    now translates the raw planes in their NATIVE class, restoring the uint16
    re-quantization DFT_rect performs, so a rejected plane is once again bit
    identical to the chain value). Previously: improved 0.0224 / 0.0428,
    0.0671 / 0.1135 and +3 0.0500 / 0.1170, 0.1729 / 2.5311. The p95 moves a
    little in both directions and the medians and sharpness do not move at
    all -- expected, since dropping a half-count quantization step is a mild
    dither difference, not a registration difference. Rows +1/+2/+4 do not
    touch chain_refine and are unchanged. (The '+2' number quoted in that
    paragraph is the pre-VST one; see the correction-2 note above.)

    wall clock: replicate 56.4 s, improved 36.5 s (the Fourier shift is
    cheaper than scipy's bilinear + uint16 requantize loop it replaces; it
    accounts for the whole difference — see the +2 row at 32.9 s). The sqrt
    domain adds one sqrt and one square per plane against a pair of 512x512
    transforms, i.e. nothing measurable: 40-volume fast_run apply stage 0.1
    min either way.

    The 'improved' and '+3' rows above are POST-GATE (test_4b/4c). Before the
    trust gate they read 0.0316 / 0.0515 and 0.0338 / 0.0855 for improved, and
    0.0718 / 0.1066, 0.1118 / 0.1838 for +3 alone. Both moved, and not all in
    the same direction — see the fourth bullet below, and read the numbers
    with the size of this subset in mind: 40 volumes over 2 chunks, where the
    runaway is a small perturbation rather than the structural failure it is
    on the full 1500-volume run (35.3% of cells past 50 px, up to 225 px at
    QUIET timepoints — cpstab/improved.py correction 3).

READ THOSE NUMBERS CAREFULLY — four things the table cannot show:

  * Correction 2 does NOT reduce motion. It changes only how a shift is
    APPLIED, so the frames move by exactly the same amounts as in replicate.
    Its large apparent drop in "residual px" is the metric getting a sharper
    image to work with: phase correlation localizes a crisp frame better than
    a bilinear-blurred one. The honest reading of correction 2 is the
    sharpness column, not the residual column.
  * The corrections that genuinely reduce MOTION are 1, 3 and 4. Replicate's
    3.39 px p95 outlier on channel 2 is the interesting event here; correction
    4 (whole-plane Z rounding, -> 0.43) fixes it on its own, and gated
    correction 3 dents it (-> 2.53) where the ungated version erased it
    (-> 0.18). Full improved mode still lands at 0.11, so the outlier is not
    left standing — but see the next bullet before reading that ungated 0.18
    as the better algorithm.
  * Correction 3 with the gate scores BETTER on channel 1 (median 0.0718 ->
    0.0500) and WORSE on channel 2's p95 (0.1838 -> 2.5311) than without it.
    The channel-2 regression is real and worth stating plainly: on THIS
    40-volume subset a couple of the large ungated "corrections" happened to
    push that outlier volume toward its neighbours. That is a coin landing
    heads. The same mechanism, on the full run, moves planes by up to 225 px
    at timepoints where nothing is moving, and the metric here cannot see it
    because the metric measures the PROJECTION and the projection averages
    over planes — which is exactly why the gate's evidence is the per-plane
    ledger (cpstab/improved.py correction 3), not this table. A refinement
    that is right on average by luck is not a restoring force.
  * Correction 3 also has a genuine trade-off independent of the gate: it
    re-references each volume to its own mean instead of to the anchor plane,
    which is the right move when the chain has drifted and a small
    perturbation when it has not.

  40 volumes over 2 chunks is also far too short to exercise correction 1,
  whose whole subject is inter-chunk drift — treat every number above as
  indicative of the mechanism, not as a result about the dataset.
"""

import math
import os
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np
import scipy.ndimage as ndi
import tifffile

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(os.path.dirname(_HERE))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from cpstab import RegistrationConfig, run_pipeline                 # noqa: E402
from cpstab import apply_project as ap                              # noqa: E402
from cpstab import improved, metrics                                # noqa: E402
from cpstab import orchestrator as orc                              # noqa: E402
from cpstab.fourier_shift import fshift2, fshift2_vst, wrap_margin  # noqa: E402
from cpstab.io_rw import VolumeSource                               # noqa: E402

# Real-data fixtures live outside the repo (they are 60+ GB). Point
# CPSTAB_WORKSPACE at them to run those tests; without it they SKIP.
WORKSPACE = os.environ.get("CPSTAB_WORKSPACE", "")
RAW_TIF = os.path.join(WORKSPACE, "FAD-F_1_T0-39.tif")
TRUTH_ZPROJ = os.path.join(WORKSPACE, "reportA", "port_run",
                           "FAD-F_1_T0-39_mean_zproj.tif")
CFG_KW = dict(refchannel=1, scale=4, chunksize=20, proj_range="quarter")


def _require(path):
    if not os.path.exists(path):
        raise unittest.SkipTest("missing test data: %s" % path)


# ---------------------------------------------------------------------------
# 1. the switchboard
# ---------------------------------------------------------------------------

def test_1_mode_switchboard_defaults_to_replicate():
    """Nothing is improved until someone asks, and asking is reversible."""
    assert improved.get_mode() == "replicate", (
        "the process must START in replicate mode")
    for f in improved.FEATURES:
        assert improved.enabled(f) is False, f

    with improved.mode_scope("improved"):
        assert improved.get_mode() == "improved"
        assert improved.use_global_median()
        assert improved.use_chain_refine()
        assert improved.use_fourier_shift()
        assert improved.use_subplane_z()
        # an override beats the mode DOWNWARD (ablate one correction out)
        with improved.feature_scope(chain_refine=False):
            assert improved.use_chain_refine() is False
            assert improved.use_global_median() is True
        assert improved.use_chain_refine() is True
    assert improved.get_mode() == "replicate", "mode_scope did not restore"

    # ... and UPWARD (add one correction to replicate — this is how test_6's
    # per-correction attribution is done)
    with improved.feature_scope(subplane_z=True):
        assert improved.get_mode() == "replicate"
        assert improved.use_subplane_z() is True
        assert improved.use_fourier_shift() is False
    assert improved.use_subplane_z() is False

    # restoration must survive an exception, or one failed improved run
    # poisons every later replicate run in the same interpreter
    for scope in (improved.mode_scope("improved"),
                  improved.feature_scope(global_median=True)):
        try:
            with scope:
                raise RuntimeError("boom")
        except RuntimeError:
            pass
    assert improved.get_mode() == "replicate"
    assert improved.enabled("global_median") is False

    # the config validates before a run starts, and normalizes to a name
    # string so it stays picklable into fast_run.py's worker job tuples
    for bad in ("fast", "improve", "better", "improved!", 1, 0.5):
        try:
            RegistrationConfig(input_path="x.tif", mode=bad)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError("mode=%r was accepted" % (bad,))
    for spelling, want in (("improved", "improved"), ("IMPROVED", "improved"),
                           (" Replicate ", "replicate"), (None, "replicate")):
        got = RegistrationConfig(input_path="x.tif", mode=spelling).mode
        assert got == want and isinstance(got, str), (spelling, got)
    assert RegistrationConfig(input_path="x.tif").mode == "replicate", (
        "the config default must be replicate")

    # unknown feature names are a typo, not a silent no-op
    for bad in ("fourier", "median", ""):
        try:
            improved.enabled(bad)
        except KeyError:
            pass
        else:
            raise AssertionError("enabled(%r) did not raise" % (bad,))


# ---------------------------------------------------------------------------
# 2. correction 2 — fshift2 is an exact translation
# ---------------------------------------------------------------------------

def _bandlimited(m, n, dy, dx):
    """A signal with a closed form under translation: a sum of sinusoids well
    below Nyquist, sampled on the integer grid after shifting by (dy, dx)."""
    y, x = np.mgrid[0:m, 0:n]
    out = np.zeros((m, n), dtype=np.float64)
    for ky, kx, a, ph in ((1, 2, 1.0, 0.3), (3, 1, 0.7, 1.1),
                          (5, 4, 0.5, 2.0), (2, 7, 0.4, 0.7)):
        out += a * np.cos(2 * np.pi * (ky * (y - dy) / m + kx * (x - dx) / n)
                          + ph)
    return out + 10.0


def test_2_fshift2_is_exact_reversible_and_clears_the_right_band():
    m = n = 64

    # --- (a) exactness against the analytic truth, vs bilinear -------------
    src = _bandlimited(m, n, 0.0, 0.0)
    worst_f = worst_b = 0.0
    for dy, dx in ((0.37, -1.62), (0.5, 0.0), (-2.25, 3.75), (0.0, -0.5)):
        truth = _bandlimited(m, n, dy, dx)
        got = fshift2(src, dy, dx, clear_wrap=False)
        lin = ndi.shift(src, (dy, dx), order=1, mode="grid-constant",
                        cval=0.0, prefilter=False)
        c = (slice(8, -8), slice(8, -8))     # interior: no fill/wrap either way
        worst_f = max(worst_f, float(np.abs(got - truth)[c].max()))
        worst_b = max(worst_b, float(np.abs(lin - truth)[c].max()))
    print("  [improved] fshift2 vs analytic truth: max|err| = %.3g "
          "(bilinear on the same input: %.3g)" % (worst_f, worst_b))
    assert worst_f < 1e-10, (
        "fshift2 is supposed to be EXACT on a band-limited signal; "
        "max|err| = %g" % worst_f)
    assert worst_b > 1e-3, (
        "bilinear was not measurably worse than the phase ramp (%g vs %g) — "
        "the test signal is probably too smooth to discriminate"
        % (worst_b, worst_f))

    # --- (b) integer shifts: bit-exact, and the CONVENTION check ----------
    rng = np.random.default_rng(3)
    img = rng.random((24, 20)) * 50 + 5
    for dy, dx in ((2, -3), (0, 4), (-1, 0), (3, 3)):
        got = fshift2(img, dy, dx)
        want = np.roll(img, (dy, dx), axis=(0, 1))
        mr, mc = wrap_margin(dy, dx)
        if dy > 0:
            want[:mr, :] = 0.0
        elif dy < 0:
            want[want.shape[0] - mr:, :] = 0.0
        if dx > 0:
            want[:, :mc] = 0.0
        elif dx < 0:
            want[:, want.shape[1] - mc:] = 0.0
        assert np.array_equal(got, want), (dy, dx)
        # the same shift through the bilinear helper it replaces must agree
        # bit-for-bit at integer shifts — this is what pins (dy, dx) to
        # _imtranslate_float's (c_shift, r_shift) and catches a swapped axis
        assert np.array_equal(got, ap._imtranslate_float(img, dx, dy)), (
            "fshift2(img, dy, dx) must equal _imtranslate_float(img, dx, dy); "
            "the row/col convention is swapped")

    # --- (c) reversibility in the interior --------------------------------
    # On content the sampling grid can actually represent, the round trip is
    # exact. It is NOT exact on white noise, and that is not a bug: a round
    # trip multiplies the Nyquist bin by cos^2(pi*d), and a Nyquist cosine
    # shifted half a sample is a sine that samples to zero. Both halves of
    # that statement are asserted, so nobody later "fixes" the wrong one.
    for dy, dx in ((0.37, -1.62), (2.5, 2.5), (-3.1, 0.0)):
        back = fshift2(fshift2(src, dy, dx, clear_wrap=False),
                       -dy, -dx, clear_wrap=False)
        k = int(math.ceil(max(abs(dy), abs(dx)))) + 6
        d = float(np.abs(back - src)[k:-k, k:-k].max())
        assert d < 1e-9, "round trip (%g, %g) left max|err| = %g" % (dy, dx, d)

    # the Nyquist bin follows |cos(pi*d)| exactly -- the theoretical limit
    yy = np.mgrid[0:m, 0:n][0]
    nyq = np.cos(np.pi * yy).astype(np.float64)
    for dy in (0.0, 0.1, 0.25, 0.4, 0.5):
        got = fshift2(nyq, dy, 0.0, clear_wrap=False)
        # the Nyquist bin of an even-length rfft is NOT half-amplitude like
        # the interior bins: cos(pi*y) has |X[m/2]| = m, so normalize by m
        amp = float(np.abs(np.fft.rfft(got[:, 0])[m // 2]) / m)
        assert abs(amp - abs(math.cos(math.pi * dy))) < 1e-9, (
            "Nyquist survival at d=%g was %.6f, expected |cos(pi*d)| = %.6f"
            % (dy, amp, abs(math.cos(math.pi * dy))))
    # ... while every frequency BELOW Nyquist keeps unit modulus, which is the
    # whole claim of correction 2 and the thing bilinear does not do. Bilinear
    # is checked against its closed-form response |(1-a) + a*exp(-2*pi*i*f)|,
    # so this documents HOW MUCH is lost, not merely that something is.
    a_ = 0.25
    lin_amps = []
    for k_ in (1, 8, 16, 24, 31):
        wave = np.cos(2 * np.pi * k_ * yy / m)
        got = fshift2(wave, a_, 0.0, clear_wrap=False)
        lin = ndi.shift(wave, (a_, 0.0), order=1, mode="grid-wrap",
                        prefilter=False)
        amp = lambda z: float(2 * np.abs(np.fft.rfft(z[:, 0])[k_]) / m)  # noqa: E731,B023
        assert abs(amp(got) - 1.0) < 1e-9, (
            "fshift2 must not attenuate k=%d; modulus %.9f" % (k_, amp(got)))
        theory = abs((1 - a_) + a_ * np.exp(-2j * np.pi * k_ / m))
        assert abs(amp(lin) - theory) < 1e-6, (k_, amp(lin), theory)
        assert amp(lin) < 1.0
        lin_amps.append(amp(lin))
    assert lin_amps[-1] < 0.55, (
        "bilinear should keep barely half the amplitude just below Nyquist; "
        "got %.4f" % lin_amps[-1])
    assert all(x > y for x, y in zip(lin_amps, lin_amps[1:])), (
        "bilinear's attenuation must grow with frequency: %r" % (lin_amps,))

    # --- (d) the cleared band is on the side content came FROM -------------
    flat = np.ones((16, 16))
    for dy, dx in ((2.3, 0.0), (-2.3, 0.0), (0.0, 3.7), (0.0, -3.7),
                   (1.5, -2.5), (-1.5, 2.5)):
        out = fshift2(flat, dy, dx)
        mr, mc = wrap_margin(dy, dx)
        assert (mr, mc) == (int(math.ceil(abs(dy))), int(math.ceil(abs(dx))))
        rows_zero = np.flatnonzero(np.all(out == 0.0, axis=1))
        cols_zero = np.flatnonzero(np.all(out == 0.0, axis=0))
        if mr:
            want = (np.arange(mr) if dy > 0
                    else np.arange(out.shape[0] - mr, out.shape[0]))
            assert np.array_equal(rows_zero, want), (
                "dy=%g should clear rows %s, cleared %s"
                % (dy, want.tolist(), rows_zero.tolist()))
        else:
            assert rows_zero.size == 0, dy
        if mc:
            want = (np.arange(mc) if dx > 0
                    else np.arange(out.shape[1] - mc, out.shape[1]))
            assert np.array_equal(cols_zero, want), (
                "dx=%g should clear cols %s, cleared %s"
                % (dx, want.tolist(), cols_zero.tolist()))
        else:
            assert cols_zero.size == 0, dx

    # --- (e) |d| >= frame size: EVERY pixel is wrap, so all of it clears ----
    # REVIEW REGRESSION. _clear_wrap used to index out[M - mr:] for dy < 0,
    # which goes NEGATIVE once mr > M and then slices from the far end: at
    # dy = -20 on a 16-row frame it cleared 4 rows instead of 16 and returned
    # 12 rows of pure wraparound as data. Note dy = -33 passed even when the
    # bug was live (mr - M >= M wraps the slice back over the whole axis), so
    # the magnitudes below deliberately straddle both regimes.
    ramp = np.arange(16 * 16, dtype=np.float64).reshape(16, 16) + 1.0
    for d in (16.0, 17.5, 20.0, 33.0):
        for sgn in (+1.0, -1.0):
            out = fshift2(ramp, sgn * d, 0.0)
            assert np.all(out == 0.0), (
                "dy=%+g on a 16-row frame: every source row is off the frame, "
                "so the result must be all zero; %d rows survived"
                % (sgn * d, int(np.count_nonzero(out.any(axis=1)))))
            out = fshift2(ramp, 0.0, sgn * d)
            assert np.all(out == 0.0), (
                "dx=%+g on a 16-col frame: every source column is off the "
                "frame, so the result must be all zero; %d cols survived"
                % (sgn * d, int(np.count_nonzero(out.any(axis=0)))))
    # wrap_margin itself stays UNCLAMPED — it knows nothing about any array.
    assert wrap_margin(20.0, -33.0) == (20, 33), wrap_margin(20.0, -33.0)


# ---------------------------------------------------------------------------
# 2b. correction 2 as it is actually SHIPPED — the sqrt-domain ramp
# ---------------------------------------------------------------------------

def _poisson_field(m, n, seed=7, npunct=300, amp=400.0, sd=0.7):
    """A stand-in for a plane of this data: a dim Poisson background (mean 6
    counts, so a hard floor at zero and shot noise all the way to Nyquist)
    carrying sparse, NEARLY UNRESOLVED bright puncta.

    The sharpness is the point. A smooth test image does not reproduce the
    bug: the same construction with sd = 2 px blobs leaves the raw ramp only
    0.4% negative, against 17% here and 25.6% on the real plane the module
    docstring measures. Gibbs ringing is driven by the content the sampling
    grid cannot represent, so a test field that is band-limited proves
    nothing about a kernel whose whole problem is content that is not.
    """
    rng = np.random.default_rng(seed)
    img = rng.poisson(6.0, (m, n)).astype(np.float64)
    yy, xx = np.mgrid[0:m, 0:n]
    ys = rng.integers(0, m, npunct)
    xs = rng.integers(0, n, npunct)
    amps = rng.gamma(2.0, amp / 2.0, npunct)
    for y, x, a in zip(ys, xs, amps):
        img += a * np.exp(-(((yy - y) ** 2 + (xx - x) ** 2) / (2 * sd ** 2)))
    return img


def test_2b_vst_shift_is_nonnegative_and_photometric():
    """fshift2_vst: the shipped correction-2 kernel.

    Four claims, in the order they matter:
      1. it CANNOT return a negative sample, on exactly the input class where
         the raw ramp returns tens of percent of them — that is the whole
         reason it exists (the negatives become black speckle in the uint16
         deliverable, see cpstab/fourier_shift.py);
      2. the raw ramp really does produce those negatives here, so this test
         fails loudly if someone "simplifies" fshift2_vst back to fshift2;
      3. total intensity survives, which is what rules out every clipping
         guard (they inflate the projection 11-14%);
      4. integer shifts still delegate to fshift2 BIT FOR BIT, so the
         convention and the exactness of the integer path are untouched.
    """
    m = n = 96
    img = _poisson_field(m, n)
    k = 12                                  # interior: no wrap band either way

    worst_raw_negfrac = 0.0
    for dy, dx in ((0.37, -1.62), (0.5, 0.5), (-2.25, 3.75), (0.0, -0.5)):
        vst = fshift2_vst(img, dy, dx, clear_wrap=False)
        raw = fshift2(img, dy, dx, clear_wrap=False)

        # (1) nonnegative, everywhere, no exceptions
        assert vst.min() >= 0.0, (
            "fshift2_vst returned %g at (%g, %g); a square cannot be negative "
            "— the sqrt-domain round trip has been broken"
            % (vst.min(), dy, dx))

        # (2) ... on an input where the raw ramp is not
        c = (slice(k, -k), slice(k, -k))
        negfrac = float((raw[c] < 0).mean())
        worst_raw_negfrac = max(worst_raw_negfrac, negfrac)

        # (3) photometry: the shift must not create or destroy photons
        ratio = float(vst[c].sum() / raw[c].sum())
        assert abs(ratio - 1.0) < 0.03, (
            "fshift2_vst moved the interior intensity by %+.2f%% at (%g, %g); "
            "a clipping guard would move it by +11-14%% and that is what this "
            "bound is there to catch" % (100 * (ratio - 1.0), dy, dx))

    print("  [improved] raw ramp negatives on a Poisson field: %.1f%% of "
          "interior px; fshift2_vst: 0.0%%" % (100 * worst_raw_negfrac))
    assert worst_raw_negfrac > 0.05, (
        "the raw phase ramp left only %.3f%% of the interior negative on this "
        "test field — the field is too smooth to reproduce the bug that "
        "fshift2_vst exists to fix, so claim (1) above proves nothing"
        % (100 * worst_raw_negfrac))

    # (4) integer shifts (and the no-op) are fshift2, bit for bit
    for dy, dx in ((2, -3), (0, 4), (-1, 0), (0, 0), (3, 3)):
        assert np.array_equal(fshift2_vst(img, dy, dx),
                              fshift2(img, dy, dx)), (dy, dx)
        assert np.array_equal(fshift2_vst(img, dy, dx, clear_wrap=False),
                              fshift2(img, dy, dx, clear_wrap=False)), (dy, dx)

    # the wraparound band is cleared exactly as fshift2 clears it
    for dy, dx in ((2.3, 0.0), (-2.3, 0.0), (0.0, 3.7), (1.5, -2.5)):
        out = fshift2_vst(np.ones((16, 16)), dy, dx)
        mr, mc = wrap_margin(dy, dx)
        rows_zero = np.flatnonzero(np.all(out == 0.0, axis=1))
        cols_zero = np.flatnonzero(np.all(out == 0.0, axis=0))
        if mr:
            want = (np.arange(mr) if dy > 0
                    else np.arange(16 - mr, 16))
            assert np.array_equal(rows_zero, want), (dy, dx, rows_zero)
        if mc:
            want = (np.arange(mc) if dx > 0
                    else np.arange(16 - mc, 16))
            assert np.array_equal(cols_zero, want), (dy, dx, cols_zero)

    # a negative sample is a caller error, and must RAISE rather than turn
    # into a NaN that pipeline.matlab_uint16 would quietly cast to 0
    try:
        fshift2_vst(img - 1.0, 0.5, 0.25)
    except ValueError:
        pass
    else:
        raise AssertionError("fshift2_vst accepted a negative sample; sqrt() "
                             "would make it NaN and the uint16 cast would "
                             "make that a silent 0")

    # ... and it is still much better than the bilinear it replaces: on a
    # mid-band sinusoid riding a positive pedestal (so the sqrt domain is
    # well-conditioned), a quarter-pixel shift keeps far more of the
    # amplitude. This is the half of correction 2 that must SURVIVE the fix.
    yy = np.mgrid[0:m, 0:n][0]
    for kk in (8, 16, 24):
        wave = 100.0 + 60.0 * np.cos(2 * np.pi * kk * yy / m)
        amp = lambda z: float(2 * np.abs(np.fft.rfft(z[:, 0])[kk]) / m)  # noqa: E731,B023
        a_vst = amp(fshift2_vst(wave, 0.25, 0.0, clear_wrap=False)) / 60.0
        a_lin = amp(ndi.shift(wave, (0.25, 0.0), order=1, mode="grid-wrap",
                              prefilter=False)) / 60.0
        assert a_vst > a_lin + 0.02, (
            "fshift2_vst kept %.4f of the k=%d amplitude, bilinear kept "
            "%.4f — correction 2's resolution gain is gone" % (a_vst, kk, a_lin))
        print("  [improved] k=%2d amplitude kept: fshift2_vst %.4f, "
              "bilinear %.4f" % (kk, a_vst, a_lin))


# ---------------------------------------------------------------------------
# 3. corrections 1 and 4 as unit facts
# ---------------------------------------------------------------------------

def test_3_median_annihilation_and_subplane_z():
    # --- correction 1: the annihilation, on a constructed matrix -----------
    rng = np.random.default_rng(7)
    per_plane = rng.normal(size=(9, 5))            # genuinely per-plane part
    plane_const = np.tile(rng.normal(size=(1, 5)), (9, 1))   # RS2 / RS_chunk
    total = per_plane + plane_const

    with improved.feature_scope(global_median=False):
        a = total - ap._median_centering(total)
        b = per_plane - ap._median_centering(per_plane)
    assert np.allclose(a, b, atol=1e-12), (
        "per-column centring was supposed to annihilate the plane-constant "
        "term; max|delta| = %g" % np.abs(a - b).max())

    with improved.feature_scope(global_median=True):
        a = total - ap._median_centering(total)
        b = per_plane - ap._median_centering(per_plane)
        assert np.asarray(ap._median_centering(total)).ndim == 0, (
            "improved centring must be a SCALAR")
    assert not np.allclose(a, b), (
        "global centring must PRESERVE the plane-constant term")

    # ZS is a (1, Nt) row vector: both modes must give the same scalar, or
    # improved mode would silently move the axial origin as well
    zs = rng.normal(size=(1, 5))
    with improved.feature_scope(global_median=False):
        z_rep = ap._median_centering(zs)
    with improved.feature_scope(global_median=True):
        z_imp = ap._median_centering(zs)
    assert float(z_rep) == float(z_imp), (z_rep, z_imp)

    # --- correction 1 on the REAL payload ---------------------------------
    if os.path.exists(TRUTH_ZPROJ):
        sp = os.path.join(os.path.dirname(TRUTH_ZPROJ),
                          "FAD-F_1_T0-39.dftshifts.npz")
        if os.path.exists(sp):
            z = np.load(sp)
            rs, rsc = z["RS"], z["RS_chunk"]
            assert np.allclose(rsc, rsc[0:1, :]), (
                "RS_chunk is supposed to be plane-constant (an imresize "
                "'nearest' stretch of a 1 x Nchunks vector)")
            with improved.feature_scope(global_median=False):
                with_chunk = (rs + rsc) - ap._median_centering(rs + rsc)
                without = rs - ap._median_centering(rs)
            d = float(np.abs(with_chunk - without).max())
            print("  [improved] real .dftshifts: per-timepoint centring "
                  "discards the whole inter-chunk correction (max|delta| "
                  "with vs without RS_chunk = %.2g px)" % d)
            assert d < 1e-12, (
                "the annihilation should be exact; got %g px" % d)

    # --- correction 4: sub-plane Z ----------------------------------------
    vol = rng.random((1, 3, 3, 6)) * 10
    assert np.array_equal(ap._apply_z_shift_subplane(vol, 0.0), vol), (
        "z = 0 must be the identity")

    half = ap._apply_z_shift_subplane(vol, 0.5)
    assert np.allclose(half[0, :, :, 1:],
                       0.5 * (vol[0, :, :, 1:] + vol[0, :, :, :-1])), (
        "z = 0.5 must be the mean of the two bracketing planes")
    assert np.allclose(half[0, :, :, 0], 0.5 * vol[0, :, :, 0]), (
        "the end plane keeps its partial weight (see the docstring: the "
        "missing contributor is a clean zero, not wrapped data)")

    for z in (1.0, -2.0):
        got = ap._apply_z_shift_subplane(vol, z)
        want = np.zeros_like(vol)
        k = int(z)
        lo, hi = max(0, k), min(6, 6 + k)
        want[:, :, :, lo:hi] = vol[:, :, :, lo - k:hi - k]
        assert np.array_equal(got, want), (
            "integer z must reduce to a zero-filled plane shift (z=%g)" % z)
        # ... and specifically must NOT clear the extra plane the original does
        assert np.count_nonzero(np.any(got, axis=(0, 1, 2))) == 6 - abs(k), z

    # z beyond the stack destroys everything: raise, do not return zeros
    try:
        ap._apply_z_shift_subplane(vol, 7.5)
    except IndexError:
        pass
    else:
        raise AssertionError("out-of-range sub-plane z did not raise")

    # _z_shift_for hands the right thing to each branch
    with improved.feature_scope(subplane_z=False):
        v = ap._z_shift_for(1.5)
        assert isinstance(v, int) and v == 2, v      # MATLAB half-away-from-0
    with improved.feature_scope(subplane_z=True):
        v = ap._z_shift_for(1.5)
        assert isinstance(v, float) and v == 1.5, v


# ---------------------------------------------------------------------------
# 4. correction 3 — the chain refinement removes chain drift
# ---------------------------------------------------------------------------

def _drifting_volume(seed, nz=15, m=96):
    """A stack of noisy, independently displaced copies of one image.

    Known truth by construction, so the chain's accumulated error is
    measurable directly rather than inferred.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:m, 0:m]
    base = (300.0
            + 120.0 * np.exp(-((yy - 44) ** 2 + (xx - 52) ** 2) / 90.0)
            + 90.0 * np.exp(-((yy - 60) ** 2 + (xx - 36) ** 2) / 50.0)
            + 40.0 * np.sin(2 * np.pi * yy / 21.0)
            * np.cos(2 * np.pi * xx / 17.0))
    true_dy = rng.normal(scale=1.2, size=nz)
    true_dx = rng.normal(scale=1.2, size=nz)
    true_dy -= true_dy[nz // 2]        # the anchor plane defines the origin
    true_dx -= true_dx[nz // 2]
    vol = np.empty((m, m, nz), dtype=np.float64)
    for i in range(nz):
        f = ndi.shift(base, (true_dy[i], true_dx[i]), order=3, mode="nearest")
        vol[:, :, i] = f + rng.normal(scale=6.0, size=(m, m))
    return np.clip(vol, 0, 65535).astype(np.uint16), true_dy, true_dx


def _rel_err(r, c, true_dy, true_dx):
    """Mean plane-to-plane alignment error, with the per-volume common offset
    removed.

    The refinement re-references the whole volume to the volume MEAN's frame
    rather than to the anchor plane's, which adds a constant to every plane.
    That constant is not an error: it is absorbed downstream by the per-volume
    3-D registration against ref2 (DFT_warp_3D_2.m L71-79) and then by the
    median centring in MakeSBXall. What the chain gets WRONG, and what this
    correction is for, is the RELATIVE spread between planes.
    """
    cen = lambda a: np.asarray(a, float).ravel() - np.mean(a)   # noqa: E731
    return float(np.hypot(cen(r) - cen(-true_dy),
                          cen(c) - cen(-true_dx)).mean())


def test_4_chain_refine_removes_chain_drift():
    """A chained registration has no restoring force; the refinement is it.

    Also the NEGATIVE CONTROL for the crop: run the same refinement on the
    full frame and it stops working, because each chained plane's hard zero
    border correlates against the volume mean's soft one. If someone later
    "simplifies" the crop away, this is the assertion that fails.
    """
    deps = orc._import_deps()
    _sbxio, dftreg, shifts2d = deps
    nz = 15
    start = int(orc.matlab_round(nz / 2.0))

    # On a 96 px frame keepingfactor=0.99 resolves to the FULL frame
    # (ceil(96*0.005)=1 .. ceil(96*0.995)=96), i.e. it exercises the same code
    # path with the crop switched off. Asserted, so a change to the crop
    # arithmetic cannot quietly turn the control into a second cropped run.
    m = 96
    assert (math.ceil(m * 0.01 / 2), math.ceil(m * 0.995)) == (1, m)

    chain, refined, uncropped = [], [], []
    for seed in range(6):
        vol_u16, tdy, tdx = _drifting_volume(seed, nz=nz, m=m)
        r, c, reg = shifts2d.dft_rect(vol_u16, start, 4)
        r2, c2, reg2 = orc._refine_chain_to_volume_mean(
            vol_u16, reg, r, c, shifts2d, dftreg)
        r3, c3, _ = orc._refine_chain_to_volume_mean(
            vol_u16, reg, r, c, shifts2d, dftreg, keepingfactor=0.99,
            blurfactor=1)
        chain.append(_rel_err(r, c, tdy, tdx))
        refined.append(_rel_err(r2, c2, tdy, tdx))
        uncropped.append(_rel_err(r3, c3, tdy, tdx))

    mc, mr, mu = np.mean(chain), np.mean(refined), np.mean(uncropped)
    print("  [improved] chain-refine over 6 known-truth volumes: relative "
          "alignment error %.4f px -> %.4f px  (same refinement without the "
          "crop: %.4f px)" % (mc, mr, mu))
    assert mr < mc, (
        "the refinement made the recovered shifts WORSE (%.4f -> %.4f px)"
        % (mc, mr))
    assert mu > mr, (
        "the uncropped refinement (%.4f px) was not worse than the cropped "
        "one (%.4f px) — either the border-mismatch bias is gone or the crop "
        "is no longer doing what _refine_chain_to_volume_mean claims"
        % (mu, mr))

    # and the volume it hands back is rebuilt at the refined total shift,
    # one interpolation deep from the ORIGINAL planes (not two from the chain),
    # IN THE ORIGINAL PLANE'S OWN CLASS. The expectation used to promote the
    # raw plane to float64 first, which is what the implementation did too --
    # and that promotion silently skipped the uint16 re-quantization
    # shifts2d.dft_rect performs on the very same planes (L172-177 ->
    # imtranslate -> matlab_compat.matlab_cast_like). Exact equality now,
    # because both sides land on the same integers.
    want = shifts2d.imtranslate(vol_u16[:, :, 0], (c2[0], r2[0]))
    assert np.array_equal(reg2[:, :, 0], want), (
        "the refined volume must be resampled once from the raw planes, in "
        "the raw planes' class")

    # The consequence that makes the trust gate's fallback honest: when EVERY
    # plane is rejected the refinement must hand back DFT_rect's own output,
    # bit for bit -- that is the whole basis of "a fully-rejected volume
    # degrades to the MATLAB-faithful result and can never be worse than the
    # status quo" (cpstab/improved.py correction 3). With the float promotion
    # in place this was false: same shifts, but 72-75% of voxels off by up to
    # half a count, which then moved RS1/CS1 downstream.
    r0, c0, reg0 = orc._refine_chain_to_volume_mean(
        vol_u16, reg, r, c, shifts2d, dftreg, cap=1e-9, min_ncc=0.999999)
    assert np.array_equal(np.ravel(r0), np.ravel(r))
    assert np.array_equal(np.ravel(c0), np.ravel(c))
    assert np.array_equal(reg0, reg), (
        "a fully-rejected volume must be DFT_rect's chain output bit for bit")


# ---------------------------------------------------------------------------
# 4b / 4c. correction 3's trust gate
# ---------------------------------------------------------------------------

def _structured_plane(m=96):
    """The deterministic content every gate test registers against."""
    yy, xx = np.mgrid[0:m, 0:m]
    return (300.0
            + 120.0 * np.exp(-((yy - 44) ** 2 + (xx - 52) ** 2) / 90.0)
            + 90.0 * np.exp(-((yy - 60) ** 2 + (xx - 36) ** 2) / 50.0)
            + 40.0 * np.sin(2 * np.pi * yy / 21.0)
            * np.cos(2 * np.pi * xx / 17.0))


def _identity_chain(vol_u16):
    """(vol_chained, r, c) for a chain that reported ZERO for every plane.

    The gate tests hand the refinement a CONSTRUCTED chain instead of running
    DFT_rect, for one reason: a real chain on a noisy plane returns a large
    shift of its own, and imtranslate then puts a hard zero border on that
    plane. Border-vs-border correlation is a documented confounder of this
    very function (see the crop section of _refine_chain_to_volume_mean), and
    it would mean the test was measuring the border rather than the missing
    content. A zero chain leaves every plane untouched, so the only thing the
    refinement can see is whether the plane shares content with the mean —
    which is the mechanism under test.
    """
    nz = vol_u16.shape[2]
    return (np.asarray(vol_u16, dtype=np.float64),
            np.zeros(nz, dtype=np.float64), np.zeros(nz, dtype=np.float64))


def _noise_volume(seed, nz=12, m=96, n_noise=4):
    """A stack whose last `n_noise` planes are pure noise — the synthetic
    stand-in for the deep planes of a real stack, which correlate with the
    volume mean at 0.04-0.25 (FAD-F_1, z >= 24)."""
    rng = np.random.default_rng(seed)
    base = _structured_plane(m)
    vol = np.empty((m, m, nz), dtype=np.float64)
    for i in range(nz - n_noise):
        vol[:, :, i] = base + rng.normal(scale=6.0, size=(m, m))
    for i in range(nz - n_noise, nz):
        # same mean and a comparable spread, so the difference from a real
        # plane is ONLY the shared structure, not the brightness or contrast
        vol[:, :, i] = rng.normal(loc=300.0, scale=45.0, size=(m, m))
    return np.clip(vol, 0, 65535).astype(np.uint16)


def test_4b_gate_rejects_planes_with_no_common_content():
    """A plane that shares nothing with the volume mean keeps the chain value.

    This is the regression for the bug the gate exists to stop: at t=1300 of
    the full FAD-F_1 run — a QUIET timepoint — 17 of 41 planes were "refined"
    by 95-225 full-resolution px, taking that volume's RS to [-10.4, +218.4]
    px against replicate's [-9.2, +19.0] and putting rectangular zero-fill
    seams into the projection.
    """
    deps = orc._import_deps()
    _sbxio, dftreg, shifts2d = deps
    nz, n_noise = 12, 4
    noise = list(range(nz - n_noise, nz))
    good = list(range(nz - n_noise))

    worst_ungated = []
    for seed in range(6):
        vol = _noise_volume(seed, nz=nz, n_noise=n_noise)
        chained, r, c = _identity_chain(vol)

        # (i) UNGATED — reproduce the bug, so this test fails loudly if the
        #     mechanism it guards against ever stops existing (in which case
        #     the gate's justification, not just its code, needs revisiting).
        st_un = {}
        r_un, c_un, _ = orc._refine_chain_to_volume_mean(
            vol, chained, r, c, shifts2d, dftreg,
            cap=float("inf"), min_ncc=0.0, stats=st_un)
        mags = [max(abs(st_un["dr"][i]), abs(st_un["dc"][i])) for i in noise]
        worst_ungated.append(max(mags))

        # (ii) GATED (the shipped defaults, read from cpstab/improved.py)
        st = {}
        r_g, c_g, vol_g = orc._refine_chain_to_volume_mean(
            vol, chained, r, c, shifts2d, dftreg, stats=st)

        for i in noise:
            assert st["accepted"][i] == False, (          # noqa: E712
                "seed %d plane %d: a pure-noise plane was TRUSTED "
                "(correction %.2f, %.2f grid px, ncc %.3f)"
                % (seed, i, st_un["dr"][i], st_un["dc"][i], st["ncc"][i]))
            # "falls back to the chain value" means EXACTLY that: bit-equal,
            # not clamped to the cap, not nudged toward the accepted planes.
            assert r_g[i] == r[i] and c_g[i] == c[i], (
                "seed %d plane %d: rejected plane did not keep its chain "
                "value (%.6f, %.6f vs chain %.6f, %.6f)"
                % (seed, i, r_g[i], c_g[i], r[i], c[i]))
            assert st["ncc"][i] < improved.CHAIN_REFINE_MIN_NCC, (
                "seed %d plane %d: ncc %.3f — the synthetic no longer models "
                "a plane without common content"
                % (seed, i, st["ncc"][i]))

        # the honest planes must come through the gate UNTOUCHED — a gate that
        # also suppressed those would "fix" the bug by disabling correction 3
        for i in good:
            assert st["accepted"][i] == True, (           # noqa: E712
                "seed %d plane %d: a structured plane was rejected "
                "(ncc %.3f, correction %.2f, %.2f)"
                % (seed, i, st["ncc"][i], st_un["dr"][i], st_un["dc"][i]))
            assert r_g[i] == r_un[i] and c_g[i] == c_un[i], (
                "seed %d plane %d: the gate changed an ACCEPTED plane"
                % (seed, i))
            assert st["ncc"][i] > 0.9, (seed, i, st["ncc"][i])

        assert st["n_rejected"] == n_noise, (seed, st["n_rejected"])
        # and the volume it returns is the rejected planes at their chain
        # shift, i.e. the raw planes untouched (chain shift is zero here)
        for i in noise:
            assert np.allclose(vol_g[:, :, i], vol[:, :, i].astype(np.float64))

    # what the gate is worth on this synthetic, in grid px (x scale = 4 for
    # full-resolution px). Reported so a future reader sees the size of the
    # thing being stopped, not just that it was stopped.
    print("  [improved] trust gate: ungated, the noise planes were 'corrected'"
          " by up to %.1f-%.1f grid px (cap %.1f)"
          % (min(worst_ungated), max(worst_ungated),
             improved.chain_refine_cap()))
    assert min(worst_ungated) > 4 * improved.chain_refine_cap(), (
        "the ungated refinement did not run away on ANY seed (worst %s grid "
        "px) — the synthetic no longer reproduces the failure this gate is "
        "for" % (worst_ungated,))


def test_4c_gate_keeps_a_real_subpixel_correction():
    """The gate must not be a way of turning correction 3 off.

    One plane carries a genuine ~1 px residual the chain missed; everything
    else is aligned. The refinement has to measure it, accept it, and fold it
    in — and the ONLY thing standing between it and rejection must be the cap,
    which the tightened-cap control at the bottom demonstrates.
    """
    deps = orc._import_deps()
    _sbxio, dftreg, shifts2d = deps
    nz, m, k = 15, 96, 9
    base = _structured_plane(m)

    for dy, dx in ((1.0, -1.0), (0.75, 1.25), (-1.25, 0.5), (1.4, -0.6)):
        rng = np.random.default_rng(3)
        vol = np.empty((m, m, nz), dtype=np.float64)
        for i in range(nz):
            f = base if i != k else ndi.shift(base, (dy, dx), order=3,
                                              mode="nearest")
            vol[:, :, i] = f + rng.normal(scale=4.0, size=(m, m))
        vol = np.clip(vol, 0, 65535).astype(np.uint16)
        chained, r, c = _identity_chain(vol)

        st = {}
        r_g, c_g, _ = orc._refine_chain_to_volume_mean(
            vol, chained, r, c, shifts2d, dftreg, stats=st)

        # dftregistration_alex returns the shift that UNDOES the offset, so a
        # plane displaced by (dy, dx) must be corrected by (-dy, -dx).
        want = np.array([-dy, -dx])
        got = np.array([r_g[k], c_g[k]])
        err = float(np.hypot(*(got - want)))
        assert st["accepted"][k] == True, (               # noqa: E712
            "(%.2f, %.2f): a REAL residual was rejected (correction %.2f, "
            "%.2f grid px, ncc %.3f)"
            % (dy, dx, st["dr"][k], st["dc"][k], st["ncc"][k]))
        assert np.all(np.abs(got - want) <= 0.5), (
            "(%.2f, %.2f): recovered (%.3f, %.3f), want (%.3f, %.3f)"
            % (dy, dx, got[0], got[1], want[0], want[1]))
        assert err <= 0.5 * float(np.hypot(*want)), (
            "(%.2f, %.2f): the refinement removed less than half the "
            "misalignment (%.3f px left of %.3f px)"
            % (dy, dx, err, float(np.hypot(*want))))
        # the aligned planes must stay aligned: no correction on any of them
        others = [i for i in range(nz) if i != k]
        assert np.all(np.asarray(r_g)[others] == 0.0), np.asarray(r_g)[others]
        assert np.all(np.asarray(c_g)[others] == 0.0), np.asarray(c_g)[others]

        # CONTROL: the cap is what decides. Tighten it below the true residual
        # and the SAME measurement is refused — which is also the proof that
        # the acceptance above was the gate passing it, not the gate missing.
        st_tight = {}
        r_t, c_t, _ = orc._refine_chain_to_volume_mean(
            vol, chained, r, c, shifts2d, dftreg, cap=0.5, stats=st_tight)
        assert st_tight["accepted"][k] == False, (dy, dx)  # noqa: E712
        assert r_t[k] == r[k] and c_t[k] == c[k], (dy, dx, r_t[k], c_t[k])

    # the thresholds are RegistrationConfig fields, validated by the module
    # that consumes them, and they reach a run through run_pipeline's scope
    cfg = RegistrationConfig(input_path="x.tif")
    assert cfg.chain_refine_cap == 3.0 and cfg.chain_refine_min_ncc == 0.30
    cfg = RegistrationConfig(input_path="x.tif", chain_refine_cap=8,
                             chain_refine_min_ncc=0)
    assert cfg.chain_refine_cap == 8.0 and cfg.chain_refine_min_ncc == 0.0
    for bad_cap in (0, -1, "big", float("nan"), True):
        try:
            RegistrationConfig(input_path="x.tif", chain_refine_cap=bad_cap)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError("chain_refine_cap=%r was accepted" % (bad_cap,))
    for bad_ncc in (-0.1, 1.0, 2, "high", True):
        try:
            RegistrationConfig(input_path="x.tif", chain_refine_min_ncc=bad_ncc)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError("chain_refine_min_ncc=%r accepted" % (bad_ncc,))

    # the process-wide setting is what the refinement reads when the caller
    # pins nothing, and it restores on the way out (mode_scope's contract)
    assert improved.chain_refine_cap() == improved.CHAIN_REFINE_CAP
    with improved.chain_refine_guard_scope(cap=0.5):
        assert improved.chain_refine_cap() == 0.5
        assert improved.chain_refine_min_ncc() == improved.CHAIN_REFINE_MIN_NCC
        st_scoped = {}
        r_s, c_s, _ = orc._refine_chain_to_volume_mean(
            vol, chained, r, c, shifts2d, dftreg, stats=st_scoped)
        assert st_scoped["accepted"][k] == False               # noqa: E712
        assert r_s[k] == r[k] and c_s[k] == c[k]
    assert improved.chain_refine_cap() == improved.CHAIN_REFINE_CAP, (
        "chain_refine_guard_scope did not restore")
    try:
        with improved.chain_refine_guard_scope(cap=0.5, min_ncc=0.9):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert (improved.chain_refine_cap(), improved.chain_refine_min_ncc()) == (
        improved.CHAIN_REFINE_CAP, improved.CHAIN_REFINE_MIN_NCC), (
        "the guard scope must restore on an exception too")


# ---------------------------------------------------------------------------
# 5. iron law (a)
# ---------------------------------------------------------------------------

def test_5_synthetic_suite_still_passes():
    """Run tests/test_synthetic.py as a subprocess; require 7/7."""
    script = os.path.join(_HERE, "test_synthetic.py")
    _require(script)
    env = dict(os.environ, PYTHONPATH=_PKG_PARENT)
    p = subprocess.run([sys.executable, script], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = p.stdout.decode("utf-8", "replace")
    assert p.returncode == 0, "test_synthetic.py failed:\n%s" % out
    assert "7/7 passed" in out, "unexpected synthetic result:\n%s" % out


# ---------------------------------------------------------------------------
# 6. iron law (b) + the improved-mode measurement
# ---------------------------------------------------------------------------

def test_6_subset_replicate_bitwise_and_improved_metrics():
    """40-volume subset: replicate bit-exact vs truth, improved characterized."""
    _require(RAW_TIF)
    _require(TRUTH_ZPROJ)
    import shutil
    tmp = tempfile.mkdtemp(prefix="cpstab_improved_")
    try:
        runs = {}
        for mode in ("replicate", "improved"):
            out_dir = os.path.join(tmp, mode)
            os.makedirs(out_dir)
            cfg = RegistrationConfig(input_path=RAW_TIF, out_dir=out_dir,
                                     mode=mode, **CFG_KW)
            t0 = time.time()
            run_pipeline(cfg)
            runs[mode] = (cfg, time.time() - t0)

        cfg_rep, t_rep = runs["replicate"]
        cfg_imp, t_imp = runs["improved"]

        # ---- IRON LAW (b) ------------------------------------------------
        got = tifffile.imread(cfg_rep.zproj_tiff_path())
        truth = tifffile.imread(TRUTH_ZPROJ)
        assert got.dtype == truth.dtype == np.uint16
        assert np.array_equal(got, truth), (
            "REPLICATE PATH BROKEN: %d/%d pixels differ from the reference, "
            "max |delta| = %d"
            % (int((got != truth).sum()), got.size,
               int(np.abs(got.astype(np.int64)
                          - truth.astype(np.int64)).max())))

        # ---- improved must actually have changed something ---------------
        imp = tifffile.imread(cfg_imp.zproj_tiff_path())
        assert imp.shape == got.shape and imp.dtype == got.dtype
        assert not np.array_equal(imp, got), (
            "improved mode produced the replicate output — the mode did not "
            "reach the pipeline")
        # corrections 1 and 3 change the SHIFTS, so the payload must differ too
        s_rep = np.load(cfg_rep.shiftpath())
        s_imp = np.load(cfg_imp.shiftpath())
        assert not np.array_equal(s_rep["RS"], s_imp["RS"]), (
            "improved mode left RS untouched — correction 3 (chain refine) "
            "did not run")

        # ---- the measurement ---------------------------------------------
        results = [metrics.stabilization_metrics(c.zproj_tiff_path(),
                                                 sample_stride=1, label=lbl)
                   for lbl, c in (("replicate", cfg_rep),
                                  ("improved", cfg_imp))]
        print(metrics.markdown_table(results))
        print("  [improved] wall clock: replicate %.1fs -> improved %.1fs"
              % (t_rep, t_imp))

        rep_m, imp_m = results
        for ch in range(len(rep_m["channels"])):
            a = rep_m["channels"][ch]
            b = imp_m["channels"][ch]
            # Deliberately loose: 40 volumes over 2 chunks is a mechanism
            # check, not a result about the dataset (see the module
            # docstring). What must hold is that improved did not make the
            # deliverable WORSE on either headline number.
            assert b["residual_px_p95"] <= a["residual_px_p95"] * 1.05, (
                "ch%d residual p95 got worse: %.4f -> %.4f px"
                % (ch + 1, a["residual_px_p95"], b["residual_px_p95"]))
            assert b["sharpness"] >= a["sharpness"] * 0.95, (
                "ch%d temporal-mean sharpness dropped: %.5f -> %.5f"
                % (ch + 1, a["sharpness"], b["sharpness"]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def test_7_no_trajectory_gate_exists():
    """Guard the FALSIFIED correction 5 against re-introduction.

    A running-median trust gate on zproj_reg's movie-level trajectory was
    implemented and measured on 2026-08-25: the 50-64 px single-frame
    deviations it flagged were REAL tissue excursions the estimator tracks to
    ~8 px of output residual, and interpolating through them raised those
    frames to 41-46 px. The feature was removed the same day. This test
    pins the removal: no traj_gate feature, no gate helper -- anyone
    re-adding one must consciously delete this test and re-run the
    output-residual measurement that killed the first attempt (see the
    comment in apply_project.zproj_reg).
    """
    assert "traj_gate" not in improved.FEATURES
    assert not hasattr(improved, "use_traj_gate")
    assert not hasattr(ap, "_traj_median_gate")


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS  %s" % name)
        except unittest.SkipTest as e:
            skipped += 1
            print("SKIP  %s: %s" % (name, e))
        except AssertionError as e:
            failed += 1
            print("FAIL  %s: %s" % (name, e))
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("ERROR %s: %r" % (name, e))
    print("\n%d/%d passed (%d skipped)"
          % (len(tests) - failed - skipped, len(tests) - skipped, skipped))
    sys.exit(1 if failed else 0)
