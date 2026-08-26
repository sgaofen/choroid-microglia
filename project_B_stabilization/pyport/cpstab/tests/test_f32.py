# -*- coding: utf-8 -*-
"""float32 fast mode (cfg.compute_dtype) — plumbing, iron laws, and the
measurement that decided where the precision boundary sits.

Runs standalone (``python test_f32.py``) or under pytest.

WHAT IS BEING PROVEN
--------------------
  test_1  the mode is actually ENGAGED and its boundary is where precision.py
          says it is: pixels become float32, the DFT correlation engine stays
          float64/complex128, MATLAB's class-preserving ops still return
          uint16, and the shift bookkeeping stays float64. Also that the
          scope restores float64 on exit, including on an exception.
  test_2  IRON LAW (a): tests/test_synthetic.py still passes 7/7 — i.e. the
          default float64 path is untouched by the whole feature.
  test_3  IRON LAW (b) + the fast-mode measurement, from ONE pair of full
          40-volume pipeline runs (the expensive part, ~100 s):
            * float64 output is bit-identical to the archived reference
              reportA/port_run/FAD-F_1_T0-39_mean_zproj.tif;
            * float32 vs float64: max|diff| / Pearson r / NRMSE at the float
              level AND at the written uint16 level;
            * the .dftshifts.npz payload is bit-identical between the modes
              (shift ESTIMATION is unaffected — that is the design claim);
            * wall-clock speedup.
  test_4  the NEGATIVE CONTROL for the boundary: force the correlation engine
          to single precision (patching only the two seam functions) and show
          the shift estimates fall apart. This is why precision.py pins the
          engine to double instead of letting it follow the compute dtype; if
          someone "simplifies" that away, this test is what fails.

The big-data tests SKIP (not fail) when the workspace files are absent, so
the file also runs on a machine that only has the repo.

MEASURED (macOS 15 / Apple silicon, numpy 2.0.2, scipy 1.13.1, python 3.9,
single process, FAD-F_1_T0-39.tif, refchannel=1 scale=4 chunksize=20
proj_range=quarter) — recorded here so a future change that moves them is
visible in the diff:

    wall clock ......... float64 55.94 s -> float32 39.96 s   (1.40x)
      registration stage ... 7.0 s/chunk -> 6.2 s/chunk       (1.12x)
      apply + refine ....... the remainder                    (~1.5x)
    zproj_mean, float level (values span 0 .. 226.3):
      max|diff| 6.94e-06, mean|diff| 5.48e-07, 88.7% of voxels differ
      max relative diff 5.41e-08
      Pearson r 0.999999999999977   (1-r = 2.3e-14)
      NRMSE (range-normalized) 3.11e-09
    written uint16 TIFF: BIT-IDENTICAL (0 of 20,971,520 pixels differ)
    .dftshifts.npz: bit-identical (RS/CS/ZS and the chunk corrections)

The uint16 bit-identity is a RESULT, not a guarantee — see test_3's
assertions, which require r > 0.999 and a <=1-count max, not equality.
"""

import os
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np
import tifffile

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(os.path.dirname(_HERE))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from cpstab import RegistrationConfig, run_pipeline            # noqa: E402
from cpstab import apply_project as ap                         # noqa: E402
from cpstab import dftreg, precision, shifts2d                 # noqa: E402
from cpstab import orchestrator as orc                         # noqa: E402
from cpstab.io_rw import VolumeSource                          # noqa: E402
from cpstab.matlab_compat import matlab_imresize               # noqa: E402

# ---------------------------------------------------------------------------
# real-data fixtures (the same 40-volume subset the port was validated on)
# ---------------------------------------------------------------------------
# Real-data fixtures live outside the repo (they are 60+ GB). Point
# CPSTAB_WORKSPACE at them to run those tests; without it they SKIP.
WORKSPACE = os.environ.get("CPSTAB_WORKSPACE", "")
RAW_TIF = os.path.join(WORKSPACE, "FAD-F_1_T0-39.tif")
TRUTH_ZPROJ = os.path.join(WORKSPACE, "reportA", "port_run",
                           "FAD-F_1_T0-39_mean_zproj.tif")
CFG_KW = dict(refchannel=1, scale=4, chunksize=20, proj_range="quarter")

# Assertion thresholds. Generous vs the observed values above on purpose: the
# point is to catch a REGIME change (a flipped correlation peak, a plane-shift
# tie tipping, the boundary being moved), not to fail on ulp noise.
TOL_PEARSON = 0.999          # brief's expectation; observed 1 - 2.3e-14
TOL_UINT16_MAX_DIFF = 1      # counts; observed 0
TOL_UINT16_FRAC = 0.01       # 1% of pixels may move by a count; observed 0


def _require(path):
    if not os.path.exists(path):
        raise unittest.SkipTest("missing test data: %s" % path)


def _pearson(a, b):
    return float(np.corrcoef(np.asarray(a, np.float64).ravel(),
                             np.asarray(b, np.float64).ravel())[0, 1])


# ---------------------------------------------------------------------------
# 1. the mode is engaged, and its boundary is where precision.py says
# ---------------------------------------------------------------------------

def test_1_boundary_is_where_precision_says():
    """Pixels drop to float32; shift estimation and bookkeeping do not.

    Without this, a plumbing regression (one missed `np.zeros(shape)`, an
    `astype(np.float64)` left behind) would silently turn fast mode back into
    a slow float64 run that still passes every numeric test.
    """
    assert precision.get_compute_dtype() == np.float64, (
        "the process must START in replicate precision")

    img_u16 = (np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) % 700)
    vol_f32 = np.stack([img_u16] * 4, axis=2).astype(np.float32)

    with precision.compute_dtype_scope("float32"):
        # --- the two classes ------------------------------------------
        assert precision.get_compute_dtype() == np.float32
        assert precision.get_correlation_dtype() == np.float64, (
            "the correlation class must stay double in fast mode")
        assert precision.get_complex_dtype() == np.complex128

        # --- pixels follow the mode -----------------------------------
        assert precision.zeros((2, 2)).dtype == np.float32
        assert precision.as_float(img_u16).dtype == np.float32
        assert shifts2d.dft_rect(vol_f32, 2, 4)[2].dtype == np.float32
        assert shifts2d.define_reference(vol_f32[..., None], 1, "mean").dtype \
            == np.float32
        assert shifts2d.apply_xy_shifts_fbs(
            vol_f32[..., None], np.zeros((4, 1)), np.zeros((4, 1))).dtype \
            == np.float32
        assert ap._project(np.zeros((1, 4, 4, 3), np.float32),
                           "mean", False).dtype == np.float32
        assert ap._process_volume(np.zeros((1, 4, 4, 2), np.uint16), 0,
                                  (0, 0, 0, 0), [None] * 2, True).dtype \
            == np.float32
        assert ap._matlab_imresize(np.zeros((8, 8, 2), np.float32),
                                   0.5).dtype == np.float32
        assert ap._quantize_u16(np.float32(3.5)).dtype == np.float32

        # --- the correlation seam does NOT ----------------------------
        assert precision.as_correlation(vol_f32).dtype == np.float64
        assert shifts2d._fft2(img_u16).dtype == np.complex128
        assert shifts2d._fft2(vol_f32[:, :, 0]).dtype == np.complex128, (
            "a float32 pixel array must be promoted before the FFT")
        assert orc._fftn(vol_f32).dtype == np.complex128
        assert ap._fft2(vol_f32[:, :, 0]).dtype == np.complex128
        assert dftreg.dftups(np.ones((4, 4), np.complex64), 4, 4, 2,
                             0, 0).dtype == np.complex128
        # ... and everything it returns is a float64 shift, never a pixel
        out = dftreg.dftregistration_alex(shifts2d._fft2(img_u16),
                                          shifts2d._fft2(img_u16), 4)
        assert out.dtype == np.float64

        # --- MATLAB class preservation is unaffected ------------------
        assert matlab_imresize(img_u16, 0.5).dtype == np.uint16, (
            "imresize must still hand the uint16 quantization chain back")
        assert shifts2d.imtranslate(img_u16, (0.5, 0.5)).dtype == np.uint16

        # --- shift bookkeeping stays double ---------------------------
        r, c = shifts2d.determine_xy_shifts_fbs(
            vol_f32[..., None], 1, 0.95,
            shifts2d.define_reference(vol_f32[..., None], 1, "mean"))
        assert r.dtype == c.dtype == np.float64
        # the float64-in / float64-out imresize (the inter-chunk shift-vector
        # stretch, DFT_warp_3D_2.m L127-129) must NOT be demoted
        assert matlab_imresize(np.arange(4.0)[None, :], output_shape=(2, 8),
                               method="nearest").dtype == np.float64

    assert precision.get_compute_dtype() == np.float64, "scope did not restore"

    # restoration must survive an exception, or one failed fast run poisons
    # every later replicate run in the same interpreter
    try:
        with precision.compute_dtype_scope("float32"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert precision.get_compute_dtype() == np.float64

    # and the config rejects anything else, before a run starts (a typo'd
    # dtype must not silently fall back to a mode the caller did not ask for)
    for bad in ("float16", "float128", "int32", "complex64", "fast"):
        try:
            RegistrationConfig(input_path="x.tif", compute_dtype=bad)
        except (ValueError, TypeError) as e:
            assert "compute dtype" in str(e) or "data type" in str(e), str(e)
        else:
            raise AssertionError("compute_dtype=%r was accepted" % bad)
    # accepted spellings all normalize to the canonical NAME string, so the
    # config stays serializable into fast_run.py's worker job tuples
    for spelling, want in ((np.float32, "float32"), ("float32", "float32"),
                           (np.dtype("float32"), "float32"),
                           ("double", "float64"), (np.float64, "float64")):
        got = RegistrationConfig(input_path="x.tif",
                                 compute_dtype=spelling).compute_dtype
        assert got == want and isinstance(got, str), (spelling, got)


# ---------------------------------------------------------------------------
# 2. iron law (a)
# ---------------------------------------------------------------------------

def test_2_synthetic_suite_still_passes():
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
# 3. iron law (b) + the fast-mode measurement (one run per mode)
# ---------------------------------------------------------------------------

def test_3_subset_replicate_bitwise_and_f32_delta():
    """40-volume subset: float64 bit-exact vs truth, float32 characterized."""
    _require(RAW_TIF)
    _require(TRUTH_ZPROJ)
    import shutil
    tmp = tempfile.mkdtemp(prefix="cpstab_f32_")
    try:
        runs = {}
        for dt in ("float64", "float32"):
            out_dir = os.path.join(tmp, dt)
            os.makedirs(out_dir)
            cfg = RegistrationConfig(input_path=RAW_TIF, out_dir=out_dir,
                                     compute_dtype=dt, **CFG_KW)
            t0 = time.time()
            z = run_pipeline(cfg)
            runs[dt] = (z, time.time() - t0, cfg)

        z64, t64, cfg64 = runs["float64"]
        z32, t32, cfg32 = runs["float32"]

        # the mode actually reached the returned array
        assert z64.dtype == np.float64, z64.dtype
        assert z32.dtype == np.float32, (
            "fast mode did not reach zproj_mean (got %s)" % z32.dtype)

        # ---- IRON LAW (b) ------------------------------------------------
        got64 = tifffile.imread(cfg64.zproj_tiff_path())
        truth = tifffile.imread(TRUTH_ZPROJ)
        assert got64.dtype == truth.dtype == np.uint16
        assert np.array_equal(got64, truth), (
            "REPLICATE PATH BROKEN: %d/%d pixels differ from the reference, "
            "max |delta| = %d"
            % (int((got64 != truth).sum()), got64.size,
               int(np.abs(got64.astype(np.int64)
                          - truth.astype(np.int64)).max())))

        # ---- shift estimation must be untouched by the mode --------------
        s64 = np.load(cfg64.shiftpath())
        s32 = np.load(cfg32.shiftpath())
        assert sorted(s64.files) == sorted(s32.files)
        for k in s64.files:
            assert s64[k].dtype == np.float64, (
                "%s left the float64 bookkeeping domain (%s)" % (k, s64[k].dtype))
            assert np.array_equal(s64[k], s32[k]), (
                "fast mode changed the shift file entry %r (max |delta| = %g) "
                "— the correlation engine is supposed to be precision-pinned"
                % (k, np.abs(s32[k] - s64[k]).max()))

        # ---- float-level characterization --------------------------------
        a = z64.astype(np.float64)
        b = z32.astype(np.float64)
        d = b - a
        rf = _pearson(a, b)
        nrmse = float(np.sqrt(np.mean(d ** 2)) / (a.max() - a.min()))
        nz = a > 0
        relmax = float((np.abs(d)[nz] / a[nz]).max()) if nz.any() else 0.0
        print("  [f32] zproj_mean float: max|diff|=%.4g mean|diff|=%.4g "
              "relmax=%.3g r=%.15f NRMSE=%.3g"
              % (np.abs(d).max(), np.abs(d).mean(), relmax, rf, nrmse))
        assert rf > TOL_PEARSON, "float-level Pearson r = %.9f" % rf

        # ---- uint16-level characterization -------------------------------
        got32 = tifffile.imread(cfg32.zproj_tiff_path())
        du = got32.astype(np.int64) - got64.astype(np.int64)
        frac = float(np.count_nonzero(du)) / du.size
        ru = _pearson(got64, got32)
        print("  [f32] written uint16: max|diff|=%d counts, %d/%d pixels "
              "differ (%.4g%%), r=%.15f"
              % (int(np.abs(du).max()), int(np.count_nonzero(du)), du.size,
                 100.0 * frac, ru))
        assert ru > TOL_PEARSON, "uint16-level Pearson r = %.9f" % ru
        assert np.abs(du).max() <= TOL_UINT16_MAX_DIFF, (
            "fast mode moved a pixel by %d counts; > %d means a registration "
            "decision flipped, not rounding"
            % (int(np.abs(du).max()), TOL_UINT16_MAX_DIFF))
        assert frac <= TOL_UINT16_FRAC, (
            "%.3g%% of pixels changed; > %.3g%% means a systematic shift, not "
            "quantization ties" % (100.0 * frac, 100.0 * TOL_UINT16_FRAC))

        # ---- timing ------------------------------------------------------
        print("  [f32] wall clock: float64 %.2fs -> float32 %.2fs (%.2fx)"
              % (t64, t32, t64 / t32))
        assert t32 < t64 * 1.10, (
            "fast mode was not faster (float64 %.2fs, float32 %.2fs) — it is "
            "allowed to be a wash on some machines, but a real slowdown means "
            "something is promoting back to double" % (t64, t32))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 4. negative control: why the correlation engine is precision-pinned
# ---------------------------------------------------------------------------

def test_4_single_precision_correlation_would_break_shifts():
    """Let the DFT engine follow the compute dtype and the shifts fall apart.

    precision.py pins the correlation class to float64/complex128 in BOTH
    modes. That looks like an unnecessary exception until you try the other
    way: on this data the phase correlation surface is nearly flat (the DC
    term is ~91% of the peak height and the winning sample beats the runner-up
    by a median 4.5e-4 relative), so complex64 does not perturb a shift by an
    ulp — it flips which cell wins and the shift jumps by a whole 1/usfac step.

    This test patches ONLY the two seam functions (shifts2d.as_correlation and
    dftreg.get_complex_dtype) so it fails loudly if someone later "simplifies"
    the engine into following get_compute_dtype().
    """
    _require(RAW_TIF)
    deps = orc._import_deps()

    def one_chunk(single_precision_engine):
        saved_asc = shifts2d.as_correlation
        saved_cdt = dftreg.get_complex_dtype
        try:
            if single_precision_engine:
                shifts2d.as_correlation = \
                    lambda x: np.asarray(x, dtype=np.float32)
                dftreg.get_complex_dtype = lambda: np.dtype(np.complex64)
            with precision.compute_dtype_scope(
                    "float32" if single_precision_engine else "float64"):
                with VolumeSource(RAW_TIF) as src:
                    nz = int(src.Nz)
                    _info, reader = orc._resolve_source(src, deps[0])
                    return orc._process_chunk(
                        reader, 0, nz * 20, 20, 1, [0, 0, 0, 0], 4,
                        [np.eye(3) for _ in range(nz)], "true", "mean",
                        1, 0.95, int(orc.matlab_round(nz / 2.0)), deps=deps)
        finally:
            shifts2d.as_correlation = saved_asc
            dftreg.get_complex_dtype = saved_cdt

    ref = one_chunk(False)
    bad = one_chunk(True)
    # index 0/1 are the combined RS/CS cells for this chunk
    d_rs = np.abs(np.asarray(bad[0]) - np.asarray(ref[0])).max()
    d_cs = np.abs(np.asarray(bad[1]) - np.asarray(ref[1])).max()
    print("  [f32] engine forced to complex64: chunk-0 RS/CS max|diff| = "
          "%.3g / %.3g px" % (d_rs, d_cs))
    assert max(d_rs, d_cs) > 1.0, (
        "a single-precision correlation engine no longer breaks the shift "
        "estimates on this data (max |delta| %.3g px). Either the fixture "
        "changed or the patched seam is no longer the one the engine uses — "
        "re-derive the boundary in cpstab/precision.py before trusting it."
        % max(d_rs, d_cs))


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

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
