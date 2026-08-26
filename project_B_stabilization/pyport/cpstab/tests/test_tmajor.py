# -*- coding: utf-8 -*-
"""T-major (.npy) ingest path: relayout.py + VolumeSource, bit-exactness.

Runs standalone (``python test_tmajor.py``) or under pytest.

WHAT IS BEING PROVEN
--------------------
The Z-major bfconvert OME-TIFF (axes ZTCYX: z SLOWEST) makes one volume read
a scatter of Nz page reads across the whole file. scripts/relayout.py rewrites
the stack once as a contiguous (T, Z, C, Y, X) .npy and io_rw.VolumeSource
reads it through the identical interface. Because that is a REPLICATE-path
change, the only acceptable outcome is bit-for-bit identity:

  test_1  every TZCYX axis permutation (including the production ZTCYX and
          the regression subset's TZCYX) relayouts to the same array, and
          VolumeSource(.npy).get_volume == VolumeSource(.tif).get_volume.
  test_2  the 40-volume real subset FAD-F_1_T0-39.tif relayouts to
          FAD-F_1_T0-39.tzcyx.npy (40, 41, 2, 512, 512) uint16.
  test_3  for EVERY (t, c) of that subset, get_volume() off the .npy equals
          get_volume() off the TIFF, exactly (np.array_equal on float64) —
          against BOTH TIFF read strategies, memmap and page-read.
  test_4  a full run_pipeline() on the .npy reproduces the reference
          projection reportA/port_run/FAD-F_1_T0-39_mean_zproj.tif bit for
          bit (the same run's TIFF-path output).
  test_5  the synthetic end-to-end suite still passes 7/7 (iron law a).
  test_6  relayout never publishes a store it did not fully write (the
          sparse output would read back as plausible zeros).
  test_7  the apply stage's bulk read (VolumeSource.read_block, used by
          fast_run.py) returns exactly what the per-volume get_volume /
          imread_fn path returns, in both compute dtypes.

The big-data tests SKIP (not fail) when the workspace files are absent, so
the file also runs on a machine that only has the repo.
"""

import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import tifffile

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(os.path.dirname(_HERE))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)
_SCRIPTS = os.path.join(_PKG_PARENT, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from cpstab import RegistrationConfig, run_pipeline, matlab_uint16   # noqa: E402
from cpstab.io_rw import VolumeSource                                # noqa: E402
from relayout import relayout                                        # noqa: E402

# ---------------------------------------------------------------------------
# real-data fixtures (the 40-volume subset used for the port validation)
# ---------------------------------------------------------------------------
# Real-data fixtures live outside the repo (they are 60+ GB). Point
# CPSTAB_WORKSPACE at them to run those tests; without it they SKIP.
WORKSPACE = os.environ.get("CPSTAB_WORKSPACE", "")
RAW_TIF = os.path.join(WORKSPACE, "FAD-F_1_T0-39.tif")
RAW_NPY = os.path.join(WORKSPACE, "FAD-F_1_T0-39.tzcyx.npy")
TRUTH_ZPROJ = os.path.join(WORKSPACE, "reportA", "port_run",
                           "FAD-F_1_T0-39_mean_zproj.tif")
# the config the reference run used (validate.py --params)
CFG_KW = dict(refchannel=1, scale=4, chunksize=20, proj_range="quarter")

EXPECT_SHAPE = (40, 41, 2, 512, 512)     # (T, Z, C, Y, X)


def _require(path):
    if not os.path.exists(path):
        raise unittest.SkipTest("missing test data: %s" % path)


_STATE = {}


def _ensure_npy():
    """Relayout RAW_TIF -> RAW_NPY once per interpreter (test_2's product)."""
    if _STATE.get("npy"):
        return RAW_NPY
    _require(RAW_TIF)
    relayout(RAW_TIF, RAW_NPY, force=True, verify=16, progress=False)
    _STATE["npy"] = True
    return RAW_NPY


# ---------------------------------------------------------------------------
# 1. axis-order independence on synthetic stacks
# ---------------------------------------------------------------------------

def test_1_axis_permutations_roundtrip():
    """relayout must READ the axis order from series.axes, not assume one.

    The production stack is ZTCYX (bfconvert) while the regression subset is
    TZCYX (ImageJ) — a hard-coded order would silently transpose one of them.
    Also checks a missing-C stack (ZTYX) and a tolerated singleton axis.
    """
    rng = np.random.default_rng(11)
    T, Z, C, Y, X = 3, 4, 2, 6, 5
    ref = rng.integers(0, 65535, size=(T, Z, C, Y, X)).astype(np.uint16)
    tmp = tempfile.mkdtemp(prefix="cpstab_tmajor_")
    try:
        for order in ("TZCYX", "ZTCYX", "ZCTYX", "CTZYX", "TCZYX"):
            src = np.transpose(ref, ["TZCYX".index(a) for a in order])
            tif = os.path.join(tmp, "s_%s.tif" % order)
            npy = os.path.join(tmp, "s_%s.npy" % order)
            tifffile.imwrite(tif, src, metadata={"axes": order})
            relayout(tif, npy, force=True, verify=6, progress=False)
            got = np.load(npy)
            assert got.shape == (T, Z, C, Y, X), "%s -> %r" % (order, got.shape)
            assert np.array_equal(got, ref), "%s: pixels permuted" % order
            with VolumeSource(npy) as a, VolumeSource(tif) as b:
                assert a.metadata == b.metadata, "%s metadata %r != %r" % (
                    order, a.metadata, b.metadata)
                assert a.dtype == b.dtype == np.uint16
                assert len(a) == len(b) == T
                for t in range(T):
                    for c in range(C):
                        va, vb = a.get_volume(t, c), b.get_volume(t, c)
                        assert va.dtype == np.float64 and va.shape == (Y, X, Z)
                        assert np.array_equal(va, vb), (
                            "%s: get_volume(t=%d,c=%d) differs" % (order, t, c))
                        assert np.array_equal(a.get_frame(t, 1, c),
                                              b.get_frame(t, 1, c))

        # no C axis at all -> C treated as size 1 (VolumeSource's own rule)
        tif = os.path.join(tmp, "noc.tif")
        npy = os.path.join(tmp, "noc.npy")
        tifffile.imwrite(tif, np.transpose(ref[:, :, 0], (1, 0, 2, 3)),
                         metadata={"axes": "ZTYX"})
        relayout(tif, npy, force=True, verify=4, progress=False)
        got = np.load(npy)
        assert got.shape == (T, Z, 1, Y, X), got.shape
        assert np.array_equal(got[:, :, 0], ref[:, :, 0])
        with VolumeSource(npy) as a, VolumeSource(tif) as b:
            assert a.metadata == b.metadata == (1, Y, X, Z, T)
            for t in range(T):
                assert np.array_equal(a.get_volume(t, 0), b.get_volume(t, 0))

        # a 5-D .npy is the ONLY accepted container shape
        bad = os.path.join(tmp, "bad.npy")
        np.save(bad, ref[0])                       # 4-D
        try:
            VolumeSource(bad)
        except ValueError as e:
            assert "5-D" in str(e), str(e)
        else:
            raise AssertionError("VolumeSource accepted a 4-D .npy")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 2. the real 40-volume subset
# ---------------------------------------------------------------------------

def test_2_relayout_real_subset():
    """FAD-F_1_T0-39.tif (ImageJ TZCYX) -> FAD-F_1_T0-39.tzcyx.npy."""
    _ensure_npy()
    arr = np.load(RAW_NPY, mmap_mode="r")
    assert arr.shape == EXPECT_SHAPE, "npy shape %r" % (arr.shape,)
    assert arr.dtype == np.uint16, arr.dtype
    expect_bytes = int(np.prod(EXPECT_SHAPE)) * 2
    got = os.path.getsize(RAW_NPY)
    assert got - expect_bytes < 4096, "size %d vs %d + header" % (got,
                                                                 expect_bytes)
    with VolumeSource(RAW_NPY) as s:
        # (Nchan, Nx=rows, Ny=cols, Nz, Nt) — MATLAB naming
        assert s.metadata == (2, 512, 512, 41, 40), s.metadata
        assert s.dtype == np.uint16


def test_3_get_volume_matches_tiff():
    """Every (t, c) volume off the .npy == the same volume off the TIFF.

    Run against BOTH TIFF read strategies, because they are different code
    paths and the regression subset alone only exercises one of them: this
    ImageJ subset is contiguous, so VolumeSource memmaps it, while the 114 GB
    production OME-TIFF is NOT memmappable (bfconvert leaves a 401,857-byte
    gap after every page) and falls back to asarray(key=[...]) page reads.
    Comparing only the memmap branch would leave the branch production
    actually runs on unverified (review finding).
    """
    _require(RAW_TIF)
    _ensure_npy()
    for prefer_memmap in (True, False):
        with VolumeSource(RAW_NPY) as a, \
                VolumeSource(RAW_TIF, prefer_memmap=prefer_memmap) as b:
            assert (b._mm is not None) == prefer_memmap, (
                "expected memmap=%s branch, got _mm=%r — this subset must be "
                "contiguous for the True case to mean anything"
                % (prefer_memmap, b._mm))
            assert a.metadata == b.metadata, "%r != %r" % (a.metadata,
                                                           b.metadata)
            assert a.dtype == b.dtype
            assert len(a) == len(b)
            nchan, _, _, nz, nt = a.metadata
            for t in range(nt):
                for c in range(nchan):
                    va, vb = a.get_volume(t, c), b.get_volume(t, c)
                    assert va.shape == (512, 512, nz), va.shape
                    assert va.dtype == np.float64 and vb.dtype == np.float64
                    assert np.array_equal(va, vb), (
                        "volume (t=%d, c=%d) differs from the TIFF path "
                        "(prefer_memmap=%s)" % (t, c, prefer_memmap))


def test_4_pipeline_bitwise_matches_truth():
    """Full run_pipeline() on the .npy == the reference TIFF-path output.

    The reference file is the port's own validated run (reportA/port_run),
    produced from FAD-F_1_T0-39.tif with refchannel=1 scale=4 chunksize=20
    proj_range=quarter. Bit-for-bit equality is the iron law: the storage
    format may change, the numbers may not.
    """
    _require(TRUTH_ZPROJ)
    _ensure_npy()
    tmp = tempfile.mkdtemp(prefix="cpstab_tmajor_run_")
    try:
        cfg = RegistrationConfig(input_path=RAW_NPY, out_dir=tmp, **CFG_KW)
        zproj = run_pipeline(cfg)                      # (C, Y, X, T) float64
        out_path = cfg.zproj_tiff_path()
        assert os.path.exists(out_path), out_path
        got = tifffile.imread(out_path)
        truth = tifffile.imread(TRUTH_ZPROJ)
        assert got.shape == truth.shape, "%r vs %r" % (got.shape, truth.shape)
        assert got.dtype == truth.dtype == np.uint16
        assert np.array_equal(got, truth), (
            "T-major run diverged from the reference projection: %d/%d pixels "
            "differ, max |delta| = %d"
            % (int((got != truth).sum()), got.size,
               int(np.abs(got.astype(np.int64)
                          - truth.astype(np.int64)).max())))
        # and the in-memory float64 return casts to exactly the same pixels
        assert np.array_equal(
            np.transpose(matlab_uint16(zproj), (3, 0, 1, 2)), truth)
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 5. iron-law regression (a): the synthetic suite must stay 7/7
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
# 6. an incomplete store must never be published (review finding)
# ---------------------------------------------------------------------------

def test_6_incomplete_store_is_never_published():
    """A store is published only if every plane was actually written.

    The output .npy is created sparse, so any region the copy loop misses
    reads back as ordinary zeros — a silent, undetectable data loss for a
    60 GiB artifact. Before the review, `block_pages <= 0` made
    `range(0, npages, step)` yield nothing, the loop body never ran, and
    os.replace published a full-size, correctly-shaped, entirely BLANK file
    with no error and exit status 0.

    Checks both layers of the guard (relayout.py DESIGN NOTES 8):
      * the argument range checks, before any file is created;
      * the `done != nplanes` invariant, which fires for ANY short copy —
        here forced by shadowing the module's `range` — and is deliberately
        independent of `verify` (a switchable random sample).
    """
    import relayout as _R
    tmp = tempfile.mkdtemp(prefix="cpstab_tmajor_guard_")
    try:
        rng = np.random.default_rng(23)
        ref = rng.integers(1, 65535, size=(2, 3, 2, 7, 11)).astype(np.uint16)
        src = os.path.join(tmp, "g.tif")
        tifffile.imwrite(src, np.transpose(ref, (1, 0, 2, 3, 4)),
                         metadata={"axes": "ZTCYX"})

        def _published(name):
            """Any output or leftover .partial for this attempt?"""
            return [f for f in os.listdir(tmp) if f.startswith(name)]

        for bad in (0, -1, -256):
            out = os.path.join(tmp, "bp%d.npy" % abs(bad))
            try:
                relayout(src, out, block_pages=bad, force=True, verify=0,
                         progress=False)
            except ValueError as e:
                assert "block_pages" in str(e), str(e)
            else:
                raise AssertionError(
                    "block_pages=%d was accepted; it publishes a blank store"
                    % bad)
            assert not _published("bp%d" % abs(bad)), _published("bp%d" % abs(bad))

        for bad in (-1, -8):
            out = os.path.join(tmp, "vf.npy")
            try:
                relayout(src, out, verify=bad, force=True, progress=False)
            except ValueError as e:
                assert "verify" in str(e), str(e)
            else:
                raise AssertionError("verify=%d was accepted" % bad)
            assert not _published("vf"), _published("vf")

        # the invariant proper: a copy loop that silently does nothing
        out = os.path.join(tmp, "short.npy")
        _R.range = lambda *a, **k: iter(())      # shadows the builtin
        try:
            relayout(src, out, force=True, verify=0, progress=False)
        except AssertionError as e:
            assert "0 of 12 planes" in str(e), str(e)
        else:
            raise AssertionError(
                "a short copy loop published a store instead of raising")
        finally:
            del _R.range
        assert not _published("short"), _published("short")

        # and the happy path still works, unchanged
        good = os.path.join(tmp, "good.npy")
        relayout(src, good, force=True, verify=6, progress=False)
        assert np.array_equal(np.load(good), ref)
        assert not [f for f in os.listdir(tmp) if f.endswith(".partial")]
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def test_7_read_block_equals_get_volume():
    """The apply stage's bulk read is the per-volume read, byte for byte.

    fast_run.py's apply worker no longer calls get_volume() once per volume
    on a T-major store: it pulls a whole run of volumes with
    VolumeSource.read_block() and slices them in RAM (io_rw PORTING NOTES
    #19). That is a REPLICATE-path change, so the only acceptable outcome is
    identity — checked here at both levels and in both compute dtypes:

      * block[j, :, c] cast to the compute class == get_volume(t0+j, c);
      * fast_run._block_volume(...) == the imread_fn that
        pipeline._apply_io_adapter hands the worker — same shape, same
        dtype, same values, including the SQUEEZED single-channel layout
        MATLAB's sbxRead returns (pmt != -1).

    Also pins the two contract details a caller could get wrong silently:
    read_block returns None (not an exception, not an emulation) for a TIFF
    source, and it range-checks its bounds.
    """
    _require(RAW_TIF)
    _ensure_npy()
    import fast_run
    from cpstab.pipeline import _apply_io_adapter
    from cpstab.precision import compute_dtype_scope

    with VolumeSource(RAW_TIF) as tif_src:
        assert tif_src.read_block(0, 2) is None, (
            "read_block must decline a TIFF source so the caller falls back "
            "to the validated per-volume path")

    for dt in ("float64", "float32"):
        with compute_dtype_scope(dt):
            with VolumeSource(RAW_NPY) as src:
                nchan, _nx, _ny, nz, nt = src.metadata
                _info_fn, imread_fn = _apply_io_adapter(src)
                fdt = np.dtype(dt)
                # a block that is neither the whole file nor aligned to it
                for t0, t1 in ((0, 7), (7, 23), (23, nt)):
                    blk = src.read_block(t0, t1)
                    assert blk.shape == (t1 - t0, nz, nchan, 512, 512), blk.shape
                    assert blk.dtype == src.dtype, blk.dtype   # native, uncast
                    for j in range(t1 - t0):
                        t = t0 + j
                        for c in range(nchan):
                            ref = src.get_volume(t, c)
                            got = np.ascontiguousarray(np.transpose(
                                np.asarray(blk[j, :, c], dtype=fdt), (1, 2, 0)))
                            assert got.dtype == ref.dtype == fdt
                            assert np.array_equal(got, ref), (
                                "block slice != get_volume at t=%d c=%d "
                                "(dtype=%s)" % (t, c, dt))
                        # both pmt conventions the driver can pass
                        for pmt in (-1, 1):
                            a = fast_run._block_volume(blk, j, nchan, pmt, fdt)
                            b = imread_fn(RAW_NPY, nz * t + 1, nz, pmt, None)
                            assert a.shape == b.shape, (a.shape, b.shape)
                            assert a.dtype == b.dtype == fdt
                            assert np.array_equal(a, b), (
                                "_block_volume != imread_fn at t=%d pmt=%d "
                                "(dtype=%s)" % (t, pmt, dt))
                for bad in ((-1, 3), (0, nt + 1), (5, 4)):
                    try:
                        src.read_block(*bad)
                    except IndexError:
                        pass
                    else:
                        raise AssertionError(
                            "read_block%r was accepted" % (bad,))


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


# ===========================================================================
# DESIGN NOTES
# ===========================================================================
# 1. Why an exhaustive per-(t, c) comparison (test_3) instead of a sample:
#    the whole risk of this change is a TRANSPOSE, and a transpose can be
#    invisible on a symmetric sample (the data is 512x512, and z-planes of
#    the same volume look alike). 80 volumes x 41 planes is ~3 GB of
#    comparisons and runs in seconds off the page cache — cheap certainty.
# 2. test_4 compares the WRITTEN TIFF, not just the returned array, because
#    the deliverable is the file; it then also checks the returned float64
#    casts to the same pixels, which pins matlab_uint16's rounding as well.
#    The reference file reportA/port_run/FAD-F_1_T0-39_mean_zproj.tif is the
#    TIFF-path run of the same config, so equality here means the storage
#    change is numerically invisible end to end (estimation + apply +
#    projection + cast + write).
# 3. test_5 shells out rather than importing test_synthetic: that module
#    keeps a module-level _STATE cache and writes into a TemporaryDirectory,
#    and importing it here would run the pipeline a second time inside this
#    process. A subprocess also proves the suite passes with a clean
#    interpreter, which is what iron-law regression (a) actually asks for.
# 4. The .npy is written NEXT TO the source TIFF (workspace), not into a
#    temp dir: it is a 1.6 GB artifact that the follow-up parallel-throughput
#    work reuses, and regenerating it per test run would be pure waste. It is
#    rewritten (force=True) on every run so a stale file can never mask a
#    relayout regression.
