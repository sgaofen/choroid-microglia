"""End-to-end synthetic test for cpstab.run_pipeline.

Runs standalone (``python test_synthetic.py``) or under pytest.

Synthetic dataset: 64 x 64 pixels, 8 z-planes, 2 channels, 7 time points,
uint16 ImageJ TIFF (axes TZCYX) — the VolumeSource ingest path. Structure is
blob-like (point sources smoothed sigma=1.2, 70% shared across planes) so the
correlation peak is sharp, approximating vessel-like 2p data. Motion applied:

  * a known whole-volume (row, col) translation per time point
    (max centered magnitude 4 px), plus
  * a small per-plane jitter, uniform in [-0.5, 0.5] px, zero-mean across
    planes per volume, plus
  * Gaussian noise, sigma = 40 counts on a ~12000-count signal (SNR ~ 300).

Pipeline config: refchannel=1, scale=2, chunksize=7 (=> 1 chunk),
proj_range='quarter' (Nz=8 -> 1-based planes 2..6, i.e. 5 planes).

TOLERANCES (recorded from the reference run on macOS/numpy 2.0; assertions
leave headroom over the observed values):

  * shift-file estimate (RS+RS_chunk / CS+CS_chunk, plane-mean,
    median-centered) vs the known volume shifts:  max abs error
    OBSERVED 0.83 px  -> asserted < 1.2 px.
    (The estimation stage self-registers each frame against a 7-frame mean
    that CONTAINS the frame, which biases estimates toward zero — inherent
    to the MATLAB algorithm at T=7, not a port defect; the in-pipeline
    zproj_reg refinement recovers most of the remainder.)
  * residual motion of the stabilized projection (phase correlation of each
    frame vs the temporal mean, usfac=50): max abs
    OBSERVED 0.72 px  -> asserted < 1.0 px; also asserted to be a >= 50%
    reduction of the raw (unstabilized) projection motion (OBSERVED 3.34 px
    -> 78% reduction).
    The floor is set by zproj_reg's usfac == scale == 2 (1-px steps at full
    resolution) plus the mean-reference bias above.
"""

import os
import sys
import tempfile

import numpy as np
import tifffile
from scipy import ndimage

_PKG_PARENT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from cpstab import RegistrationConfig, run_pipeline, matlab_uint16
from cpstab.dftreg import dftregistration_alex

# ---------------------------------------------------------------------------
# synthetic dataset
# ---------------------------------------------------------------------------

Y, X, Z, C, T = 64, 64, 8, 2, 7
PAD = 16  # generation canvas margin so shifted content never runs out

# known per-volume (row, col) shifts; median-centered magnitudes up to 4 px
VOL_SHIFTS = np.array(
    [[0, 0], [2, -1], [-2, 2], [3, 1], [-1, -3], [1, 3], [-3, 2]],
    dtype=np.float64)

TOL_EST = 1.2       # px, shift-file estimate vs truth (observed 0.83)
TOL_RESID = 1.0     # px, stabilized-projection residual (observed 0.72)
MIN_RAW_MOTION = 2.0    # px, sanity: the test must be non-trivial (obs 3.34)
MIN_REDUCTION = 0.50    # stabilization must remove >=50% of raw motion


def _blob_plane(rng, shape, n, sigma=1.2):
    img = np.zeros(shape)
    ys = rng.integers(4, shape[0] - 4, n)
    xs = rng.integers(4, shape[1] - 4, n)
    img[ys, xs] = rng.uniform(0.4, 1.0, n)
    img = ndimage.gaussian_filter(img, sigma)
    return img / img.max()


def make_synthetic(seed=7):
    """Build the uint16 (T, Z, C, Y, X) stack plus the applied truth shifts.

    Returns (data, vol_shifts, jitter): jitter is (T, Z, 2), zero-mean across
    planes for each volume, so vol_shifts IS the per-volume truth.
    """
    rng = np.random.default_rng(seed)
    big = (Y + 2 * PAD, X + 2 * PAD)
    shared = _blob_plane(rng, big, 80)
    planes = np.stack(
        [0.7 * shared + 0.3 * _blob_plane(rng, big, 60) for _ in range(Z)])

    jitter = rng.uniform(-0.5, 0.5, size=(T, Z, 2))
    jitter -= jitter.mean(axis=1, keepdims=True)

    data = np.zeros((T, Z, C, Y, X), dtype=np.uint16)
    for t in range(T):
        for z in range(Z):
            dy, dx = VOL_SHIFTS[t] + jitter[t, z]
            shifted = ndimage.shift(planes[z], (dy, dx), order=1,
                                    mode="nearest")
            crop = shifted[PAD:PAD + Y, PAD:PAD + X]
            for c in range(C):
                g = 12000.0 * crop * (1.0 if c == 0 else 0.6) + 800.0
                g = g + rng.normal(0.0, 40.0, size=g.shape)
                data[t, z, c] = np.clip(np.round(g), 0, 65535).astype(np.uint16)
    return data, VOL_SHIFTS.copy(), jitter


# ---------------------------------------------------------------------------
# one shared pipeline run (lazy; reused by every test)
# ---------------------------------------------------------------------------

_STATE = {}


def _run_once():
    if _STATE:
        return _STATE
    data, vol_shifts, jitter = make_synthetic()
    tmp = tempfile.TemporaryDirectory(prefix="cpstab_test_synth_")
    tif = os.path.join(tmp.name, "synth.tif")
    tifffile.imwrite(tif, data, imagej=True, metadata={"axes": "TZCYX"})

    cfg = RegistrationConfig(input_path=tif, refchannel=1, scale=2,
                             chunksize=7, proj_range="quarter")
    zproj = run_pipeline(cfg)

    _STATE.update(dict(tmpdir=tmp, data=data, vol_shifts=vol_shifts,
                       jitter=jitter, cfg=cfg, zproj=zproj))
    return _STATE


def _phase_shifts(stack_yxt, usfac=50):
    """(row, col) shift of each frame vs the temporal mean; (T, 2) float."""
    ref_f = np.fft.fft2(stack_yxt.mean(axis=2))
    out = []
    for t in range(stack_yxt.shape[2]):
        s = dftregistration_alex(ref_f, np.fft.fft2(stack_yxt[:, :, t]), usfac)
        out.append([float(s[0]), float(s[1])])
    return np.asarray(out)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_pipeline_completes_and_shapes():
    st = _run_once()
    zproj = st["zproj"]
    # MakeSBXall returns (Nc, Y, X, Nt) float64
    assert zproj.shape == (C, Y, X, T), "zproj shape %r" % (zproj.shape,)
    assert zproj.dtype == np.float64
    assert np.isfinite(zproj).all()
    assert zproj.max() > 0


def test_output_tiff_written_correctly():
    st = _run_once()
    path = st["cfg"].zproj_tiff_path()
    assert os.path.exists(path), "projection TIFF missing: %s" % path
    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        axes, shape = series.axes, tuple(series.shape)
        arr = series.asarray()
    assert series_dtype_is_uint16(arr), "output must be uint16"
    # writer emits ImageJ hyperstack; T slowest, C fastest (Z singleton dropped)
    assert axes == "TCYX", "unexpected axes %r" % axes
    assert shape == (T, C, Y, X), "unexpected shape %r" % (shape,)
    # pixel-level: file content is exactly matlab_uint16(zproj)
    expect = matlab_uint16(st["zproj"])           # (C, Y, X, T)
    got = np.transpose(arr, (1, 2, 3, 0))         # TCYX -> CYXT
    assert np.array_equal(got, expect), "TIFF pixels differ from zproj cast"


def series_dtype_is_uint16(arr):
    return arr.dtype == np.uint16


def test_shift_file_contents():
    st = _run_once()
    sf = np.load(st["cfg"].shiftpath())
    for key, shape in (("RS", (Z, T)), ("CS", (Z, T)), ("ZS", (1, T)),
                       ("RS_chunk", (Z, T)), ("CS_chunk", (Z, T)),
                       ("ZS_chunk", (1, T))):
        assert key in sf.files, "shift file missing %r" % key
        assert sf[key].shape == shape, "%s shape %r != %r" % (
            key, sf[key].shape, shape)
    assert sf["tforms_optotune_full"].shape == (Z, 3, 3)
    # opttype='none': stored transforms are exact identities
    assert np.array_equal(sf["tforms_optotune_full"],
                          np.tile(np.eye(3), (Z, 1, 1)))
    # no z motion was injected; ZS must stay small
    assert np.abs(sf["ZS"] + sf["ZS_chunk"]).max() <= 0.5, "spurious Z shift"


def test_known_shifts_recovered():
    st = _run_once()
    sf = np.load(st["cfg"].shiftpath())
    rs_tot = sf["RS"] + sf["RS_chunk"]            # (Z, T)
    cs_tot = sf["CS"] + sf["CS_chunk"]
    est = np.stack([rs_tot.mean(axis=0), cs_tot.mean(axis=0)], axis=1)
    est_c = est - np.median(est, axis=0)
    true_c = st["vol_shifts"] - np.median(st["vol_shifts"], axis=0)
    # dftregistration returns the shift to APPLY, i.e. minus the displacement
    err = np.abs(est_c + true_c)
    assert err.max() < TOL_EST, (
        "shift-file estimate error %.2f px >= %.2f px\nest:\n%r\ntruth:\n%r"
        % (err.max(), TOL_EST, np.round(est_c, 2), true_c))


def test_stabilized_projection_residual():
    st = _run_once()
    zproj = st["zproj"]
    resid = _phase_shifts(zproj[0])               # channel 1 (refchannel)
    max_resid = np.abs(resid).max()

    # raw (unstabilized) motion of the same projection window, channel 1:
    # 'quarter' with Nz=8 -> 1-based planes 2..6 -> 0-based 1..5
    raw_proj = st["data"][:, 1:6, 0].mean(axis=1).astype(np.float64)
    raw = _phase_shifts(np.transpose(raw_proj, (1, 2, 0)))
    max_raw = np.abs(raw).max()

    assert max_raw > MIN_RAW_MOTION, (
        "synthetic raw motion %.2f px too small — test degenerate" % max_raw)
    assert max_resid < TOL_RESID, (
        "stabilized residual %.2f px >= %.2f px (raw was %.2f px)\n%r"
        % (max_resid, TOL_RESID, max_raw, np.round(resid, 2)))
    assert max_resid < (1.0 - MIN_REDUCTION) * max_raw, (
        "stabilization removed only %.0f%% of motion (%.2f -> %.2f px)"
        % (100 * (1 - max_resid / max_raw), max_raw, max_resid))


def test_second_channel_follows():
    """Channel 2 receives the same shifts; its projection must be stabilized
    to the same tolerance (it was generated at 0.6x brightness)."""
    st = _run_once()
    resid = _phase_shifts(st["zproj"][1])
    assert np.abs(resid).max() < TOL_RESID, (
        "channel-2 residual %.2f px" % np.abs(resid).max())


def test_multichunk_and_write_registered():
    """Glue paths not hit by the shared run: Nchunks > 1 (inter-chunk
    stitching + RS_chunk 'nearest' stretch) and write_registered=True
    (PASS2 -> io_rw.RegWriter '.sbxall' byte stream).

    chunksize=4, T=7 -> Nchunks = round(7/4) = 2, t_chunk = floor(7/2) = 3,
    so 6 volumes are registered/applied (the 7th is dropped, as in MATLAB).
    """
    st = _run_once()
    import shutil
    tmp = tempfile.mkdtemp(prefix="cpstab_test_mc_")
    try:
        raw = os.path.join(tmp, "synth.tif")
        shutil.copy(st["cfg"].input_path, raw)
        cfg = RegistrationConfig(input_path=raw, refchannel=1, scale=2,
                                 chunksize=4, write_registered=True,
                                 out_dir=os.path.join(tmp, "out"))
        zproj = run_pipeline(cfg)
        n_reg = 6
        assert zproj.shape == (C, Y, X, n_reg), zproj.shape
        sf = np.load(cfg.shiftpath())
        assert sf["RS"].shape == (Z, n_reg)
        assert sf["RS_chunk"].shape == (Z, n_reg)
        # registered stack: C*Y*X*Z*T_reg uint16 records
        sbxall = cfg.registered_stack_path()
        assert os.path.exists(sbxall), "missing %s" % sbxall
        expect_bytes = C * Y * X * Z * n_reg * 2
        got = os.path.getsize(sbxall)
        assert got == expect_bytes, "sbxall %d bytes != %d" % (got, expect_bytes)
        # stabilization quality on the multi-chunk path, too
        resid = _phase_shifts(zproj[0])
        assert np.abs(resid).max() < TOL_RESID, (
            "multi-chunk residual %.2f px" % np.abs(resid).max())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS  %s" % name)
        except AssertionError as e:
            failed += 1
            print("FAIL  %s: %s" % (name, e))
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("ERROR %s: %r" % (name, e))
    print("\n%d/%d passed" % (len(tests) - failed, len(tests)))
    sys.exit(1 if failed else 0)
