"""Self-tests for cpstab.dftreg (port of dftregistrationAlex.m / dftregistration3D.m).

Runs standalone (``python test_dftreg.py``) or under pytest. Ground truth is
synthetic: known integer shifts via np.roll (circular, exact) and known
subpixel shifts via exact Fourier phase ramps (circular, band-limited).

Sign convention being verified: moving = roll(reference, (+dy, +dx)) ==>
returned shift == (-dy, -dx) (the translation to apply to the moving image).
"""

import os
import sys

import numpy as np

# Import through the PACKAGE, not as a flat module off cpstab/. dftreg.py now
# carries a relative import (`from .precision import ...` -- the float32 fast
# mode's precision boundary), so a flat `import dftreg` raises
# "attempted relative import with no known parent package". This matches what
# test_synthetic.py / test_tmajor.py already do.
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from cpstab.dftreg import (dftregistration_alex, dftregistration3d, dftups,
                           ftpad, ftpad3, _argmax_first_colmajor,
                           _matlab_round)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _smooth_image(shape, seed):
    """Random image, gaussian-smoothed so the correlation peak is unambiguous."""
    rng = np.random.RandomState(seed)
    img = rng.rand(*shape)
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(img, 2.0).astype(np.float64)


def _phase_ramp_shift_ft(ft, shifts):
    """Exact circular subpixel shift in the Fourier domain.

    Returns the FFT of g where g(n) = f(n - d) (circular, band-limited):
    G(k) = F(k) * exp(-2i*pi*sum_axis(k_axis * d_axis / N_axis)).
    """
    out = np.array(ft, dtype=np.complex128, copy=True)
    for ax, d in enumerate(shifts):
        n = ft.shape[ax]
        k = np.fft.ifftshift(np.arange(-(n // 2), (n + 1) // 2))
        shape = [1] * ft.ndim
        shape[ax] = n
        out = out * np.exp(-2j * np.pi * k.reshape(shape) * d / n)
    return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def test_matlab_round():
    cases = [(0.5, 1.0), (-0.5, -1.0), (1.5, 2.0), (-1.5, -2.0),
             (2.5, 3.0), (-2.5, -3.0), (0.49, 0.0), (-0.49, 0.0),
             (3.0, 3.0), (-3.0, -3.0), (0.0, 0.0)]
    for x, want in cases:
        got = _matlab_round(x)
        assert got == want, "round(%r): got %r want %r" % (x, got, want)


def test_argmax_colmajor_tiebreak():
    a = np.array([[0.0, 5.0],
                  [5.0, 0.0]])
    # column-major scan: (0,0),(1,0),(0,1),(1,1) -> first max at (1,0).
    # (C-order argmax would give (0,1).)
    assert _argmax_first_colmajor(a) == (1, 0)
    b = np.zeros((3, 4, 5))
    b[2, 1, 3] = 7.0
    assert _argmax_first_colmajor(b) == (2, 1, 3)


# ---------------------------------------------------------------------------
# FTpad
# ---------------------------------------------------------------------------

def test_ftpad_roundtrip():
    rng = np.random.RandomState(0)
    for shape in [(6, 8), (7, 9), (6, 9)]:  # even/odd mixes
        x = rng.randn(*shape) + 1j * rng.randn(*shape)
        big = ftpad(x, (2 * shape[0], 2 * shape[1] + 1))
        back = ftpad(big, shape)
        assert np.allclose(back, x, atol=1e-12), "ftpad roundtrip %s" % (shape,)


def test_ftpad_dirichlet_samples():
    # 2x zero-pad interpolation must reproduce the original samples exactly
    # at the original grid points (even sizes, unsplit Nyquist included).
    f = _smooth_image((16, 24), seed=3)
    F2 = ftpad(np.fft.fft2(f), (32, 48))
    f2 = np.fft.ifft2(F2)
    assert np.allclose(f2[::2, ::2].real, f, atol=1e-10)
    assert np.max(np.abs(f2[::2, ::2].imag)) < 1e-10


def test_ftpad_rejects_non2d():
    try:
        ftpad(np.zeros((2, 2, 2), dtype=complex), (4, 4))
    except ValueError:
        pass
    else:
        raise AssertionError("ftpad accepted a 3D array")


def test_ftpad3_roundtrip():
    rng = np.random.RandomState(1)
    for shape in [(4, 6, 5), (5, 7, 3)]:
        x = rng.randn(*shape) + 1j * rng.randn(*shape)
        big = ftpad3(x, tuple(2 * s for s in shape))
        back = ftpad3(big, shape)
        assert np.allclose(back, x, atol=1e-12), "ftpad3 roundtrip %s" % (shape,)


# ---------------------------------------------------------------------------
# dftups vs brute-force zero-padded FFT
# ---------------------------------------------------------------------------

def test_dftups_bruteforce():
    rng = np.random.RandomState(7)
    nr, nc = 8, 6
    x = rng.randn(nr, nc) + 1j * rng.randn(nr, nc)
    for usfac, nor, noc, roff, coff in [(3, 7, 5, 4, 2),
                                        (4, 6, 6, -3, 9),
                                        (2, 5, 4, 0, 0)]:
        NR, NC = usfac * nr, usfac * nc
        # centered embedding == FTpad without its amplitude scale
        big = ftpad(x, (NR, NC)) * (nr * nc) / (NR * NC)
        BF = np.fft.fft2(big)
        expect = np.empty((nor, noc), dtype=np.complex128)
        for p in range(nor):
            for q in range(noc):
                expect[p, q] = BF[(p - roff) % NR, (q - coff) % NC]
        got = dftups(x, nor, noc, usfac, roff, coff)
        err = np.max(np.abs(got - expect)) / np.max(np.abs(expect))
        assert err < 1e-10, "dftups mismatch (usfac=%d): rel err %.3g" % (usfac, err)


def test_dftups_defaults():
    rng = np.random.RandomState(8)
    x = rng.randn(5, 6) + 1j * rng.randn(5, 6)
    # all defaults: nor=nr, noc=nc, usfac=1, roff=coff=0
    assert dftups(x).shape == (5, 6)
    assert np.allclose(dftups(x), dftups(x, 5, 6, 1, 0, 0), atol=1e-12)


# ---------------------------------------------------------------------------
# 2D registration: integer shifts (np.roll ground truth)
# ---------------------------------------------------------------------------

def test_2d_integer_shifts():
    f = _smooth_image((96, 128), seed=42)   # non-square to catch axis swaps
    F = np.fft.fft2(f)
    for (dy, dx) in [(0, 0), (5, -3), (-7, 11), (2, 9), (-1, -1)]:
        g = np.roll(f, (dy, dx), axis=(0, 1))   # g(n) = f(n - d)
        G = np.fft.fft2(g)
        for usfac in [1, 2, 4, 100]:
            out = dftregistration_alex(F, G, usfac)
            assert out.shape == (2,)
            got = (out[0], out[1])
            want = (-float(dy), -float(dx))
            assert abs(got[0] - want[0]) < 1e-9 and abs(got[1] - want[1]) < 1e-9, \
                "2D integer shift (%d,%d) usfac=%d: got %s want %s" % (
                    dy, dx, usfac, got, want)


def test_2d_odd_size_integer_shifts():
    f = _smooth_image((65, 97), seed=5)
    F = np.fft.fft2(f)
    for (dy, dx) in [(4, -6), (-9, 2)]:
        G = np.fft.fft2(np.roll(f, (dy, dx), axis=(0, 1)))
        for usfac in [1, 2, 4, 100]:
            out = dftregistration_alex(F, G, usfac)
            assert abs(out[0] + dy) < 1e-9 and abs(out[1] + dx) < 1e-9, \
                "odd-size shift (%d,%d) usfac=%d: got %s" % (dy, dx, usfac, out)


# ---------------------------------------------------------------------------
# 2D registration: subpixel shifts (Fourier phase-ramp ground truth)
# ---------------------------------------------------------------------------

def test_2d_subpixel_shifts():
    f = _smooth_image((96, 128), seed=13)
    F = np.fft.fft2(f)
    cases = [
        # (dy, dx), usfac, tolerance
        ((3.5, -2.5), 2, 1e-9),        # on the half-pixel grid: exact
        ((1.25, -0.75), 4, 1e-9),      # on the quarter-pixel grid: exact
        ((3.37, -1.83), 100, 0.5 / 100 + 1e-9),   # nearest 1/100 grid point
        ((-4.62, 2.14), 100, 0.5 / 100 + 1e-9),
        ((0.3, -0.4), 2, 0.25 + 1e-9),            # off-grid, usfac=2
        ((3.37, -1.83), 1, 0.5 + 1e-9),           # integer regime
    ]
    for (dy, dx), usfac, tol in cases:
        G = _phase_ramp_shift_ft(F, (dy, dx))
        out = dftregistration_alex(F, G, usfac)
        err = max(abs(out[0] + dy), abs(out[1] + dx))
        assert err <= tol, \
            "2D subpixel (%.3f,%.3f) usfac=%d: got %s, err %.4g > tol %.4g" % (
                dy, dx, usfac, out, err, tol)


def test_2d_usfac_odd_refine():
    # odd usfac exercises the half-away-from-zero round at L102-L103
    f = _smooth_image((64, 64), seed=21)
    F = np.fft.fft2(f)
    for (dy, dx) in [(2.4, -1.8), (-0.6, 3.2)]:
        G = _phase_ramp_shift_ft(F, (dy, dx))
        out = dftregistration_alex(F, G, 5)
        err = max(abs(out[0] + dy), abs(out[1] + dx))
        assert err <= 0.5 / 5 + 1e-9, "usfac=5 err %.4g (shift %s)" % (err, (dy, dx))


def test_2d_usfac0_and_default():
    f = _smooth_image((32, 32), seed=2)
    F = np.fft.fft2(f)
    G = np.fft.fft2(np.roll(f, (3, 4), axis=(0, 1)))
    assert np.all(dftregistration_alex(F, G, 0) == 0.0)
    # default usfac=1 == explicit usfac=1
    assert np.allclose(dftregistration_alex(F, G),
                       dftregistration_alex(F, G, 1), atol=0)


def test_2d_singleton_row():
    # nr == 1: row shift is forced to 0 in the usfac > 1 regimes
    rng = np.random.RandomState(9)
    from scipy.ndimage import gaussian_filter1d
    f = gaussian_filter1d(rng.rand(64), 2.0).reshape(1, 64)
    F = np.fft.fft2(f)
    G = np.fft.fft2(np.roll(f, 3, axis=1))
    for usfac in [2, 4, 100]:
        out = dftregistration_alex(F, G, usfac)
        assert out[0] == 0.0, "usfac=%d row_shift %r != 0" % (usfac, out[0])
        assert abs(out[1] + 3) < 1e-9, "usfac=%d col_shift %r" % (usfac, out[1])


def test_2d_constant_image_tie():
    # Constant image -> CC magnitude ties everywhere -> first col-major max
    # is index (0,0)/(1,1 in MATLAB) -> zero shift for usfac 1 and 2.
    # For usfac > 2 MATLAB genuinely returns a NONZERO shift: the dftups
    # window is all-ties too (verified std == 0), find-first lands on its
    # (1,1) corner, rloc = 1 - dftshift - 1 = -dftshift, so
    # shift = -dftshift/usfac = -fix(ceil(1.5*usfac)/2)/usfac = -0.75 for
    # usfac=4. The port must reproduce that degenerate behavior exactly.
    f = np.ones((16, 20))
    F = np.fft.fft2(f)
    for usfac in [1, 2]:
        out = dftregistration_alex(F, F, usfac)
        assert np.all(out == 0.0), "constant image usfac=%d: %s" % (usfac, out)
    out = dftregistration_alex(F, F, 4)
    assert np.all(out == -0.75), "constant image usfac=4: %s" % (out,)


def test_2d_bad_usfac_raises():
    F = np.fft.fft2(np.ones((8, 8)))
    try:
        dftregistration_alex(F, F, 0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("usfac=0.5 did not raise")


# ---------------------------------------------------------------------------
# optional cross-check against skimage (same algorithm)
# ---------------------------------------------------------------------------

def test_2d_skimage_crosscheck():
    try:
        from skimage.registration import phase_cross_correlation
    except ImportError:
        print("  [skip] skimage not available")
        return
    f = _smooth_image((96, 128), seed=31)
    F = np.fft.fft2(f)
    for (dy, dx) in [(3.37, -1.83), (-2.6, 0.45)]:
        G = _phase_ramp_shift_ft(F, (dy, dx))
        ours = dftregistration_alex(F, G, 100)
        try:
            sk = phase_cross_correlation(F, G, upsample_factor=100,
                                         space="fourier", normalization=None)
        except TypeError:   # older skimage: no `normalization` kwarg
            sk = phase_cross_correlation(F, G, upsample_factor=100,
                                         space="fourier")
        sk_shift = np.asarray(sk[0] if isinstance(sk, tuple) else sk, dtype=float)
        assert np.allclose(ours, sk_shift, atol=1.5 / 100), \
            "skimage disagreement: ours %s vs skimage %s" % (ours, sk_shift)


# ---------------------------------------------------------------------------
# 3D registration
# ---------------------------------------------------------------------------

def _smooth_volume(shape, seed):
    rng = np.random.RandomState(seed)
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(rng.rand(*shape), 1.5).astype(np.float64)


def test_3d_integer_shifts():
    v = _smooth_volume((32, 48, 12), seed=17)   # [Y, X, Z], all different
    V = np.fft.fftn(v)
    for (dy, dx, dz) in [(0, 0, 0), (2, -3, 1), (-4, 5, -2), (1, 1, 3)]:
        W = np.fft.fftn(np.roll(v, (dy, dx, dz), axis=(0, 1, 2)))
        for usfac in [1, 2]:
            out = dftregistration3d(V, W, usfac)
            assert out.shape == (3,)
            want = (-float(dy), -float(dx), -float(dz))
            assert np.allclose(out, want, atol=1e-9), \
                "3D integer (%d,%d,%d) usfac=%d: got %s" % (dy, dx, dz, usfac, out)


def test_3d_halfvoxel_shifts():
    v = _smooth_volume((32, 48, 12), seed=23)
    V = np.fft.fftn(v)
    for (dy, dx, dz) in [(1.5, -2.5, 0.5), (-0.5, 3.5, -1.5)]:
        W = _phase_ramp_shift_ft(V, (dy, dx, dz))
        out = dftregistration3d(V, W, 2)
        want = np.array([-dy, -dx, -dz])
        assert np.allclose(out, want, atol=1e-9), \
            "3D half-voxel (%s): got %s want %s" % ((dy, dx, dz), out, want)


def test_3d_offgrid_shifts():
    # usfac=2 quantizes to the half-voxel grid: error <= 0.25 + estimator eps
    v = _smooth_volume((32, 48, 16), seed=29)
    V = np.fft.fftn(v)
    for (dy, dx, dz) in [(1.3, -2.7, 0.6), (-0.2, 0.9, -1.1)]:
        W = _phase_ramp_shift_ft(V, (dy, dx, dz))
        out = dftregistration3d(V, W, 2)
        err = np.max(np.abs(out - np.array([-dy, -dx, -dz])))
        assert err <= 0.25 + 1e-9, \
            "3D off-grid (%s): got %s, err %.4g" % ((dy, dx, dz), out, err)


def test_3d_odd_dims():
    v = _smooth_volume((31, 47, 11), seed=37)
    V = np.fft.fftn(v)
    W = np.fft.fftn(np.roll(v, (3, -5, 2), axis=(0, 1, 2)))
    out = dftregistration3d(V, W, 2)
    assert np.allclose(out, [-3.0, 5.0, -2.0], atol=1e-9), "odd dims: %s" % (out,)


def test_3d_single_plane_raises():
    # np_ == 1: MATLAB errors inside FTpad (size() drops the trailing
    # singleton -> `centerout - center` is 1x3 minus 1x2); the port must
    # raise, not return a fabricated pl_shift (PORTING NOTES 8).
    v = _smooth_volume((6, 6, 1), seed=41)
    V = np.fft.fftn(v)
    W = np.fft.fftn(np.roll(v, (1, 2, 0), axis=(0, 1, 2)))
    try:
        out = dftregistration3d(V, W, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("np_==1 did not raise (returned %s)" % (out,))


def test_3d_fractional_usfac():
    # MATLAB has NO usfac validation: any usfac > 0 whose products
    # usfac*[nr nc np] are all integers runs (zeros() succeeds). usfac=1.5
    # on all-even dims must therefore be accepted (PORTING NOTES 6) and
    # recover shifts on the 1/1.5 grid.
    v = _smooth_volume((16, 24, 8), seed=43)
    V = np.fft.fftn(v)
    W = np.fft.fftn(np.roll(v, (2, -4, 2), axis=(0, 1, 2)))
    out = dftregistration3d(V, W, 1.5)
    assert np.allclose(out, [-2.0, 4.0, -2.0], atol=1e-9), \
        "usfac=1.5 even dims: %s" % (out,)
    # non-integer product (odd dim * 1.5) -> MATLAB zeros() errors -> raise
    v = _smooth_volume((5, 6, 4), seed=44)
    V = np.fft.fftn(v)
    try:
        out = dftregistration3d(V, V, 1.5)
    except ValueError:
        pass
    else:
        raise AssertionError("usfac=1.5 on odd dim did not raise (%s)" % (out,))


def test_nan_input_raises():
    # MATLAB: max() skips NaN and ==NaN is all-false, so find() comes back
    # EMPTY and MATLAB errors/propagates empties; the port raises instead of
    # silently registering to a NaN peak (PORTING NOTES 12).
    img = _smooth_image((8, 8), seed=47)
    bad = img.copy()
    bad[3, 4] = np.nan
    F = np.fft.fft2(img)
    G = np.fft.fft2(bad)
    for usfac in [1, 2, 4]:
        try:
            with np.errstate(invalid="ignore"):
                out = dftregistration_alex(F, G, usfac)
        except ValueError:
            pass
        else:
            raise AssertionError("NaN usfac=%d did not raise (%s)" % (usfac, out))
    # usfac == 0 never inspects the data: stays [0, 0] exactly like MATLAB
    assert np.all(dftregistration_alex(F, G, 0) == 0.0)
    vol = _smooth_volume((8, 8, 4), seed=48)
    vol[1, 1, 1] = np.nan
    Vb = np.fft.fftn(vol)
    try:
        with np.errstate(invalid="ignore"):
            out = dftregistration3d(Vb, Vb, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("NaN 3D did not raise (%s)" % (out,))


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
