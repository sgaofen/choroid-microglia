"""Adversarial verification of dftreg.py vs literal MATLAB transliteration.

The transliteration below re-implements dftregistrationAlex.m /
dftregistration3D.m mechanically with 1-BASED indexing (converted at the
last moment), MATLAB colon ranges, MATLAB round/fix semantics via
independent formulas (sign*floor(abs+.5), np.trunc), and MATLAB find()
semantics (including the no-'first' all-ties vector case at L84).
Then fuzz both against the port across sizes/usfac regimes and edge cases.
"""
import os
import sys
import numpy as np
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))   # pyport/
from cpstab.dftreg import (dftregistration_alex, dftregistration3d,
                           dftups, ftpad, ftpad3)
from cpstab.dftreg import _matlab_round

rng = np.random.default_rng(20260824)
FAIL = []

def check(name, cond, detail=""):
    if not cond:
        FAIL.append((name, detail))
        print("FAIL:", name, detail)

# ----- independent MATLAB semantics helpers ------------------------------
def m_round(x):
    # MATLAB round(): half away from zero. Independent formula.
    return float(np.sign(x) * np.floor(np.abs(x) + 0.5))

def m_colon(a, b):
    # MATLAB a:b inclusive (integer step 1)
    return np.arange(a, b + 1)

def m_find_first_colmajor(mask):
    lin = np.flatnonzero(mask.ravel(order="F"))
    return int(lin[0])  # 0-based linear (column-major)

def m_ftpad2(imFT, outsize):
    Nout = [int(v) for v in outsize]
    Nin = list(imFT.shape)
    im = np.fft.fftshift(imFT)
    out = np.zeros(Nout, dtype=complex)
    center = [n // 2 + 1 for n in Nin]        # floor(size/2)+1, 1-based
    centerout = [n // 2 + 1 for n in Nout]
    cc = [co - ci for co, ci in zip(centerout, center)]
    d = [(max(cc[k] + 1, 1), min(cc[k] + Nin[k], Nout[k])) for k in range(2)]
    s = [(max(-cc[k] + 1, 1), min(-cc[k] + Nout[k], Nin[k])) for k in range(2)]
    out[d[0][0]-1:d[0][1], d[1][0]-1:d[1][1]] = im[s[0][0]-1:s[0][1], s[1][0]-1:s[1][1]]
    return np.fft.ifftshift(out) * Nout[0] * Nout[1] / (Nin[0] * Nin[1])

def m_ftpad3(imFT, outsize):
    Nout = [int(v) for v in outsize]
    Nin = list(imFT.shape)
    im = np.fft.fftshift(imFT)
    out = np.zeros(Nout, dtype=complex)
    center = [n // 2 + 1 for n in Nin]
    centerout = [n // 2 + 1 for n in Nout]
    cc = [co - ci for co, ci in zip(centerout, center)]
    d = [(max(cc[k] + 1, 1), min(cc[k] + Nin[k], Nout[k])) for k in range(3)]
    s = [(max(-cc[k] + 1, 1), min(-cc[k] + Nout[k], Nin[k])) for k in range(3)]
    out[d[0][0]-1:d[0][1], d[1][0]-1:d[1][1], d[2][0]-1:d[2][1]] = \
        im[s[0][0]-1:s[0][1], s[1][0]-1:s[1][1], s[2][0]-1:s[2][1]]
    return (np.fft.ifftshift(out)
            * Nout[0] * Nout[1] * Nout[2] / (Nin[0] * Nin[1] * Nin[2]))

def m_dftups(in_, nor, noc, usfac, roff, coff):
    nr, nc = in_.shape
    kernc = np.exp((-1j * 2 * np.pi / (nc * usfac))
                   * np.outer(np.fft.ifftshift(np.arange(nc)) - np.floor(nc / 2),
                              np.arange(noc) - coff))
    kernr = np.exp((-1j * 2 * np.pi / (nr * usfac))
                   * np.outer(np.arange(nor) - roff,
                              np.fft.ifftshift(np.arange(nr)) - np.floor(nr / 2)))
    return kernr @ in_ @ kernc

def m_dftreg_alex(buf1ft, buf2ft, usfac):
    """Literal transliteration; returns the FULL MATLAB output vector
    (may exceed 2 elements in the usfac==1 tie case)."""
    nr, nc = buf2ft.shape
    Nr = np.fft.ifftshift(m_colon(-np.fix(nr / 2), np.ceil(nr / 2) - 1))
    Nc = np.fft.ifftshift(m_colon(-np.fix(nc / 2), np.ceil(nc / 2) - 1))
    if usfac == 0:
        row_shift = np.array([0.0]); col_shift = np.array([0.0])
    elif usfac == 1:
        CC = np.fft.ifft2(buf1ft * np.conj(buf2ft))
        CCabs = np.abs(CC)
        mask = (CCabs == CCabs.max())
        lin = np.flatnonzero(mask.ravel(order="F"))          # all ties, colmajor
        rows = lin % nr + 1                                   # 1-based
        cols = lin // nr + 1
        row_shift = Nr[rows - 1].astype(float)
        col_shift = Nc[cols - 1].astype(float)
    elif usfac > 1:
        CC = np.fft.ifft2(m_ftpad2(buf1ft * np.conj(buf2ft), [2 * nr, 2 * nc]))
        CCabs = np.abs(CC)
        lin = m_find_first_colmajor(CCabs == CCabs.max())
        r1 = lin % (2 * nr) + 1; c1 = lin // (2 * nr) + 1     # 1-based subs
        Nr2 = np.fft.ifftshift(m_colon(-np.fix(nr), np.ceil(nr) - 1))
        Nc2 = np.fft.ifftshift(m_colon(-np.fix(nc), np.ceil(nc) - 1))
        row_shift = np.array([Nr2[r1 - 1] / 2.0])
        col_shift = np.array([Nc2[c1 - 1] / 2.0])
        if usfac > 2:
            row_shift = np.array([m_round(row_shift[0] * usfac) / usfac])
            col_shift = np.array([m_round(col_shift[0] * usfac) / usfac])
            dftshift = float(np.fix(np.ceil(usfac * 1.5) / 2))
            nl = int(np.ceil(usfac * 1.5))
            CC = np.conj(m_dftups(buf2ft * np.conj(buf1ft), nl, nl, usfac,
                                  dftshift - row_shift[0] * usfac,
                                  dftshift - col_shift[0] * usfac))
            CCabs = np.abs(CC)
            lin = m_find_first_colmajor(CCabs == CCabs.max())
            rloc = lin % nl + 1; cloc = lin // nl + 1          # 1-based
            rloc = rloc - dftshift - 1
            cloc = cloc - dftshift - 1
            row_shift = np.array([row_shift[0] + rloc / usfac])
            col_shift = np.array([col_shift[0] + cloc / usfac])
        if nr == 1:
            row_shift = np.array([0.0])
        if nc == 1:
            col_shift = np.array([0.0])
    else:
        raise RuntimeError("MATLAB: undefined row_shift")
    return np.concatenate([np.atleast_1d(row_shift), np.atleast_1d(col_shift)])

def m_dftreg_3d(buf1ft, buf2ft, usfac):
    nr, nc, npl = buf2ft.shape
    if npl == 1:
        raise RuntimeError("MATLAB errors here: size() drops trailing "
                           "singleton -> centerout(1x3) - center(1x2) mismatch")
    CC = np.fft.ifftn(m_ftpad3(buf1ft * np.conj(buf2ft),
                               [usfac * nr, usfac * nc, usfac * npl]))
    CCabs = np.abs(CC)
    lin = int(np.flatnonzero((CCabs == CCabs.max()).ravel(order="F"))[0])
    # ind2sub, 1-based
    # int(): the products are exact integers whenever MATLAB's zeros() runs
    # (holds for fractional usfac too, e.g. 1.5 on even dims)
    sr = int(usfac * nr); sc = int(usfac * nc)
    r = lin % sr + 1
    c = (lin // sr) % sc + 1
    p = lin // (sr * sc) + 1
    Nr2 = np.fft.ifftshift(m_colon(-np.fix(nr * usfac / 2), np.ceil(nr * usfac / 2) - 1))
    Nc2 = np.fft.ifftshift(m_colon(-np.fix(nc * usfac / 2), np.ceil(nc * usfac / 2) - 1))
    Np2 = np.fft.ifftshift(m_colon(-np.fix(npl * usfac / 2), np.ceil(npl * usfac / 2) - 1))
    row_shift = Nr2[r - 1] / usfac
    col_shift = Nc2[c - 1] / usfac
    pl_shift = Np2[p - 1] / usfac
    if nr == 1: row_shift = 0.0
    if nc == 1: col_shift = 0.0
    return np.array([row_shift, col_shift, pl_shift], dtype=float)

# ======================= 1. matlab_round exact check ======================
for u in (3, 5, 7, 9, 15, 4, 8, 20):
    for k in range(-41, 42):
        x = (k / 2.0) * u          # value row_shift*usfac actually takes
        exact = Fraction(x).limit_denominator(2) * 1
        f = Fraction(x)
        # exact half-away-from-zero on the rational value
        n = f.numerator; d = f.denominator
        import math
        if d == 1:
            want = float(n)
        else:
            want = float(math.floor(abs(f) + Fraction(1, 2)) * (1 if n >= 0 else -1))
        check("matlab_round u=%d k=%d" % (u, k), _matlab_round(x) == want,
              "got %r want %r x=%r" % (_matlab_round(x), want, x))

# ======================= 2. FTpad 2D/3D fuzz ==============================
sizes2 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
for tr in range(400):
    ni = (int(rng.choice(sizes2)), int(rng.choice(sizes2)))
    no = (int(rng.choice(sizes2 + [10, 12, 16])), int(rng.choice(sizes2 + [10, 12, 16])))
    a = rng.standard_normal(ni) + 1j * rng.standard_normal(ni)
    got = ftpad(a, no)
    want = m_ftpad2(a, no)
    check("ftpad %s->%s" % (ni, no), np.array_equal(got, want),
          "maxdiff %g" % np.max(np.abs(got - want)))

sizes3 = [1, 2, 3, 4, 5, 6]
for tr in range(200):
    ni = tuple(int(rng.choice(sizes3)) for _ in range(3))
    no = tuple(int(rng.choice(sizes3 + [8, 9])) for _ in range(3))
    a = rng.standard_normal(ni) + 1j * rng.standard_normal(ni)
    got = ftpad3(a, no)
    want = m_ftpad3(a, no)
    check("ftpad3 %s->%s" % (ni, no), np.array_equal(got, want),
          "maxdiff %g" % np.max(np.abs(got - want)))

# ======================= 3. dftups fuzz ===================================
for tr in range(300):
    nr = int(rng.integers(1, 9)); nc = int(rng.integers(1, 9))
    u = int(rng.choice([1, 2, 3, 4, 5, 10]))
    nor = int(rng.integers(1, 3 * u + 2)); noc = int(rng.integers(1, 3 * u + 2))
    roff = float(rng.integers(-5, 6)) + float(rng.choice([0.0, 0.5, 0.25]))
    coff = float(rng.integers(-5, 6)) + float(rng.choice([0.0, 0.5, 0.25]))
    a = rng.standard_normal((nr, nc)) + 1j * rng.standard_normal((nr, nc))
    with np.errstate(all="ignore"):
        got = dftups(a, nor, noc, u, roff, coff)
        want = m_dftups(a, nor, noc, u, roff, coff)
    check("dftups nr%d nc%d u%d" % (nr, nc, u), np.array_equal(got, want),
          "maxdiff %g" % np.max(np.abs(got - want)))

# ======================= 4. dftregistration_alex fuzz =====================
def rand_img(nr, nc, smooth=False):
    im = rng.standard_normal((nr, nc))
    if smooth and nr > 4 and nc > 4:
        f = np.fft.fft2(im)
        ky = np.abs(np.fft.fftfreq(nr)); kx = np.abs(np.fft.fftfreq(nc))
        f *= np.exp(-8 * (ky[:, None] ** 2 + kx[None, :] ** 2))
        im = np.real(np.fft.ifft2(f))
    return im

shapes = [(1, 1), (1, 7), (7, 1), (2, 2), (3, 3), (4, 4), (4, 6), (5, 7),
          (6, 5), (8, 8), (7, 7), (16, 12), (13, 17)]
usfacs = [0, 1, 2, 3, 4, 5, 10, 1.5, 2.5]
nmismatch = 0
for tr in range(600):
    nr, nc = shapes[tr % len(shapes)]
    u = usfacs[(tr // len(shapes)) % len(usfacs)]
    a = rand_img(nr, nc, smooth=(tr % 2 == 0))
    dy = int(rng.integers(0, nr)); dx = int(rng.integers(0, nc))
    b = np.roll(a, (dy, dx), axis=(0, 1)) + 0.01 * rng.standard_normal((nr, nc))
    A = np.fft.fft2(a); B = np.fft.fft2(b)
    with np.errstate(all="ignore"):
        got = dftregistration_alex(A, B, u)
        want = m_dftreg_alex(A, B, u)
    ok = (len(want) == 2 and np.array_equal(got, want))
    if not ok:
        nmismatch += 1
        if nmismatch < 8:
            print("MISMATCH alex nr=%d nc=%d u=%r got=%s want=%s" %
                  (nr, nc, u, got, want))
check("dftreg_alex fuzz", nmismatch == 0, "%d mismatches" % nmismatch)

# exact .5 rounding regime: force initial 2x estimate at half-integer, odd usfac
for u in (3, 5, 7):
    for nr, nc in [(8, 8), (9, 7), (12, 10)]:
        a = rand_img(nr, nc, smooth=True)
        # shift by half-pixels via Fourier phase ramp
        for dy2, dx2 in [(0.5, -1.5), (-2.5, 0.5), (1.5, 1.5), (-0.5, -0.5)]:
            ky = np.fft.fftfreq(nr)[:, None]; kx = np.fft.fftfreq(nc)[None, :]
            B = np.fft.fft2(a) * np.exp(-2j * np.pi * (ky * dy2 + kx * dx2))
            A = np.fft.fft2(a)
            with np.errstate(all="ignore"):
                got = dftregistration_alex(A, B, u)
                want = m_dftreg_alex(A, B, u)
            check("halfpix u=%d %s" % (u, (dy2, dx2)),
                  np.array_equal(got, want), "got %s want %s" % (got, want))

# constant image (all ties): usfac=1 -- MATLAB vector output vs port
c = np.full((4, 4), 3.7)
Cf = np.fft.fft2(c)
want = m_dftreg_alex(Cf, Cf.copy(), 1)
got = dftregistration_alex(Cf, Cf.copy(), 1)
print("tie case usfac=1: MATLAB output vector =", want,
      " caller sees S(1)=%g S(2)=%g" % (want[0], want[1]),
      "| port =", got)

# constant image usfac=4 (documented -0.75 case)
with np.errstate(all="ignore"):
    got = dftregistration_alex(Cf, Cf.copy(), 4)
    want = m_dftreg_alex(Cf, Cf.copy(), 4)
check("tie usfac=4", np.array_equal(got, want), "got %s want %s" % (got, want))

# sign convention, integer shifts, all usfac
for u in (1, 2, 4, 10):
    a = rand_img(16, 12, smooth=True)
    mv = np.roll(a, (3, -2), axis=(0, 1))
    with np.errstate(all="ignore"):
        out = dftregistration_alex(np.fft.fft2(a), np.fft.fft2(mv), u)
    check("signconv u=%d" % u, np.allclose(out, [-3, 2]), str(out))

# subpixel recovery accuracy usfac=20
a = rand_img(32, 24, smooth=True)
ky = np.fft.fftfreq(32)[:, None]; kx = np.fft.fftfreq(24)[None, :]
B = np.fft.fft2(a) * np.exp(-2j * np.pi * (ky * 1.3 + kx * (-0.7)))
out = dftregistration_alex(np.fft.fft2(a), B, 20)
check("subpix20", np.max(np.abs(out - [-1.3, 0.7])) <= 1.0 / 20 + 1e-9, str(out))

# ======================= 5. dftregistration3d fuzz ========================
shapes3 = [(4, 4, 4), (5, 4, 3), (3, 5, 2), (6, 6, 5), (1, 5, 4), (5, 1, 4),
           (2, 2, 2), (7, 6, 3)]
nm3 = 0
for tr in range(240):
    nr, nc, npl = shapes3[tr % len(shapes3)]
    u = [1, 2, 3, 4][(tr // len(shapes3)) % 4]
    a = rng.standard_normal((nr, nc, npl))
    dz = int(rng.integers(0, npl))
    b = np.roll(a, (int(rng.integers(0, nr)), int(rng.integers(0, nc)), dz),
                axis=(0, 1, 2)) + 0.01 * rng.standard_normal((nr, nc, npl))
    A = np.fft.fftn(a); B = np.fft.fftn(b)
    got = dftregistration3d(A, B, u)
    want = m_dftreg_3d(A, B, u)
    if not np.array_equal(got, want):
        nm3 += 1
        if nm3 < 8:
            print("MISMATCH 3d %s u=%d got=%s want=%s" % ((nr, nc, npl), u, got, want))
check("dftreg3d fuzz", nm3 == 0, "%d mismatches" % nm3)

# 3D sign convention
a = rng.standard_normal((8, 8, 6))
mv = np.roll(a, (2, -1, 1), axis=(0, 1, 2))
out = dftregistration3d(np.fft.fftn(a), np.fft.fftn(mv), 2)
check("signconv3d", np.allclose(out, [-2, 1, -1]), str(out))

# np==1 plane: MATLAB ERRORS (size() drops the trailing singleton,
# cenout_cen is a 1x3 - 1x2 mismatch). Post-review fix: the port now raises
# ValueError instead of silently returning a value (finding F1).
a = rng.standard_normal((6, 6, 1))
b = np.roll(a, (1, 2, 0), axis=(0, 1, 2))
try:
    got = dftregistration3d(np.fft.fftn(a), np.fft.fftn(b), 2)
except ValueError as e:
    check("np_==1 raises (F1 fix)", True, str(e)[:60])
else:
    check("np_==1 raises (F1 fix)", False, "silently returned %s" % (got,))

print("\n%d FAILURES" % len(FAIL))
for name, det in FAIL[:20]:
    print(" -", name, det)
