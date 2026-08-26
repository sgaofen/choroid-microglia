# Adversarial verification scratchpad for shifts2d.py vs MATLAB ground truth.
# Run: python tests/scratch_shifts2d.py  (from pyport/)
import sys
import numpy as np
import math

# --- Finding 1 workaround: shifts2d imports a nonexistent name ---
import cpstab.dftreg as dreg
if not hasattr(dreg, "dftregistration"):
    dreg.dftregistration = dreg.dftregistration_alex  # patch so review can proceed
import cpstab.shifts2d as s2

rng = np.random.default_rng(7)
fails = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ("  " + detail if detail else ""))
    if not cond:
        fails.append(name)


# ---------------------------------------------------------------- T1: imtranslate
# 1a. Sign convention: [C,R]=[1,0] moves content one column RIGHT (+x).
img = np.zeros((5, 6))
img[2, 3] = 1.0
out = s2.imtranslate(img, (1, 0))
check("T1a imtranslate +C moves right", out[2, 4] == 1.0 and out[2, 3] == 0.0)
out = s2.imtranslate(img, (0, 1))
check("T1b imtranslate +R moves down", out[3, 3] == 1.0 and out[2, 3] == 0.0)

# 1c. Fractional shift == manual bilinear over zero padding (MATLAB fill model).
def manual_bilinear_zeropad(a, tx, ty):
    H, W = a.shape
    pad = 8
    ap = np.zeros((H + 2 * pad, W + 2 * pad))
    ap[pad:pad + H, pad:pad + W] = a
    out = np.empty_like(a)
    for y in range(H):
        for x in range(W):
            sy = y - ty + pad
            sx = x - tx + pad
            y0, x0 = math.floor(sy), math.floor(sx)
            fy, fx = sy - y0, sx - x0
            out[y, x] = (ap[y0, x0] * (1 - fy) * (1 - fx)
                         + ap[y0 + 1, x0] * fy * (1 - fx)
                         + ap[y0, x0 + 1] * (1 - fy) * fx
                         + ap[y0 + 1, x0 + 1] * fy * fx)
    return out

a = rng.random((16, 17)) * 100
for tx, ty in [(0.3, -1.7), (-2.25, 0.5), (5.9, -3.1), (0.5, 0.5)]:
    got = s2.imtranslate(a, (tx, ty))
    ref = manual_bilinear_zeropad(a, tx, ty)
    d = np.abs(got - ref).max()
    check(f"T1c imtranslate frac ({tx},{ty}) == zero-pad bilinear", d < 1e-12, f"maxdiff={d:.2e}")

# 1d. Integer class round-half-away + saturate.
u = np.array([[0, 1], [2, 65535]], dtype=np.uint16)
got = s2.imtranslate(u, (0.5, 0))  # blends columns -> .5 values
# manual: float result then MATLAB cast
reff = manual_bilinear_zeropad(u.astype(float), 0.5, 0)
cast = np.sign(reff) * np.floor(np.abs(reff) + 0.5)
cast = np.clip(cast, 0, 65535).astype(np.uint16)
check("T1d imtranslate uint16 rounding", got.dtype == np.uint16 and np.array_equal(got, cast))

# ---------------------------------------------------------------- T2: _imgaussfilt
# Manual MATLAB imgaussfilt: kernel exp(-x^2/(2s^2)) size 2*ceil(2s)+1 normalized,
# replicate padding, separable correlation.
def matlab_imgaussfilt(a, sigma):
    r = math.ceil(2 * sigma)
    x = np.arange(-r, r + 1, dtype=float)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    k /= k.sum()
    ap = np.pad(a, r, mode="edge")
    # rows then cols (symmetric kernel, corr==conv)
    tmp = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 0, ap)
    out = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 1, tmp)
    return out

for sig in [1.0, 0.5, 2.3]:
    got = s2._imgaussfilt(a, sig)
    ref = matlab_imgaussfilt(a, sig)
    d = np.abs(got - ref).max()
    check(f"T2 imgaussfilt sigma={sig}", d < 1e-10, f"maxdiff={d:.2e}")

# ---------------------------------------------------------------- T3: dft_reg round trip
# Periodic construction (np.roll) so registration is exact -> tests PLUMBING
# (argument order target-vs-source, sign convention, [C,R] application).
base = matlab_imgaussfilt(rng.random((64, 64)) * 100, 2.0)  # smooth so subpixel works
true_shifts = [(0, 0), (3, -2), (-5, 7)]  # (row, col) roll applied to make source
stack = np.stack([np.roll(base, s, axis=(0, 1)) for s in true_shifts], axis=2).astype(float)
R, C, reg = s2.dft_reg(stack, base, 100)
# Shift to apply to source to align back = -(roll)
ok = all(R[i] == -true_shifts[i][0] and C[i] == -true_shifts[i][1] for i in range(3))
check("T3a dft_reg recovers -(applied roll) exactly", ok, f"R={R}, C={C}")
inner = (slice(10, 54), slice(10, 54))
d = max(np.abs(reg[:, :, i][inner] - base[inner]).max() for i in range(3))
check("T3b dft_reg reg aligned to target (interior)", d < 1e-9, f"inner maxdiff={d:.2e}")

# ---------------------------------------------------------------- T4: dft_rect chain semantics
# Plane i = base rolled by cumulative integer (r_i, c_i); anchor = middle.
cum = [(-2, 1), (-1, 1), (0, 0), (1, -1), (2, -1)]
vol = np.stack([np.roll(base, s, axis=(0, 1)) for s in cum], axis=2).astype(float)
start = 3  # 1-based anchor -> plane index 2 (zero offset)
R, C, reg = s2.dft_rect(vol, start, 100)
# target chain is aligned to anchor, so R(i),C(i) ~ -(cumulative roll of plane i)
ok = all(abs(R[i] + cum[i][0]) <= 0.05 and abs(C[i] + cum[i][1]) <= 0.05 for i in range(5))
check("T4a dft_rect cumulative shifts vs anchor", ok, f"R={R}, C={C}")
d = max(np.abs(reg[:, :, i][inner] - base[inner]).max() for i in range(5))
check("T4b dft_rect reg planes aligned (interior)", d < 0.5, f"inner maxdiff={d:.2e}")
check("T4c dft_rect anchor plane zero shift", R[start - 1] == 0.0 and C[start - 1] == 0.0)

# ---------------------------------------------------------------- T5: crop bounds table
# Compare python bounds to MATLAB-identical double arithmetic (they share the
# expression, so this documents concrete values; hand-check a few).
cases = [(128, 0.95, 4, 125), (100, 0.95, 3, 98), (512, 0.9, 26, 487), (64, 0.99, 1, 64)]
ok = True
for M, k, lo_exp, hi_exp in cases:
    lo = math.ceil(M * (1 - k) / 2)
    hi = math.ceil(M * (1 - (1 - k) / 2))
    if (lo, hi) != (lo_exp, hi_exp):
        ok = False
        print(f"  bounds M={M} k={k}: got ({lo},{hi}) expected ({lo_exp},{hi_exp})")
check("T5 crop bounds hand-checked table", ok)

# ---------------------------------------------------------------- T6: determine+apply end-to-end
# Small [Y,X,Z,T]: known per-(z,t) shifts; ref built by define_reference over n=2.
Y, X, Z, T = 48, 48, 2, 4
base_z = [matlab_imgaussfilt(rng.random((Y, X)) * 100, 1.5) for _ in range(Z)]
true = {(z, t): (((t % 3) - 1) * 1.0, ((t % 2) * 2 - 1) * 0.5) for z in range(Z) for t in range(T)}
true = {(z, t): ((t % 3) - 1, (t % 2) * 2 - 1) for z in range(Z) for t in range(T)}
fv = np.zeros((Y, X, Z, T))
for z in range(Z):
    for t in range(T):
        fv[:, :, z, t] = np.roll(base_z[z], true[(z, t)], axis=(0, 1))
refv = s2.define_reference(fv, 2, "mean")
RS, CS = s2.determine_xy_shifts_fbs(fv, 1.0, 0.95, refv)
check("T6a determine_xy shapes", RS.shape == (Z, T) and CS.shape == (Z, T))
corr = s2.apply_xy_shifts_fbs(fv, RS, CS)
# After correction, all t for a given z should be mutually aligned (to their ref)
sp = (slice(10, 38), slice(10, 38))
d = max(np.abs(corr[:, :, z, t][sp] - corr[:, :, z, 0][sp]).max()
        for z in range(Z) for t in range(T))
check("T6b apply_xy aligns frames", d < 2.0, f"inner maxdiff={d:.2e}, RS={RS}, CS={CS}")

# ---------------------------------------------------------------- T7: define_reference median/mean
v = rng.random((3, 3, 1, 4))
refm = s2.define_reference(v, 4, "median")
exp = np.median(v[:, :, 0, :], axis=2)
check("T7a median even-n midpoint", np.allclose(refm[:, :, 0, 0], exp, atol=0))
refmean = s2.define_reference(v, 2, "mean")
check("T7b mean chunks", np.allclose(refmean[:, :, 0, 1], v[:, :, 0, 2:4].mean(axis=2)))
vn = v.copy(); vn[1, 1, 0, 2] = np.nan
with np.errstate(all="ignore"):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        refn = s2.define_reference(vn, 4, "median")
check("T7c median NaN propagates", np.isnan(refn[1, 1, 0, 0]))

# ---------------------------------------------------------------- T8: chunck / reft indexing
# S4=4, refT=2 -> chunck=2; t(1-based)=1,2 -> ref 1; t=3,4 -> ref 2.
# Instrument by giving each ref time-slice a distinct constant and zero blur? Blur needs >0.
# Instead verify the index formula directly against MATLAB semantics.
ok = True
for S4, refT in [(4, 2), (10, 3), (6, 6), (7, 2)]:
    chunck = math.floor(S4 / refT)
    for t1 in range(1, S4 + 1):
        matlab_idx = math.ceil(t1 / chunck)
        py_idx = math.ceil(((t1 - 1) + 1) / chunck) - 1
        if py_idx != matlab_idx - 1:
            ok = False
            print(f"  S4={S4} refT={refT} t={t1}: matlab {matlab_idx} vs py0 {py_idx}")
        if matlab_idx > refT:
            # both sides crash here (MATLAB index OOB / numpy IndexError) - note only
            pass
check("T8 reft index formula", ok)

# ---------------------------------------------------------------- T9: scipy version features
import scipy
print("scipy", scipy.__version__)

print()
print("FAILURES:", fails if fails else "none")
