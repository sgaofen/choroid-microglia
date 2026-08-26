"""Adversarial review scratch experiments for orchestrator.py / matlab_compat.py.

Run: python tests/scratch_orchestrator.py
(from pyport/, i.e. with cpstab importable)
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))   # pyport/

from cpstab import dftreg
# shifts2d imports `dftregistration` from dftreg, which does not exist (bug).
# Patch so we can exercise the numerics anyway.
if not hasattr(dftreg, "dftregistration"):
    dftreg.dftregistration = dftreg.dftregistration_alex
sys.modules.pop("cpstab.shifts2d", None)
from cpstab import shifts2d  # noqa: E402
from cpstab import matlab_compat as mc  # noqa: E402

rng = np.random.default_rng(0)


def matlab_uint16_cast(x):
    """MATLAB double->uint16: round half away from zero + saturate."""
    r = np.sign(x) * np.floor(np.abs(x) + 0.5)
    return np.clip(r, 0, 65535).astype(np.uint16)


print("=" * 70)
print("A. imtranslate edge semantics: matlab_compat vs shifts2d (same MATLAB fn)")
img = np.full((4, 5), 100.0)
t_mc = mc.matlab_imtranslate(img, (0.5, 0.0))       # [C,R] = shift +0.5 col
t_sh = shifts2d.imtranslate(img, (0.5, 0.0))
print("matlab_compat col0:", t_mc[:, 0])
print("shifts2d      col0:", t_sh[:, 0])
T = np.array([[1.0, 0, 0], [0, 1.0, 0], [0.5, 0.0, 1.0]])  # x += 0.5
w_mc = mc.matlab_imwarp_affine2d(img, T)
print("matlab_imwarp col0:", w_mc[:, 0], " (imtranslate-equivalent warp)")
print("interior equal:", np.allclose(t_mc[:, 1:], t_sh[:, 1:]))

print("=" * 70)
print("B. uint16 vs float64 flow through imresize + DFT_rect")
# Synthetic 'realistic' volume: smooth structure + photon-ish noise, uint16.
ny, nx, nz = 64, 64, 8
yy, xx = np.mgrid[0:ny, 0:nx]
base = np.zeros((ny, nx, nz))
rng = np.random.default_rng(42)
for z in range(nz):
    # blob that drifts slightly with z (what DFT_rect corrects)
    cy, cx = 32 + 1.3 * (z - nz / 2), 30 - 0.9 * (z - nz / 2)
    base[:, :, z] = 3000 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 12.0 ** 2))
vol_u16 = np.clip(base + rng.normal(300, 60, base.shape), 0, 65535)
vol_u16 = matlab_uint16_cast(vol_u16)

scale = 2.0  # pipeline downsamples by 1/scale
# --- MATLAB flow: imresize on uint16 returns uint16 (round+clamp); DFT_rect
#     then runs on uint16 (its imtranslate rounds the running target).
small_f = mc.matlab_imresize(vol_u16.astype(np.float64), 1.0 / scale)
small_matlab = matlab_uint16_cast(small_f)                    # what MATLAB has
# --- port flow: float64 throughout.
small_port = small_f                                          # what the port has

print("imresize output max |float - uint16round| :",
      np.max(np.abs(small_port - small_matlab.astype(np.float64))))
frac = np.mean(small_port != small_matlab.astype(np.float64))
print("fraction of pixels differing:", round(float(frac), 4))

start = int(mc.matlab_round(nz / 2.0))
r_m, c_m, reg_m = shifts2d.dft_rect(small_matlab, start, 4)   # uint16 in
r_p, c_p, reg_p = shifts2d.dft_rect(small_port, start, 4)     # float64 in
print("RS0 (uint16 flow):", r_m)
print("RS0 (float64 flow):", r_p)
print("CS0 (uint16 flow):", c_m)
print("CS0 (float64 flow):", c_p)
print("RS0/CS0 all equal:", np.array_equal(r_m, r_p) and np.array_equal(c_m, c_p))
print("chunk_reg0 max |diff|:", np.max(np.abs(reg_m - reg_p)))
print("chunk_reg0 mean |diff|:", np.mean(np.abs(reg_m - reg_p)))

# and the usfac=100 second stage sensitivity: register plane to itself+noise
diffs = 0
trials = 40
for t in range(trials):
    im = np.clip(base[:, :, 3] + rng.normal(300, 60, (ny, nx)), 0, 65535)
    im_u = matlab_uint16_cast(im)
    im_small_f = mc.matlab_imresize(im_u.astype(np.float64), 1.0 / scale)
    im_small_u = matlab_uint16_cast(im_small_f).astype(np.float64)
    ref = mc.matlab_imresize(
        matlab_uint16_cast(np.clip(base[:, :, 3] + rng.normal(300, 60, (ny, nx)),
                                   0, 65535)).astype(np.float64), 1.0 / scale)
    ref_u = matlab_uint16_cast(ref).astype(np.float64)
    s_f = dftreg.dftregistration_alex(np.fft.fft2(ref), np.fft.fft2(im_small_f), 100)
    s_u = dftreg.dftregistration_alex(np.fft.fft2(ref_u), np.fft.fft2(im_small_u), 100)
    if not np.array_equal(s_f, s_u):
        diffs += 1
print("usfac=100 shift pairs differing (rounded vs unrounded imresize): "
      "%d / %d" % (diffs, trials))

print("=" * 70)
print("C. nearest 'stretch' of chunk corrections == exact per-chunk repeat?")
for nchunks, T, nzq in [(4, 5, 8), (3, 7, 30), (20, 13, 15), (2, 1, 4)]:
    vec = rng.normal(size=nchunks)
    out = mc.matlab_imresize(vec[None, :], output_shape=(nzq, nchunks * T),
                             method="nearest")
    expect = np.tile(np.repeat(vec, T), (nzq, 1))
    ok = np.array_equal(out, expect)
    print("Nchunks=%2d T=%2d Nz=%2d  exact-repeat: %s" % (nchunks, T, nzq, ok))

print("=" * 70)
print("D. reshape order sanity (column-major z-fastest)")
nx_, ny_, nzr, tr = 3, 4, 5, 6
frames = np.zeros((nx_, ny_, nzr * tr))
for f in range(nzr * tr):
    frames[:, :, f] = f
res = frames.reshape(nx_, ny_, tr, nzr).transpose(0, 1, 3, 2)
ok = all(res[0, 0, z, t] == t * nzr + z for z in range(nzr) for t in range(tr))
print("f = t*Nz + z mapping ok:", ok)

print("=" * 70)
print("E. matlab_imresize scalar-scale output size + nearest known-value checks")
a = np.arange(1.0, 5.0)[None, :]  # [1 2 3 4] as 1x4
half = mc.matlab_imresize(np.repeat(a, 4, axis=0), 0.5, method="nearest")
print("imresize([1 2 3 4]x4, 0.5, nearest) row:", half[0], " (MATLAB: [2 4])")
sz = mc.matlab_imresize(np.zeros((5, 7)), 1.0 / 2.0).shape
print("ceil(5/2), ceil(7/2) ->", sz, "(expect (3, 4))")
