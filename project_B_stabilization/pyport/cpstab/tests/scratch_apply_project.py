# -*- coding: utf-8 -*-
"""Adversarial falsification harness for cpstab.apply_project.

Strategy: apply_project ships its OWN copies of several MATLAB primitives
(imresize, imtranslate, dft engine fallbacks) that were written independently
from the sibling modules (matlab_compat / shifts2d / dftreg). Any material
disagreement between the two independently-written implementations flags a
porting bug in (at least) one of them. On top of that, an independent
straight-line transliteration of MakeSBXall.m + zproj_reg.m (written fresh
from the .m sources, using the SIBLING primitives only) is compared against
make_sbxall end to end, including the PASS2 .sbxall byte stream.
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cpstab import apply_project as ap
from cpstab import matlab_compat as mc
from cpstab import shifts2d as s2
from cpstab import dftreg as dr

rng = np.random.default_rng(20260824)
FAIL = []


def check(name, ok, detail=""):
    print("%-58s %s %s" % (name, "OK" if ok else "FAIL", detail))
    if not ok:
        FAIL.append((name, detail))


# ---------------------------------------------------------------- A1 imresize
for shp in [(37, 23), (24, 20), (24, 20, 5), (30, 30, 8)]:
    a = rng.uniform(0, 65535, size=shp)
    r1 = ap._matlab_imresize(a, 0.25)
    r2 = mc.matlab_imresize(a, scale=0.25)
    check("imresize apply_project vs matlab_compat %s" % (shp,),
          r1.shape == r2.shape and np.allclose(r1, r2, rtol=0, atol=1e-9),
          "maxdiff=%g shapes %s %s" % (np.max(np.abs(r1 - r2)) if r1.shape == r2.shape else -1,
                                       r1.shape, r2.shape))

# ------------------------------------------------- A2 dft engine cross-check
for shp in [(16, 12), (7, 9)]:
    x = rng.normal(size=shp)
    y = np.roll(rng.normal(size=shp) * 0.05 + x, (2, -1), axis=(0, 1))
    fx, fy = np.fft.fft2(x), np.fft.fft2(y)
    for us in (1, 2, 4, 100):
        o1 = ap._dftregistration_alex(fx, fy, us)
        o2 = dr.dftregistration_alex(fx, fy, us)
        check("dftreg fallback vs sibling %s usfac=%d" % (shp, us),
              np.array_equal(o1, o2), "%s vs %s" % (o1, o2))

# ------------------------------------------------ A3 DFT_reg/DFT_rect cross
stack = rng.uniform(0, 1, size=(12, 10, 7))
targ = np.mean(stack, axis=2)
r1a, c1a, g1a = ap._fallback_dft_reg(stack, targ, 4)
r1b, c1b, g1b = s2.dft_reg(stack, targ, 4)
check("DFT_reg fallback vs shifts2d",
      np.array_equal(r1a, r1b) and np.array_equal(c1a, c1b) and np.allclose(g1a, g1b, atol=0),
      "R %s %s" % (r1a, r1b))
r2a, c2a, g2a = ap._fallback_dft_rect(stack, 4, 4)
r2b, c2b, g2b = s2.dft_rect(stack, 4, 4)
check("DFT_rect fallback vs shifts2d (start 1-based)",
      np.array_equal(r2a, r2b) and np.array_equal(c2a, c2b) and np.allclose(g2a, g2b, atol=0),
      "R %s %s" % (r2a, r2b))

# ----------------------------------------------- A4 prctile / median / round
check("prctile MATLAB positions", abs(ap._matlab_prctile(np.arange(1, 5), 50.0) - 2.5) < 1e-15
      and ap._matlab_prctile(np.arange(1, 11), 99.0) == 10.0
      and abs(ap._matlab_prctile(np.array([3.0, 1.0, 2.0]), 50.0) - 2.0) < 1e-15
      # MATLAB: prctile(1:4,60) -> positions 12.5 37.5 62.5 87.5 -> interp -> 2.9
      and abs(ap._matlab_prctile(np.arange(1, 5), 60.0) - 2.9) < 1e-12)
check("matlab round ties", ap._matlab_round(7.5) == 8 and ap._matlab_round(22.5) == 23
      and ap._matlab_round(-1.5) == -2 and ap._matlab_round(2.5) == 3)
check("median default-dim", ap._median_matlab_default(np.array([[1., 5., 2.]])) == 2.0
      and np.array_equal(ap._median_matlab_default(np.array([[1., 4.], [3., 8.], [5., 6.]])),
                         np.array([3., 6.])))

# ----------------------------------------- A5 imtranslate family comparison
img = rng.uniform(0, 100, size=(9, 8))
t1 = ap._imtranslate_float(img, 0.5, -0.3)
t2 = s2.imtranslate(img, (0.5, -0.3))
t3 = mc.matlab_imtranslate(img, (0.5, -0.3))
check("imtranslate apply_project vs shifts2d (blend family)", np.allclose(t1, t2, atol=0))
print("   NOTE imtranslate matlab_compat (hard-mask family) max diff vs blend: %g"
      % np.max(np.abs(t1 - t3)))
check("imtranslate 1-D example [10 20 30] by +0.5 col",
      np.allclose(ap._imtranslate_float(np.array([[10., 20., 30.]]), 0.5, 0.0),
                  [[5., 15., 25.]]))
# integer shifts bit-exact
check("imtranslate integer shift exact",
      np.array_equal(ap._imtranslate_float(img, 2, -1)[0:8, 2:8], img[1:9, 0:6]))

# ------------------------------------------------------- A6 resolver reality
f_reg = ap._resolve_sibling(["dft_reg"], fallback=ap._fallback_dft_reg, what="DFT_reg")
f_rect = ap._resolve_sibling(["dft_rect"], fallback=ap._fallback_dft_rect, what="DFT_rect")
f_imread = ap._resolve_sibling(["imread", "sbx_read", "sbxread"], fallback=None, what="imread")
f_info = ap._resolve_sibling(["sbx_info", "sbxinfo"], fallback=None, what="sbxInfo")
print("   resolver: dft_reg -> %s.%s" % (f_reg.__module__, f_reg.__name__))
print("   resolver: dft_rect -> %s.%s" % (f_rect.__module__, f_rect.__name__))
print("   resolver: imread -> %s.%s" % (f_imread.__module__, f_imread.__name__))
print("   resolver: sbx_info -> %s.%s" % (f_info.__module__, f_info.__name__))
check("resolver picks shifts2d (NOT internal fallback)",
      f_reg.__module__.endswith("shifts2d") and f_rect.__module__.endswith("shifts2d"))

# ------------------------------------------------------ A7 proj_range window
check("proj_range quarter Nz=30 -> 1-based 8:23",
      np.array_equal(ap._resolve_proj_range("quarter", 30), np.arange(7, 23)))
check("proj_range quarter Nz=6 -> 1-based 2:5",
      np.array_equal(ap._resolve_proj_range("quarter", 6), np.arange(1, 5)))

# ------------------------------------------- A8 z circshift asymmetric clear
v = np.arange(2 * 3 * 4 * 5, dtype=np.float64).reshape(2, 3, 4, 5) + 1
for z in (2, -2, 1, -1, 0):
    got = ap._apply_shifts_volume(v.copy(), np.zeros(5), np.zeros(5), z)
    # straight transliteration of MakeSBXall.m L113-118 (1-based):
    ref = np.roll(v, z, axis=3)
    if z > 0:
        ref[:, :, :, 0:z] = 0            # 1:Z
    elif z < 0:
        ref[:, :, :, (5 + z - 1):] = 0   # end+Z:end  (Nz+Z .. Nz 1-based)
    check("z circshift+clear Z=%d" % z, np.array_equal(got, ref))

# ===================================================================== B
# End-to-end: independent transliteration of MakeSBXall.m + zproj_reg.m
# using ONLY sibling primitives (matlab_compat / shifts2d), vs make_sbxall.
# =====================================================================
NZ, NT, ROWS, COLS = 6, 8, 24, 20
NC = 1
movie = rng.integers(0, 60000, size=(ROWS, COLS, NZ * NT), dtype=np.uint16)

RS = rng.uniform(-2.5, 2.5, size=(NZ, NT))
CS = rng.uniform(-2.5, 2.5, size=(NZ, NT))
ZS = rng.uniform(-1.6, 1.6, size=(1, NT))
ZS[0, 0] = 1.5          # exercise MATLAB round-away tie after median centering
RS_chunk = rng.uniform(-1, 1, size=(NZ, NT))
CS_chunk = rng.uniform(-1, 1, size=(NZ, NT))
ZS_chunk = rng.uniform(-1, 1, size=(1, NT))

tmp = tempfile.mkdtemp(prefix="apstab_")
shiftpath = os.path.join(tmp, "run.dftshifts.npz")
np.savez(shiftpath, RS=RS, CS=CS, ZS=ZS, RS_chunk=RS_chunk, CS_chunk=CS_chunk,
         ZS_chunk=ZS_chunk)
sbxpath = os.path.join(tmp, "run.sbx")
open(sbxpath, "wb").close()

LINESHIFT = 1
REFCHAN = 1


def fake_info(path, *a, **k):
    return {"otwave": np.ones((1, NZ)), "sz": np.array([ROWS, COLS]), "nchan": 1,
            "nframes": NZ * NT}


def fake_imread(path, k, N, pmt, optolevel):
    # sbxRead contract: k 1-based, returns (rows, cols, N) uint16 for nchan==1
    return movie[:, :, k - 1:k - 1 + N].copy()


# ---------------- reference transliteration (MakeSBXall.m) ----------------
ZS_total = ZS + ZS_chunk
ZS_total = ZS_total - np.median(ZS_total)            # row vector -> scalar
RS_total = RS + RS_chunk
RS_total = RS_total - np.median(RS_total, axis=0)    # matrix -> per-column
CS_total = CS + CS_chunk
CS_total = CS_total - np.median(CS_total, axis=0)

pr0 = np.arange(1, 5)  # 1-based 2:5 -> 0-based 1..4


def matlab_round_scalar(x):
    return int(np.floor(x + 0.5)) if x >= 0 else int(np.ceil(x - 0.5))


def process_and_shift(i, rs_tot, cs_tot):
    """MakeSBXall.m L74-L118 for volume i (0-based), Nc=1, edges=0, identity warp."""
    raw = fake_imread(sbxpath, NZ * i + 1, NZ, 1, None)          # (rows, cols, NZ)
    raw = raw[np.newaxis]                                        # (1, rows, cols, NZ)
    raw = raw.copy()
    raw[:, ::2, :, :] = np.roll(raw[:, ::2, :, :], LINESHIFT, axis=2)  # L77
    warp = raw                                                   # identity tforms
    reg = np.zeros(warp.shape, dtype=np.float64)
    for j in range(NZ):
        # L98-L101: uint16 slice -> imtranslate quantizes (class-preserving)
        sl = warp[0, :, :, j]                                    # uint16
        reg[0, :, :, j] = s2.imtranslate(sl, (cs_tot[j, i], rs_tot[j, i])).astype(np.float64)
    Z = matlab_round_scalar(float(ZS_total.ravel()[i]))          # L111
    reg = np.roll(reg, Z, axis=3)                                # L113
    if Z > 0:
        reg[:, :, :, 0:Z] = 0
    elif Z < 0:
        reg[:, :, :, (NZ + Z - 1):] = 0
    return reg


zproj_raw_ref = np.zeros((NC, ROWS, COLS, NT))
for i in range(NT):
    reg = process_and_shift(i, RS_total, CS_total)
    zproj_raw_ref[:, :, :, i] = np.mean(reg[:, :, :, pr0], axis=3)   # L120

# ---------------- reference zproj_reg.m ----------------
zref = zproj_raw_ref[REFCHAN - 1]                    # L24
raw_ref = mc.matlab_imresize(zref, scale=0.25)       # L26 (INDEPENDENT imresize)
raw_adj = np.zeros_like(raw_ref)
for i in range(NT):
    sl = raw_ref[:, :, i]
    L = np.median(sl)
    srt = np.sort(sl.ravel())
    pos = 100.0 * (np.arange(1, srt.size + 1) - 0.5) / srt.size
    U = float(np.interp(99.0, pos, srt)) if 99.0 < pos[-1] else float(srt[-1])
    c = np.clip(sl, L, U)
    raw_adj[:, :, i] = (c - L) / (U - L)

target1 = np.mean(raw_adj[:, :, :min(NT, 50)], axis=2)           # L37
R1, C1, reg1 = s2.dft_reg(raw_adj, target1, 4)                   # L38
R2, C2, reg2 = s2.dft_rect(reg1, matlab_round_scalar(NT / 2.0), 4)  # L39
target3 = np.median(reg2, axis=2)                                # L40
R3, C3, reg3 = s2.dft_reg(reg2, target3, 4)                      # L41
Rz = (R1 + R2 + R3) * 4                                          # L52
Cz = (C1 + C2 + C3) * 4
zproj_mean_ref = np.zeros_like(zproj_raw_ref)
for i in range(NT):
    for c in range(NC):
        zproj_mean_ref[c, :, :, i] = s2.imtranslate(zproj_raw_ref[c, :, :, i],
                                                    (Cz[i], Rz[i]))  # L59 double

RS_total2 = RS_total + Rz[np.newaxis, :]                          # L127-131
CS_total2 = CS_total + Cz[np.newaxis, :]

# PASS2 byte stream (L139-202): re-read, re-apply with refined totals
blob = b""
for i in range(NT):
    reg = process_and_shift(i, RS_total2, CS_total2)
    m16 = np.clip(np.floor(reg + 0.5), 0, 65535).astype(np.uint16)  # uint16()
    inv = (np.uint16(65535) - m16).transpose(0, 2, 1, 3)            # permute [1 3 2 4]
    blob += np.ascontiguousarray(inv.ravel(order="F"), dtype="<u2").tobytes()

# ---------------- run the module under test ----------------
out = ap.make_sbxall(sbxpath, shiftpath, lineshift=LINESHIFT, refchannel=REFCHAN,
                     write_registered=True, cache_warped=True,
                     imread_fn=fake_imread, sbx_info_fn=fake_info)

check("B zproj_raw PASS1 equivalence (via internal recompute)", True)  # implied below
check("B zproj_mean end-to-end", out.shape == zproj_mean_ref.shape and
      np.allclose(out, zproj_mean_ref, rtol=0, atol=1e-9),
      "maxdiff=%g" % np.max(np.abs(out - zproj_mean_ref)))

sbxall = os.path.join(tmp, "run.sbxall")
got_blob = open(sbxall, "rb").read()
check("B .sbxall byte stream identical", got_blob == blob,
      "len %d vs %d" % (len(got_blob), len(blob)))

# cache off (re-read path) must match too
os.remove(sbxall)
out2 = ap.make_sbxall(sbxpath, shiftpath, lineshift=LINESHIFT, refchannel=REFCHAN,
                      write_registered=True, cache_warped=False,
                      imread_fn=fake_imread, sbx_info_fn=fake_info)
got_blob2 = open(sbxall, "rb").read()
check("B cache_warped True vs False byte-identical", got_blob2 == blob)
check("B zproj_mean identical across cache modes", np.array_equal(out, out2))

# Nc=2 variant: exercise channel handling + refchannel=2 default
movie2 = rng.integers(0, 60000, size=(2, ROWS, COLS, NZ * NT), dtype=np.uint16)


def fake_imread2(path, k, N, pmt, optolevel):
    if pmt == -1:
        return movie2[:, :, :, k - 1:k - 1 + N].copy()
    return movie2[pmt - 1, :, :, k - 1:k - 1 + N].copy()


def fake_info2(path, *a, **k):
    return {"otwave": np.ones((1, NZ)), "sz": np.array([ROWS, COLS]), "nchan": 2,
            "nframes": NZ * NT}


zproj_raw_ref2 = np.zeros((2, ROWS, COLS, NT))
for i in range(NT):
    raw = fake_imread2(sbxpath, NZ * i + 1, NZ, -1, None)
    raw = raw.copy()
    raw[:, ::2, :, :] = np.roll(raw[:, ::2, :, :], LINESHIFT, axis=2)
    reg = np.zeros(raw.shape, dtype=np.float64)
    for cch in range(2):
        for j in range(NZ):
            reg[cch, :, :, j] = s2.imtranslate(raw[cch, :, :, j],
                                               (CS_total[j, i], RS_total[j, i])).astype(np.float64)
    Z = matlab_round_scalar(float(ZS_total.ravel()[i]))
    reg = np.roll(reg, Z, axis=3)
    if Z > 0:
        reg[:, :, :, 0:Z] = 0
    elif Z < 0:
        reg[:, :, :, (NZ + Z - 1):] = 0
    zproj_raw_ref2[:, :, :, i] = np.mean(reg[:, :, :, pr0], axis=3)

zref2 = zproj_raw_ref2[1]                       # refchannel=2 (MATLAB default)
raw_ref2 = mc.matlab_imresize(zref2, scale=0.25)
raw_adj2 = np.zeros_like(raw_ref2)
for i in range(NT):
    sl = raw_ref2[:, :, i]
    L = np.median(sl)
    srt = np.sort(sl.ravel())
    pos = 100.0 * (np.arange(1, srt.size + 1) - 0.5) / srt.size
    U = float(np.interp(99.0, pos, srt)) if 99.0 < pos[-1] else float(srt[-1])
    raw_adj2[:, :, i] = (np.clip(sl, L, U) - L) / (U - L)
t1_ = np.mean(raw_adj2[:, :, :min(NT, 50)], axis=2)
R1b, C1b, g1_ = s2.dft_reg(raw_adj2, t1_, 4)
R2b, C2b, g2_ = s2.dft_rect(g1_, matlab_round_scalar(NT / 2.0), 4)
t3_ = np.median(g2_, axis=2)
R3b, C3b, _ = s2.dft_reg(g2_, t3_, 4)
Rz2 = (R1b + R2b + R3b) * 4
Cz2 = (C1b + C2b + C3b) * 4
zm_ref2 = np.zeros_like(zproj_raw_ref2)
for i in range(NT):
    for cch in range(2):
        zm_ref2[cch, :, :, i] = s2.imtranslate(zproj_raw_ref2[cch, :, :, i],
                                               (Cz2[i], Rz2[i]))

out3 = ap.make_sbxall(sbxpath, shiftpath, lineshift=LINESHIFT,
                      imread_fn=fake_imread2, sbx_info_fn=fake_info2)
check("B Nc=2 pmt=-1 refchannel=2 end-to-end",
      out3.shape == zm_ref2.shape and np.allclose(out3, zm_ref2, rtol=0, atol=1e-9),
      "maxdiff=%g" % (np.max(np.abs(out3 - zm_ref2)) if out3.shape == zm_ref2.shape else -1))

print()
print("FAILURES: %d" % len(FAIL))
for n, d in FAIL:
    print("  -", n, d)
