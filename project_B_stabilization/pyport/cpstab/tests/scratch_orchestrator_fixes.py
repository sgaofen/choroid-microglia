# -*- coding: utf-8 -*-
"""Verification of the review-fix round for orchestrator.py (findings F1-F4).

Run with:  python scratch_orchestrator_fixes.py

Checks:
  F1  import seam works end-to-end; dft_warp_3d_2 runs from (a) a .sbx path,
      (b) an SbxFile object, (c) a VolumeSource OME-TIFF object, and all
      three produce IDENTICAL saved shifts.
  F2  matlab_imresize / matlab_imwarp_affine2d / apply_optotune_warp preserve
      uint16 (MATLAB class rules); quantization actually happens (output
      differs from the float64 path); float64 behavior is unchanged.
  F3  matlab_compat imtranslate/imwarp now agree bit-for-bit with the
      independently verified shifts2d.imtranslate on the boundary ring
      (constant-100 image, 0.5 px shift -> edge 50, not 0).
  F4  no behavior change; nothing to test beyond nz == numel(otwave).
"""

import os
import sys
import tempfile

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cpstab import io_rw, matlab_compat, orchestrator, shifts2d
from cpstab.matlab_compat import (matlab_imresize, matlab_imtranslate,
                                  matlab_imwarp_affine2d)

rng = np.random.default_rng(7)
FAIL = []


def check(name, cond):
    print(("PASS  " if cond else "FAIL  ") + name)
    if not cond:
        FAIL.append(name)


# ---------------------------------------------------------------- F1: imports
try:
    sbxio, dftreg, s2d = orchestrator._import_deps()
    check("F1 _import_deps resolves io_rw/dftreg/shifts2d", True)
    check("F1 dft_rect reached via shifts2d", hasattr(s2d, "dft_rect"))
    check("F1 dftregistration3d reached via dftreg", hasattr(dftreg, "dftregistration3d"))
except ImportError as e:
    check("F1 _import_deps resolves io_rw/dftreg/shifts2d (%s)" % e, False)

# ------------------------------------------------------- F3: boundary parity
img = rng.integers(0, 4000, size=(9, 11)).astype(np.float64)
for tr in [(0.5, 0.0), (0.0, 0.5), (0.3, -0.7), (-1.2, 2.6), (2.0, -3.0)]:
    a = matlab_imtranslate(img, tr)
    b = shifts2d.imtranslate(img, tr)
    check("F3 imtranslate == shifts2d.imtranslate tr=%s" % (tr,),
          np.allclose(a, b, rtol=0, atol=1e-12))

const = np.full((4, 4), 100.0)
out = matlab_imtranslate(const, (0.5, 0.0))
check("F3 constant-100 shifted 0.5px: leading edge col = 50 (blend, not 0)",
      np.allclose(out[:, 0], 50.0) and np.allclose(out[:, 1:], 100.0))

# imwarp with a pure-translation affine == imtranslate
tx, ty = 0.5, -1.3
T = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [tx, ty, 1.0]])
w = matlab_imwarp_affine2d(img, T)
t = matlab_imtranslate(img, (tx, ty))
# atol: imwarp maps through 1-based coords + a numeric matrix inverse, so
# coordinates carry ~1e-15 noise -> values (~4e3) differ by O(1e-12).
check("F3 imwarp(pure translation) == imtranslate (incl. boundary ring)",
      np.allclose(w, t, rtol=0, atol=1e-9))

# far-outside queries are pure fill
big = matlab_imtranslate(const, (10.0, 0.0))
check("F3 far-out-of-domain -> exact fill 0", np.all(big[:, :4] == 0.0)
      if big.shape[1] >= 4 else False)

# ------------------------------------------------------ F2: class preservation
u16 = rng.integers(0, 65535, size=(16, 16, 3, 4)).astype(np.uint16)
r_int = matlab_imresize(u16, 0.5)
r_f64 = matlab_imresize(u16.astype(np.float64), 0.5)
check("F2 imresize uint16 -> uint16", r_int.dtype == np.uint16)
manual = np.clip(np.sign(r_f64) * np.floor(np.abs(r_f64) + 0.5), 0, 65535).astype(np.uint16)
check("F2 imresize uint16 == round-half-away(f64 result) saturated",
      np.array_equal(r_int, manual))
check("F2 imresize quantization is real (uint16 path != float64 path)",
      not np.allclose(r_int.astype(np.float64), r_f64))
check("F2 imresize float64 -> float64 unchanged", r_f64.dtype == np.float64)

wu = matlab_imwarp_affine2d(u16[:, :, 0, 0], T)
check("F2 imwarp uint16 -> uint16", wu.dtype == np.uint16)

# apply_optotune_warp: identity fast path returns THE input; class preserved
ident = [np.eye(3) for _ in range(3)]
aw = orchestrator.apply_optotune_warp(u16, ident)
check("F2 apply_optotune_warp identity returns input array unchanged", aw is u16)
near = [np.eye(3) for _ in range(3)]
near[1] = T.copy()
aw2 = orchestrator.apply_optotune_warp(u16, near)
check("F2 apply_optotune_warp non-identity keeps uint16", aw2.dtype == np.uint16)
check("F2 apply_optotune_warp identity planes copied exactly",
      np.array_equal(aw2[:, :, 0], u16[:, :, 0]))
aw3 = orchestrator.apply_optotune_warp(u16.astype(np.float64), near)
check("F2 apply_optotune_warp float64 stays float64", aw3.dtype == np.float64)

# dft_rect fed uint16 runs and chains uint16 targets internally (smoke)
r_, c_, reg_ = s2d.dft_rect(u16[:, :, :, 0].astype(np.uint16), 2, 4)
check("F2 dft_rect(uint16) runs; reg is float64 (MATLAB zeros)", reg_.dtype == np.float64)

# --------------------------------------- F1(4)+F2: end-to-end on three sources
tmp = tempfile.mkdtemp(prefix="cpstab_fix_")
NX, NY, NZ, NT = 32, 32, 5, 8
base = rng.integers(200, 3000, size=(NX, NY, NZ)).astype(np.float64)
frames = np.zeros((NX, NY, NZ, NT))
for tt in range(NT):
    for zz in range(NZ):
        sh = int(rng.integers(-2, 3))
        frames[:, :, zz, tt] = np.roll(np.roll(base[:, :, zz], sh, axis=0),
                                       -sh, axis=1)
frames += rng.normal(0, 40, frames.shape)
frames = np.clip(np.round(frames), 0, 65535).astype(np.uint16)

# (a) legacy .sbx + sidecar
sbx_path = os.path.join(tmp, "m_260824_001.sbx")
info = io_rw.spoof_sbx_info_3d(NX, NY, NZ, NT, 1)
io_rw.save_sbx_info(os.path.join(tmp, "m_260824_001.mat"), info)
rw = io_rw.RegWriter(sbx_path, info, ".sbx", True)
# .sbx frame order: z fastest then t -> [rows, cols, F]
flat = frames.reshape(NX, NY, NZ * NT, order="F")  # (y,x,f) f=z+t*NZ? check:
# frames[...,z,t] -> column-major merge of (Z,T) gives f = z + t*NZ: z fastest. good.
rw.write(flat)
rw.delete()

# (b) OME-TIFF for VolumeSource: pages (T, Z, Y, X), same values
tif_path = os.path.join(tmp, "m_260824_001.ome.tif")
pages = np.transpose(frames, (3, 2, 0, 1))  # (T, Z, Y, X)
tifffile.imwrite(tif_path, pages, imagej=True, metadata={"axes": "TZYX"})

tforms = [np.eye(3) for _ in range(NZ)]
common = dict(refchannel=1, scale=2, nchunks=2, tforms_optotune=tforms,
              reftype="mean", save=True)

res_path = orchestrator.dft_warp_3d_2(
    sbx_path, os.path.join(tmp, "a.dftshifts"), common["refchannel"],
    common["scale"], common["nchunks"], tforms, reftype="mean")

sbxfile = io_rw.SbxFile(sbx_path)
res_obj = orchestrator.dft_warp_3d_2(
    sbxfile, os.path.join(tmp, "b.dftshifts"), 1, 2, 2, tforms, reftype="mean")

vsrc = io_rw.VolumeSource(tif_path)
check("F1 VolumeSource dtype is uint16", vsrc.dtype == np.uint16)
res_vs = orchestrator.dft_warp_3d_2(
    vsrc, os.path.join(tmp, "c.dftshifts"), 1, 2, 2, tforms, reftype="mean")

for key in ("RS", "CS", "ZS", "RS_chunk", "CS_chunk", "ZS_chunk"):
    check("F1 path vs SbxFile identical: %s" % key,
          np.array_equal(res_path[key], res_obj[key]))
    check("F1 path vs VolumeSource identical: %s" % key,
          np.array_equal(res_path[key], res_vs[key]))
check("F1 RS shape (Nz, Nchunks*T)", res_path["RS"].shape == (NZ, NT))
check("F1 ZS shape (1, Nchunks*T)", res_path["ZS"].shape == (1, NT))
with open(os.path.join(tmp, "a.dftshifts"), "rb") as f:
    saved = np.load(f)
    check("F1 .dftshifts loads with RS/CS/ZS keys",
          all(k in saved.files for k in ("RS", "CS", "ZS")))

# F2 macro effect: shifts from the uint16 chain differ from an all-float64 run
frames_f = frames.astype(np.float64)


class FloatSource(object):
    """VolumeSource stand-in with no native integer dtype (float chain)."""
    Nx, Ny, Nz, Nt = NX, NY, NZ, NT

    def get_volume(self, t, channel=0):
        return frames_f[:, :, :, t]


res_float = orchestrator.dft_warp_3d_2(
    FloatSource(), os.path.join(tmp, "d.dftshifts"), 1, 2, 2, tforms,
    reftype="mean")
check("F2 uint16 chain vs float64 chain: RS differs at this SNR (quantization live)",
      not np.array_equal(res_path["RS"], res_float["RS"]))

# calculate_optotune_warp on both source kinds
tf1 = orchestrator.calculate_optotune_warp(sbx_path, 1, 2, regtype="none")
tf2 = orchestrator.calculate_optotune_warp(vsrc, 1, 2, regtype="none")
check("F1 calculate_optotune_warp('none') gives Nz identities (path & source)",
      len(tf1) == NZ and len(tf2) == NZ and
      all(np.array_equal(t_, np.eye(3)) for t_ in tf1 + tf2))

# pipeline seam: VolumeSource resolvable, _dims works on it
from cpstab import pipeline as pl
VS = pl._resolve(pl._VOLUME_SOURCE_CANDIDATES, "VolumeSource", "pipe.io")
check("F1 pipeline resolves VolumeSource from io_rw", VS is io_rw.VolumeSource)
check("F1 pipeline._dims(VolumeSource) -> (Nz, Nt)", pl._dims(vsrc) == (NZ, NT))
wr = pl._resolve(pl._WRITER_CANDIDATES, "write_zproj_tiff", "write2chanTiff.m")
check("F1 pipeline resolves writer from io_rw", wr is io_rw.write_zproj_tiff)

print()
if FAIL:
    print("FAILURES (%d):" % len(FAIL))
    for f_ in FAIL:
        print("  -", f_)
    sys.exit(1)
print("ALL CHECKS PASSED")
