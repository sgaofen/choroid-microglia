# -*- coding: utf-8 -*-
"""Verification of the adversarial-review fixes in apply_project / pipeline /
config (findings F1, F3 + minors a/b; F2 is docs-only, F4 deferred pending
real MATLAB).

F1: MAT shift file with present-but-undecodable tforms_optotune_full must
    RAISE when optotune is truthy, proceed when falsy or overridden.
F3: pipeline.run_pipeline must resolve apply_project.make_sbxall and run
    end-to-end on a TIFF VolumeSource (previously ImportError at the apply
    stage).
(a) _apply_shifts_volume out-of-range Z guards (boundaries exact).
(b) _resolve_proj_range rejects non-integer sequences.
"""
import os
import sys
import tempfile
import warnings

import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cpstab import apply_project as ap

rng = np.random.default_rng(20260824)
FAIL = []


def check(name, ok, detail=""):
    print("%-64s %s %s" % (name, "OK" if ok else "FAIL", detail))
    if not ok:
        FAIL.append((name, detail))


tmp = tempfile.mkdtemp(prefix="apfix_")

NZ, NT, ROWS, COLS = 6, 8, 24, 20
RS = rng.uniform(-2.5, 2.5, size=(NZ, NT))
CS = rng.uniform(-2.5, 2.5, size=(NZ, NT))
ZS = rng.uniform(-1.6, 1.6, size=(1, NT))
RS_chunk = rng.uniform(-1, 1, size=(NZ, NT))
CS_chunk = rng.uniform(-1, 1, size=(NZ, NT))
ZS_chunk = rng.uniform(-1, 1, size=(1, NT))
movie = rng.integers(0, 60000, size=(ROWS, COLS, NZ * NT), dtype=np.uint16)


def fake_info(path, *a, **k):
    return {"otwave": np.ones((1, NZ)), "sz": np.array([ROWS, COLS]), "nchan": 1,
            "nframes": NZ * NT}


def fake_imread(path, k, N, pmt, optolevel):
    return movie[:, :, k - 1:k - 1 + N].copy()


# ============================================================ F1: undecodable
# Simulate what scipy.loadmat yields for a MATLAB affine2d array: an object
# array that cannot be coerced to float64 (ragged / non-numeric contents).
bad = np.empty((NZ,), dtype=object)
for j in range(NZ):
    bad[j] = np.eye(3) if j % 2 == 0 else "affine2d-opaque"
matpath = os.path.join(tmp, "run_mat.dftshifts")
sio.savemat(matpath, {"RS": RS, "CS": CS, "ZS": ZS, "RS_chunk": RS_chunk,
                      "CS_chunk": CS_chunk, "ZS_chunk": ZS_chunk,
                      "tforms_optotune_full": bad})

sh = ap._load_shifts(matpath)
check("F1 _load_shifts flags undecodable (no warn, no raise)",
      sh["tforms_optotune_full"] is None and sh["tforms_optotune_undecodable"] is True)

# truthy optotune (the MATLAB default 'true') -> must raise
try:
    ap.make_sbxall(os.path.join(tmp, "run.sbx"), matpath,
                   imread_fn=fake_imread, sbx_info_fn=fake_info)
    check("F1 truthy optotune + undecodable tforms raises", False, "no exception")
except ValueError as e:
    check("F1 truthy optotune + undecodable tforms raises",
          "tforms_optotune_full" in str(e) and "affine2d" in str(e))

# even the MATLAB quirk optotune='false' (nonempty string == truthy) raises
try:
    ap.make_sbxall(os.path.join(tmp, "run.sbx"), matpath, optotune="false",
                   imread_fn=fake_imread, sbx_info_fn=fake_info)
    check("F1 optotune='false' (MATLAB-truthy!) still raises", False, "no exception")
except ValueError:
    check("F1 optotune='false' (MATLAB-truthy!) still raises", True)

# falsy optotune -> warp branch dead, must complete (with a warning)
with warnings.catch_warnings(record=True) as wlist:
    warnings.simplefilter("always")
    out_falsy = ap.make_sbxall(os.path.join(tmp, "run.sbx"), matpath, optotune=0, refchannel=1,
                               imread_fn=fake_imread, sbx_info_fn=fake_info)
check("F1 falsy optotune completes with warning",
      out_falsy.shape == (1, ROWS, COLS, NT)
      and any("tforms_optotune_full" in str(w.message) for w in wlist))

# explicit numeric override -> completes without touching the opaque payload
out_override = ap.make_sbxall(os.path.join(tmp, "run.sbx"), matpath, refchannel=1,
                              tforms_optotune_full=np.eye(3),
                              imread_fn=fake_imread, sbx_info_fn=fake_info)
check("F1 explicit tforms_optotune_full override completes",
      out_override.shape == (1, ROWS, COLS, NT))

# identity-equivalence: falsy-optotune vs identity-override paths must agree
check("F1 falsy vs identity-override numerically identical",
      np.array_equal(out_falsy, out_override))

# npz with numeric (Nz,3,3) tforms (orchestrator contract) -> decodable, runs
npzpath = os.path.join(tmp, "run_npz.dftshifts.npz")
np.savez(npzpath, RS=RS, CS=CS, ZS=ZS, RS_chunk=RS_chunk, CS_chunk=CS_chunk,
         ZS_chunk=ZS_chunk, tforms_optotune_full=np.tile(np.eye(3), (NZ, 1, 1)))
out_npz = ap.make_sbxall(os.path.join(tmp, "run.sbx"), npzpath, refchannel=1,
                         imread_fn=fake_imread, sbx_info_fn=fake_info)
check("F1 npz numeric tforms path unaffected",
      np.array_equal(out_npz, out_override))

# absent tforms (MakeSBXall.m L61-63) -> identity, still no raise
npz2 = os.path.join(tmp, "run_npz2.dftshifts.npz")
np.savez(npz2, RS=RS, CS=CS, ZS=ZS, RS_chunk=RS_chunk, CS_chunk=CS_chunk,
         ZS_chunk=ZS_chunk)
out_abs = ap.make_sbxall(os.path.join(tmp, "run.sbx"), npz2, refchannel=1,
                         imread_fn=fake_imread, sbx_info_fn=fake_info)
check("F1 absent tforms -> identity (MATLAB L61-63) unaffected",
      np.array_equal(out_abs, out_override))

# ============================================================ (a) Z guards
v = np.arange(1 * 3 * 4 * 5, dtype=np.float64).reshape(1, 3, 4, 5) + 1
nz5 = 5
ok = True
for z, should_raise in [(nz5, False), (nz5 + 1, True),
                        (-(nz5 - 1), False), (-nz5, True)]:
    try:
        ap._apply_shifts_volume(v.copy(), np.zeros(nz5), np.zeros(nz5), z)
        raised = False
    except IndexError:
        raised = True
    if raised != should_raise:
        ok = False
        print("   Z=%d raised=%r expected=%r" % (z, raised, should_raise))
check("(a) Z guard boundaries (Z=Nz ok, Z>Nz raise, Z=-Nz raise)", ok)

# ============================================================ (b) proj_range
try:
    ap._resolve_proj_range([1.5, 2.5], 6)
    check("(b) non-integer proj_range raises", False, "no exception")
except ValueError as e:
    check("(b) non-integer proj_range raises", "non-integer" in str(e))
check("(b) integer proj_range still accepted",
      np.array_equal(ap._resolve_proj_range([1, 2, 3], 6), np.array([0, 1, 2]))
      and np.array_equal(ap._resolve_proj_range(np.array([2, 5], dtype=np.int64), 6),
                         np.array([1, 4]))
      and np.array_equal(ap._resolve_proj_range(np.array([2.0, 3.0]), 6),
                         np.array([1, 2])))

# ============================================================ F3: pipeline e2e
import tifffile
from cpstab.config import RegistrationConfig
from cpstab.pipeline import run_pipeline, _apply_io_adapter
from cpstab import io_rw

NT2, NZ2, R2, C2 = 8, 6, 24, 20
data = rng.integers(0, 60000, size=(NT2, NZ2, R2, C2), dtype=np.uint16)
tif = os.path.join(tmp, "e2e_stack.tif")
tifffile.imwrite(tif, data, imagej=True, metadata={"axes": "TZYX"})

cfg = RegistrationConfig(input_path=tif, refchannel=1, scale=4, chunksize=4,
                         write_registered=True, out_dir=tmp)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    zproj = run_pipeline(cfg)
check("F3 run_pipeline completes end-to-end on TIFF source",
      zproj.shape == (1, R2, C2, NT2), "shape %r" % (zproj.shape,))
check("F3 projection TIFF written", os.path.exists(cfg.zproj_tiff_path()))
check("F3 .sbxall written where config reports it",
      os.path.exists(cfg.registered_stack_path())
      and os.path.getsize(cfg.registered_stack_path()) == R2 * C2 * NZ2 * NT2 * 2)

# adapter numerics: same call through the adapter reproduces run_pipeline's
# apply stage bit-for-bit (idempotence smoke; DFT stage reuses shiftpath).
src = io_rw.VolumeSource(tif)
sbx_info_fn, imread_fn = _apply_io_adapter(src)
zproj2 = ap.make_sbxall(tif, cfg.shiftpath(), refchannel=1,
                        proj_range=cfg.proj_planes_1based(NZ2),
                        sbx_info_fn=sbx_info_fn, imread_fn=imread_fn)
check("F3 adapter re-run reproduces run_pipeline output bit-exactly",
      np.array_equal(zproj, zproj2),
      "maxdiff=%g" % (np.max(np.abs(zproj - zproj2))
                      if zproj.shape == zproj2.shape else -1))

# adapter contract details: 1-based k, whole-volume reads, layout
raw = imread_fn(tif, NZ2 * 3 + 1, NZ2, 1, None)
check("F3 adapter imread layout (Y,X,Z) & values",
      raw.shape == (R2, C2, NZ2)
      and np.array_equal(raw.astype(np.uint16),
                         np.transpose(data[3], (1, 2, 0))))
try:
    imread_fn(tif, 2, NZ2, 1, None)
    check("F3 adapter rejects volume-misaligned read", False, "no exception")
except ValueError:
    check("F3 adapter rejects volume-misaligned read", True)

info = sbx_info_fn(tif)
check("F3 adapter info contract",
      int(np.asarray(info["otwave"]).size) == NZ2
      and list(np.asarray(info["sz"]).ravel().astype(int)) == [R2, C2]
      and info["nchan"] == 1)

print()
print("FAILURES: %d" % len(FAIL))
for n, d in FAIL:
    print("  -", n, d)
sys.exit(1 if FAIL else 0)
