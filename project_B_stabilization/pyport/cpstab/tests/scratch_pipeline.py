"""Adversarial scratch checks for cpstab pipeline.py / config.py vs MATLAB semantics."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
from cpstab.config import RegistrationConfig, matlab_round
from cpstab.pipeline import matlab_uint16

fails = []
def check(name, got, want):
    ok = got == want
    print(("PASS" if ok else "FAIL"), name, "got", got, "want", want)
    if not ok:
        fails.append(name)

# ---- matlab_round vs MATLAB round() ground truth ----
# MATLAB: round(7.5)=8 round(22.5)=23 round(-2.5)=-3 round(11.5)=12 round(0.5)=1
for x, want in [(7.5, 8), (22.5, 23), (-2.5, -3), (11.5, 12), (0.5, 1),
                (10.5, 11), (-0.5, -1), (0.25, 0), (0.75, 1), (2.0, 2), (-3.0, -3)]:
    check("matlab_round(%r)" % x, matlab_round(x), want)

# double-rounding hazard: largest double < 0.5. MATLAB round() returns 0.
x = np.nextafter(0.5, 0.0)
print("double-rounding probe: matlab_round(nextafter(0.5,0)) =", matlab_round(x),
      " (MATLAB round gives 0)")

# ---- matlab_uint16 vs MATLAB uint16() ground truth ----
got = matlab_uint16(np.array([2.5, -3.7, 70000.0, np.nan, 0.5, 1.5, -0.4,
                              65534.5, np.inf, -np.inf]))
want = np.array([3, 0, 65535, 0, 1, 2, 0, 65535, 65535, 0], dtype=np.uint16)
check("matlab_uint16 vector", got.tolist(), want.tolist())
check("matlab_uint16 dtype", got.dtype == np.uint16, True)
# double rounding probe
print("uint16 double-rounding probe:", matlab_uint16(np.nextafter(0.5, 0.0)),
      "(MATLAB uint16 gives 0)")
# integer input passthrough
check("matlab_uint16 int scalar", int(matlab_uint16(65535)), 65535)

# ---- nchunks == MATLAB round(Nt/chunksize) ----
def matlab_ref_round(x):
    # emulate MATLAB round via decimal-free logic on exact binary values
    import math
    f = math.floor(x); frac = x - f
    if x >= 0:
        return f + 1 if frac >= 0.5 else f
    return -matlab_ref_round(-x)

cfg = RegistrationConfig(input_path="/tmp/x.sbx")
for nt in [1, 9, 10, 11, 19, 20, 21, 29, 30, 31, 50, 200, 230, 249, 250, 251, 610, 630]:
    check("nchunks(nt=%d,cs=20)" % nt, cfg.nchunks(nt), matlab_ref_round(nt / 20))
cfg15 = RegistrationConfig(input_path="/tmp/x.sbx", chunksize=15)
for nt in [7, 8, 22, 23, 37, 38]:
    check("nchunks(nt=%d,cs=15)" % nt, cfg15.nchunks(nt), matlab_ref_round(nt / 15))

# ---- proj_planes_1based 'quarter' == MATLAB round(0.25*Nz):round(0.75*Nz) ----
for nz in range(2, 65):
    lo = matlab_ref_round(0.25 * nz); hi = matlab_ref_round(0.75 * nz)
    want_planes = list(range(lo, hi + 1))
    got_planes = cfg.proj_planes_1based(nz).tolist()
    if got_planes != want_planes:
        check("proj_planes quarter nz=%d" % nz, got_planes, want_planes)
print("PASS proj_planes 'quarter' matches MATLAB colon for Nz=2..64")

got_full = RegistrationConfig(input_path="/tmp/x.sbx", proj_range="full").proj_planes_1based(30).tolist()
check("proj_planes full nz=30", got_full, list(range(1, 31)))

# Nz=1 'quarter' -> MATLAB indexes 0 -> error; port must raise
try:
    cfg.proj_planes_1based(1)
    check("proj_planes nz=1 raises", False, True)
except ValueError:
    check("proj_planes nz=1 raises", True, True)

# ---- paths ----
c = RegistrationConfig(input_path="/data/run002.sbx", out_dir="/out")
check("shiftpath", c.shiftpath(), "/out/run002.dftshifts.npz")
check("zproj path", c.zproj_tiff_path(), "/out/run002_mean_zproj.tif")
c2 = RegistrationConfig(input_path="/data/run002.sbx", proj_type="max")
check("zproj path proj_type", c2.zproj_tiff_path(), "/data/run002_max_zproj.tif")

# ---- package import (no siblings present) ----
import cpstab
check("package imports", hasattr(cpstab, "run_pipeline"), True)

print()
print("FAILURES:", fails if fails else "none")
