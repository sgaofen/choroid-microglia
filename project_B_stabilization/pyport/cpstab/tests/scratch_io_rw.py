# -*- coding: utf-8 -*-
"""Adversarial probes for io_rw.py vs MATLAB ground truth."""
import os, sys, tempfile, warnings
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from cpstab import io_rw
from cpstab.io_rw import (matlab_uint16, sbx_read, sbx_info, RegWriter,
                          spoof_sbx_info_3d, save_sbx_info, write2chan_tiff,
                          SbxFile, _rescale)

tmp = tempfile.mkdtemp(prefix="scratch_io_rw_")
print("tmp:", tmp)

# ---------------------------------------------------------------- probe 1
# NaN through matlab_uint16 (MATLAB uint16(NaN) == 0, deterministically)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    out = matlab_uint16(np.array([np.nan, 1.5, -np.inf, np.inf]))
    print("P1 matlab_uint16([nan,1.5,-inf,inf]) ->", out,
          "warnings:", [str(x.message) for x in w])

# NaN through RegWriter._clamp_uint16 path (clip keeps NaN as NaN)
x = np.clip(np.array([np.nan]), 0, 65535)
print("P1b np.clip(nan,0,65535) =", x)

# ---------------------------------------------------------------- probe 2
# Build a tiny sbx: nchan=2, 4x4 frames, Nz=2, Nt=3 -> roundtrip + pmt probes
rows, cols, Nz, Nt, Nchan = 4, 5, 2, 3, 2   # NON-square on purpose? -> RegWriter
# RegWriter with swapped spoof only passes square; use MakeSBXall order (correct)
info = spoof_sbx_info_3d(rows, cols, Nz, Nt, Nchan)   # sz=[rows,cols]
base = os.path.join(tmp, "t1")
save_sbx_info(base + ".mat", info)

rng = np.random.default_rng(0)
data = rng.integers(0, 60000, size=(Nchan, rows, cols, Nz * Nt)).astype(np.uint16)
rw = RegWriter(base + ".sbx", info, ".sbx", True)
rw.write(data)
rw.delete()

# roundtrip identity
back = sbx_read(base + ".sbx", 1, -1, -1, None)
print("P2 roundtrip identical:", np.array_equal(back, data))

# pmt=0: MATLAB x(0,...) ERRORS; python?
try:
    ch = sbx_read(base + ".sbx", 1, 2, 0, None)
    print("P2b pmt=0 silently returned; equals LAST channel:",
          np.array_equal(ch, data[1, :, :, 0:2]),
          "equals first channel:", np.array_equal(ch, data[0, :, :, 0:2]))
except Exception as e:
    print("P2b pmt=0 raised:", type(e).__name__, e)

# pmt=3 (nchan=2): MATLAB x(3,...) errors; python x[2] -> IndexError?
try:
    ch = sbx_read(base + ".sbx", 1, 2, 3, None)
    print("P2c pmt=3 silently returned shape", ch.shape)
except Exception as e:
    print("P2c pmt=3 raised:", type(e).__name__)

# ---------------------------------------------------------------- probe 3
# N==0 shapes (MATLAB: squeeze(x(1,:,:)) on [c,r,cols,0] -> [r, 0])
try:
    z = sbx_read(base + ".sbx", info["nframes"] + 1, 5, 1, None)
    print("P3 N==0 shape:", z.shape, "(MATLAB would be (rows, 0) 2-D)")
except Exception as e:
    print("P3 raised:", type(e).__name__, e)

# ---------------------------------------------------------------- probe 4
# optolevel read vs manual
v = sbx_read(base + ".sbx", 1, -1, 1, optolevel=2)
manual = data[0, :, :, 1::Nz]
print("P4 optolevel=2 identical:", np.array_equal(v, manual), v.shape)

# optolevel overrun clamp: k=Nt (1-based t index), N=5 > remaining
v2 = sbx_read(base + ".sbx", Nt, 5, 1, optolevel=1)
man2 = data[0, :, :, (Nt - 1) * Nz::Nz]
print("P4b overrun clamp identical:", (np.array_equal(v2, man2) if v2.ndim == 3
      else np.array_equal(v2, man2[:, :, 0])), v2.shape, man2.shape)

# ---------------------------------------------------------------- probe 5
# SbxFile.read out-of-range k: sbx_read raises; SbxFile.read?
sf = SbxFile(base + ".sbx")
try:
    r1 = sbx_read(base + ".sbx", info["nframes"] + 3, 2, 1, None)
    print("P5 sbx_read k>nframes returned", r1.shape)
except Exception as e:
    print("P5 sbx_read k>nframes raised:", type(e).__name__)
try:
    r2 = sf.read(info["nframes"] + 3, 2, 1, None)
    print("P5 SbxFile.read k>nframes returned", getattr(r2, "shape", r2))
except Exception as e:
    print("P5 SbxFile.read k>nframes raised:", type(e).__name__)

# SbxFile.read vs sbx_read equality on valid input
print("P5b SbxFile.read == sbx_read:",
      np.array_equal(sf.read(2, 3, 2, None), sbx_read(base + ".sbx", 2, 3, 2, None)))
sf.close()

# ---------------------------------------------------------------- probe 6
# write2chan_tiff with trailing singleton T (MATLAB ndims==3 -> disp, NO file)
p6 = os.path.join(tmp, "t6.tif")
mov = rng.integers(0, 60000, size=(2, 4, 5, 1)).astype(np.uint16)  # C,Y,X,T=1
write2chan_tiff(mov, p6)
print("P6 (C,Y,X,1) wrote file:", os.path.exists(p6),
      "(MATLAB: ndims==3 -> 'Movie dim must be 4' and NO file)")

# 5-D with T=1: MATLAB sees ndims 4 -> labels Z as frames; python labels slices
p6b = os.path.join(tmp, "t6b.tif")
mov5 = rng.integers(0, 60000, size=(2, 4, 5, 3, 1)).astype(np.uint16)
write2chan_tiff(mov5, p6b)
import tifffile
with tifffile.TiffFile(p6b) as tf:
    print("P6b python 5D T=1 imagej metadata:", tf.imagej_metadata)

# ---------------------------------------------------------------- probe 7
# rescale degenerate input
print("P7 _rescale(const) =", _rescale(np.full((2, 2), 7.0)).ravel()[:2])

# ---------------------------------------------------------------- probe 8
# uint16 scalar arithmetic dtype under numpy 2
arr = np.array([[1, 2]], dtype=np.uint16)
print("P8 (uint16max - arr).dtype =", (io_rw.UINT16_MAX - arr).dtype)

# ---------------------------------------------------------------- probe 9
# sbx_info cache aliasing: mutate returned dict, re-request
i1 = sbx_info(base + ".sbx")
i1["nframes"] = 99999
i2 = sbx_info(base + ".sbx")
print("P9 cache poisoned by caller mutation:", i2["nframes"] == 99999)

# ---------------------------------------------------------------- probe 10
# GetDimensions fallback when sidecar unreadable AND .tif.frames deleted:
from cpstab.io_rw import get_dimensions
try:
    get_dimensions(os.path.join(tmp, "nonexistent.sbx"), tmp, "nope")
except Exception as e:
    print("P10 get_dimensions (no sidecar, no frames dir) raised:",
          type(e).__name__, "(MATLAB with v7.3 sidecar present: succeeds)")

# ---------------------------------------------------------------- probe 11
# imread N as 0-d array
from cpstab.io_rw import imread as io_imread
r = io_imread(base + ".sbx", 1, np.array(2), 1, None)
print("P11 imread(N=np.array(2)) frames returned:",
      r.shape, "(MATLAB N=2 -> 2 frames; port treats as -1 -> all?)")
