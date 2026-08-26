#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""make_sbx.py -- convert T-major .npy / OME-TIFF into the .sbx + .mat that MATLAB can read.

A newly designed tool (no .m counterpart), serving one purpose: get the
MATLAB original pipeline (clean/run_registration.m) on the lab Dell to
consume pixels that are **bit-identical** to what the Python side gets, so
the timings on both sides are comparable and the results can be
cross-checked.

Why the format conversion is necessary
---------------------------------------
The MATLAB-side entry point `run_registration.m` only recognizes `.sbx`
(+ a same-name `.mat` sidecar): `GetDimensions` goes through
`pipe.io.sbxInfo`, and `MakeSBXall`/`DFT_warp_3D_2` go through
`pipe.imread` -> `sbxRead`'s fseek/fread. It can't read OME-TIFF, let alone
`.npy`. ConvertOIR_SBX.m can convert from a FluoView `.tif.frames`
directory, but that needs the original export directory, and all we have
is the OME-TIFF from bfconvert and the T-major storage from
scripts/relayout.py.

Uses the existing contract, not a newly written byte format
-------------------------------------------------------------
Writing goes through `cpstab.io_rw.RegWriter` (a line-by-line port of
RegWriter.m L1-111); the sidecar goes through `cpstab.io_rw.spoof_sbx_info_3d`
(SpoofSBXinfo3D.m L1-34) + `save_sbx_info` (ConvertOIR_SBX.m L39). In other
words there is no "my understanding of the sbx format" here -- it's a port
of the same two functions MATLAB itself uses for conversion, and right
after converting it reads back with `cpstab.io_rw.SbxFile` and compares
bit-for-bit.

The 65535 inversion is applied by RegWriter on write and undone by sbxRead
on read (see the io_rw module docstring), so the values MATLAB sees == the
values VolumeSource hands the Python pipeline.

Usage
-----
    # 40-frame subset (1.6 GB, for validating the conversion itself)
    python matlab_bench/make_sbx.py \
        --in  /Users/.../shipley_workspace/FAD-F_1_T0-39.tzcyx.npy \
        --out-base /Users/.../shipley_workspace/matlab_bench/FAD-F_1_T0-39 \
        --verify-all

    # full run (60 GiB, don't casually run this on the Mac -- confirm the
    # target disk has space first)
    python matlab_bench/make_sbx.py \
        --in  /Users/.../shipley_workspace/FAD-F_1_raw.tzcyx.npy \
        --out-base /Volumes/BIG/FAD-F_1_raw --verify 16

    # just want to see how big it'll be / where it'll go
    python matlab_bench/make_sbx.py --in X.npy --out-base Y --dry-run

Outputs
-------
    <out-base>.sbx    Nt*Nz records, each rows*cols*2*Nchan bytes
    <out-base>.mat    info struct (MAT v5), readable directly by MATLAB `load`

Both files must be in the same directory with the same name -- that's how
MATLAB's sbxInfo finds the sidecar.
"""

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYPORT = os.path.dirname(_HERE)
if _PYPORT not in sys.path:
    sys.path.insert(0, _PYPORT)

from cpstab.io_rw import (RegWriter, SbxFile, VolumeSource,   # noqa: E402
                          save_sbx_info, spoof_sbx_info_3d)


def matlabize(info):
    """Coerce the integers in info to double -- MATLAB-side numeric semantics depend on this.

    Numeric literals in MATLAB are all double, so the struct
    SpoofSBXinfo3D.m produces is **all double** except for
    `area_line = true` (logical) and `opto2pow = []` (empty double array).
    But Python-side `int` / `np.arange` get stored as int64 by
    scipy.io.savemat, and MATLAB reads them back as an integer type -- so
    the same sidecar has different arithmetic semantics on the two sides:

        double: 1719664640/512/512*1/4 - 1  ->  1639
        int64 : the same expression chain **rounds** at every step
                (MATLAB's integer division rounds, it doesn't truncate),
                and integer types are contagious -- double combined with
                int64 gives int64.

    On our data every step divides evenly, so the two forms happen to
    agree; but the same expression chain propagates through sbxInfo.m L96
    (`max_idx`), GetDimensions.m L16 (`Nt = floor(...)`), and
    run_registration.m L67 (`Nchunks = round(Nt/chunksize)`) -- the moment
    one of those doesn't divide evenly, int64 silently gives a different
    integer than the double version, and that's exactly the quantity that
    decides chunk splitting and frame count. This kind of "blows up only
    on different data" bug can't be left in.

    logical / float / empty arrays pass through unchanged.
    """
    out = {}
    for k, v in info.items():
        if isinstance(v, (bool, np.bool_)):
            out[k] = bool(v)                    # area_line = true
        elif isinstance(v, (int, np.integer)):  # NB: bool is a subclass of int,
            out[k] = float(v)                   #     so it must be checked first
        elif isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.integer):
            out[k] = v.astype(np.float64)       # otwave = 1:Nz
        else:
            out[k] = v
    return out


def build_info(source, nt=None):
    """The call site for spoof_sbx_info_3d, plus the type finalization for the sidecar.

    Argument order follows the MakeSBXall.m L134 version, (Nx, Ny, ...) =
    (rows, cols) -- the two call sites in SpoofSBXinfo3D.m have the
    argument order reversed (see the "ARGUMENT-ORDER TRAP" note in
    spoof_sbx_info_3d's docstring); only this version makes
    info.sz == [rows, cols], which is the order SbxFile / sbxInfo assume
    on readback. This project's frames are square (512x512), so it would
    still read back correctly even if reversed -- which is exactly why the
    order must be pinned explicitly here, or a non-square dataset would
    get silently transposed on some other machine.

    matlabize() only applies at the sidecar boundary; spoof_sbx_info_3d
    itself is untouched -- it serves the in-process info, where Python's
    int is fine.
    """
    nchan, nx, ny, nz, nt_all = (int(v) for v in source.metadata)  # nx = rows
    nt = int(nt_all if nt is None else min(nt, nt_all))
    info = spoof_sbx_info_3d(nx, ny, nz, nt, nchan)
    return matlabize(info), nt


def convert(in_path, out_base, nt=None, force=False, verify=8,
            verify_all=False, progress=True):
    """Write <out_base>.sbx + <out_base>.mat, then read back and verify.

    Returns (sbx_path, mat_path, nt_written).
    """
    src = VolumeSource(in_path)
    nchan, nx, ny, nz, _nt_all = (int(v) for v in src.metadata)
    info, nt = build_info(src, nt)
    store_dtype = np.dtype(src.dtype)

    sbx_path = out_base + ".sbx"
    mat_path = out_base + ".mat"
    total = nt * nz * nx * ny * 2 * nchan
    if progress:
        print("make_sbx: %s" % in_path)
        print("  (Nchan, Nx=rows, Ny=cols, Nz, Nt) = (%d, %d, %d, %d, %d)  %s"
              % (nchan, nx, ny, nz, nt, store_dtype))
        print("  -> %s  (%d records, %.1f GiB)"
              % (sbx_path, nt * nz, total / 2.0 ** 30))
        print("  -> %s  (info sidecar, MAT v5)" % mat_path)

    out_dir = os.path.dirname(os.path.abspath(sbx_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    # PRE-FLIGHT the overwrite guard, next to the disk-space one. RegWriter
    # enforces it too (io_rw.py L578-579), but it only opens AFTER the sidecar
    # below is written -- so pointing a DIFFERENT geometry at an existing
    # --out-base without --force used to overwrite the good .mat, then abort,
    # leaving the old .sbx bytes paired with a sidecar describing the new
    # rows/cols/Nz. Nothing downstream notices: sbx_info re-derives nframes
    # from the file SIZE (io_rw.py L96) and never cross-checks the sidecar, so
    # pipe.imread hands MATLAB silently transposed / mis-z-sliced volumes.
    # That is exactly the "sidecar that does not match the data" DESIGN NOTES
    # 3 calls the harmful case.
    if os.path.exists(sbx_path) and not force:
        raise IOError(
            "refusing to overwrite existing %s (pass --force to overwrite). "
            "Check first whether it came from a different input -- same "
            "name, different geometry will make .mat and .sbx mismatch, "
            "and MATLAB will read it without error but get it wrong."
            % sbx_path)
    st = os.statvfs(out_dir)
    free = st.f_bavail * st.f_frsize
    if free < total:
        raise IOError("target disk only has %.1f GiB free, need %.1f GiB (%s)"
                      % (free / 2.0 ** 30, total / 2.0 ** 30, out_dir))

    t0 = time.time()
    # Write the sidecar first: RegWriter only handles bytes, and without the
    # .mat MATLAB can't even read the dimensions. The reverse order (.sbx
    # then .mat) leaves a "visible but unreadable" large file if it dies
    # partway; writing the sidecar first at least makes the failure mode a
    # detectable "record count doesn't match".
    save_sbx_info(mat_path, info)

    with RegWriter(sbx_path, info, extension=".sbx", force=force) as w:
        for t in range(nt):
            # (Nc, rows, cols, Nz) -- this is exactly what RegWriter.write's
            # 4-D branch expects: MATLAB's [C, rows, cols, frames].
            vol = np.stack([src.get_volume(t, c) for c in range(nchan)])
            if np.issubdtype(store_dtype, np.integer):
                # get_volume returns float, but the values are the stored
                # integers unchanged (no arithmetic applied), so this
                # downcast is lossless and skips RegWriter's internal
                # clip+round float pass. Non-integer storage (e.g. a
                # float32 export) doesn't take this path -- it's left to
                # RegWriter's MATLAB-style uint16() rounding.
                vol = vol.astype(np.uint16)
            w.write(vol)
            if progress and (t % 20 == 0 or t == nt - 1):
                el = time.time() - t0
                done = (t + 1) * nz * nx * ny * 2 * nchan
                sys.stdout.write(
                    "\r  %d/%d volumes  %5.1f%%  %6.1f MB/s  eta %5.1f min"
                    % (t + 1, nt, 100.0 * (t + 1) / nt,
                       done / max(el, 1e-9) / 2.0 ** 20,
                       (nt - t - 1) / max(t + 1, 1) * el / 60.0))
                sys.stdout.flush()
        # RegWriter.close() calls print(curframe) -- MATLAB's disp -- keep
        # it as a record-count receipt; __exit__ goes through delete()
        # (silent), so this checks explicitly.
        wrote = w.curframe
    if progress:
        sys.stdout.write("\n")
    if wrote != nt * nz:
        raise AssertionError("RegWriter only accepted %d records, expected %d"
                             % (wrote, nt * nz))
    size = os.path.getsize(sbx_path)
    if size != total:
        raise AssertionError("%s is %d bytes, expected %d" % (sbx_path, size, total))

    n_checked = _verify(src, sbx_path, nt, nchan, verify, verify_all, progress)
    if progress:
        print("  verify: %d (t, c) volumes bit-identical" % n_checked)
        print("  wrote %s + %s  (%.1f GiB, %.1f min)"
              % (sbx_path, mat_path, total / 2.0 ** 30,
                 (time.time() - t0) / 60.0))
    src.close()
    return sbx_path, mat_path, nt


def _verify(src, sbx_path, nt, nchan, verify, verify_all, progress):
    """Read the .sbx back and compare bit-for-bit against the same volume in the source.

    Reads back via SbxFile(path) (no dims passed), i.e. **through the
    sidecar** -- so this verifies not just the bytes but also whether the
    .mat's geometry is self-consistent: get the record layout, rows/cols
    order, or otlevels (=Nz) wrong anywhere and the volumes read back will
    be misaligned.
    """
    if verify_all:
        pairs = [(t, c) for t in range(nt) for c in range(nchan)]
    elif verify > 0:
        rng = np.random.default_rng(0)
        idx = rng.choice(nt * nchan, size=min(int(verify), nt * nchan),
                         replace=False)
        pairs = sorted((int(i) // nchan, int(i) % nchan) for i in idx)
    else:
        return 0
    with SbxFile(sbx_path) as sb:
        if sb.metadata != src.metadata[:4] + (nt,):
            raise AssertionError("readback metadata %r != source %r"
                                 % (sb.metadata, src.metadata[:4] + (nt,)))
        for k, (t, c) in enumerate(pairs):
            a = sb.get_volume(t, c)
            b = src.get_volume(t, c)
            if not np.array_equal(a, b):
                d = np.abs(a - b)
                raise AssertionError(
                    "verify FAILED at (t=%d, c=%d): max|d|=%g, %d/%d pixels differ"
                    % (t, c, d.max(), int(np.count_nonzero(d)), d.size))
            if progress and verify_all and k % 20 == 0:
                sys.stdout.write("\r  verify %d/%d" % (k + 1, len(pairs)))
                sys.stdout.flush()
    if progress and verify_all:
        sys.stdout.write("\r" + " " * 40 + "\r")
    return len(pairs)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True,
                    help="T-major .npy (scripts/relayout.py) or OME-TIFF")
    ap.add_argument("--out-base", required=True,
                    help="output path without extension; writes <out-base>.sbx and .mat")
    ap.add_argument("--limit-t", type=int, default=None,
                    help="only convert the first N time points (for making a small sample)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing .sbx (RegWriter refuses to overwrite by default)")
    ap.add_argument("--verify", type=int, default=8,
                    help="how many (t, c) volumes to spot-check randomly (default 8, 0 = skip)")
    ap.add_argument("--verify-all", action="store_true",
                    help="compare every (t, c) -- use this on a subset")
    ap.add_argument("--dry-run", action="store_true",
                    help="only print dimensions/size/paths, write no files")
    a = ap.parse_args(argv)

    if a.dry_run:
        src = VolumeSource(a.inp)
        nchan, nx, ny, nz, nt_all = (int(v) for v in src.metadata)
        nt = nt_all if a.limit_t is None else min(a.limit_t, nt_all)
        total = nt * nz * nx * ny * 2 * nchan
        print("make_sbx (dry-run): %s" % a.inp)
        print("  (Nchan, Nx=rows, Ny=cols, Nz, Nt) = (%d, %d, %d, %d, %d)"
              % (nchan, nx, ny, nz, nt))
        print("  will write %s.sbx  = %d records, %.1f GiB"
              % (a.out_base, nt * nz, total / 2.0 ** 30))
        print("  will write %s.mat  = info sidecar" % a.out_base)
        src.close()
        return 0

    convert(a.inp, a.out_base, nt=a.limit_t, force=a.force,
            verify=a.verify, verify_all=a.verify_all)
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ===========================================================================
# DESIGN NOTES
# ===========================================================================
# 1. Why not just read TIFF on the MATLAB side. MATLAB's imread can read
#    TIFF, but the pipeline's whole IO seam is pipe.imread(path, k, N, pmt,
#    optolevel) -> sbxRead's fseek/fread semantics (1-based frame numbers,
#    pmt channel selection, otlevels frame picking). Making MATLAB read
#    TIFF would mean replacing sbxRead -- that changes the subject under
#    test, and the benchmark would no longer be "the original pipeline's
#    timing." Converting the data instead of the code is the only way to
#    keep the two sides comparable.
# 2. Why verification goes through SbxFile(path) and not SbxFile(path,
#    dims=...). Passing dims would bypass the sidecar and only verify
#    bytes. But the part of this conversion most prone to error, and most
#    silently so, is exactly the sidecar: the [rows, cols] order of sz
#    (the two call sites of spoof have it reversed), whether otlevels is
#    Nz, and the nchan<->channels inversion (sbxInfo.m L77-89 derives
#    nchan back from info.channels -- get the channel count backwards and
#    everything shifts). Reading back through the sidecar means an error
#    in any of these three spots blows up the bit-for-bit comparison.
# 3. Why .mat is written before .sbx. See the comment in convert():
#    failure modes need to be detectable. Also, RegWriter's force
#    semantics only protect .sbx -- the sidecar is overwritten
#    unconditionally. This is deliberate: an orphaned sidecar is harmless,
#    but a sidecar that doesn't match the data is the harmful case.
# 4. Completeness: record count (RegWriter.curframe) and file byte count
#    are both checked explicitly, rather than assuming "written means
#    correct." Same lesson as scripts/relayout.py DESIGN NOTES 8 -- .sbx
#    has no self-describing header, so a file short a few records just
#    reads back as "fewer frames," and GetDimensions derives Nt from file
#    size, so a short file silently turns into a "shorter experiment" with
#    no error raised.
# 5. What this doesn't do: no lineshift (ConvertOIR_SBX's 'lineshift'
#    option). The Python-side pipeline runs the apply stage with
#    lineshift=0 (the circshift in MakeSBXall.m L77 is the identity when
#    lineshift=0), and the two sides must agree, so what's written here is
#    the raw pixels with no row shift applied in advance.
