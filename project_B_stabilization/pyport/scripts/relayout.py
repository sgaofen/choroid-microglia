#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""relayout.py — rewrite a bfconvert OME-TIFF into a T-MAJOR .npy volume store.

New-design tool (no .m counterpart): the MATLAB pipeline read `.sbx`, whose
on-disk record order is already t-slowest/z-fastest, so one volume was one
contiguous run of bytes. The Python port ingests bfconvert output instead,
and bfconvert writes the OIR series as **ZTCYX** — z SLOWEST. Reading the
82 pages of a single (t) volume therefore means 82 seeks spread over the
whole 114 GB file, and with 10 worker processes the drive's request queue
degenerates (measured 2.4x on 10 processes instead of ~8x).

This script performs the one-off transpose to the layout the pipeline
actually consumes:

    (T, Z, C, Y, X) uint16, C-order, in a plain np.lib.format .npy

so that `source.get_volume(t, c)` becomes ONE contiguous 41*512*512*2-byte
run per channel (strided only by C), and N parallel workers each stream a
different, sequential region. `cpstab.io_rw.VolumeSource` accepts the
resulting `.npy` directly (same interface, same values — see io_rw
DESIGN NOTES "T-major .npy"), so nothing else in the pipeline changes.

Usage
-----
    python scripts/relayout.py --in FAD-F_1_raw.ome.tif \
                               --out FAD-F_1_raw.tzcyx.npy

    # axis order is read from series.axes; --axes overrides a wrong/absent one
    python scripts/relayout.py --in x.tif --out x.tzcyx.npy --axes ZTCYX

Disk-friendliness (the whole point) — see DESIGN NOTES 1/2:
  * READ: pages are consumed strictly in FILE ORDER (page 0 .. page N-1) in
    blocks of `--block-pages`, i.e. one purely sequential sweep. For the
    production ZTCYX file that sweep visits, for each z, the full contiguous
    run of that z's (t, c) pages — page p = (z*T + t)*C + c — which is the
    z-strip read the task asks for, without hard-coding the axis order.
  * WRITE: each page lands at out[t, z, c]; consecutive source pages differ
    in c (fastest) so writes are contiguous 2*Y*X*C-byte bursts, never
    single-page dribbles. Every output byte is written exactly once into a
    freshly allocated (sparse) file, so no read-modify-write ever happens.
"""

import argparse
import os
import sys
import time

import numpy as np
import tifffile

TARGET_AXES = "TZCYX"


def describe_series(series, axes=None):
    """Normalize a TIFF series to the (T, Z, C, Y, X) frame of reference.

    New-design helper; the axis-tolerance rules mirror
    cpstab.io_rw.VolumeSource.__init__ verbatim (unknown axes are accepted
    only when singleton, Y/X must be the last two axes) so that a file
    relayout accepts is exactly a file VolumeSource accepts.

    Returns
    -------
    sizes : dict         {'T': nt, 'Z': nz, 'C': nc, 'Y': ny, 'X': nx};
                         missing axes default to 1.
    plane_axes : list    the non-YX axis letters in FILE order; lowercase
                         entries are tolerated singleton axes.
    plane_dims : list    their sizes (C-order plane raveling, slowest first).
    """
    s_axes = (axes or series.axes).upper()
    shape = tuple(series.shape)
    if len(s_axes) != len(shape):
        raise ValueError("axes %r do not match shape %r" % (s_axes, shape))

    sizes = {}
    keep = []
    for ax, n in zip(s_axes, shape):
        if ax in TARGET_AXES:
            if ax in sizes:
                raise ValueError("axis %r appears twice in %r" % (ax, s_axes))
            sizes[ax] = int(n)
            keep.append(ax)
        elif n == 1:
            keep.append(ax.lower())          # tolerated singleton (S/Q/...)
        else:
            raise ValueError("unsupported axis %r of size %d (axes %r)"
                             % (ax, n, s_axes))
    kept = "".join(keep)
    if "Y" not in sizes or "X" not in sizes:
        raise ValueError("series has no Y/X axes: %r" % s_axes)
    if not kept.upper().endswith("YX"):
        raise ValueError("expected Y,X as the last axes, got %r — pass "
                         "--axes to override if the file is nonstandard"
                         % s_axes)
    for ax in TARGET_AXES:
        sizes.setdefault(ax, 1)

    plane_axes = [a for a in kept if a.upper() not in "YX"]
    plane_dims = [sizes[a] if a.isupper() else 1 for a in plane_axes]
    return sizes, plane_axes, plane_dims


def plane_destinations(plane_axes, plane_dims):
    """Per-plane (t, z, c) destination indices, in file order.

    Plane p of a tifffile series is the C-order ravel of the non-YX axes in
    the file's own axis order (the same mapping VolumeSource._page_axes
    inverts). For the production ZTCYX file this is
    p = (z*T + t)*C + c. Returns three int64 vectors of length prod(dims).
    """
    nplanes = int(np.prod(plane_dims)) if plane_dims else 1
    sub = np.unravel_index(np.arange(nplanes, dtype=np.int64),
                           plane_dims or [1])
    zero = np.zeros(nplanes, dtype=np.int64)
    out = []
    for ax in "TZC":
        out.append(np.asarray(sub[plane_axes.index(ax)], dtype=np.int64)
                   if ax in plane_axes else zero)
    return out[0], out[1], out[2]


def relayout(in_path, out_path, axes=None, block_pages=None, force=False,
             verify=8, progress=True, flush_bytes=2 << 30):
    """Write `in_path` (any TZCYX permutation) as a (T,Z,C,Y,X) .npy memmap.

    New-design function (no .m counterpart). Values are copied verbatim —
    same dtype, no scaling, no inversion (io_rw module docstring: the
    65535-inversion lives only inside `.sbx` bytes).

    Parameters
    ----------
    block_pages : int, optional   pages per sequential read, >= 1 (default:
                                  as many as fit in ~128 MB).
    verify : int                  re-read this many random planes afterwards
                                  and compare against the .npy (0 disables).
                                  Must be >= 0. NOT a substitute for the
                                  completeness invariant (DESIGN NOTES 8).
    flush_bytes : int             msync the output roughly this often
                                  (default 2 GiB; see DESIGN NOTES 2).

    Returns the output path.
    """
    if os.path.exists(out_path) and not force:
        raise IOError("refusing to overwrite %s (use --force)" % out_path)
    verify = int(verify)
    if verify < 0:
        raise ValueError("verify must be >= 0, got %d" % verify)

    t_start = time.time()
    with tifffile.TiffFile(in_path) as tif:
        series = tif.series[0]
        sizes, plane_axes, plane_dims = describe_series(series, axes)
        dtype = np.dtype(series.dtype)
        nt, nz, nc = sizes["T"], sizes["Z"], sizes["C"]
        ny, nx = sizes["Y"], sizes["X"]                 # rows, cols
        nplanes = int(np.prod(plane_dims)) if plane_dims else 1
        if nplanes != nt * nz * nc:
            raise ValueError("plane count %d != T*Z*C = %d (axes %r)"
                             % (nplanes, nt * nz * nc, series.axes))

        # One IFD usually holds one [Y, X] plane, but tifffile also writes
        # "depth" pages (several planes per IFD). key= indexes IFDs, so map
        # planes -> IFDs explicitly instead of assuming 1:1.
        pshape = tuple(series.pages[0].shape)
        if pshape[-2:] != (ny, nx):
            raise ValueError(
                "page shape %r does not end in (Y, X) = (%d, %d) — planar/RGB "
                "samples are not supported" % (pshape, ny, nx))
        planes_per_page = int(np.prod(pshape[:-2])) if len(pshape) > 2 else 1
        npages = len(series.pages)
        if npages * planes_per_page != nplanes:
            raise ValueError(
                "%d pages x %d planes/page != %d planes (axes %r)"
                % (npages, planes_per_page, nplanes, series.axes))

        plane_bytes = ny * nx * dtype.itemsize
        if block_pages is None:
            block_pages = max(1, (128 << 20)
                              // max(plane_bytes * planes_per_page, 1))
        block_pages = int(block_pages)
        if block_pages < 1:
            # A non-positive step makes range() yield nothing, so the copy
            # loop would be skipped and an all-zero store published as if it
            # were complete (DESIGN NOTES 8).
            raise ValueError("block_pages must be >= 1, got %d" % block_pages)

        tt, zz, cc = plane_destinations(plane_axes, plane_dims)

        total = nt * nz * nc * plane_bytes
        if progress:
            print("relayout: %s" % in_path)
            print("  axes %s %r  ->  TZCYX (%d, %d, %d, %d, %d) %s  [%.1f GiB]"
                  % (axes or series.axes, tuple(series.shape),
                     nt, nz, nc, ny, nx, dtype, total / 2.0 ** 30))
            print("  %d planes in %d pages, %d pages/read (%.0f MB)"
                  % (nplanes, npages, block_pages,
                     block_pages * planes_per_page * plane_bytes / 2.0 ** 20))

        tmp_path = out_path + ".partial"
        out = np.lib.format.open_memmap(
            tmp_path, mode="w+", dtype=dtype, shape=(nt, nz, nc, ny, nx))
        try:
            done = 0
            since_flush = 0
            for g0 in range(0, npages, block_pages):
                g1 = min(g0 + block_pages, npages)
                data = tif.asarray(key=slice(g0, g1), series=0)
                data = np.asarray(data).reshape((-1, ny, nx))
                p0 = g0 * planes_per_page
                for k in range(data.shape[0]):
                    p = p0 + k
                    out[tt[p], zz[p], cc[p]] = data[k]
                done = p0 + data.shape[0]
                since_flush += data.shape[0] * plane_bytes
                if since_flush >= flush_bytes:  # bound the dirty-page backlog
                    out.flush()                 # (msync walks the WHOLE map,
                    since_flush = 0             #  so not once per block)
                if progress:
                    el = time.time() - t_start
                    rate = done * plane_bytes / max(el, 1e-9) / 2.0 ** 20
                    sys.stdout.write(
                        "\r  %d/%d planes  %5.1f%%  %6.1f MB/s  eta %5.1f min"
                        % (done, nplanes, 100.0 * done / nplanes, rate,
                           (nplanes - done) / max(done, 1) * el / 60.0))
                    sys.stdout.flush()
            if progress:
                sys.stdout.write("\n")
            if done != nplanes:
                # The design guarantees every output byte is written exactly
                # once; assert the "at least once" half before publishing,
                # so no skipped-plane bug can ever ship a store whose gaps
                # read back as legitimate zeros (DESIGN NOTES 8).
                raise AssertionError(
                    "wrote %d of %d planes — refusing to publish a partially "
                    "filled store" % (done, nplanes))
            del out
        except BaseException:
            del out
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        os.replace(tmp_path, out_path)          # atomic: no half-written .npy

        if verify:
            rng = np.random.default_rng(0)
            chk = rng.choice(nplanes, size=min(verify, nplanes),
                             replace=False)
            got = np.load(out_path, mmap_mode="r")
            for p in sorted(int(v) for v in chk):
                g = p // planes_per_page
                ref = np.asarray(tif.asarray(key=slice(g, g + 1), series=0))
                ref = ref.reshape((-1, ny, nx))[p % planes_per_page]
                if not np.array_equal(got[tt[p], zz[p], cc[p]], ref):
                    raise AssertionError(
                        "verify FAILED at plane %d -> (t=%d, z=%d, c=%d)"
                        % (p, tt[p], zz[p], cc[p]))
            del got
            if progress:
                print("  verify: %d random planes bit-identical" % len(chk))

    if progress:
        print("  wrote %s  (%.1f GiB, %.1f min)"
              % (out_path, total / 2.0 ** 30, (time.time() - t_start) / 60.0))
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True,
                    help="input OME-TIFF / ImageJ TIFF (any TZCYX order)")
    ap.add_argument("--out", default=None,
                    help="output .npy (default: <input stem>.tzcyx.npy)")
    ap.add_argument("--axes", default=None,
                    help="override series axis order (e.g. ZTCYX)")
    ap.add_argument("--block-pages", type=int, default=None,
                    help="pages per sequential read, >= 1 "
                         "(default ~128 MB worth)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing output file")
    ap.add_argument("--verify", type=int, default=8,
                    help="random pages re-checked after the write (0 = none, "
                         "must be >= 0)")
    args = ap.parse_args(argv)

    out = args.out
    if out is None:
        stem = args.inp
        for ext in (".ome.tif", ".ome.tiff", ".tif", ".tiff"):
            if stem.lower().endswith(ext):
                stem = stem[:-len(ext)]
                break
        out = stem + ".tzcyx.npy"
    relayout(args.inp, out, axes=args.axes, block_pages=args.block_pages,
             force=args.force, verify=args.verify)
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ===========================================================================
# DESIGN NOTES
# ===========================================================================
# 1. Why file-order reads instead of an explicit "for z: read that z's pages"
#    loop. The two are the same sweep for the production ZTCYX file (z is the
#    slowest page axis, so file order IS z-strip order) but the file-order
#    formulation also handles the ImageJ TZCYX subsets used for regression
#    (FAD-F_1_T0-39.tif is T-major already) and any other permutation
#    bfconvert/Fiji might emit — with zero seeks in every case. The axis
#    order is never assumed: it comes from series.axes (or --axes) and is
#    inverted through page_destinations(), the same page-raveling contract
#    VolumeSource uses.
# 2. Write pattern. Destination offset of a page is
#    ((t*Z + z)*C + c)*Y*X*itemsize. Source pages advance with c fastest for
#    both ZTCYX and TZCYX inputs, so each burst of C pages is contiguous in
#    the output (1 MB for the production 512x512x2 case) and every output
#    byte is written exactly once. open_memmap creates the file sparse, so
#    first touch of a page is a zero-fill, NOT a read from disk — the
#    conversion costs one read pass plus one write pass, nothing more.
#    Periodic out.flush() keeps the dirty-page backlog bounded (a 60 GB
#    conversion otherwise leaves writeback entirely to the VM system) — but
#    np.memmap.flush() msyncs the ENTIRE mapping, so it is called per
#    ~flush_bytes (2 GiB) of progress, not per block: ~30 msyncs for the
#    production stack instead of ~480 over a 60 GB map.
# 3. .npy, not .zarr/.h5/.sbx. Requirements are numpy/scipy/tifffile only;
#    np.lib.format.open_memmap gives a self-describing header (shape, dtype,
#    C-order) that np.load(mmap_mode='r') reopens with zero copies and no
#    third-party dependency. Chunked/compressed containers would defeat the
#    purpose: the point is that a volume is a plain contiguous byte range.
# 4. dtype is taken from the series, not forced to uint16. All production
#    input is uint16; a float32 export would still relayout correctly, and
#    VolumeSource.dtype then reports float32 so the orchestrator's
#    "cast back to the native integer class" step correctly does nothing.
# 5. .partial + os.replace. A 114 GB conversion that dies halfway must not
#    leave a plausible-looking .npy behind: readers would silently see zeros
#    in the untouched region. The output name only appears once the file is
#    complete and verified.
# 6. verify samples PLANES (random (t,z,c) triples), not volumes, because a
#    transposition bug shows up as a permuted plane, which a per-plane
#    compare catches with certainty for the planes it draws. It is a cheap
#    smoke test, not a proof — cpstab/tests/test_tmajor.py does the
#    exhaustive per-(t,c) equality check against the TIFF VolumeSource.
# 7. Planes vs TIFF pages. tifffile's key= selects IFDs, and an IFD may hold
#    more than one [Y, X] plane (its "depth" pages, which tifffile itself
#    emits for some 5-D writes). Assuming 1 plane == 1 page silently reads
#    the wrong data for such files, so planes_per_page is derived from
#    series.pages[0].shape and every read is reshaped to (-1, Y, X). The
#    C-order plane ravel is unaffected by that grouping: pages are stored in
#    the series' own axis order, so a contiguous run of pages is still a
#    contiguous run of planes. Planar/RGB samples (last axis == samples) are
#    rejected outright rather than mis-transposed.
# 8. Completeness invariant (added in review). The sparse-output trick of
#    note 2 has a sharp edge: an UNWRITTEN region of the store reads back as
#    legitimate zeros, indistinguishable from real data. So "the copy loop
#    ran to completion" must be checked, not assumed. It was not: passing a
#    non-positive block_pages made `range(0, npages, step)` yield nothing,
#    the loop body never executed, and `os.replace` published a full-size,
#    correctly-shaped, entirely BLANK .npy with no error (reproduced:
#    relayout(..., block_pages=-1, verify=0) -> np.load(...).any() is False).
#    Two guards now stand between that and a published file:
#      * block_pages and verify are range-checked before any file is created;
#      * `done != nplanes` after the loop raises, and the raise lands in the
#        existing BaseException handler that unlinks the .partial — so the
#        output name never appears.
#    The invariant is deliberately independent of `verify`: verify is a
#    random SAMPLE and is switchable off (--verify 0), whereas this is an
#    exact count that costs nothing and cannot be disabled. Together with
#    note 2's "written exactly once", the pair pins both halves of the
#    guarantee — every plane at least once, and no plane twice.
# 9. The output NAME changes the pipeline's output names (review finding).
#    config.RegistrationConfig.out_base() derives every artifact from a
#    single os.path.splitext() of the input basename, so the container swap
#    is bit-identical in PIXELS but not in FILENAMES:
#      FAD-F_1_raw.ome.tif   -> FAD-F_1_raw.ome_mean_zproj.tif
#      FAD-F_1_raw.tzcyx.npy -> FAD-F_1_raw.tzcyx_mean_zproj.tif
#    (likewise .dftshifts.npz and .sbxall). That is pre-existing behavior —
#    out_base() strips ONE extension, which is why the TIFF run already
#    carries the '.ome' — but a reader comparing a T-major run against an
#    earlier TIFF run will find the files under different names. Naming is
#    deliberately NOT patched here: out_base() is on the replicate path and
#    the validated reference is literally named FAD-F_1_T0-39_mean_zproj.tif.
#    To keep the stem identical, name the store so that stripping its last
#    extension reproduces the TIFF's stem:
#      --out FAD-F_1_raw.ome.npy   (stem 'FAD-F_1_raw.ome', same as the TIFF)
#    The default --out (<stem>.tzcyx.npy) favors a self-describing filename
#    over stem parity; pass --out explicitly when parity matters.
