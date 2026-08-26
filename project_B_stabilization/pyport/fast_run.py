#!/usr/bin/env python
"""cpstab parallel driver -- multiprocess version of run_pipeline.

Registration stage: 75 chunks are mutually independent -> process-pool
parallel (orchestrator._process_chunk, same implementation as the serial
version); the parent process does the inter-chunk stitching and
.dftshifts save (same assembly code as dft_warp_3d_2 L110-152).
Apply stage: independent per volume -> process-pool parallel by T segment
(apply_project's _process_volume / _apply_shifts_volume / _project, setup
code mirrors make_sbxall L26-L63/L88).
zproj_reg's temporal refinement and TIFF write stay serial (chained
dependency / one-shot).

Numerical contract: each chunk / each volume's computation is
self-contained, parallelism only changes scheduling, not the math ->
output should match the serial run_pipeline bit-for-bit (verify with
--compare).

Input: --raw accepts bfconvert's OME-TIFF / ImageJ TIFF, and also accepts
the T-major .npy volume store produced by scripts/relayout.py
(T, Z, C, Y, X). Both are read through the same io_rw.VolumeSource
interface, bit-for-bit identical values (io_rw PORTING NOTES #16); on
.npy the apply stage additionally addresses by block (--read-mb controls
block size; that's a mmap view, not a real read, see DESIGN NOTES 1).

Precision: --dtype float32 (alias --compute-dtype) turns on fast mode
(cpstab/precision.py). Pixel domain drops to single precision, the
correlation (registration) math stays double precision -- see
precision.py for the measured rationale.

Algorithm: --mode improved turns on all four science corrections
(cpstab/improved.py). Default replicate matches the serial version
bit-for-bit.

Both of these settings are per-process globals, so every worker must set
them once in its own process: both ride the job tuple and get installed
on the first lines of the worker function. Forgetting to set them means
the worker runs float64/replicate (slower but MATLAB-faithful), never the
other way -- a half-improved result cannot silently happen.

Usage:
  python fast_run.py --raw RAW.tzcyx.npy --out-dir DIR [--workers 10]
                     [--refchannel 1] [--dtype float32]
                     [--mode improved] [--read-mb 2048]
                     [--compare serial-version-zproj.tif]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cpstab import orchestrator as orc
from cpstab import apply_project as ap
from cpstab.config import RegistrationConfig
from cpstab.io_rw import VolumeSource, write_zproj_tiff
from cpstab.improved import FEATURES, set_chain_refine_guard, set_features, set_mode
from cpstab.pipeline import matlab_uint16, _apply_io_adapter
from cpstab.precision import set_compute_dtype, get_compute_dtype


# ---------------------------------------------------------------------------
# workers (module-level: picklable under spawn)
# ---------------------------------------------------------------------------
def _block_volume(blk, j, nchan, pmt, fdt):
    """One volume out of a read_block() slab, in the imread_fn layout.

    This is literally pipeline._apply_io_adapter's imread_fn composed with
    io_rw.VolumeSource.get_volume, with `self._npy[t, :, c]` replaced by the
    already-resident `blk[j, :, c]` — same integers, same cast, same
    transpose, so the two routes are bit-for-bit identical (io_rw PORTING
    NOTES #19). Returns (Nc, Y, X, Nz) for pmt == -1, else the squeezed
    (Y, X, Nz) that MATLAB's sbxRead returns for a single channel.
    """
    chans = range(int(nchan)) if pmt == -1 else [int(pmt) - 1]
    vols = np.stack([
        np.ascontiguousarray(
            np.transpose(np.asarray(blk[j, :, c], dtype=fdt), (1, 2, 0)))
        for c in chans])
    return vols if pmt == -1 else vols[0]


def _block_volume_pm(blk, j, nchan, pmt, fdt):
    """_block_volume's volume in the PLANE-MAJOR layout: (Nc, Nz, Y, X).

    Same integers, same cast, one fewer shuffle: the store is already
    plane-major (blk[t, z, c] is a Y-by-X frame), so this is the identical
    uint16 -> float cast with the transpose left out. `_block_volume`'s
    (Y, X, Nz) result only ever gets transposed straight back inside
    _apply_shifts_volume; see apply_project's _PLANE_MAJOR_NOTE.
    """
    chans = range(int(nchan)) if pmt == -1 else [int(pmt) - 1]
    vols = np.stack([np.asarray(blk[j, :, c], dtype=fdt) for c in chans])
    return vols if pmt == -1 else vols[np.newaxis, 0]


def _tail_threads():
    n = os.environ.get("CPSTAB_TAIL_THREADS")
    if n:
        try:
            return max(1, int(n))
        except ValueError:
            pass
    return max(1, min(os.cpu_count() or 1, 18))


def _uint16_by_t(zproj, block=32):
    """pipeline.matlab_uint16 applied one T block at a time.

    Identical output, bounded transient. matlab_uint16 forces float64 and
    builds an eight-deep chain of full-size temporaries (isnan / abs /
    minimum / floor / compare / sign / multiply / clip) with almost no reuse:
    measured high-water is 6.375x the float64 size of its input, dead linear
    in Nt (Nt=80 -> 1.99 GiB, 160 -> 3.99, 320 -> 7.97, 750 -> 18.68). The
    production projection is (2, 512, 512, 1500) = 5.86 GiB as float64, so
    the one-shot cast alone transients ~37 GiB and the whole tail of main()
    was measured at 8.4x = ~49 GiB. --dtype float32 does not help: the cast
    forces float64 on purpose (RegistrationMasterPipeline.m L44).

    Slicing the LAST axis is safe because the cast is elementwise -- no
    reduction, no neighbourhood -- so every block sees exactly the values it
    would have seen in one call. Verified bitwise, NaN path included.
    """
    zproj = np.asarray(zproj)
    if zproj.ndim < 1 or zproj.shape[-1] <= int(block):
        return matlab_uint16(zproj)
    out = np.empty(zproj.shape, dtype=np.uint16)
    spans = [(t0, min(t0 + int(block), zproj.shape[-1]))
             for t0 in range(0, zproj.shape[-1], int(block))]

    def one(s):
        out[..., s[0]:s[1]] = matlab_uint16(zproj[..., s[0]:s[1]])

    # The blocks were already independent (that is what makes the split
    # bitwise safe); running them on a thread pool only changes the order
    # they are visited in. numpy's ufuncs release the GIL, and this cast is
    # an eight-deep chain of them over 5.9 GB.
    nw = min(_tail_threads(), len(spans))
    if nw <= 1:
        for s in spans:
            one(s)
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=nw) as ex:
            list(ex.map(one, spans))
    return out


def _reg_worker(args):
    (raw_path, chunk, chunkframes, t_chunk, refchannel, edges, scale,
     tforms, optotune, reftype, blurfactor, keepingfactor, start_plane,
     compute_dtype, mode, cr_cap, cr_min_ncc, features) = args
    set_compute_dtype(compute_dtype)   # per-process globals; see module
    set_mode(mode)                     # docstring -- both, on the first lines
    # correction 3's trust gate is a THIRD per-process setting the registration
    # stage reads (cpstab/improved.py); it must ride the job tuple for the same
    # reason the other two do -- a worker that misses it would silently run a
    # different algorithm from the parent.
    set_chain_refine_guard(cr_cap, cr_min_ncc)
    if features is not None:           # ablation: per-correction sub-switches
        set_features(**features)
    deps = orc._import_deps()
    src = VolumeSource(raw_path)
    _info, reader = orc._resolve_source(src, deps[0])
    out = orc._process_chunk(reader, chunk, chunkframes, t_chunk, refchannel,
                             edges, scale, tforms, optotune, reftype,
                             blurfactor, keepingfactor, start_plane, deps=deps)
    return chunk, out


def _apply_worker(args):
    (raw_path, t_lo, t_hi, nz, nc, pmt, edges, tforms,
     rs_slab, cs_slab, zs_slab, idx0, compute_dtype, mode, read_vols,
     features) = args
    set_compute_dtype(compute_dtype)   # per-process globals; see module
    set_mode(mode)                     # docstring -- both, on the first lines
    if features is not None:           # ablation: per-correction sub-switches
        set_features(**features)
    src = VolumeSource(raw_path)
    _sbx_info_fn, imread_fn = _apply_io_adapter(src)
    fdt = get_compute_dtype()
    read_vols = max(1, int(read_vols))
    # PLANE-MAJOR fast path (apply_project's _PLANE_MAJOR_NOTE).  Eligible
    # only where _process_volume reduces to its as_float cast, i.e. no line
    # shift (fast_run passes 0 literally, below), no crop, and an exactly
    # identity optotune tform on every plane -- which is what opttype 'none'
    # produces.  Improved mode is excluded because its sub-plane Z helper
    # indexes axis 3.  Anything that fails the gate, and every non-.npy
    # source (blk is None), runs the ordinary path unchanged.
    plane_major = (mode == "replicate"
                   and all(int(e) == 0 for e in edges)
                   and all(ap._is_identity_tform(t) for t in tforms))
    out = None
    for b0 in range(t_lo, t_hi, read_vols):
        b1 = min(b0 + read_vols, t_hi)
        # A memmap VIEW of the sub-slab, not a bulk read -- pages still fault
        # in on demand below (io_rw.read_block); None on any other source, and
        # then the per-volume path runs unchanged (DESIGN NOTES 1).
        blk = src.read_block(b0, b1)
        for i in range(b0, b1):
            j = i - t_lo
            z = ap._z_shift_for(zs_slab[j])                          # L111
            if blk is not None and plane_major:
                warp4 = _block_volume_pm(blk, i - b0, nc, pmt, fdt)
                if (warp4.ndim != 4 or warp4.shape[0] != nc
                        or warp4.shape[1] != nz):
                    raise ValueError("plane-major volume has shape %r; "
                                     "expected (Nc=%d, Nz=%d, Y, X)"
                                     % (warp4.shape, nc, nz))
                reg = ap._apply_shifts_volume(warp4, rs_slab[:, j],
                                              cs_slab[:, j], z,
                                              plane_major=True)
                # back to (Nc, Y, X, Nplanes) for the projection: _project's
                # mean is over the LAST axis of a C-contiguous block, and
                # that layout is what fixes its summation order.  reg[:, idx0]
                # is a fancy index, so this materializes the same 21 planes
                # the (Nc, Y, X, Nz) path hands it, in the same order.
                proj = ap._project(
                    np.ascontiguousarray(np.moveaxis(reg[:, idx0], 1, 3)),
                    "mean", False)                                   # L120
                if out is None:
                    out = np.zeros(proj.shape + (t_hi - t_lo,),
                                   dtype=proj.dtype)
                out[..., j] = proj
                continue
            if blk is None:
                k = nz * i + 1                           # MakeSBXall.m L74
                mov = imread_fn(raw_path, k, nz, pmt, None)
            else:
                mov = _block_volume(blk, i - b0, nc, pmt, fdt)
            raw = ap._normalize_movie(mov, nc, nz)
            warp4 = ap._process_volume(raw, 0, edges, tforms, True)  # L77-91
            reg = ap._apply_shifts_volume(warp4, rs_slab[:, j],
                                          cs_slab[:, j], z)
            proj = ap._project(reg[:, :, :, idx0], "mean", False)    # L120
            if out is None:
                # follow proj's class, or a float32 slab would be widened
                # back to float64 here and shipped double-size through the
                # pickle
                out = np.zeros(proj.shape + (t_hi - t_lo,), dtype=proj.dtype)
            out[..., j] = proj
        blk = None                     # drops the VIEW object only; the pages
                                       # it touched are the kernel's to reclaim
    return t_lo, out


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--raw", required=True,
                    help="OME-TIFF / ImageJ TIFF, or a T-major .npy volume "
                         "store from scripts/relayout.py")
    pa.add_argument("--out-dir", required=True)
    pa.add_argument("--workers", type=int, default=10)
    pa.add_argument("--refchannel", type=int, default=1)
    pa.add_argument("--scale", type=int, default=4)
    pa.add_argument("--chunksize", type=int, default=20)
    pa.add_argument("--proj-range", default="quarter")
    pa.add_argument("--dtype", "--compute-dtype", dest="compute_dtype",
                    default="float64", choices=["float64", "float32"],
                    help="float64=replicate precision (default); "
                         "float32=fast mode, pixel domain single precision, "
                         "registration correlation still double precision "
                         "(cpstab/precision.py)")
    pa.add_argument("--mode", default="replicate",
                    choices=["replicate", "improved"],
                    help="replicate=MATLAB-faithful (default, matches serial "
                         "version bit-for-bit); improved=all four science "
                         "corrections on (cpstab/improved.py)")
    pa.add_argument("--chain-refine-cap", type=float, default=3.0,
                    help="correction 3 trust gate: max per-plane refine "
                         "magnitude, unit=registration grid px (x scale for "
                         "full-resolution px); planes over the limit fall "
                         "back to the DFT_rect chain value. "
                         "inf = disable magnitude gate (improved mode only)")
    pa.add_argument("--chain-refine-min-ncc", type=float, default=0.30,
                    help="correction 3 trust gate: lower bound on the "
                         "plane-vs-volume-mean demeaned normalized "
                         "correlation [0,1); planes below this fall back to "
                         "the chain value. 0 = disable correlation gate")
    pa.add_argument("--features", default=None,
                    help="improved-mode ablation: comma list of corrections "
                         "to ENABLE (%s); the rest are disabled. Omit for "
                         "the mode's defaults." % ",".join(FEATURES))
    pa.add_argument("--read-mb", type=int, default=2048,
                    help="upper bound on the block size each apply-stage "
                         "worker addresses at once (MB, .npy input only; "
                         "0 = the whole slab as one block). Note this is "
                         "the mmap view's block size, not a real read into "
                         "memory -- see DESIGN NOTES 1")
    pa.add_argument("--compare", default=None,
                    help="serial-version zproj tif, for bit-for-bit "
                         "comparison")
    a = pa.parse_args()

    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    cfg = RegistrationConfig(input_path=a.raw, out_dir=a.out_dir,
                             refchannel=a.refchannel, scale=a.scale,
                             chunksize=a.chunksize, proj_range=a.proj_range,
                             compute_dtype=a.compute_dtype, mode=a.mode,
                             chain_refine_cap=a.chain_refine_cap,
                             chain_refine_min_ncc=a.chain_refine_min_ncc)
    # the parent does the assembly, the median centring, the zproj_reg
    # refinement and the write, so it needs the settings too (workers get
    # them through their job tuples)
    set_compute_dtype(cfg.compute_dtype)
    set_mode(cfg.mode)
    set_chain_refine_guard(cfg.chain_refine_cap, cfg.chain_refine_min_ncc)
    feat = None
    if a.features is not None:
        chosen = {f.strip() for f in a.features.split(",") if f.strip()}
        bad = chosen - set(FEATURES)
        if bad:
            raise SystemExit("--features: unknown correction(s) %s; valid: %s"
                             % (sorted(bad), ",".join(FEATURES)))
        feat = {f: (f in chosen) for f in FEATURES}
        set_features(**feat)
        print("fast_run: features override = %s" % feat, flush=True)
    os.makedirs(a.out_dir, exist_ok=True)

    source = VolumeSource(a.raw)
    nchan, nx, ny, nz, nt = (int(v) for v in source.metadata)
    # read_block returns None for anything but a T-major store, so an
    # EMPTY block is the zero-cost way to ask "is bulk reading available?"
    tmajor = source.read_block(0, 0) is not None
    itemsize = np.dtype(source.dtype).itemsize
    vol_bytes = nz * nchan * nx * ny * itemsize
    nchunks = cfg.nchunks(nt)
    chunkframes = int(nz * (nt // nchunks))
    t_chunk = chunkframes // nz
    start_plane = int(orc.matlab_round(nz / 2.0))
    edges = [0, 0, 0, 0]
    tforms = [np.eye(3, dtype=np.float64) for _ in range(nz)]   # opttype none
    print("fast_run: Nt=%d Nz=%d Nchunks=%d workers=%d dtype=%s mode=%s"
          % (nt, nz, nchunks, a.workers, cfg.compute_dtype, cfg.mode),
          flush=True)
    print("fast_run: source=%s (%s), volume=%.1f MB"
          % (os.path.basename(a.raw),
             "T-major .npy, bulk reads ON" if tmajor
             else "TIFF, per-volume reads",
             vol_bytes / 2.0 ** 20), flush=True)

    # ---- registration stage: chunk parallel -----------------------------
    jobs = [(a.raw, c, chunkframes, t_chunk, a.refchannel, edges, a.scale,
             tforms, "true", "mean", 1, 0.95, start_plane, cfg.compute_dtype,
             cfg.mode, cfg.chain_refine_cap, cfg.chain_refine_min_ncc, feat)
            for c in range(nchunks)]
    results = [None] * nchunks
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for chunk, out in ex.map(_reg_worker, jobs):
            results[chunk] = out
            print("reg chunk %d/%d  %.0fs" % (chunk + 1, nchunks,
                                              time.time() - t0), flush=True)
    t_reg = time.time() - t0
    print("registration parallel done %.1f min" % (t_reg / 60), flush=True)

    # ---- assembly (same as dft_warp_3d_2 L110-152) -----------------------
    _ta = time.time()
    deps = orc._import_deps()
    dftreg = deps[1]
    rs_cells = [r[0] for r in results]
    cs_cells = [r[1] for r in results]
    zs_cells = [r[2] for r in results]
    ref_all = [r[3] for r in results]

    ref_final_f = np.fft.fftn(ref_all[0])                       # L110-114
    interchunk = np.zeros((nchunks, 3))

    def _inter(j):
        # one chunk reference against the first, independent of every other
        # j; dftregistration3d is a pure function and numpy's FFT releases
        # the GIL, so the pool only reorders the visits
        interchunk[j, :] = np.ravel(
            dftreg.dftregistration3d(ref_final_f, np.fft.fftn(ref_all[j]), 2))

    from concurrent.futures import ThreadPoolExecutor
    _nw = min(_tail_threads(), nchunks)
    if _nw <= 1:
        for j in range(nchunks):
            _inter(j)
    else:
        with ThreadPoolExecutor(max_workers=_nw) as ex:
            list(ex.map(_inter, range(nchunks)))
    RS = np.concatenate(rs_cells, axis=1)                       # L121-123
    CS = np.concatenate(cs_cells, axis=1)
    ZS = np.concatenate(zs_cells, axis=1)
    RS_chunk = orc.matlab_imresize((interchunk[:, 0] * a.scale)[None, :],
                                   output_shape=RS.shape, method="nearest")
    CS_chunk = orc.matlab_imresize((interchunk[:, 1] * a.scale)[None, :],
                                   output_shape=CS.shape, method="nearest")
    ZS_chunk = orc.matlab_imresize(interchunk[:, 2][None, :],
                                   output_shape=ZS.shape, method="nearest")
    tf_full = []
    for t in tforms:
        t2 = np.array(t, copy=True)
        t2[2, 0:2] *= a.scale
        tf_full.append(t2)
    shift_result = {"RS": RS, "CS": CS, "ZS": ZS, "RS_chunk": RS_chunk,
                    "CS_chunk": CS_chunk, "ZS_chunk": ZS_chunk,
                    "tforms_optotune_full": np.stack(tf_full)}
    shiftpath = cfg.shiftpath()
    with open(shiftpath, "wb") as f:
        np.savez(f, **shift_result)
    print("[timing] assemble+shiftsave %.1fs" % (time.time() - _ta), flush=True)

    # ---- apply stage setup (same as make_sbxall L26-L63) -----------------
    # _median_centering: MATLAB's per-column median (replicate) or a global
    # scalar (improved correction 1) -- see cpstab/improved.py
    zs_total = np.ravel((ZS + ZS_chunk)
                        - ap._median_centering(ZS + ZS_chunk))
    rs_total = (RS + RS_chunk)
    rs_total = rs_total - ap._median_centering(rs_total)
    cs_total = (CS + CS_chunk)
    cs_total = cs_total - ap._median_centering(cs_total)
    # The apply stage must be sized by the number of columns the REGISTRATION
    # stage actually produced, not by the source's Nt. Nchunks*(Nt//Nchunks)
    # is < Nt whenever Nt % Nchunks != 0 (chunkframes = Nz*floor(Nt/Nchunks),
    # orchestrator L27), and using Nt then walked off the end of every slab:
    # `zs_slab[j]` in _apply_worker raised IndexError. Nt=1500/chunksize=20
    # divides evenly, which is the only reason production never hit it; 94.4%
    # of Nt in [100, 2000] do not. The serial path already does exactly this
    # (apply_project.make_sbxall L1217-1219 takes nt from cs.shape[1]), so
    # this is also what makes fast_run and run_pipeline agree on such a file.
    nt_reg = int(RS.shape[1])
    if nt_reg != nt:
        print("fast_run: registration produced %d of %d timepoints "
              "(Nt %% Nchunks != 0); applying to those %d, like run_pipeline"
              % (nt_reg, nt, nt_reg), flush=True)
    nc = nchan
    pmt = -1 if nc == 2 else 1
    tforms_n = ap._normalize_tforms(np.stack(tf_full), nz)
    proj_planes = cfg.proj_planes_1based(nz)
    idx0 = ap._resolve_proj_range(proj_planes, nz)

    # ---- apply stage: T-segment parallel (PASS1, make_sbxall L72-123) ---
    n_slabs = a.workers * 3        # subdivide to smooth out fast/slow workers
    bounds = np.linspace(0, nt_reg, n_slabs + 1).astype(int)
    slab_max = int(np.max(np.diff(bounds))) if n_slabs else nt_reg
    # how many volumes to read in sequence at once: 0 = the whole segment,
    # otherwise converted from the --read-mb memory budget (per worker;
    # DESIGN NOTES 1)
    read_vols = (slab_max if a.read_mb <= 0
                 else max(1, (int(a.read_mb) << 20) // max(vol_bytes, 1)))
    if tmajor:
        # "addressed at once", NOT "resident": read_block returns a memmap
        # view, so this is the loop granularity, not a memory budget being
        # spent (io_rw.read_block / DESIGN NOTES 1). Saying "bulk read = N
        # volumes (X MB) per worker" here reported a residency that never
        # materialized.
        print("fast_run: apply block = %d volumes (%.0f MB of mmap addressed "
              "at once) per worker, %d slabs"
              % (min(read_vols, slab_max),
                 min(read_vols, slab_max) * vol_bytes / 2.0 ** 20,
                 len(bounds) - 1), flush=True)
    jobs = [(a.raw, int(lo), int(hi), nz, nc, pmt, edges, tforms_n,
             rs_total[:, lo:hi], cs_total[:, lo:hi], zs_total[lo:hi], idx0,
             cfg.compute_dtype, cfg.mode, read_vols, feat)
            for lo, hi in zip(bounds[:-1], bounds[1:]) if hi > lo]
    zproj_raw = None
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for t_lo, out in ex.map(_apply_worker, jobs):
            if zproj_raw is None:
                # dtype from the workers' slabs, so fast mode is not undone
                # by a float64 accumulator in the parent
                zproj_raw = np.zeros(out.shape[:-1] + (nt_reg,),
                                     dtype=out.dtype)
            zproj_raw[..., t_lo:t_lo + out.shape[-1]] = out
            done += 1
            print("apply slab %d/%d  %.0fs" % (done, len(jobs),
                                               time.time() - t0), flush=True)
    t_apply = time.time() - t0 - t_reg
    print("apply parallel done %.1f min" % (t_apply / 60), flush=True)

    # ---- zproj_reg refine + write (serial, make_sbxall L125 / master L43-44) --
    _tz = time.time()
    zproj_mean, _r, _c, _tf = ap.zproj_reg(
        1, nt_reg, pmt, idx0 + 1, refchan=a.refchannel, zproj_raw=zproj_raw)
    print("[timing] zproj_reg %.1fs" % (time.time() - _tz), flush=True)
    # movie-level trajectory, DIAGNOSTIC ONLY (dropped on the floor until now).
    # Phase-1 of the event-frame plan needs its jump distribution to place a
    # data-derived trust cap; saving it changes no pixel anywhere.
    np.savez(shiftpath.replace(".dftshifts.npz", ".zprojtraj.npz"),
             R=np.asarray(_r, dtype=np.float64),
             C=np.asarray(_c, dtype=np.float64))
    del zproj_raw                  # zproj_reg returned a fresh array (it does
                                   # np.zeros_like); holding both to the end
                                   # doubled the parent's floor for nothing
    savepath = cfg.zproj_tiff_path()
    _tc = time.time()
    _u16 = _uint16_by_t(zproj_mean)
    print("[timing] uint16cast %.1fs" % (time.time() - _tc), flush=True)
    del zproj_mean
    _tw = time.time()
    write_zproj_tiff(_u16, savepath)
    print("[timing] tiffwrite %.1fs" % (time.time() - _tw), flush=True)
    del _u16
    total = time.time() - t0
    print("fast_run DONE %.1f min (reg %.1f + apply %.1f + refine/write %.1f)"
          % (total / 60, t_reg / 60, t_apply / 60,
             (total - t_reg - t_apply) / 60), flush=True)
    print("wrote %s" % savepath, flush=True)

    # ---- compare ----------------------------------------------------------
    if a.compare:
        import tifffile
        aa = tifffile.imread(savepath)
        bb = tifffile.imread(a.compare)
        same = np.array_equal(aa, bb)
        print("bitwise identical to %s : %s" % (a.compare, same), flush=True)
        if not same:
            d = aa.astype(np.int64) - bb.astype(np.int64)
            print("  diff: max|d|=%d, frac nonzero=%.2e"
                  % (np.abs(d).max(), np.count_nonzero(d) / d.size), flush=True)


if __name__ == "__main__":
    main()


# ===========================================================================
# DESIGN NOTES
# ===========================================================================
# 1. Apply-stage blocking (--read-mb). On a T-major .npy store a RUN of
#    volumes is one contiguous byte range, so each worker addresses its
#    sub-slab as one array via VolumeSource.read_block() and then walks it
#    volume by volume. read_block() returns None for TIFF sources and the
#    loop falls back to the ordinary per-volume imread_fn path, unchanged.
#    There is deliberately no TIFF emulation: two code paths that must agree
#    bit for bit are a liability, and the fallback is the already-validated
#    one.
#    CORRECTION (this note used to claim otherwise). read_block returns a
#    memmap VIEW — np.asarray on an np.memmap copies nothing — so there is
#    no "single sequential read", no kernel prefetch of the slab, and no
#    overlap of one worker's read with another's compute. The page-fault
#    pattern is byte-for-byte the one the per-volume path had. Both the
#    claim and its remedy were measured (numbers in io_rw.read_block's
#    docstring): forcing a real np.array() copy is a 15-25% LOSS
#    single-process and lands inside the run-to-run noise at 10 workers,
#    while making ~2.1 GB per worker genuinely resident. So the code keeps
#    the view and the claim is withdrawn. What survives is a flatter loop
#    and one bounds check per block instead of per volume — small, real,
#    and not what the note used to advertise.
# 2. Why --read-mb has a nonzero default (2048). Read it as loop
#    granularity, NOT as a memory budget: with a view there is nothing to
#    budget, and the ~21 GB across 10 workers this note used to quote was
#    never allocated (the kernel's page cache holds whatever it likes and
#    reclaims it freely). The default is kept because it costs nothing, it
#    bounds residency the day read_block is ever made to copy, and changing
#    it cannot change any value — only how many volumes are addressed at a
#    time. --read-mb 0 means "the whole slab as one block".
# 3. --dtype is an ALIAS of --compute-dtype, not a new setting: both spell
#    the same argparse dest, so old command lines and scripts keep working
#    and there is exactly one field (cfg.compute_dtype) carrying it into the
#    workers. Same for --mode, which was already the pass-through of
#    cfg.mode. Neither is inferred from anything: a run is replicate/float64
#    unless the operator says otherwise, so the failure mode of a forgotten
#    flag is "slower and MATLAB-faithful", never "silently improved".
# 4. Per-slab apply progress. The apply stage used to print nothing between
#    "registration done" and "apply done", which on the production stack is
#    a ~40 minute silence — indistinguishable from a hang, and the first
#    thing anyone benchmarking wants is a rate. ex.map yields in job order,
#    so the counter is monotone even though the workers finish out of order.
# 5. NOT changed: the registration stage still reads through
#    orchestrator._resolve_source one chunk at a time. Its reads are already
#    coarse (a whole chunk of downsampled reference planes) and it is the
#    smaller half of the wall clock; adding a second bulk-read path there
#    would duplicate note 1's risk for a much smaller prize.
