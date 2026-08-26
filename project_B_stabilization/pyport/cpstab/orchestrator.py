"""Port of the Shipley2020 registration orchestration layer.

MATLAB sources (ground truth; every line read):
  registration/DFT_warp_3D_2.m        (166 lines)  -> dft_warp_3d_2
  registration/CalculateOptotuneWarp.m ( 49 lines) -> calculate_optotune_warp
  registration/ApplyOptotuneWarp.m    (  7 lines)  -> apply_optotune_warp

Array conventions (MATLAB dim order, NOT C-order habits):
  single frame [Y, X]; volume [Y, X, Z]; time block [Y, X, Z, T].
  The IO layer handles the channel dim; this module only sees one channel.

Dtype (review F2 -- IMPORTANT): the raw chunk is kept in its NATIVE class
(uint16 for .sbx / uint16 TIFF sources) through crop -> reshape -> imresize
-> optotune warp -> DFT_rect, exactly as in MATLAB, because MATLAB's
imresize/imwarp/imtranslate re-quantize integer classes and that
quantization measurably changes the estimated shifts at realistic SNR.
Everything from chunk_reg0 (a double zeros() fill) onward is float64, again
exactly as in MATLAB -- unless the run opted into the float32 fast mode
(cfg.compute_dtype, cpstab/precision.py), which lowers those pixel arrays to
single while leaving the uint16 stage above and the DFT correlation below in
place. See PORTING NOTES.

Sibling modules (parallel ports; imported lazily):
  cpstab/io_rw.py    : sbx_info(path), imread(path, k, N, pmt, optolevel)
                       -- k is 1-based (mirrors pipe.imread); returns
                       [rows, cols, N] uint16 (sbxRead.m contract); plus the
                       new-design VolumeSource / SbxFile source objects
                       accepted by _resolve_source below.
  cpstab/dftreg.py   : dftregistration3d(buf1ft, buf2ft, usfac).
  cpstab/shifts2d.py : dft_rect(vol, start, upscale)  -- start is the MATLAB
                       1-based center-plane index;
                       define_reference(volume, n, type),
                       determine_xy_shifts_fbs(full_vol, blurfactor,
                       keepingfactor, reference_volume),
                       apply_xy_shifts_fbs(unshifted, row_shifts, col_shifts).
"""

import math
import os
import sys

import numpy as np

from . import improved as _improved
from .matlab_compat import matlab_imresize, matlab_imwarp_affine2d, matlab_round
# Float working class of the image domain (float64 replicate / float32 fast).
from .precision import as_correlation, as_float, zeros as _fzeros

__all__ = [
    "dft_warp_3d_2",
    "calculate_optotune_warp",
    "apply_optotune_warp",
]


# ---------------------------------------------------------------------------
# small private helpers
# ---------------------------------------------------------------------------

def _fftn(a):
    """fftn for the registration engine: ALWAYS float64 -> complex128.

    Same rule as shifts2d._fft2 -- the volume-vs-reference (L71-75) and
    inter-chunk (L110-118) 3-D registrations below are argmaxes that pick a
    shift, so they keep double even when the volume itself is float32
    (cpstab/precision.py). A no-op promotion in the default mode.
    """
    return np.fft.fftn(as_correlation(a))


def _fft2(a):
    """fft2 for the registration engine: ALWAYS float64 -> complex128.

    The 2-D counterpart of _fftn above, used by the improved-mode chain
    refinement. Same rule, same reason (cpstab/precision.py): the transform
    feeds an argmax that picks a shift, so it keeps double even when the
    planes it is handed are float32.
    """
    return np.fft.fft2(as_correlation(a))


def _refine_chain_to_volume_mean(vol_raw, vol_chained, r, c, shifts2d, dftreg,
                                 usfac=4, blurfactor=1, keepingfactor=0.95,
                                 cap=None, min_ncc=None, stats=None):
    """Global refinement of a DFT_rect chain (cpstab/improved.py correction 3).

    DFT_rect (registration/DFT_rect.m, called at DFT_warp_3D_2.m L56) rectifies
    a volume by CHAINING: plane i is registered to the translated plane i-1,
    running outward from the centre plane in both directions. Chained
    estimates random-walk. Each link's error enters every link after it and
    nothing pulls the sequence back, so the planes at the two ends of the stack
    drift away from the anchor with no restoring force -- the classic failure
    of sequential registration, and the reason template-based schemes exist.

    This adds the missing restoring force: build the mean of the chain-aligned
    volume (a template every plane has already been brought close to, so it is
    sharp enough to register against), measure each plane against it, and fold
    the correction into the chain's own R/C -- but only where that measurement
    is TRUSTWORTHY, see the next section.

    THE TRUST GATE IS LOAD-BEARING TOO
    ----------------------------------
    A plane deep in the stack shares almost no content with the volume mean.
    On FAD-F_1 the plane-vs-mean correlation falls from 0.30-0.76 (z <= 23) to
    0.039-0.245 (z >= 24), where the correlation surface is flat to within
    1-4% and its argmax is a uniform draw over the +-N/2 search domain. Folded
    in unconditionally -- which is what this function used to do -- that draw
    became a real shift: at the QUIET timepoint t=1300, 17 of 41 planes moved
    95-225 full-resolution px, RS went to [-10.4, +218.4] px against
    replicate's [-9.2, +19.0], and the resulting zero-fill bands are the
    rectangular seams and ghosting in the projection. It compounds, too: the
    volume below is rebuilt at the bogus total shift, so DetermineXYShiftsFBS
    (the very next stage) measures a mostly-black frame and its own CS spread
    went 4.1 px -> 189 px. Across the full run, 35.3% of (plane, timepoint)
    cells exceeded 50 px, in 1492 of 1500 timepoints.

    So a plane's correction is applied only when BOTH hold:

        max(|dR|, |dC|) <= cap      and      NCC_zm(plane, mean) >= min_ncc

    and otherwise the plane keeps its chain value unchanged (dR = dC = 0),
    which is precisely what replicate does for that plane -- so the worst case
    of a rejection is the MATLAB-faithful result, never something new. The
    defaults, their derivation from a 4100-sample survey, and why the fallback
    is "keep the chain value" rather than "clamp to cap" are all in
    cpstab/improved.py correction 3. Ungated behaviour (for reproducing that
    measurement) is cap=float('inf'), min_ncc=0.0.

    THE CROP IS LOAD-BEARING, NOT HYGIENE
    -------------------------------------
    Registering against the volume mean on the FULL frame makes the result
    worse, not better, and the reason is not subtle once seen: every chained
    plane carries a hard zero border of its own width (imtranslate fills with
    0), so their mean carries a SOFT one -- a ramp averaged over the different
    shifts. Correlating a hard border against a soft border produces a strong,
    plane-dependent signal that has nothing to do with the image content, and
    on data with a DC pedestal this large (cpstab/precision.py: ~91% of peak
    height) it wins. Measured over 6 synthetic volumes with known truth, mean
    residual after the chain vs after the refinement:

        full frame ................. 0.239 px -> 0.239 px   (no gain)
        crop 0.95, no blur ......... 0.239 px -> 0.151 px
        crop 0.95 + blur 1 ......... 0.239 px -> 0.106 px
        crop 0.90 + blur 1 ......... 0.239 px -> 0.089 px

    Raising `usfac` does not help the uncropped case at all (4 / 10 / 20 / 50 /
    100 all land at 0.62 px in an earlier version of this measurement), which
    is what identifies the failure as a bias rather than quantization noise.

    The shipped defaults are `keepingfactor` 0.95 and `blurfactor` 1 -- NOT the
    0.90 that scored best above. They are the values DFT_warp_3D_2 already
    passes to DetermineXYShiftsFBS (its own L9-10 defaults) for the very same
    sub-problem, so the refinement inherits the pipeline's existing
    conditioning instead of introducing a constant tuned on one synthetic.

    Parameters
    ----------
    vol_raw : [Y, X, Z] -- the volume DFT_rect was given (native uint16 here).
    vol_chained : [Y, X, Z] -- DFT_rect's `reg` output.
    r, c : (Z,) -- DFT_rect's chain shifts.
    shifts2d, dftreg : the sibling modules (passed in; this runs inside
        _process_chunk, which already resolved them).
    usfac : subpixel factor for the refinement, 4 -- the same one the chain
        itself uses at L56.
    blurfactor, keepingfactor : the DetermineXYShiftsFBS conditioning, see
        above. The crop bounds are computed with that function's exact
        expression (DetermineXYShiftsFBS.m L23-26) so the two stages measure
        the same region.
    cap : float, optional
        Trust gate, max |correction| per plane in REGISTRATION-GRID px (the
        units of this call site's dftregistration_alex output; x scale for
        full-resolution px). None -> improved.chain_refine_cap(), the
        process-wide setting run_pipeline installs from
        cfg.chain_refine_cap.
    min_ncc : float, optional
        Trust gate, minimum zero-mean normalized correlation between the plane
        and the volume mean. None -> improved.chain_refine_min_ncc().
    stats : dict, optional
        If given, filled in with per-plane gate telemetry:
        'accepted' (bool (Z,)), 'ncc' (float (Z,)), 'dr'/'dc' (the RAW
        corrections BEFORE the gate, float (Z,)), 'n_rejected' (int).
        Purely diagnostic -- the numerics do not depend on it.

    Returns
    -------
    (r_total, c_total, vol_refined) with the same shapes/classes as the inputs
    it replaces. Shifts stay float64 (bookkeeping, cpstab/precision.py); the
    volume follows the compute dtype.

    NOTES
    -----
    The shifts are ESTIMATED on the cropped, blurred planes but APPLIED to the
    full-size original ones -- the crop conditions the measurement, it does not
    shrink the data.

    The refined volume is rebuilt from the ORIGINAL planes at the corrected
    TOTAL shift rather than by translating the already-translated planes again:
    the chain output is one interpolation deep and re-shifting it would make it
    two. Shift composition is additive for translations, so composing and
    resampling once keeps the refined volume exactly as smooth as the
    unrefined one -- which matters, because it is what DetermineXYShiftsFBS
    (L65) measures next.

    Application uses shifts2d.imtranslate, NOT the improved-mode Fourier shift:
    this is the ESTIMATION side, and correction 2 is scoped to the apply side
    precisely so the two corrections can be attributed separately.
    """
    nz = vol_chained.shape[2]
    s1, s2 = vol_chained.shape[0], vol_chained.shape[1]
    keep = float(keepingfactor)
    # DetermineXYShiftsFBS.m L23-L26, 1-based inclusive -> half-open slice.
    r_lo = math.ceil(s1 * (1 - keep) / 2)
    r_hi = math.ceil(s1 * (1 - (1 - keep) / 2))
    c_lo = math.ceil(s2 * (1 - keep) / 2)
    c_hi = math.ceil(s2 * (1 - (1 - keep) / 2))
    if r_lo < 1 or c_lo < 1 or r_hi > s1 or c_hi > s2:
        raise ValueError(
            "chain refinement crop out of range for keepingfactor=%r on a "
            "%dx%d frame" % (keepingfactor, s1, s2))
    crop = (slice(r_lo - 1, r_hi), slice(c_lo - 1, c_hi))

    # _imgaussfilt is shifts2d's MATLAB imgaussfilt mirror; reused rather than
    # re-derived so the refinement blurs exactly like DetermineXYShiftsFBS.
    blur = shifts2d._imgaussfilt
    cropped = vol_chained[crop[0], crop[1], :]
    target = blur(np.mean(cropped, axis=2), blurfactor)
    target_ft = _fft2(target)

    # Trust gate (cpstab/improved.py correction 3). Read from the process-wide
    # setting unless the caller pinned one, exactly like the mode itself.
    cap = _improved.resolve_chain_refine_cap(
        _improved.chain_refine_cap() if cap is None else cap)
    min_ncc = _improved.resolve_chain_refine_min_ncc(
        _improved.chain_refine_min_ncc() if min_ncc is None else min_ncc)
    # NCC_zm's fixed half: the template, mean-removed once, in double (this is
    # a decision statistic, same rule as _fft2 above).
    t_zm = as_correlation(target) - float(np.mean(target))
    t_norm = float(np.sqrt(np.sum(t_zm * t_zm)))

    r_out = np.array(r, dtype=np.float64).ravel()
    c_out = np.array(c, dtype=np.float64).ravel()
    # NOT as_float(vol_raw): the planes must be translated in the class
    # DFT_rect uses (native uint16 here, shifts2d.dft_rect L172-177), because
    # shifts2d.imtranslate -> matlab_compat.matlab_cast_like re-quantizes an
    # integer input (round-half-away + saturate) and that re-quantization is
    # part of the MATLAB numerics -- the same dtype-chain rule this module
    # states in its DESIGN NOTES. Promoting to float here skipped exactly that
    # one cast, so a REJECTED plane (dR = dC = 0) came out different from the
    # chain value replicate keeps, breaking the "a fully-rejected volume
    # degrades to the MATLAB-faithful result" guarantee: measured on the
    # 40-frame subset it moved 72.7% of chunk_reg0 voxels by up to 0.5 counts
    # and carried through into RS1/CS1. `out` is np.empty_like(vol_chained),
    # already the compute dtype, so no cast is needed for the assignment.
    src = vol_raw
    out = np.empty_like(vol_chained)
    accepted = np.zeros(nz, dtype=bool)
    ncc_all = np.zeros(nz, dtype=np.float64)
    dr_all = np.zeros(nz, dtype=np.float64)
    dc_all = np.zeros(nz, dtype=np.float64)
    for i in range(nz):
        plane = blur(cropped[:, :, i], blurfactor)
        d = dftreg.dftregistration_alex(target_ft, _fft2(plane), usfac)
        dr, dc = float(d[0]), float(d[1])
        # How much of a peak was that argmax actually sitting on? A flat
        # correlation surface (deep planes: no shared content with the volume
        # mean) still HAS an argmax, and it is noise. Measured with the DC
        # pedestal removed from both sides -- with it in place a mislocked
        # plane scores 0.93 against a good plane's 0.89 and the statistic
        # cannot separate them at all.
        p_zm = as_correlation(plane) - float(np.mean(plane))
        den = t_norm * float(np.sqrt(np.sum(p_zm * p_zm)))
        ncc = float(np.sum(t_zm * p_zm) / den) if den > 0.0 else 0.0
        # min_ncc == 0.0 means DISABLED, not ">= 0". NCC_zm is SIGNED, so a
        # bare `ncc >= 0.0` still rejects every anti-correlated plane -- on the
        # 40-frame subset 10/1640 planes (all z >= 37) score down to -0.163,
        # so the configuration documented everywhere as "ungated" was in fact
        # still dropping the very planes with the largest corrections. That
        # made the gated-vs-ungated A/B measure something other than its label.
        # The default (0.30) path is unchanged: 0.30 > 0, so the test below is
        # the ordinary one.
        ok = ((max(abs(dr), abs(dc)) <= cap)
              and (min_ncc <= 0.0 or ncc >= min_ncc))
        if not ok:
            # Keep DFT_rect's chain value for this plane -- see the TRUST GATE
            # section above for why this and not a clamp to `cap`.
            dr = dc = 0.0
        accepted[i] = ok
        ncc_all[i] = ncc
        dr_all[i] = float(d[0])
        dc_all[i] = float(d[1])
        r_out[i] += dr
        c_out[i] += dc
        out[:, :, i] = shifts2d.imtranslate(src[:, :, i],
                                            (c_out[i], r_out[i]))  # [C, R]

    n_rejected = int(nz - accepted.sum())
    if stats is not None:
        stats.update(accepted=accepted, ncc=ncc_all, dr=dr_all, dc=dc_all,
                     n_rejected=n_rejected)
    if n_rejected * 2 > nz:
        # Not an error -- a rejected plane is a plane that keeps the MATLAB
        # answer -- but a majority rejection means correction 3 has nothing to
        # work with on this data, which is worth seeing once per process
        # rather than never. Once, not per volume: this loop runs Nt times.
        _warn_chain_refine_rejection(n_rejected, nz, cap, min_ncc)
    return r_out, c_out, out


_CHAIN_REFINE_WARNED = False


def _warn_chain_refine_rejection(n_rejected, nz, cap, min_ncc):
    """First-time-only notice that correction 3's gate is rejecting a majority
    of planes (see _refine_chain_to_volume_mean). Reset the module flag to
    re-arm it; tests do."""
    global _CHAIN_REFINE_WARNED
    if _CHAIN_REFINE_WARNED:
        return
    _CHAIN_REFINE_WARNED = True
    print("cpstab: chain-refine trust gate rejected %d/%d planes of a volume "
          "(cap=%g grid px, min_ncc=%g) -- those planes keep their DFT_rect "
          "chain value. A majority rejection means the deep planes share too "
          "little content with the volume mean for correction 3 to measure "
          "anything here; see cpstab/improved.py correction 3. Reported once "
          "per process." % (n_rejected, nz, cap, min_ncc), file=sys.stderr)


def _info_get(info, name):
    """Read a field from the IO layer's info (attribute or mapping access)."""
    if hasattr(info, name):
        return getattr(info, name)
    try:
        return info[name]
    except (TypeError, KeyError) as e:
        raise KeyError("sbx info has no field %r" % name) from e


def _matlab_truthy(value):
    """Interpret DFT_warp_3D_2's `optotune` flag.

    MATLAB default is the CHAR ARRAY 'true' (DFT_warp_3D_2.m L6), and
    `if p.optotune` on any non-empty char array is true -- so in MATLAB even
    'false' selects the warp branch and the else branch is unreachable
    (CODEMAP section 10). DIVERGENCE (deliberate fix): here 'false'/'0'/''
    and False actually select the no-warp branch. On the default path
    ('true') behaviour is identical.
    """
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "1"):
            return True
        if s in ("false", "0", ""):
            return False
        raise ValueError("optotune flag must be boolean-like, got %r" % value)
    return bool(value)


def _import_deps():
    """Lazily import the parallel-ported sibling modules with a clear error.

    Review F1: the IO port landed as io_rw.py (not sbxio.py) and dft_rect
    lives in shifts2d.py (not dftreg.py); this is the single place the
    module-name seam is wired.
    """
    try:
        from . import dftreg, io_rw as sbxio, shifts2d
    except ImportError as e:
        raise ImportError(
            "orchestrator.py needs sibling modules from the parallel ports: "
            "cpstab/io_rw.py (sbx_info, imread), cpstab/dftreg.py "
            "(dftregistration3d), cpstab/shifts2d.py (dft_rect, "
            "define_reference, determine_xy_shifts_fbs, apply_xy_shifts_fbs)."
            " Import failed: %s" % e
        ) from e
    return sbxio, dftreg, shifts2d


def _resolve_source(src, sbxio):
    """Normalize dft_warp_3d_2/calculate_optotune_warp's first argument.

    The MATLAB functions take a .sbx PATH; the ported pipeline driver
    (pipeline.py, IO-seam signature change documented in its PORTING NOTES
    #1) passes a SOURCE OBJECT (io_rw.VolumeSource / io_rw.SbxFile).
    Review F1(4): both are accepted here.

    Returns (info, reader) where
      info   : mapping with at least sz ([rows, cols]), otwave (row vector,
               numel == Nz) and nframes (total 2-D records), the three
               fields this module consumes;
      reader : reader(k, N, pmt) -> [rows, cols, N] raw frames, k 1-BASED
               (MATLAB pipe.imread contract), in the source's NATIVE class
               (uint16 for .sbx and for integer-typed TIFF sources) so the
               MATLAB uint16 quantization chain is preserved (review F2).
    """
    # 1) path string -> the MATLAB-mirror IO functions (uint16 out).
    if isinstance(src, (str, bytes, os.PathLike)):
        info = sbxio.sbx_info(src)                      # L16 / L13
        def reader(k, n, pmt):
            return sbxio.imread(src, k, n, pmt, None)   # L38 / L26
        return info, reader

    # 2) SbxFile-like object: has the sbxRead-compatible .read (uint16 out)
    #    and a ready info dict.
    if hasattr(src, "read") and hasattr(src, "info"):
        def reader(k, n, pmt):
            return src.read(k, n, pmt, None)
        return src.info, reader

    # 3) VolumeSource-like object: 0-based get_volume(t, channel) returning
    #    [Y, X, Z] compute-class floats plus Nx/Ny/Nz/Nt metadata. Frames
    #    are stacked in
    #    acquisition order (z fastest, f = t*Nz + z -- same mapping the L40
    #    reshape inverts) and cast back to the file's native integer class
    #    when it has one: get_volume performs no arithmetic, so float64 AND
    #    float32 both hold the uint16 values exactly (17 bits) and the cast
    #    back is lossless in either precision mode.
    if hasattr(src, "get_volume"):
        nz = int(src.Nz)
        nt = int(src.Nt)
        info = {
            "sz": np.array([int(src.Nx), int(src.Ny)], dtype=np.float64),
            # only numel/size-along-row of otwave is consumed downstream;
            # synthesize the standard (1, Nz) row vector (sbxInfo shape).
            "otwave": np.arange(1, nz + 1, dtype=np.float64)[None, :],
            "nframes": nz * nt,
        }
        native = getattr(src, "dtype", None)

        def reader(k, n, pmt):
            k0 = int(k) - 1
            if k0 % nz or int(n) % nz:
                raise ValueError(
                    "volume-source reads must be volume-aligned: first frame"
                    " %r / count %r not multiples of Nz=%d" % (k, n, nz))
            t0 = k0 // nz
            n = int(n)
            if n <= 0:
                # np.concatenate([]) used to raise here; keep it an error, but
                # a legible one (dft_warp_3d_2 already rejects chunkframes<=0).
                raise ValueError("volume-source read of %r frames" % (n,))
            # PRE-ALLOCATE in the output class and fill volume by volume. The
            # obvious `np.concatenate([get_volume(t) ...]).astype(native)` held
            # three copies at once -- the float64 list, the float64 concat and
            # the integer result -- so reading one production chunk
            # (chunkframes = 41*20) peaked at 4.10 GB of RSS for a 0.40 GB
            # answer, x10 workers. Filling in place peaks at 0.97 GB. The
            # per-volume float -> integer store is the same unsafe truncation
            # .astype() performs and the values are exact integers either way,
            # so the bytes are unchanged (verified bit-for-bit in float64 and
            # float32 modes).
            out_dtype = (np.dtype(native)
                         if native is not None
                         and np.issubdtype(np.dtype(native), np.integer)
                         else None)
            out = None
            for i in range(n // nz):
                vol = src.get_volume(t0 + i, channel=int(pmt) - 1)
                if out is None:
                    out = np.empty(vol.shape[:2] + (n,),
                                   dtype=out_dtype if out_dtype is not None
                                   else vol.dtype)
                out[:, :, i * nz:(i + 1) * nz] = vol    # [Y, X, N], z fastest
            return out
        return info, reader

    raise TypeError(
        "first argument must be a .sbx path or an io_rw source object "
        "(VolumeSource / SbxFile); got %r" % type(src))


def _as_volume_3d(ref):
    """Drop the trailing singleton time axis of a defineReference output.

    MATLAB's ref2 is Y x X x Z x 1, which MATLAB itself treats as 3-D (trailing
    singleton dims do not exist), so fftn(ref2) is a 3-D FFT. The Python
    define_reference may return either 3-D or 4-D-with-singleton; normalize.
    """
    ref = as_float(ref)
    if ref.ndim == 4:
        if ref.shape[3] != 1:
            raise ValueError(
                "expected a single-volume reference, got T=%d" % ref.shape[3]
            )
        ref = ref[:, :, :, 0]
    if ref.ndim != 3:
        raise ValueError("reference must be a 3-D volume [Y, X, Z]")
    return ref


# ---------------------------------------------------------------------------
# CalculateOptotuneWarp
# ---------------------------------------------------------------------------

def calculate_optotune_warp(path, refchannel, scale, edges=(0, 0, 0, 0),
                            regtype="affine", refsize=30, save=False):
    """registration/CalculateOptotuneWarp.m L1-49.

    For regtype='none' (the piezo / demo path) MATLAB early-returns
    repmat(affine2d(eye(3)), [1, Nz]) at L20-23, BEFORE touching Fiji and
    BEFORE the save at L46-48. This port returns the same thing as a list of
    Nz identity 3x3 T matrices (MATLAB affine2d row-vector convention:
    [x y 1] = [x y 1] * T).

    DIVERGENCE (documented, out of scope): the 'affine'/'rigid' branches
    (L35-43) drive Fiji MultiStackReg/TurboReg through MIJ and are dead on
    the production path (CODEMAP sections 4 and 8); they raise
    NotImplementedError here instead of silently doing something different.
    MATLAB's trailing else (L40-42) prints a message and returns an undefined
    variable (a latent crash at the caller); this port raises ValueError.

    Parameters mirror the MATLAB signature and defaults: path, refchannel,
    scale, then keyword edges=[0,0,0,0], regtype='affine', refsize=30,
    save=False. `path` may also be an io_rw source object (VolumeSource /
    SbxFile), matching the pipeline driver's IO seam. `refchannel`, `scale`,
    `edges`, `refsize` and `save` are only consumed by the unported Fiji
    branches (L26-33, L46-48).
    """
    sbxio, _, _ = _import_deps()
    info, _reader = _resolve_source(path, sbxio)             # L13
    # L18 is size(info.otwave,2); numel() is used here (as MATLAB itself does
    # at DFT_warp_3D_2.m L24) under the assumption that otwave is the
    # standard (1, Nz) ROW vector sbxInfo/loadmat produce, where the two
    # agree. A column-vector otwave would make MATLAB run with Nz=1 (itself
    # broken downstream); it does not occur on the supported paths.
    nz = int(np.asarray(_info_get(info, "otwave")).ravel().size)  # L18

    if regtype == "none":                                    # L20-23
        return [np.eye(3, dtype=np.float64) for _ in range(nz)]

    if regtype in ("affine", "rigid"):                       # L35-39
        raise NotImplementedError(
            "CalculateOptotuneWarp regtype=%r needs Fiji MultiStackReg/"
            "TurboReg (MATLAB L26-43); that path is dead on the piezo demo "
            "path (opttype='none') and is not ported." % regtype
        )
    raise ValueError(                                        # L40-42
        "Invalid registration type for optotune correction: %r" % regtype
    )


# ---------------------------------------------------------------------------
# ApplyOptotuneWarp
# ---------------------------------------------------------------------------

def apply_optotune_warp(chunk, tforms_optotune):
    """registration/ApplyOptotuneWarp.m L1-7.

    Per-slice imwarp of the per-z affine transforms:
      for j (time), i (plane):
        unwarped(:,:,i,j) = imwarp(chunk(:,:,i,j), tforms(i),
                                   'OutputView', imref2d(size(slice)))
    with 'linear' interpolation and FillValues=0 (MATLAB defaults).

    Optimization (mandated; numerically exact): when a transform is EXACTLY
    the identity (the whole demo path, where CalculateOptotuneWarp('none')
    returns eye(3) for every plane), the imwarp is skipped -- bilinear
    resampling at exact integer coordinates reproduces the input bit-for-bit,
    so skipping is equivalent and much faster. Near-identity transforms are
    still warped, exactly as MATLAB would.

    Parameters
    ----------
    chunk : [Y, X, Z, T]; any class. Review F2: the input CLASS IS
        PRESERVED, exactly like MATLAB imwarp -- the production path feeds
        uint16 (sbxRead -> imresize keep uint16) and imwarp's per-slice
        re-quantization back to uint16 is part of the MATLAB numerics.
        matlab_imwarp_affine2d performs that cast-back itself.
    tforms_optotune : sequence of Z (3, 3) affine2d T matrices.

    Returns [Y, X, Z, T] in chunk's class. If every transform is identity
    the INPUT ARRAY ITSELF is returned (no copy); callers in this pipeline
    never mutate it afterwards (DFT_warp_3D_2 only reads it).
    """
    chunk = np.asarray(chunk)
    if chunk.ndim != 4:
        raise ValueError("chunk must be [Y, X, Z, T]")
    tf = [np.asarray(t, dtype=np.float64) for t in tforms_optotune]
    if len(tf) < chunk.shape[2]:
        raise ValueError(
            "need one transform per plane: %d < %d" % (len(tf), chunk.shape[2])
        )
    eye = np.eye(3)
    if all(np.array_equal(t, eye) for t in tf[: chunk.shape[2]]):
        return chunk

    out = np.empty_like(chunk)
    for j in range(chunk.shape[3]):          # L3 (time)
        for i in range(chunk.shape[2]):      # L4 (plane)
            if np.array_equal(tf[i], eye):
                out[:, :, i, j] = chunk[:, :, i, j]
            else:
                out[:, :, i, j] = matlab_imwarp_affine2d(chunk[:, :, i, j], tf[i])  # L6
    return out


# ---------------------------------------------------------------------------
# DFT_warp_3D_2
# ---------------------------------------------------------------------------

def _process_chunk(reader, chunk, chunkframes, t_chunk, refchannel, edges,
                   scale, tforms_optotune, optotune, reftype, blurfactor,
                   keepingfactor, start_plane, deps=None):
    """One iteration of the DFT_warp_3D_2.m L36-98 chunk loop.

    Extracted verbatim from the dft_warp_3d_2 loop so a process pool can run
    chunks in parallel (chunks are mutually independent; only the inter-chunk
    stitch L110-118 needs their refs together). Numerics identical to the
    former inline body. Returns (rs_cell, cs_cell, zs_cell, ref2_vol,
    rs0, cs0, rs1, cs1, rs2, cs2).
    """
    sbxio, dftreg, shifts2d = deps if deps is not None else _import_deps()

    # 1) load reference chunk (L38-41)
    raw_chunk = np.asarray(
        reader(chunkframes * chunk + 1, chunkframes, refchannel))
    raw_chunk = raw_chunk[
        edges[2]: raw_chunk.shape[0] - edges[3],
        edges[0]: raw_chunk.shape[1] - edges[1],
        :,
    ]
    nx, ny = raw_chunk.shape[0], raw_chunk.shape[1]
    nz = raw_chunk.shape[2] // t_chunk
    raw_chunk = raw_chunk.reshape(nx, ny, t_chunk, nz).transpose(0, 1, 3, 2)
    raw_chunk = matlab_imresize(raw_chunk, 1.0 / scale)

    # 2) optotune warp (L45-49)
    if _matlab_truthy(optotune):
        unwarped_chunk = apply_optotune_warp(raw_chunk, tforms_optotune)  # L46
    else:
        unwarped_chunk = raw_chunk

    # rectify each volume with dft (L52-57)
    nzc = unwarped_chunk.shape[2]
    # rs0/cs0 are SHIFT bookkeeping and stay float64 in both precision modes
    # (see cpstab/precision.py); chunk_reg0 is the image accumulator and
    # follows the compute dtype -- it is also the largest array in the chunk.
    rs0 = np.zeros((nzc, t_chunk))
    cs0 = np.zeros((nzc, t_chunk))
    chunk_reg0 = _fzeros(unwarped_chunk.shape)
    for i in range(t_chunk):
        r, c, reg = shifts2d.dft_rect(unwarped_chunk[:, :, :, i],
                                      start_plane, 4)             # L56
        if _improved.use_chain_refine():
            # cpstab/improved.py correction 3: the L56 chain has no restoring
            # force, so add one global pass against the chain-aligned volume
            # mean and fold it into RS0/CS0.
            # blurfactor/keepingfactor are FORWARDED, not left to the callee's
            # own defaults: its docstring promises "the refinement inherits the
            # pipeline's existing conditioning" and that the crop uses
            # DetermineXYShiftsFBS's exact expression "so the two stages
            # measure the same region" -- neither held while this call dropped
            # them. No-op on every shipped path (fast_run passes 1/0.95 and
            # pipeline takes dft_warp_3d_2's 1/0.95 defaults, which are the
            # values the callee defaulted to anyway).
            r, c, reg = _refine_chain_to_volume_mean(
                unwarped_chunk[:, :, :, i], reg, r, c, shifts2d, dftreg,
                blurfactor=blurfactor, keepingfactor=keepingfactor)
        rs0[:, i] = np.ravel(r)
        cs0[:, i] = np.ravel(c)
        chunk_reg0[:, :, :, i] = reg

    # 3) first round of XY DFT registration (L61-69)
    ref1 = shifts2d.define_reference(chunk_reg0, t_chunk, reftype)      # L63
    rs1, cs1 = shifts2d.determine_xy_shifts_fbs(
        chunk_reg0, blurfactor, keepingfactor, ref1)                    # L65
    chunk_reg1 = shifts2d.apply_xy_shifts_fbs(chunk_reg0, rs1, cs1)     # L67
    ref2 = shifts2d.define_reference(chunk_reg1, t_chunk, reftype)      # L69
    ref2_vol = _as_volume_3d(ref2)

    # whole-volume 3D shift vs ref2 (L71-75)
    shifts = np.zeros((t_chunk, 3))
    ref2_f = _fftn(ref2_vol)
    for j in range(t_chunk):
        vol = chunk_reg1[:, :, :, j]
        shifts[j, :] = np.ravel(
            dftreg.dftregistration3d(ref2_f, _fftn(vol), 2))            # L74

    rs2 = shifts[:, 0] * scale                                          # L77
    cs2 = shifts[:, 1] * scale                                          # L78
    zs1 = shifts[:, 2]                                                  # L79

    # 6) combine row/column shifts (L83-85)
    rs_cell = rs0 * scale + rs1 * scale + np.tile(rs2, (nzc, 1))
    cs_cell = cs0 * scale + cs1 * scale + np.tile(cs2, (nzc, 1))
    zs_cell = zs1[None, :].copy()

    return (rs_cell, cs_cell, zs_cell, ref2_vol,
            rs0, cs0, np.asarray(rs1, dtype=np.float64),
            np.asarray(cs1, dtype=np.float64), rs2, cs2)


def dft_warp_3d_2(path, shiftpath, refchannel, scale, nchunks, tforms_optotune,
                  edges=(0, 0, 0, 0), nt=None, optotune="true",
                  reftype="median", blurfactor=1, keepingfactor=0.95,
                  planescorr=3, save=True, save_debug=False):
    """registration/DFT_warp_3D_2.m L1-166. Core chunked-shift orchestration.

    Per chunk (<= Nt/Nchunks volumes): read raw frames, crop edges, reshape to
    [Y, X, Z, T], imresize by 1/scale (MATLAB bicubic + antialiasing, via
    matlab_compat.matlab_imresize), optionally apply the optotune warp, then
      1) DFT_rect per volume (axial rectify, usfac=4)          -> RS0, CS0
      2) defineReference + DetermineXYShiftsFBS (usfac=100)    -> RS1, CS1
         + ApplyXYShiftsFBS, second defineReference            -> ref2
      3) dftregistration3D volume-vs-ref2 (usfac=2)            -> RS2, CS2, ZS1
    Shifts combine as RS = RS0*scale + RS1*scale + RS2' (L83; RS2/CS2 already
    carry *scale from L77-78, ZS stays in plane units). Chunks are anchored to
    chunk 1 by registering each chunk's ref2 to chunk 1's (L110-118) and the
    per-chunk corrections are stretched framewise with imresize 'nearest'
    (L127-129). Results are persisted to `shiftpath`.

    Parameters mirror the MATLAB signature/defaults: positional path,
    shiftpath, refchannel, scale, nchunks, tforms_optotune; keyword
    edges=[0,0,0,0], nt=None (MATLAB Nt=[] -> all volumes), optotune='true',
    reftype='median', blurfactor=1, keepingfactor=0.95, planescorr=3 (parsed
    but NEVER used by the MATLAB body either -- kept for signature parity),
    save=True. Extra Python-only flag: save_debug=False. `path` may be a
    .sbx path string (MATLAB-verbatim) or an io_rw source object
    (VolumeSource / SbxFile) -- the pipeline driver's IO-seam convention
    (see _resolve_source).

    Persistence (deliberate format change, keys keep MATLAB variable names):
    MATLAB `save(shiftpath, ..., '-mat')` becomes np.savez to the exact
    `shiftpath` given (no .npz suffix appended). By default only the keys
    MakeSBXall actually reads are stored (CODEMAP section 5 row F):
      RS, CS       (Nz, Nchunks*T)   float64
      ZS           (1,  Nchunks*T)   float64  (MATLAB row-vector shape kept)
      RS_chunk, CS_chunk, ZS_chunk   same shapes as RS/CS/ZS
      tforms_optotune_full           (Nz, 3, 3) stacked affine2d T matrices,
                                     translations T[2, 0:2] scaled by `scale`
    The dead weight MATLAB also saved (`ref_all`, `intermediate_shifts`,
    `scale` -- unread by MakeSBXall) is stored only when save_debug=True, as
    keys 'scale', 'ref_all' (Nchunks, Y, X, Z) and
    'intermediate_shifts_<field>' (flattened struct fields, MATLAB shapes:
    RS0/RS1 fields (Nz, Nchunks*T); RS2 fields (T, Nchunks)).

    Returns a dict with the saved keys (MATLAB returns nothing; the dict is a
    Python-side convenience and also carries the debug arrays when
    save_debug=True).
    """
    sbxio, dftreg, shifts2d = _import_deps()

    edges = [int(e) for e in edges]
    if len(edges) != 4:
        raise ValueError("edges must have 4 entries [left, right, top, bottom]")

    info, reader = _resolve_source(path, sbxio)                   # L16
    # L17: fdir = fileparts(path) -- computed but never used; omitted.
    sz = np.asarray(_info_get(info, "sz")).ravel()
    otwave = np.asarray(_info_get(info, "otwave")).ravel()
    nframes = int(_info_get(info, "nframes"))

    nx = int(sz[0]) - edges[2] - edges[3]                         # L19
    ny = int(sz[1]) - edges[0] - edges[1]                         # L20
    # L21 is size(info.otwave,2), L24 numel(info.otwave); numel is used for
    # both here under the (documented) assumption that otwave is the standard
    # (1, Nz) ROW vector produced by sbxInfo/loadmat -- where the two agree.
    nz = int(otwave.size)                                         # L21

    if nt is None:                                                # L23-25
        nt = nframes / nz          # float division, exactly like MATLAB
    chunkframes = int(nz * math.floor(nt / nchunks))              # L27
    if chunkframes <= 0:
        raise ValueError(
            "chunkframes = Nz*floor(Nt/Nchunks) = 0 (Nt=%r, Nchunks=%d); "
            "MATLAB would read zero frames and crash later" % (nt, nchunks)
        )
    t_chunk = chunkframes // nz

    rs_cells = []                                                 # L29-31
    cs_cells = []
    zs_cells = []
    ref_all = []                                                  # L89
    rs0_all, cs0_all = [], []
    rs1_all, cs1_all = [], []
    rs2_all, cs2_all = [], []

    # L33: parfor_progressbar replaced by plain progress prints (GUI hazard,
    # CODEMAP section 9 row 8).
    print("DFT registration: %d chunks" % nchunks, file=sys.stderr, flush=True)

    # MATLAB round(Nz/2): half away from zero, kept as a 1-based plane index
    # for the mirrored dft_rect signature (L56).
    start_plane = int(matlab_round(nz / 2.0))

    for chunk in range(nchunks):                                  # L36
        # Loop body extracted to _process_chunk (chunks are independent; see
        # its docstring). All L36-98 comments and review notes live there.
        (rs_cell, cs_cell, zs_cell, ref2_vol,
         rs0, cs0, rs1, cs1, rs2, cs2) = _process_chunk(
            reader, chunk, chunkframes, t_chunk, refchannel, edges, scale,
            tforms_optotune, optotune, reftype, blurfactor, keepingfactor,
            start_plane, deps=(sbxio, dftreg, shifts2d))

        rs_cells.append(rs_cell)                       # L83-85
        cs_cells.append(cs_cell)
        zs_cells.append(zs_cell)

        # 7) keep per-chunk pieces for stitching / debug (L89-95).
        ref_all.append(ref2_vol)
        rs0_all.append(rs0)
        cs0_all.append(cs0)
        rs1_all.append(rs1)
        cs1_all.append(cs1)
        rs2_all.append(rs2)
        cs2_all.append(cs2)

        print("DFT registration: chunk %d/%d done" % (chunk + 1, nchunks),
              file=sys.stderr, flush=True)             # L98 H.iterate(1)

    # intermediate_shifts struct (L101-106). MATLAB horzcats the cells:
    # RS0/RS1 cells are (Nz, T) -> (Nz, Nchunks*T); RS2 cells are (T, 1)
    # columns -> (T, Nchunks).
    intermediate_shifts = {
        "RS0_all": np.concatenate(rs0_all, axis=1),
        "RS1_all": np.concatenate(rs1_all, axis=1),
        "RS2_all": np.stack(rs2_all, axis=1),
        "CS0_all": np.concatenate(cs0_all, axis=1),
        "CS1_all": np.concatenate(cs1_all, axis=1),
        "CS2_all": np.stack(cs2_all, axis=1),
    }

    # Fix inter-chunk discontinuities: anchor every chunk to chunk 1
    # (L110-118). Chunk 1 vs itself yields exactly zero shift.
    ref_final = ref_all[0]                                        # L110
    ref_final_f = _fftn(ref_final)
    interchunk_shifts = np.zeros((nchunks, 3))                    # L112
    for j in range(nchunks):
        interchunk_shifts[j, :] = np.ravel(
            dftreg.dftregistration3d(ref_final_f, _fftn(ref_all[j]), 2))  # L114
    rs_chunk_vec = interchunk_shifts[:, 0] * scale                # L116
    cs_chunk_vec = interchunk_shifts[:, 1] * scale                # L117
    zs_chunk_vec = interchunk_shifts[:, 2]                        # L118

    # Cells -> matrices (L121-123). ZS keeps MATLAB's (1, N) row shape.
    RS = np.concatenate(rs_cells, axis=1)
    CS = np.concatenate(cs_cells, axis=1)
    ZS = np.concatenate(zs_cells, axis=1)

    # Stretch inter-chunk corrections framewise (L127-129): MATLAB
    # imresize(vec', size(RS), 'nearest') -- 1 x Nchunks stretched to the
    # full shift-matrix shape with MATLAB nearest index mapping.
    RS_chunk = matlab_imresize(rs_chunk_vec[None, :],
                               output_shape=RS.shape, method="nearest")
    CS_chunk = matlab_imresize(cs_chunk_vec[None, :],
                               output_shape=CS.shape, method="nearest")
    ZS_chunk = matlab_imresize(zs_chunk_vec[None, :],
                               output_shape=ZS.shape, method="nearest")

    # Scale the optotune transforms (L133-136): copy (MATLAB affine2d is a
    # value class, so L133 copies), then T(3,1:2) *= scale -> T[2, 0:2].
    tforms_optotune_full = [np.array(t, dtype=np.float64, copy=True)
                            for t in tforms_optotune]
    for t in tforms_optotune_full:
        t[2, 0:2] *= scale

    result = {
        "RS": RS,
        "CS": CS,
        "ZS": ZS,
        "RS_chunk": RS_chunk,
        "CS_chunk": CS_chunk,
        "ZS_chunk": ZS_chunk,
        "tforms_optotune_full": np.stack(tforms_optotune_full),
    }
    if save_debug:
        result["scale"] = np.float64(scale)
        result["ref_all"] = np.stack(ref_all)      # (Nchunks, Y, X, Z)
        for key, val in intermediate_shifts.items():
            result["intermediate_shifts_" + key] = val

    # L139-141: save. Write through an open handle so np.savez cannot append
    # '.npz' and silently change the '.dftshifts' filename.
    if save:
        with open(shiftpath, "wb") as f:
            np.savez(f, **result)

    print("DFT registration: done", file=sys.stderr, flush=True)  # L142 close(H)
    return result


# PORTING NOTES
# -------------
# * Ground truth: references/Shipley2020/registration/ was not present on disk
#   (references/README.md says the clone is git-ignored). The port was written
#   against clean/registration/{DFT_warp_3D_2,CalculateOptotuneWarp,
#   ApplyOptotuneWarp}.m, verified to match every CODEMAP line citation for
#   these files (including the L48 `unwarp_chunk` typo and the L20-23 'none'
#   early return), so they are byte-faithful mirrors for the ported lines.
#   Line numbers in docstrings/comments refer to those files.
# * Sibling-module contract (INTEGRATED, review F1): the IO port is io_rw.py
#   (sbx_info/imread; imread keeps MATLAB 1-based first-frame k and returns
#   [rows, cols, N] in the file's native class -- uint16, per sbxRead.m L75);
#   dft_rect lives in shifts2d.py (start is the MATLAB 1-based plane index
#   round(Nz/2)); dftreg.py provides dftregistration3d. _import_deps() and
#   _resolve_source() are the only seam points. The first argument accepts
#   the MATLAB path string or the pipeline driver's source object
#   (VolumeSource / SbxFile), see _resolve_source.
# * IMPROVED MODE (port extension, cpstab/improved.py, cfg.mode): exactly ONE
#   of the four corrections lives in this module -- correction 3, the global
#   refinement pass bolted onto the L52-57 DFT_rect loop
#   (_refine_chain_to_volume_mean). In the default 'replicate' mode the guard
#   is False and the loop is literally the code it always was, which is what
#   the iron-law regressions assert. Two consequences worth knowing when
#   editing this file:
#     - the correction changes RS0/CS0, hence RS/CS, hence the .dftshifts.npz
#       this function writes. An improved run's shift file is NOT comparable
#       to a replicate run's; that is expected, not a regression.
#     - it also changes chunk_reg0, which is what defineReference (L63) and
#       DetermineXYShiftsFBS (L65) then measure, so RS1/CS1 move too. The
#       refinement is applied once per volume, before any of that, precisely so
#       the downstream stages see a chain that has already been de-drifted.
#   The other three corrections are in apply_project (its PORTING NOTES #17).
# * FLOAT32 FAST MODE (port extension, cpstab/precision.py): the float
#   working class of this module's image arrays is the process-wide compute
#   dtype -- float64 (replicate, default, bit-identical to before) or float32
#   (cfg.compute_dtype='float32'). Exactly two arrays here change class:
#   chunk_reg0 (the [Y,X,Z,T] rectified chunk, by far the largest allocation
#   in the loop) and ref2_vol via _as_volume_3d. The 3-D registrations do NOT
#   follow them: _fftn promotes back to float64 so dftregistration3d keeps
#   complex128 in both modes -- fast mode lowers pixels, never the correlation
#   that decides a shift (the measurement behind that rule is in
#   cpstab/precision.py). The uint16 stage described in the next bullet is
#   UNCHANGED: the raw chunk is still read, cropped, imresize'd and warped as
#   uint16, so MATLAB's re-quantization points survive in both modes -- fast
#   mode only lowers the class those ops compute IN (matlab_compat's
#   work_dtype_for). Everything this function SAVES stays float64:
#   RS/CS/ZS/RS_chunk/CS_chunk/ZS_chunk and the scaled tforms are shift
#   bookkeeping, deliberately excluded from the fast path so the .dftshifts
#   payload has one precision regardless of how it was produced.
# * Dtype chain (review F2): raw chunk stays in its native class (uint16)
#   through crop/reshape/imresize/optotune-warp/DFT_rect, mirroring MATLAB
#   where imresize and imwarp/imtranslate re-quantize integer classes
#   (round half away from zero + saturate). Those two re-quantization
#   points measurably change the estimated RS/CS/ZS at realistic SNR; from
#   chunk_reg0 (double zeros() fill) onward everything is float64 exactly
#   as in MATLAB. VolumeSource-backed reads are cast back to the file's
#   native integer class (lossless: get_volume does no arithmetic).
# * MATLAB reshape is column-major: reshape([Y,X,F] -> Y,X,Z,T) fills z
#   fastest (f = t*Nz + z). Ported as .reshape(Y, X, T, Z).transpose(0,1,3,2).
#   This was the single most dangerous silent-transposition spot.
# * `Nt` stays a float when derived (nframes/Nz, L24) exactly like MATLAB;
#   only Nz*floor(Nt/Nchunks) is integerized. Frames beyond
#   Nchunks*chunkframes are ignored, as in MATLAB.
# * ref2 is Y x X x Z x 1 in MATLAB, which IS 3-D there (trailing singletons
#   vanish), so fftn(ref2) is a 3-D FFT. _as_volume_3d() reproduces that for a
#   define_reference that returns either 3-D or 4-D-with-singleton.
# * fftn(ref2) is hoisted out of the L72-75 loop (MATLAB recomputes the same
#   FFT every iteration); identical computation, identical result.
# * imresize 'nearest' stretch (L127-129): MATLAB's nearest mapping
#   u = x/scale + 0.5*(1-1/scale), index floor(u+0.5) via the box kernel --
#   reproduced by matlab_compat.matlab_imresize with an explicit output size.
#   For N = Nchunks*T this lands exactly on chunk boundaries (no ties occur:
#   u can never be an exact half-integer there).
# * ZS is saved with MATLAB's (1, N) row-vector shape (ZS{chunk} = ZS1'),
#   RS/CS as (Nz, N), ZS_chunk as (1, N) -- MakeSBXall's reader can use
#   MATLAB-identical indexing.
# * tforms_optotune_full is stored as an (Nz, 3, 3) stack of affine2d T
#   matrices (row-vector convention, translations in [2, 0:2]) because .npz
#   cannot hold object arrays of transforms. The L135 translation scaling
#   T(3,1:2)*scale is applied to a deep copy, mirroring MATLAB affine2d value
#   semantics (the caller's tforms_optotune is left untouched).
# * Persisted keys (deliberate, per instructions): only what MakeSBXall reads
#   (CODEMAP section 5 row F): RS/CS/ZS, RS_chunk/CS_chunk/ZS_chunk,
#   tforms_optotune_full. ref_all + intermediate_shifts + scale (unread dead
#   weight, the .dftshifts bloat) are only written with save_debug=True;
#   intermediate_shifts is flattened to 'intermediate_shifts_<field>' keys
#   because np.savez cannot store a struct/dict.
# * `planescorr` is parsed but never used -- true in the MATLAB source too
#   (appears only at L10); kept for signature parity.
# * `optotune` truthiness: MATLAB's `if p.optotune` on the default char 'true'
#   is always true, making the else branch (with its typo) unreachable; port
#   fixes the typo and gives 'false'/False a working meaning (DIVERGENCE
#   documented at _matlab_truthy; identical behaviour on the default path).
# * Identity-transform fast path in apply_optotune_warp is numerically exact:
#   bilinear resampling at unshifted integer grid points returns the original
#   samples; MATLAB imwarp with eye(3) produces the same values.
# * parfor_progressbar (GUI) replaced with stderr prints; MATLAB parfor loops
#   run serially here (results are order-independent: each chunk/volume is
#   independent).
# * fdir = fileparts(path) at L17 is computed and never used in MATLAB;
#   omitted.
